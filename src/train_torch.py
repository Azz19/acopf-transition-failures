"""Train the Lagrangian MLP and GNN surrogates, then recover by AC power flow.

Run from the folder holding data_<case>.pkl and eval_<case>.pkl:
    python train_torch.py 300_ieee 1800 --workers 3

Writes eval_<case>_torch.pkl, which also carries ridge, for `claims.py <case> <ntrain> torch`
and `robust.py <case> <ntrain> torch`. Settings are chosen on the last 15% of the training
data by recovered reactive violation, as in tune.py. Only non-convergence counts are shown
for test and path instances. Tensor shapes: B batch, N buses, G generators, E directed edges.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

import recovery
from common import PD, QD, PMAX, PMIN, QMAX, QMIN, VMAX, VMIN, RATE_A, load_case

GRID = {
    "lagmlp": [{"hidden": 256, "rho": r, "weight_decay": wd} for r in (1.0, 10.0, 100.0) for wd in (0.0, 1e-5)],
    "gnn": [{"hidden": h, "layers": k} for h in (64, 128) for k in (3, 5)],
}
EPOCHS = {"lagmlp": 400, "gnn": 300}
NONCONVERGENCE_LIMIT = 0.01
BATCH_SIZE = 64


class Grid:
    """Tensors for AC equations, with the branch model used by MATPOWER."""

    def __init__(self, case, device):
        bus, gen, br = case["bus"], case["gen"], case["branch"][case["branch"][:, 10] > 0]
        self.base = float(case["baseMVA"])
        self.nb, self.ng = bus.shape[0], gen.shape[0]
        ids = {int(b): i for i, b in enumerate(bus[:, 0])}
        self.gb_np = np.array([ids[int(x)] for x in gen[:, 0]])
        f = np.array([ids[int(x)] for x in br[:, 0]])
        t = np.array([ids[int(x)] for x in br[:, 1]])
        ys = 1.0 / (br[:, 2] + 1j * br[:, 3])
        tap = np.where(br[:, 8] == 0, 1.0, br[:, 8]) * np.exp(1j * np.deg2rad(br[:, 9]))
        ytt = ys + 0.5j * br[:, 4]
        yff = ytt / (tap * np.conj(tap))
        yft, ytf = -ys / np.conj(tap), -ys / tap
        ybus = np.zeros((self.nb, self.nb), dtype=complex)
        np.add.at(ybus, (f, f), yff)
        np.add.at(ybus, (t, t), ytt)
        np.add.at(ybus, (f, t), yft)
        np.add.at(ybus, (t, f), ytf)
        ybus[np.arange(self.nb), np.arange(self.nb)] += (bus[:, 4] + 1j * bus[:, 5]) / self.base
        tensor = lambda a, kind=torch.float32: torch.as_tensor(a, dtype=kind, device=device)
        self.ybus = tensor(ybus, torch.complex64)
        self.f, self.t = tensor(f, torch.long), tensor(t, torch.long)
        self.yff, self.yft = tensor(yff, torch.complex64), tensor(yft, torch.complex64)
        self.ytf, self.ytt = tensor(ytf, torch.complex64), tensor(ytt, torch.complex64)
        rate = br[:, 5] / self.base
        self.rated = tensor(rate > 0, torch.bool)
        self.rate = tensor(rate)
        self.gb = tensor(self.gb_np, torch.long)
        self.pmin, self.pmax = tensor(gen[:, PMIN] / self.base), tensor(gen[:, PMAX] / self.base)
        self.qmin, self.qmax = tensor(gen[:, QMIN] / self.base), tensor(gen[:, QMAX] / self.base)
        self.vmin, self.vmax = tensor(bus[:, VMIN]), tensor(bus[:, VMAX])

    def residuals(self, pd, qd, pg, qg, vm, va):
        """Return power balance residuals, (B, 2N), and thermal excess, (B, 2*rated), in per unit."""
        v = torch.polar(vm, va)  # (B, N)
        sbus = v * torch.conj(v @ self.ybus.T)
        inj_p = torch.zeros_like(vm).index_add(1, self.gb, pg) - pd
        inj_q = torch.zeros_like(vm).index_add(1, self.gb, qg) - qd
        balance = torch.cat([sbus.real - inj_p, sbus.imag - inj_q], dim=1)
        vf, vt = v[:, self.f], v[:, self.t]
        sf = vf * torch.conj(self.yff * vf + self.yft * vt)
        st = vt * torch.conj(self.ytf * vf + self.ytt * vt)
        flows = torch.cat([sf.abs()[:, self.rated] - self.rate[self.rated], st.abs()[:, self.rated] - self.rate[self.rated]], 1)
        return balance, torch.relu(flows)


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class GNN(nn.Module):
    """Edge-conditioned message passing with residual updates and generator readout."""

    def __init__(self, grid, node_static, gen_static, edge_static, hidden, layers):
        super().__init__()
        self.register_buffer("node_static", node_static)  # (N, Fs)
        self.register_buffer("gen_static", gen_static)  # (G, Fg)
        self.register_buffer("edge_static", edge_static)  # (E, Fe)
        f, t = grid.f, grid.t
        self.register_buffer("src", torch.cat([f, t]))
        self.register_buffer("dst", torch.cat([t, f]))
        self.register_buffer("gb", grid.gb)
        counts = torch.zeros(grid.nb, device=f.device).index_add(0, self.dst, torch.ones_like(self.dst, dtype=torch.float32))
        self.register_buffer("counts", counts.clamp_min(1.0))
        self.encode = nn.Linear(2 + node_static.shape[1], hidden)
        self.message = nn.ModuleList(nn.Sequential(nn.Linear(2 * hidden + edge_static.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers))
        self.update_ = nn.ModuleList(nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.readout = nn.Sequential(nn.Linear(hidden + gen_static.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        b = x.shape[0]  # x: (B, N, 2)
        h = self.encode(torch.cat([x, self.node_static.expand(b, -1, -1)], -1))  # (B, N, H)
        edge = self.edge_static.expand(b, -1, -1)  # (B, E, Fe)
        for message, update, norm in zip(self.message, self.update_, self.norm):
            m = message(torch.cat([h[:, self.src], h[:, self.dst], edge], -1))  # (B, E, H)
            agg = torch.zeros_like(h).index_add(1, self.dst, m) / self.counts[None, :, None]
            h = norm(h + update(torch.cat([h, agg], -1)))
        gen_h = torch.cat([h[:, self.gb], self.gen_static.expand(b, -1, -1)], -1)  # (B, G, H+Fg)
        out = self.readout(gen_h)  # (B, G, 2)
        return torch.cat([out[..., 0], out[..., 1]], 1)  # (B, 2G)


def zscore(a, floor=1e-9):
    return (a - a.mean(0)) / np.maximum(a.std(0), floor)


def train_lagmlp(cfg, seed, data, grid, device, epochs):
    torch.manual_seed(seed)
    tr = data["fit"]
    x, y = data["x_all"], data["y_full"]
    nb, ng = grid.nb, grid.ng
    mean, scale = y[tr].mean(0), np.maximum(y[tr].std(0), data["full_floor"])
    xm, xs = x[tr].mean(0), np.maximum(x[tr].std(0), 1e-6)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    xt, yt, mt, st = t((x - xm) / xs), t((y - mean) / scale), t(mean), t(scale)
    pd_t, qd_t = t(data["pd"] / grid.base), t(data["qd"] / grid.base)
    model = MLP(x.shape[1], y.shape[1], cfg["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    idx = torch.as_tensor(tr, device=device)
    lam = None
    for _ in range(epochs):
        total, count = None, 0
        for batch in idx[torch.randperm(len(idx), device=device)].split(BATCH_SIZE):
            pred = model(xt[batch])
            phys = pred * st + mt
            pg, qg = phys[:, :ng] / grid.base, phys[:, ng:2 * ng] / grid.base
            vm, va = phys[:, 2 * ng:2 * ng + nb], torch.deg2rad(phys[:, 2 * ng + nb:])
            balance, thermal = grid.residuals(pd_t[batch], qd_t[batch], pg, qg, vm, va)
            viol = torch.cat([balance.abs(), thermal, torch.relu(pg - grid.pmax), torch.relu(grid.pmin - pg),
                              torch.relu(qg - grid.qmax), torch.relu(grid.qmin - qg),
                              torch.relu(vm - grid.vmax), torch.relu(grid.vmin - vm)], 1)
            if lam is None:
                lam = torch.zeros(viol.shape[1], device=device)
            loss = nn.functional.mse_loss(pred, yt[batch]) + (lam * viol).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            part = viol.detach().sum(0)
            total = part if total is None else total + part
            count += len(batch)
        lam = (lam + cfg["rho"] * total / count).clamp_max(1e4)
        sched.step()
    model.eval()
    with torch.no_grad():
        out = (model(xt) * st + mt).cpu().numpy()
    return np.c_[out[:, :ng], out[:, 2 * ng:2 * ng + nb][:, grid.gb_np]]


def train_gnn(cfg, seed, data, grid, device, epochs):
    torch.manual_seed(seed)
    tr = data["fit"]
    y = data["y_short"]
    mean, scale = y[tr].mean(0), np.maximum(y[tr].std(0), data["short_floor"])
    pd, qd = data["pd"], data["qd"]
    pm, ps = pd[tr].mean(0), np.maximum(pd[tr].std(0), 1e-6)
    qm, qs = qd[tr].mean(0), np.maximum(qd[tr].std(0), 1e-6)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    xt = t(np.stack([(pd - pm) / ps, (qd - qm) / qs], -1))  # (S, N, 2)
    yt, mt, st = t((y - mean) / scale), t(mean), t(scale)
    model = GNN(grid, data["node_static"].to(device), data["gen_static"].to(device), data["edge_static"].to(device), cfg["hidden"], cfg["layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    idx = torch.as_tensor(tr, device=device)
    for _ in range(epochs):
        for batch in idx[torch.randperm(len(idx), device=device)].split(BATCH_SIZE):
            loss = nn.functional.mse_loss(model(xt[batch]), yt[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return torch.cat([model(b) for b in xt.split(256)]).mul(st).add(mt).cpu().numpy()


TRAINERS = {"lagmlp": train_lagmlp, "gnn": train_gnn}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("ntrain", type=int)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--families", nargs="+", default=["lagmlp", "gnn"], choices=list(GRID))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epoch-scale", type=float, default=1.0, help="for quick tests only")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--rho", type=float, nargs="+", help="Lagrangian penalty steps; 0 is a no-penalty diagnostic, never selected")
    ap.add_argument("--no-penalty", action="store_true", help="train the full-state MLP without penalties, labeled fsmlp")
    ap.add_argument("--tag", default="torch", help="results are written to eval_<case>_<tag>.pkl and tuned_<case>_<tag>.json")
    ap.add_argument("--pglib-dir", default=os.environ.get("PGLIB_DIR", "pglib-opf"))
    args = ap.parse_args()

    if args.no_penalty:
        args.rho = [0.0]
    if args.rho:
        GRID["lagmlp"] = [{"hidden": 256, "rho": r, "weight_decay": 1e-5} for r in args.rho]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    case_path = os.path.join(args.pglib_dir, f"pglib_opf_case{args.case}.m")
    case = load_case(case_path)
    grid = Grid(case, device)
    res = [r for r in pickle.load(open(f"data_{args.case}.pkl", "rb")) if r and r["ok"]]
    rand = [r for r in res if r["kind"] == "rand"]
    path = [r for r in res if r["kind"] == "path"]
    train, test = rand[: args.ntrain], rand[args.ntrain :]
    samples = train + test + path
    n_all, n_train, n_val = len(samples), args.ntrain, int(round(args.ntrain * args.val_fraction))
    fit, val = np.arange(n_train - n_val), np.arange(n_train - n_val, n_train)
    held_out = np.arange(n_train, n_all)
    bus, gen = case["bus"], case["gen"]
    ng, nb = gen.shape[0], bus.shape[0]
    gb = grid.gb_np
    lb = np.r_[gen[:, PMIN], bus[gb, VMIN]]
    ub = np.r_[gen[:, PMAX], bus[gb, VMAX]]
    qr = np.maximum(gen[:, QMAX] - gen[:, QMIN], 1.0)
    pd = np.array([r["pd"] for r in samples])
    qd = np.array([r["qd"] for r in samples])
    loads = np.where((bus[:, PD] != 0) | (bus[:, QD] != 0))[0]
    q_slice = slice(2 * ng, 4 * ng)

    with mp.Pool(args.workers, initializer=recovery.init, initargs=(case_path,)) as pool:
        va = pool.map(recovery.angles, [(r["pd"], r["qd"], r["pg"], r["vg"]) for r in train], chunksize=16)
        if any(a is None for a in va):
            raise SystemExit("Power flow failed on a training label, so angle targets are unavailable.")
        va = np.array(va)
        # Check the AC model against solved labels before training on it.
        tt = lambda a: torch.as_tensor(a[:8], dtype=torch.float32, device=device)
        balance, thermal = grid.residuals(tt(pd) / grid.base, tt(qd) / grid.base,
                                          tt(np.array([r["pg"] for r in train])) / grid.base, tt(np.array([r["qg"] for r in train])) / grid.base,
                                          tt(np.array([r["vm"] for r in train])), torch.deg2rad(tt(va)))
        print(f"label check: max balance residual {balance.abs().max().item():.2e} pu, max thermal excess {thermal.max().item():.2e} pu", flush=True)

        y_short = np.array([np.r_[r["pg"], r["vg"]] for r in train])
        y_short = np.vstack([y_short, np.zeros((n_all - n_train, 2 * ng))])
        y_full = np.array([np.r_[r["pg"], r["qg"], r["vm"], a] for r, a in zip(train, va)])
        y_full = np.vstack([y_full, np.zeros((n_all - n_train, y_full.shape[1]))])
        base = grid.base
        branch = case["branch"][case["branch"][:, 10] > 0]
        bus_gen = np.zeros(nb)
        np.add.at(bus_gen, gb, 1.0)
        node_static = np.c_[bus[:, VMIN], bus[:, VMAX], bus_gen > 0, bus[:, 1] == 3, bus[:, 4] / base, bus[:, 5] / base]
        gen_static = np.c_[gen[:, PMIN] / base, gen[:, PMAX] / base, gen[:, QMIN] / base, gen[:, QMAX] / base,
                           case["gencost"][:, 4] * base ** 2 * 1e-4, case["gencost"][:, 5] * base * 1e-4]
        fwd = np.c_[branch[:, 2], branch[:, 3], branch[:, 4], branch[:, 5] / base, np.ones(len(branch))]
        rev = fwd.copy()
        rev[:, 4] = -1.0
        data = dict(fit=fit, x_all=np.c_[pd[:, loads], qd[:, loads]], y_full=y_full, y_short=y_short, pd=pd, qd=qd,
                    full_floor=np.r_[np.full(ng, 0.1), np.full(ng, 0.1), np.full(nb, 1e-4), np.full(nb, 1e-2)],
                    short_floor=np.r_[np.full(ng, 0.1), np.full(ng, 1e-4)],
                    node_static=torch.as_tensor(np.nan_to_num(zscore(node_static)), dtype=torch.float32),
                    gen_static=torch.as_tensor(np.nan_to_num(zscore(gen_static)), dtype=torch.float32),
                    edge_static=torch.as_tensor(np.nan_to_num(np.vstack([fwd, rev])), dtype=torch.float32))

        def recover_all(pred, index):
            clipped = np.clip(pred, lb, ub)
            return pool.map(recovery.recover, [(pd[i], qd[i], clipped[i]) for i in index], chunksize=16), clipped

        output = {"ridge": pickle.load(open(f"eval_{args.case}.pkl", "rb"))["ridge"]}
        log = {}
        for family in args.families:
            epochs = max(2, int(EPOCHS[family] * args.epoch_scale))
            records = []
            for cfg in GRID[family]:
                t0 = time.time()
                pred = TRAINERS[family](cfg, 0, data, grid, device, epochs)
                rec, clipped = recover_all(pred, val)
                solved = [np.sum(-x[0][q_slice] * np.r_[qr, qr]) for x in rec if x is not None]
                row = dict(family=family, cfg=cfg, val_q_mvar=float(np.mean(solved)) if solved else None, nonconverged=1 - len(solved) / len(rec),
                           val_mae_pg_pct=float(100 * (np.abs(clipped[val, :ng] - y_short[val, :ng]) / np.maximum(gen[:, PMAX] - gen[:, PMIN], 1.0)).mean()),
                           val_mae_vg_pu=float(np.abs(clipped[val, ng:] - y_short[val, ng:]).mean()), seconds=round(time.time() - t0, 1))
                records.append(row)
                print(json.dumps(row), flush=True)
            usable = [r for r in records if r["val_q_mvar"] is not None and (args.no_penalty or r["cfg"].get("rho", 1) > 0)]
            eligible = [r for r in usable if r["nonconverged"] <= NONCONVERGENCE_LIMIT] or usable
            best = min(eligible, key=lambda r: r["val_q_mvar"])["cfg"]
            log[family] = dict(best=best, records=records)
            print("Selected:", family, json.dumps(best), flush=True)
            final = dict(data, fit=np.arange(n_train))
            for seed in range(args.seeds):
                t0 = time.time()
                pred = TRAINERS[family](best, seed + 1, final, grid, device, epochs)
                rec, clipped = recover_all(pred, held_out)
                label = "fsmlp" if args.no_penalty else family
                output[f"{label}_s{seed}"] = dict(yh=clipped[held_out].astype(np.float32), rec=rec)
                print(f"{label}_s{seed} seconds {time.time() - t0:.0f} nonconverged {sum(x is None for x in rec)}", flush=True)
                with open(f"eval_{args.case}_{args.tag}.pkl", "wb") as fh:
                    pickle.dump(output, fh)
        with open(f"tuned_{args.case}_{args.tag}.json", "w") as fh:
            json.dump(log, fh, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
