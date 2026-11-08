"""Constraint-aware feasibility, failure analysis and selective re-solve (Figure C).

Usage, from the folder holding data and eval files:
    python triage.py 300_ieee 1800 --workers 3 --time-sample 30 --retry 0

4  Hard feasibility after recovery, and violation rates on constraints binding versus not binding.
5  Where AC-OPF solves and recoveries fail, by load level and active-set size.
6  Selective re-solve: flag a share of test instances by a score available at deployment, re-solve
   those, and accept the rest. Re-solved instances take the stored optimal solution; a timed sample
   of warm-started re-solves gives the cost and success rate. Scores: predicted binding intensity
   from load (active-set aware), ensemble disagreement, total load, random, and an oracle bound.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import pickle
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import ipopt_backend
import recovery
from common import (GST, OPF_OPT, PD, PG, PMAX, PMIN, QD, QG, QMAX, QMIN, VA, VM, VMAX, VMIN, load_case)
from mechanism import FAMILIES, LABELS, load_evals
from pypower.api import runopf

TAU = 1e-4
BUDGETS = (0.1, 0.2, 0.3)
_case = None


def _init(case_path, max_iter):
    global _case
    _case = load_case(case_path)
    recovery.init(case_path)
    ipopt_backend.IPOPT_OPTIONS["max_iter"] = max_iter
    ipopt_backend.install()


def _timed_pf(args):
    t0 = time.perf_counter()
    out = recovery.pf_state(args)
    return out, time.perf_counter() - t0


def _resolve(args):
    """Solve AC-OPF; warm start from a recovered point when one is given."""
    pd, qd, warm = args
    c = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in _case.items()}
    c["bus"][:, PD], c["bus"][:, QD] = pd, qd
    if warm is not None:
        pg, qg, vm, va = warm
        c["gen"][:, PG], c["gen"][:, QG], c["bus"][:, VM], c["bus"][:, VA] = pg, qg, vm, va
    t0 = time.perf_counter()
    try:
        r = runopf(c, OPF_OPT)
        ok, f = bool(r["success"]), float(r["f"])
    except Exception:
        ok, f = False, np.nan
    return ok, time.perf_counter() - t0, f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("ntrain", type=int)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--time-sample", type=int, default=30, help="test instances timed for recovery and re-solve")
    ap.add_argument("--retry", type=int, default=0, help="failed random AC-OPF attempts to retry with 1000 iterations")
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--pglib-dir", default=os.environ.get("PGLIB_DIR", "pglib-opf"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    name = args.case
    case_path = os.path.join(args.pglib_dir, f"pglib_opf_case{name}.m")
    case = load_case(case_path)
    res = [r for r in pickle.load(open(f"data_{name}.pkl", "rb")) if r and r["ok"]]
    rand = [r for r in res if r["kind"] == "rand"]
    train, test = rand[: args.ntrain], rand[args.ntrain:]
    n = len(test)
    models = load_evals(name, args.families)
    fams = [f for f in args.families if any(m.split("_s")[0] == f for m in models)]
    seeds = {f: sorted(m for m in models if m.split("_s")[0] == f) for f in fams}
    gen, bus, branch = case["gen"], case["bus"], case["branch"]
    nb, ng, nl = bus.shape[0], gen.shape[0], branch.shape[0]
    ids = {int(b): i for i, b in enumerate(bus[:, 0])}
    gb = np.array([ids[int(x)] for x in gen[:, 0]])
    typ = np.array(["Pmax"] * ng + ["Pmin"] * ng + ["Qmax"] * ng + ["Qmin"] * ng + ["Vmax"] * nb + ["Vmin"] * nb + ["Sf"] * nl + ["St"] * nl)
    slack_g = np.where(bus[gb, 1] == 3)[0]
    clamped = np.zeros(typ.size, bool)
    clamped[: 2 * ng] = True
    clamped[slack_g] = clamped[ng + slack_g] = False
    clamped[4 * ng + gb] = clamped[4 * ng + nb + gb] = True
    vr = np.maximum(bus[:, VMAX] - bus[:, VMIN], 1e-3)
    qr = np.maximum(gen[:, QMAX] - gen[:, QMIN], 1.0)
    # Strict tolerance: 0.1% of range for P, Q and flows; 0.001 pu for voltage.
    tol = np.full(typ.size, 1e-3)
    tol[4 * ng: 4 * ng + 2 * nb] = 1e-3 / np.r_[vr, vr]
    # Practical tolerance, fixed before 300- and 200-bus scoring: 5 MVAr, 0.005 pu, 2% of range or rating.
    tol_p = np.full(typ.size, 0.02)
    tol_p[2 * ng: 4 * ng] = 5.0 / np.r_[qr, qr]
    tol_p[4 * ng: 4 * ng + 2 * nb] = 5e-3 / np.r_[vr, vr]
    qidx = np.where(np.isin(typ, ["Qmax", "Qmin"]))[0]
    q_mvar = np.r_[qr, qr]
    act_test = np.array([(np.abs(r["mu"]) > TAU) for r in test])
    act_train = np.array([(np.abs(r["mu"]) > TAU) for r in train])
    freq = act_train.mean(0)
    switching = (freq > 0) & (freq < 1) & ~clamped
    load_test = np.array([r["pd"].sum() for r in test]) / bus[:, PD].sum()
    out = dict(case=name, n_test=n, families={})

    # Section 4: constraint-aware feasibility metrics.
    viol, feasible, feasible_p, q_total = {}, {}, {}, {}
    for m, e in models.items():
        v = np.full((n, typ.size), np.nan)
        for i, x in enumerate(e["rec"][:n]):
            if x is not None:
                v[i] = -x[0]
        viol[m] = v
        conv = np.isfinite(v).all(1)
        feasible[m] = conv & np.all(np.nan_to_num(v) <= tol, axis=1)
        feasible_p[m] = conv & np.all(np.nan_to_num(v) <= tol_p, axis=1)
        q_total[m] = np.where(conv, np.nansum(v[:, qidx] * q_mvar, axis=1), np.nan)
    elig = ~clamped & np.isin(typ, ["Qmax", "Qmin", "Vmax", "Vmin", "Sf", "St", "Pmax", "Pmin"])
    for f in fams:
        rows = []
        for m in seeds[f]:
            v = viol[m]
            conv = np.isfinite(v).all(1)
            over = (np.nan_to_num(v) > tol) & conv[:, None]
            binding = act_test & elig
            free = ~act_test & elig
            qb = np.where(act_test[:, qidx], v[:, qidx] * q_mvar, 0)
            qn = np.where(~act_test[:, qidx], v[:, qidx] * q_mvar, 0)
            rows.append([100 * feasible[m].mean(), 100 * feasible_p[m].mean(), 100 * (~conv).mean(),
                         100 * over[binding].mean(), 100 * over[free].mean(),
                         np.nanmean(np.nansum(qb, 1)), np.nanmean(np.nansum(qn, 1)),
                         np.nanpercentile(np.nanmax(v[:, elig], axis=1), 95)])
        r = np.array(rows).mean(0)
        out["families"][f] = dict(feasibility=dict(strict_feasible_pct=round(r[0], 2), practical_feasible_pct=round(r[1], 2),
                                                   nonconverged_pct=round(r[2], 2),
                                                   binding_violation_rate_pct=round(r[3], 3), nonbinding_violation_rate_pct=round(r[4], 4),
                                                   q_overshoot_binding_mvar=round(r[5], 2), q_overshoot_nonbinding_mvar=round(r[6], 2),
                                                   p95_max_normalized_violation=round(r[7], 4)))

    # Section 5: failures.
    fail = {}
    ckpt = f"data_{name}.ckpt"
    if os.path.exists(ckpt):
        st = pickle.load(open(ckpt, "rb"))
        cfg = st["config"]
        lo, hi = cfg.get("load_min", 0.85), cfg.get("load_max", 1.15)
        b = np.array([np.random.default_rng([cfg["seed"], 0, r["t"]]).uniform(lo, hi) for r in st["rand"]])
        ok = np.array([r["ok"] for r in st["rand"]])
        edges = np.quantile(b, np.linspace(0, 1, 6))
        q = np.clip(np.digitize(b, edges[1:-1]), 0, 4)
        fail["opf_failure_rate_by_load_quintile_pct"] = [round(100 * float(1 - ok[q == j].mean()), 1) for j in range(5)]
        fail["opf_load_factor_quintile_edges"] = [round(float(x), 3) for x in edges]
        fail["opf_failures"] = int((~ok).sum())
        fail["opf_attempts"] = int(ok.size)
        if st.get("path_fail_at"):
            fail["path_failure_stop_points"] = st["path_fail_at"]
        failed_ids = [r["t"] for r in st["rand"] if not r["ok"]]
        fail["_failed_ids"] = failed_ids
        fail["_cfg"] = cfg
    n_act = act_test[:, elig].sum(1)
    for f in fams:
        rows = []
        for m in seeds[f]:
            nc = ~np.isfinite(viol[m]).all(1)
            rows.append(nc)
        nc = np.any(rows, axis=0)
        rec = dict(nonconverged_instances_any_seed=int(nc.sum()))
        if 0 < nc.sum() < n:
            rec["load_nonconverged_vs_converged"] = [round(float(np.median(load_test[nc])), 4), round(float(np.median(load_test[~nc])), 4),
                                                     float(f"{mannwhitneyu(load_test[nc], load_test[~nc]).pvalue:.3g}")]
            rec["active_count_nonconverged_vs_converged"] = [float(np.median(n_act[nc])), float(np.median(n_act[~nc])),
                                                             float(f"{mannwhitneyu(n_act[nc], n_act[~nc]).pvalue:.3g}")]
        out["families"][f]["recovery_failures"] = rec

    # Section 6: scores available at deployment.
    loads = np.where((bus[:, PD] != 0) | (bus[:, QD] != 0))[0]
    xs = lambda S: np.array([np.r_[r["pd"][loads], r["qd"][loads]] for r in S])
    target = lambda S: np.log1p(np.array([np.abs(r["mu"])[switching] for r in S]) / TAU)
    sc = StandardScaler().fit(xs(train))
    reg = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=500, random_state=0).fit(sc.transform(xs(train)), target(train))
    pred_bind = np.maximum(reg.predict(sc.transform(xs(test))), 0)
    true_bind = target(test)
    scores_common = {"Active-set aware": pred_bind.sum(1), "Total load": load_test}
    out["binding_score_vs_true_rho"] = round(float(spearmanr(pred_bind.sum(1), true_bind.sum(1))[0]), 3)
    grid = np.linspace(0, 1, 101)

    def curve(score, feas, qv):
        order = np.argsort(-score, kind="stable")
        hfr, qres = [], []
        for fr in grid:
            k = int(round(fr * n))
            fixed = np.zeros(n, bool)
            fixed[order[:k]] = True
            hfr.append(100 * np.mean(feas | fixed))
            qres.append(float(np.nanmean(np.where(fixed, 0, np.nan_to_num(qv, nan=np.nanmax(qv))))))
        return np.array(hfr), np.array(qres)

    curves = {}
    for f in fams:
        per = {}
        for m in seeds[f]:
            yh = models[m]["yh"][:n]
            sc_m = dict(scores_common)
            if len(seeds[f]) > 1:
                stack = np.array([models[s]["yh"][:n] for s in seeds[f]])
                rngp = np.maximum(gen[:, PMAX] - gen[:, PMIN], 1.0)
                sc_m["Ensemble disagreement"] = (stack[:, :, :ng].std(0) / rngp).mean(1) + (stack[:, :, ng:].std(0) / 0.01).mean(1)
            sc_m["Oracle"] = np.where(np.isfinite(q_total[m]), q_total[m], np.inf)
            for k, s in sc_m.items():
                per.setdefault(k, []).append(curve(s, feasible_p[m], q_total[m]))
            base = 100 * feasible_p[m].mean()
            per.setdefault("Random", []).append((base + grid * (100 - base), np.nanmean(q_total[m]) * (1 - grid)))
        curves[f] = {k: (np.mean([c[0] for c in v], 0), np.mean([c[1] for c in v], 0)) for k, v in per.items()}
        out["families"][f]["triage_practical_feasible_pct"] = {k: {f"{int(100 * bdg)}%": round(float(v[0][int(round(bdg * 100))]), 2) for bdg in BUDGETS} for k, v in curves[f].items()}
        out["families"][f]["triage_residual_q_mvar"] = {k: {f"{int(100 * bdg)}%": round(float(v[1][int(round(bdg * 100))]), 2) for bdg in (0.0,) + BUDGETS} for k, v in curves[f].items()}
        # Bootstrap over test instances for the 20% budget, first seed.
        m = seeds[f][0]
        diffs = {"vs_load": [], "vs_random": [], "vs_ensemble": []}
        s_aw = scores_common["Active-set aware"]
        ens = None
        if len(seeds[f]) > 1:
            stack = np.array([models[s]["yh"][:n] for s in seeds[f]])
            rngp = np.maximum(gen[:, PMAX] - gen[:, PMIN], 1.0)
            ens = (stack[:, :, :ng].std(0) / rngp).mean(1) + (stack[:, :, ng:].std(0) / 0.01).mean(1)

        qv = np.nan_to_num(q_total[m], nan=np.nanmax(q_total[m]))

        def resid_at(score, idx, budget=0.2):
            """Mean residual Q violation (MVAr) over idx after re-solving the top-scored share."""
            k = int(round(budget * idx.size))
            pos = np.argsort(-score[idx], kind="stable")[:k]
            keep = np.ones(idx.size, bool)
            keep[pos] = False
            return float(np.mean(np.where(keep, qv[idx], 0)))
        for _ in range(args.boot):
            idx = rng.integers(0, n, n)
            a = resid_at(s_aw, idx)
            diffs["vs_load"].append(a - resid_at(load_test, idx))
            diffs["vs_random"].append(a - 0.8 * float(np.mean(qv[idx])))
            if ens is not None:
                diffs["vs_ensemble"].append(a - resid_at(ens, idx))
        out["families"][f]["triage_20pct_residual_q_difference_ci"] = {k: [round(float(np.percentile(v, 2.5)), 2), round(float(np.mean(v)), 2), round(float(np.percentile(v, 97.5)), 2)]
                                                           for k, v in diffs.items() if v}

    # Costs: timed recovery and re-solves on a sample, plus optional failure retries.
    lead = next((f for f in ("fsmlp", "mlp256", "mlp64") if f in fams), fams[0])
    cost = {}
    with mp.Pool(args.workers, initializer=_init, initargs=(case_path, 100)) as pool:
        if args.time_sample:
            sample = rng.choice(n, min(args.time_sample, n), replace=False)
            yh = models[seeds[lead][0]]["yh"]
            pf = pool.map(_timed_pf, [(test[i]["pd"], test[i]["qd"], yh[i][:ng], yh[i][ng:]) for i in sample], chunksize=1)
            warm = [(test[i]["pd"], test[i]["qd"], None if p[0] is None else (p[0][2], p[0][1], p[0][3], p[0][4])) for i, p in zip(sample, pf)]
            w = pool.map(_resolve, warm, chunksize=1)
            c = pool.map(_resolve, [(test[i]["pd"], test[i]["qd"], None) for i in sample], chunksize=1)
            opt_f = np.array([test[i]["f"] for i in sample])
            cost = dict(model=seeds[lead][0], sample=int(sample.size),
                        recovery_pf_ms=round(1000 * float(np.median([p[1] for p in pf])), 1),
                        warm_resolve_ms=round(1000 * float(np.median([x[1] for x in w])), 1),
                        cold_resolve_ms=round(1000 * float(np.median([x[1] for x in c])), 1),
                        warm_success_pct=round(100 * float(np.mean([x[0] for x in w])), 1),
                        cold_success_pct=round(100 * float(np.mean([x[0] for x in c])), 1),
                        warm_cost_gap_pct=round(float(np.nanmedian([100 * (x[2] - o) / o for x, o in zip(w, opt_f) if x[0]])), 4))
        if args.retry and fail.get("_failed_ids"):
            cfg = fail["_cfg"]
            pd0, qd0 = bus[:, PD], bus[:, QD]
            picks = fail["_failed_ids"][: args.retry]
            jobs = []
            for a in picks:
                g = np.random.default_rng([cfg["seed"], 0, a])
                b = g.uniform(cfg.get("load_min", 0.85), cfg.get("load_max", 1.15))
                jobs.append((pd0 * b * g.uniform(0.9, 1.1, pd0.size), qd0 * b * g.uniform(0.9, 1.1, pd0.size), None))
    if args.retry and fail.get("_failed_ids"):
        with mp.Pool(args.workers, initializer=_init, initargs=(case_path, 1000)) as pool:
            rr = pool.map(_resolve, jobs, chunksize=1)
        fail["retry_1000_iterations"] = dict(tried=len(rr), solved=int(sum(x[0] for x in rr)),
                                             median_seconds=round(float(np.median([x[1] for x in rr])), 1))
    fail.pop("_failed_ids", None)
    fail.pop("_cfg", None)
    out["opf_failures"] = fail
    out["cost"] = cost
    with open(f"triage_{name}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "families"}, indent=0), flush=True)
    for f in fams:
        print(f, json.dumps(out["families"][f]), flush=True)
    figure(name, curves, lead, cost, grid)
    print("done", flush=True)


def figure(name, curves, lead, cost, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                         "axes.spines.top": False, "axes.spines.right": False})
    style = {"Active-set aware": dict(color="#1f5fbf", lw=1.6), "Ensemble disagreement": dict(color="#d98c1f"),
             "Total load": dict(color="#555555"), "Random": dict(color="#999999", ls="--"), "Oracle": dict(color="k", ls=":", lw=0.9)}
    if cost:
        x = cost["recovery_pf_ms"] + grid * cost["warm_resolve_ms"]
        xlab = "Mean compute per instance (ms)"
    else:
        x = 100 * grid
        xlab = "Instances re-solved (%)"
    fig, ax = plt.subplots(2, 1, figsize=(3.45, 3.9), sharex=True)
    c = curves[lead]
    for k in ("Oracle", "Active-set aware", "Ensemble disagreement", "Total load", "Random"):
        if k in c:
            ax[0].plot(x, c[k][0], label=k, **style[k])
            ax[1].plot(x, c[k][1], **style[k])
    for a, j in ((ax[0], 0), (ax[1], 1)):
        a.plot([x[0]], [c["Random"][j][0]], "o", ms=3.5, color="k")
        a.plot([x[-1]], [c["Random"][j][-1]], "s", ms=3.5, color="k")
    ax[0].annotate("uncorrected", (x[0], c["Random"][0][0]), xytext=(5, 5), textcoords="offset points", fontsize=6)
    ax[0].annotate("re-solve all", (x[-1], c["Random"][0][-1]), xytext=(-34, 6), textcoords="offset points", fontsize=6)
    ax[0].set_ylabel("Within practical tolerance (%)")
    ax[1].set_ylabel("Residual Q violation (MVAr)")
    ax[1].set_xlabel(xlab)
    ax[0].legend(frameon=False, fontsize=6, loc="lower right")
    ax[0].set_title(f"(a) {LABELS.get(lead, lead)}, feasibility", loc="left", fontsize=7)
    ax[1].set_title("(b) Remaining reactive violation", loc="left", fontsize=7)
    fig.tight_layout(h_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(f"fig_C_{name}.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
