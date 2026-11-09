"""Mechanism analyses for H2, with Figures A and B. Run from the folder holding data and eval files.

Usage: python mechanism.py 300_ieee 1800 --workers 3

1  Bending: derivative of signed slack just before and after activation, optimal versus learned.
   bend ratio r = (learned derivative jump) / (optimal derivative jump); under-bending means r < 1.
2  Depth versus load: same constraint and same 1%-of-load bin across paths, with and without
   load-pattern controls; within-episode fits with a quadratic load trend.
3  AC sensitivity: linearized violation e = S(x*) du, split into sensitivity size, control-error
   size and alignment, each regressed on depth.
8  Seed-level slopes and path-cluster bootstrap intervals for the depth slope.
All analyses here are exploratory unless docs/go_no_go_300bus.md says otherwise.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr

import recovery
from common import GST, PD, PMAX, PMIN, QD, QMAX, QMIN, VMAX, VMIN, load_case
from sensitivity import Sensitivity

FAMILIES = ["mlp64", "mlp256", "ridge", "fsmlp", "gnn", "gnnfs", "pifs"]
LABELS = {"mlp64": "MLP (64)", "mlp256": "MLP (256,256)", "ridge": "Ridge", "fsmlp": "Full-state MLP",
          "gnn": "GNN", "gnnfs": "Full-state GNN", "pifs": "Physics-tuned MLP"}
TAU = 1e-4


def load_evals(name, wanted):
    files = [f"eval_{name}.pkl"] + sorted(f for f in glob.glob(f"eval_{name}_*.pkl") if f != f"eval_{name}.pkl")
    models = {}
    for path in files:
        for m, e in pickle.load(open(path, "rb")).items():
            if m.split("_s")[0] in wanted and m not in models:
                models[m] = e
    return models


def demean(y, keys):
    y = np.asarray(y, float).copy()
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    sums = np.add.reduceat(y[order], starts)
    counts = np.diff(np.r_[starts, k.size])
    means = np.repeat(sums / counts, counts)
    out = np.empty_like(y)
    out[order] = y[order] - means
    return out


def fit(y, cols, fixed, cluster):
    x = np.column_stack([demean(c, fixed) for c in cols])
    keep = np.abs(x).sum(1) > 0
    r = sm.OLS(demean(y, fixed)[keep], x[keep]).fit(cov_type="cluster", cov_kwds={"groups": cluster[keep]})
    b, se = float(r.params[0]), float(r.bse[0])
    return dict(slope=round(b, 4), ci=[round(b - 1.96 * se, 4), round(b + 1.96 * se, 4)], p=float(f"{r.pvalues[0]:.3g}"),
                n=int(keep.sum()), clusters=int(np.unique(cluster[keep]).size))


def boot_slope(y, d, load, fixed, cluster, reps, rng):
    """Path-cluster bootstrap of the depth slope, using per-path sufficient statistics."""
    x = np.column_stack([demean(d, fixed), demean(load, fixed)])
    yt = demean(y, fixed)
    paths = np.unique(cluster)
    xx = np.array([x[cluster == p].T @ x[cluster == p] for p in paths])
    xy = np.array([x[cluster == p].T @ yt[cluster == p] for p in paths])
    draws = []
    for _ in range(reps):
        pick = rng.integers(0, paths.size, paths.size)
        try:
            draws.append(np.linalg.solve(xx[pick].sum(0), xy[pick].sum(0))[0])
        except np.linalg.LinAlgError:
            pass
    return [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 or np.ptp(x) <= 0:
        return np.nan
    try:
        return float(np.polyfit(x, y, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("ntrain", type=int)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--near-points", type=int, default=8, help="points on each side used for bend derivatives")
    ap.add_argument("--pglib-dir", default=os.environ.get("PGLIB_DIR", "pglib-opf"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    name = args.case
    case_path = os.path.join(args.pglib_dir, f"pglib_opf_case{name}.m")
    case = load_case(case_path)
    res = [r for r in pickle.load(open(f"data_{name}.pkl", "rb")) if r and r["ok"]]
    rand = [r for r in res if r["kind"] == "rand"]
    path = [r for r in res if r["kind"] == "path"]
    train, test = rand[: args.ntrain], rand[args.ntrain:]
    n = len(test)
    all_s = test + path
    models = load_evals(name, args.families)
    fams = [f for f in args.families if any(m.split("_s")[0] == f for m in models)]
    print("families:", fams, flush=True)

    gen, bus, branch = case["gen"], case["bus"], case["branch"]
    nb, ng, nl = bus.shape[0], gen.shape[0], branch.shape[0]
    base = float(case["baseMVA"])
    ids = {int(b): i for i, b in enumerate(bus[:, 0])}
    gb = np.array([ids[int(x)] for x in gen[:, 0]])
    typ = np.array(["Pmax"] * ng + ["Pmin"] * ng + ["Qmax"] * ng + ["Qmin"] * ng + ["Vmax"] * nb + ["Vmin"] * nb + ["Sf"] * nl + ["St"] * nl)
    grp = np.array([{"P": "P", "Q": "Q", "V": "V", "S": "S"}[t[0]] for t in typ])
    slack_g = np.where(bus[gb, 1] == 3)[0]
    clamped = np.zeros(typ.size, bool)
    clamped[: 2 * ng] = True
    clamped[slack_g] = clamped[ng + slack_g] = False
    clamped[4 * ng + gb] = clamped[4 * ng + nb + gb] = True
    act = lambda S: np.array([(np.abs(r["mu"]) > TAU).astype(np.int8) for r in S])
    freq = act(train).mean(0)
    switching = (freq > 0) & (freq < 1) & ~clamped
    a_path = act(path)
    pid = np.array([r["pid"] for r in path])
    step = np.array([r["t"] for r in path])
    npts = int(step.max()) + 1
    paths = {int(k): np.where(pid == k)[0][np.argsort(step[pid == k])] for k in np.unique(pid)}
    paths = {k: v for k, v in paths.items() if v.size == npts}
    x_path = np.array([np.r_[r["pd"], r["qd"]] for r in path])
    x_norm = np.linalg.norm(np.r_[bus[:, PD], bus[:, QD]])
    load = np.array([r["pd"].sum() for r in path]) / bus[:, PD].sum()
    slk_opt = np.array([r["slk"] for r in path])

    episodes = []
    for k, idx in paths.items():
        for c in np.where(switching)[0]:
            a = a_path[idx, c]
            ch = np.flatnonzero(np.diff(a))
            if ch.size == 1:
                b = int(ch[0])
                mid = 0.5 * (x_path[idx[b]] + x_path[idx[b + 1]])
                dist = 100 * np.linalg.norm(x_path[idx] - mid, axis=1) / x_norm
                dist = np.where(a == 1, dist, -dist)  # negative on the inactive side
                depth = np.where(a == 1, np.where(a[-1] == 1, np.arange(npts) - b - 1, b - np.arange(npts)), -1)
                episodes.append(dict(k=k, c=int(c), idx=idx, active=a.astype(bool), dist=dist, depth=depth))
    print(f"episodes: {len(episodes)}", flush=True)

    def violation(m):
        v = np.full((len(all_s), typ.size), np.nan)
        for i, x in enumerate(models[m]["rec"]):
            if x is not None:
                v[i] = -x[0]
        return v[n:]

    seeds = {f: sorted(m for m in models if m.split("_s")[0] == f) for f in fams}
    out = dict(case=name, episodes=len(episodes), families={})

    # Active-side rows shared by sections 2 and 8.
    rows = []
    for e_id, ep in enumerate(episodes):
        for j in np.flatnonzero(ep["active"]):
            rows.append((ep["k"], ep["c"], e_id, ep["depth"][j], ep["idx"][j]))
    rows = np.array(rows)
    k_, c_, e_, d_, i_ = rows.T
    logd, load_r = np.log1p(d_), load[i_]
    xs = x_path - x_path.mean(0)
    pcs = xs @ np.linalg.svd(xs, full_matrices=False)[2][:5].T  # load-pattern components
    lbin = np.floor(load_r * 100).astype(int)
    cbin = c_ * 1000 + lbin

    for fam in fams:
        rec = {}
        v_seed = {m: violation(m) for m in seeds[fam]}
        v = np.nanmean(list(v_seed.values()), axis=0)
        y = v[i_, c_]
        ok = np.isfinite(y)
        logy = np.log10(y + 1e-4)
        r = lambda yy, cols, fe: fit(yy[ok], [c[ok] for c in cols], fe[ok], k_[ok])
        rec["depth_slope"] = r(logy, [logd, load_r], e_)
        rec["boot_ci_path"] = boot_slope(logy[ok], logd[ok], load_r[ok], e_[ok], k_[ok], args.boot, rng)
        rec["seed_slopes"] = {m: fit(np.log10(v_seed[m][i_, c_] + 1e-4)[ok], [logd[ok], load_r[ok]], e_[ok], k_[ok])["slope"] for m in seeds[fam]}
        sv = np.array(list(rec["seed_slopes"].values()))
        rec["seed_mean_sd"] = [round(float(sv.mean()), 4), round(float(sv.std(ddof=1)), 4) if sv.size > 1 else None]
        # Section 2: separate depth from load across paths.
        rec["same_constraint_load_bin"] = r(logy, [logd], cbin)
        rec["same_constraint_load_bin_pcs"] = r(logy, [logd] + [pcs[i_, j] for j in range(5)], cbin)
        rec["episode_quadratic_load"] = r(logy, [logd, load_r, load_r ** 2], e_)
        rec["bins_with_depth_contrast"] = int(sum(np.ptp(d_[cbin == b]) > 0 for b in np.unique(cbin)))
        # Depth-binned violation rates, a constraint-aware metric on paths.
        viol = y > 1e-3
        rec["violation_rate_by_depth"] = {lab: round(float(np.nanmean(viol[(d_ >= lo) & (d_ <= hi) & ok])), 4)
                                          for lab, lo, hi in [("0-2", 0, 2), ("3-6", 3, 6), ("7-12", 7, 12), ("13+", 13, 99)]}
        out["families"][fam] = rec
        print(fam, json.dumps({k: rec[k] for k in ("depth_slope", "boot_ci_path", "seed_mean_sd", "same_constraint_load_bin_pcs")}), flush=True)

    # Sections 1 and 3 need signed recovered quantities, so rerun recovery for seed 0 of each family.
    reps = {f: seeds[f][0] for f in fams}
    path_rows = np.arange(n, len(all_s))
    with mp.Pool(args.workers, initializer=recovery.init, initargs=(case_path,)) as pool:
        opt = pool.map(recovery.pf_state, [(r["pd"], r["qd"], r["pg"], r["vg"]) for r in path], chunksize=8)
        learned = {}
        for f, m in reps.items():
            yh = models[m]["yh"][path_rows]
            learned[f] = pool.map(recovery.pf_state, [(r["pd"], r["qd"], y[:ng], y[ng:]) for r, y in zip(path, yh)], chunksize=8)
            print(f"recovered {f}: {sum(x is None for x in learned[f])} non-converged", flush=True)

    sens = Sensitivity(case)
    qr = np.maximum(gen[:, QMAX] - gen[:, QMIN], 1.0)
    vr = np.maximum(bus[:, VMAX] - bus[:, VMIN], 1e-3)
    scale = np.r_[np.maximum(gen[sens.p_gen, PMAX] - gen[sens.p_gen, PMIN], 1.0) / base, np.full(sens.v_gen.size, 0.01)]
    on_pos = {g: i for i, g in enumerate(sens.online)}
    pq_pos = {b: i for i, b in enumerate(sens.pq)}
    ep_cons = sorted({ep["c"] for ep in episodes})

    def row_of(c, sq, sv):
        """Sensitivity row (scaled controls), direction sign and range for constraint c, or None."""
        t = typ[c]
        if t in ("Qmax", "Qmin"):
            g = c - (2 * ng if t == "Qmax" else 3 * ng)
            if g not in on_pos:
                return None
            return sq[on_pos[g]] * base * scale, (1.0 if t == "Qmax" else -1.0), qr[g]
        if t in ("Vmax", "Vmin"):
            b = c - (4 * ng if t == "Vmax" else 4 * ng + nb)
            if b not in pq_pos:
                return None
            return sv[pq_pos[b]] * scale, (1.0 if t == "Vmax" else -1.0), vr[b]
        return None

    sens_norm = np.full((len(path), typ.size), np.nan)
    amp = {f: np.full((len(path), typ.size), np.nan) for f in fams}
    du_norm = {f: np.full(len(path), np.nan) for f in fams}
    lin = {f: [] for f in fams}
    for i, r in enumerate(path):
        if opt[i] is None:
            continue
        sq, sv = sens.compute(opt[i][3].astype(float), opt[i][4].astype(float))
        rows_c = {c: row_of(c, sq, sv) for c in ep_cons}
        for c, rc in rows_c.items():
            if rc is not None:
                sens_norm[i, c] = np.linalg.norm(rc[0]) / rc[2]
        for f in fams:
            y = models[reps[f]]["yh"][n + i]
            du = sens.control_delta(y[:ng], y[ng:], r["pg"], r["vg"]) / scale
            du_norm[f][i] = np.linalg.norm(du)
            for c, rc in rows_c.items():
                if rc is not None:
                    amp[f][i, c] = rc[1] * float(rc[0] @ du) / rc[2]
            if learned[f][i] is not None:
                pred = base * sq @ (du * scale)
                lin[f].append((pred, (learned[f][i][1] - opt[i][1])[sens.online]))

    fig_a, fig_b = {}, {}
    for f in fams:
        rec = out["families"][f]
        sl_hat = np.array([x[0] if x is not None else np.full(typ.size, np.nan, np.float32) for x in learned[f]])
        p, a_ = (np.concatenate([u[0] for u in lin[f]]), np.concatenate([u[1] for u in lin[f]])) if lin[f] else (np.array([0.0]), np.array([0.0]))
        rec["linearization_q"] = dict(corr=round(float(np.corrcoef(p, a_)[0, 1]), 4), slope=round(slope(p, a_), 4))
        # Section 1: bend ratios near activation.
        bends = []
        for e_id, ep in enumerate(episodes):
            c, idx, dist, active = ep["c"], ep["idx"], ep["dist"], ep["active"]
            if grp[c] not in ("Q", "V"):
                continue
            ins, outs = np.flatnonzero(~active), np.flatnonzero(active)
            ins = ins[np.argsort(-dist[ins])][: args.near_points]
            outs = outs[np.argsort(dist[outs])][: args.near_points]
            if ins.size < 4 or outs.size < 4:
                continue
            so, sh = slk_opt[idx, c], sl_hat[idx, c]
            if not (np.isfinite(sh[ins]).all() and np.isfinite(sh[outs]).all()):
                continue
            ob, oa = slope(dist[ins], so[ins]), slope(dist[outs], so[outs])
            hb, ha = slope(dist[ins], sh[ins]), slope(dist[outs], sh[outs])
            jump = oa - ob
            if not (ob < 0 and jump > 0):
                continue
            allout = np.flatnonzero(active)
            growth = slope(dist[allout], np.maximum(-sh[allout], 0)) if allout.size >= 4 else np.nan
            bends.append(dict(e=e_id, k=ep["k"], c=c, grp=grp[c], ratio=(ha - hb) / jump, track=hb / ob, kink=jump,
                              growth=growth, ob=ob, oa=oa, hb=hb, ha=ha))
        if bends:
            ratio = np.array([b["ratio"] for b in bends])
            ks = np.array([b["k"] for b in bends])
            med = []
            uk = np.unique(ks)
            for _ in range(args.boot):
                pick = rng.choice(uk, uk.size)
                med.append(np.median(np.concatenate([ratio[ks == p] for p in pick])))
            g = np.array([b["growth"] for b in bends])
            under = np.array([(1 - b["ratio"]) * b["kink"] for b in bends])
            fin = np.isfinite(g)
            rec["bend"] = dict(episodes=len(bends), by_group={q: int(sum(b["grp"] == q for b in bends)) for q in ("Q", "V")},
                               median_ratio=round(float(np.median(ratio)), 3),
                               median_ratio_ci_path=[round(float(np.percentile(med, 2.5)), 3), round(float(np.percentile(med, 97.5)), 3)],
                               share_ratio_below_1=round(float(np.mean(ratio < 1)), 3),
                               median_tracking=round(float(np.median([b["track"] for b in bends])), 3),
                               growth_vs_underbend_rho=round(float(spearmanr(under[fin], g[fin])[0]), 3),
                               growth_vs_kink_rho=round(float(spearmanr(np.array([b["kink"] for b in bends])[fin], g[fin])[0]), 3))
            for q in ("Q", "V"):
                rq = ratio[np.array([b["grp"] == q for b in bends])]
                if rq.size:
                    rec["bend"][f"median_ratio_{q}"] = round(float(np.median(rq)), 3)
        # Section 3: decomposition of the linearized violation on active-side points.
        rr = [(e_id, ep["k"], ep["c"], ep["depth"][j], ep["idx"][j]) for e_id, ep in enumerate(episodes) for j in np.flatnonzero(ep["active"])]
        rr = np.array(rr)
        ee, kk, cc, dd, ii = rr.T
        e_lin = amp[f][ii, cc]
        s_n = sens_norm[ii, cc]
        u_n = du_norm[f][ii]
        vv = violation(reps[f])[ii, cc]
        good = (e_lin > 1e-6) & np.isfinite(s_n) & np.isfinite(vv) & (u_n > 0)
        if good.sum() > 50:
            ld = np.log1p(dd[good])
            comp = {"linearized_violation": np.log10(e_lin[good]), "sensitivity": np.log10(s_n[good]),
                    "control_error": np.log10(u_n[good]), "alignment": np.log10(e_lin[good] / (s_n[good] * u_n[good]))}
            rec["decomposition"] = {k: fit(v_, [ld, load[ii[good]]], ee[good], kk[good]) for k, v_ in comp.items()}
            yv = np.log10(vv[good] + 1e-4)
            rec["decomposition"]["actual_violation"] = fit(yv, [ld, load[ii[good]]], ee[good], kk[good])
            rec["decomposition"]["actual_given_linearized"] = fit(yv, [ld, load[ii[good]], np.log10(e_lin[good])], ee[good], kk[good])
            rec["decomposition"]["share_points_positive_linearized"] = round(float(np.mean(e_lin[np.isfinite(e_lin)] > 0)), 3)
            # Unselected check: signed alignment cosine on all active-side points, linear outcome.
            allp = np.isfinite(e_lin) & np.isfinite(s_n) & (u_n > 0)
            cosine = e_lin[allp] / (s_n[allp] * u_n[allp])
            rec["decomposition"]["alignment_cosine_all_points"] = fit(cosine, [np.log1p(dd[allp]), load[ii[allp]]], ee[allp], kk[allp])
            rec["decomposition"]["sensitivity_all_points"] = fit(np.log10(s_n[allp]), [np.log1p(dd[allp]), load[ii[allp]]], ee[allp], kk[allp])
            rec["decomposition"]["control_error_all_points"] = fit(np.log10(u_n[allp]), [np.log1p(dd[allp]), load[ii[allp]]], ee[allp], kk[allp])
            fig_b[f] = dict(sens=s_n[good], viol=vv[good], depth=dd[good], grp=grp[cc[good]])
        if bends:
            fig_a[f] = dict(bends=bends, sl_hat=sl_hat, qg_hat=np.array([x[1] if x is not None else np.full(ng, np.nan, np.float32) for x in learned[f]]))
        print(f, json.dumps({k: rec.get(k) for k in ("linearization_q", "bend")}), flush=True)
        if "decomposition" in rec:
            print("  decomposition", json.dumps({k: (v_["slope"], v_["ci"]) if isinstance(v_, dict) else v_ for k, v_ in rec["decomposition"].items()}), flush=True)

    with open(f"mechanism_{name}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    try:
        with open(f"figdata_{name}.pkl", "wb") as fh:
            pickle.dump(dict(fams=fams, fig_a=fig_a, fig_b=fig_b, episodes=episodes, path=path, slk_opt=slk_opt,
                             typ=typ, gen=gen, ng=ng, out=out), fh)
    except Exception as exc:  # figure data is a convenience, never fatal
        print("figdata not saved:", exc, flush=True)
    make_figures(name, fams, fig_a, fig_b, episodes, path, slk_opt, typ, gen, ng, out)
    print("done", flush=True)


def make_figures(name, fams, fig_a, fig_b, episodes, path, slk_opt, typ, gen, ng, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"mlp256": "#1f5fbf", "mlp64": "#6f9be0", "fsmlp": "#c0392b", "gnn": "#2e8b57", "gnnfs": "#7fbf7f",
              "ridge": "#7f7f7f", "pifs": "#8e44ad"}
    lead = next((f for f in ("mlp256", "fsmlp", "mlp64") if f in fig_a), None)
    if lead:
        fig, ax = plt.subplots(2, 1, figsize=(3.45, 4.1))
        bends = [b for b in fig_a[lead]["bends"] if b["grp"] == "Q"] or fig_a[lead]["bends"]
        med = np.median([b["ratio"] for b in bends])
        rep = min(bends, key=lambda b: abs(b["ratio"] - med))  # rule: closest to the family median
        ep = episodes[rep["e"]]
        c, idx, dist = ep["c"], ep["idx"], ep["dist"]
        o = np.argsort(dist)
        if typ[c] in ("Qmax", "Qmin"):
            g = c - (2 * ng if typ[c] == "Qmax" else 3 * ng)
            q_opt = np.array([path[i]["qg"][g] for i in idx])
            q_hat = fig_a[lead]["qg_hat"][idx, g]
            lim = gen[g, QMAX] if typ[c] == "Qmax" else gen[g, QMIN]
            ax[0].axhline(lim, color="k", ls="--", lw=0.7)
            ax[0].plot(dist[o], q_opt[o], color="k", label="Optimal")
            ax[0].plot(dist[o], q_hat[o], color=colors[lead], label=f"{LABELS[lead]}, recovered")
            ax[0].set_ylabel("Generator Q (MVAr)")
            ax[0].set_title(f"(a) Example: {typ[c]}, gen. {g + 1}, bend ratio {rep['ratio']:.2f}", loc="left", fontsize=7)
        ax[0].axvline(0, color="#999", lw=0.6)
        ax[0].set_xlabel("Load distance from activation (% of base load)")
        ax[0].legend(frameon=False, loc="best")
        if not ax[0].get_title(loc="left"):
            ax[0].set_title("(a) Representative episode", loc="left", fontsize=7)
        # Panel (b): event-aligned mean signed slack, Q episodes.
        qeps = [e for e in episodes if typ[e["c"]] in ("Qmax", "Qmin")]
        alld = np.concatenate([e["dist"] for e in qeps])
        lim_d = np.percentile(np.abs(alld), 80)
        edges = np.linspace(-lim_d, lim_d, 17)
        ctr = 0.5 * (edges[1:] + edges[:-1])

        def binned(vals):
            d_all = np.concatenate([e["dist"] for e in qeps])
            v_all = np.concatenate(vals)
            b = np.digitize(d_all, edges) - 1
            return np.array([np.nanmean(v_all[(b == j) & np.isfinite(v_all)]) if np.any((b == j) & np.isfinite(v_all)) else np.nan for j in range(ctr.size)])
        ax[1].axhline(0, color="k", ls="--", lw=0.7)
        ax[1].plot(ctr, 100 * binned([slk_opt[e["idx"], e["c"]] for e in qeps]), color="k", label="Optimal")
        for f in [f for f in ("mlp256", "fsmlp", "gnn", "ridge") if f in fig_a]:
            ax[1].plot(ctr, 100 * binned([fig_a[f]["sl_hat"][e["idx"], e["c"]] for e in qeps]), color=colors[f], label=LABELS[f])
        ax[1].axvline(0, color="#999", lw=0.6)
        ax[1].set_xlabel("Load distance from activation (% of base load)")
        ax[1].set_ylabel("Slack to Q limit (% of range)")
        ax[1].legend(frameon=False, fontsize=6, ncol=2, loc="upper right")
        ax[1].set_title(f"(b) Mean over {len(qeps)} reactive-limit episodes", loc="left", fontsize=7)
        fig.tight_layout(h_pad=0.8)
        for ext in ("pdf", "png"):
            fig.savefig(f"fig_A_{name}.{ext}", dpi=300)
        plt.close(fig)
    lead_b = next((f for f in ("mlp256", "fsmlp", "mlp64") if f in fig_b), None)
    if lead_b:
        fig, ax = plt.subplots(2, 1, figsize=(3.45, 4.1))
        fb = fig_b[lead_b]
        sel = fb["grp"] == "Q"
        bins = [("0–2", 0, 2), ("3–6", 3, 6), ("7–12", 7, 12), ("13+", 13, 99)]
        shades = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
        for (lab, lo, hi), col in zip(bins, shades):
            s = sel & (fb["depth"] >= lo) & (fb["depth"] <= hi) & (fb["viol"] > 1e-5)
            ax[0].scatter(fb["sens"][s], 100 * fb["viol"][s], s=3, color=col, alpha=0.6, lw=0, label=f"{lab} steps")
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        from matplotlib.ticker import NullFormatter
        ax[0].xaxis.set_minor_formatter(NullFormatter())
        ax[0].set_xlabel("Q sensitivity to scaled controls (range per unit)")
        ax[0].set_ylabel("Recovered Q violation (% of range)")
        ax[0].legend(frameon=False, title="Depth", fontsize=6, title_fontsize=6, markerscale=3)
        ax[0].set_title(f"(a) {LABELS[lead_b]}, reactive limits", loc="left", fontsize=7)
        comps = [("actual_violation", "Actual\nviolation"), ("linearized_violation", "Linearized\nviolation"),
                 ("sensitivity", "Sensitivity"), ("control_error", "Control\nerror"), ("alignment", "Alignment")]
        show = [f for f in ("mlp256", "fsmlp", "gnn", "ridge") if "decomposition" in out["families"].get(f, {})]
        w = 0.8 / max(len(show), 1)
        for j, f in enumerate(show):
            dec = out["families"][f]["decomposition"]
            vals = [dec[k]["slope"] for k, _ in comps]
            err = np.array([[dec[k]["slope"] - dec[k]["ci"][0], dec[k]["ci"][1] - dec[k]["slope"]] for k, _ in comps]).T
            ax[1].bar(np.arange(len(comps)) + (j - (len(show) - 1) / 2) * w, vals, w, yerr=err, color=colors[f], label=LABELS[f],
                      error_kw=dict(lw=0.6, capsize=1.5))
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_xticks(np.arange(len(comps)), [lab for _, lab in comps], fontsize=6)
        ax[1].set_ylabel("Depth slope (log$_{10}$ per log step)")
        ax[1].legend(frameon=False, fontsize=6)
        ax[1].set_title("(b) What grows with depth", loc="left", fontsize=7)
        fig.tight_layout(h_pad=0.8)
        for ext in ("pdf", "png"):
            fig.savefig(f"fig_B_{name}.{ext}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--replot":
        with open(f"figdata_{sys.argv[2]}.pkl", "rb") as fh:
            d = pickle.load(fh)
        make_figures(sys.argv[2], **d)
    else:
        main()
