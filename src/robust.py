"""Robustness checks for H2. Run after claims.py, from the folder holding its inputs.

Usage: python robust.py 300_ieee 1800

Within one load path, total load changes linearly with path position, and so does depth.
Episode fixed effects therefore separate them only through curvature. Checks R3, R4 and R5
break that link, so read them first when asking whether H2 is more than a load effect.

R1 same regression, clustered by path
R2 alternative log floors, plus linear and any-violation outcomes
R3 constraint fixed effects across paths, with load control: same constraint, similar load, different depth
R4 placebo: always-active constraints given the same pseudo-activation pattern
R5 random test set, no paths: violation against the constraint's multiplier, with load control
R6 slope by constraint type
R7 under-bending: do steeper pre-activation approaches produce steeper post-activation growth?
All checks here are exploratory. They do not change the locked go/no-go decision.
"""
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr

from common import PD, PMAX, PMIN, QMAX, QMIN, RATE_A, VMAX, VMIN, load_case

name, ntrain = sys.argv[1], int(sys.argv[2])
case = load_case(os.path.join(os.environ.get("PGLIB_DIR", "pglib-opf"), f"pglib_opf_case{name}.m"))
res = [r for r in pickle.load(open(f"data_{name}.pkl", "rb")) if r and r["ok"]]
rand = [r for r in res if r["kind"] == "rand"]
path = [r for r in res if r["kind"] == "path"]
train, test = rand[:ntrain], rand[ntrain:]
all_s = test + path
n = len(test)
ev = pickle.load(open(f"eval_{name}.pkl", "rb"))
gen, bus, branch = case["gen"], case["bus"], case["branch"]
nb, ng, nl = bus.shape[0], gen.shape[0], branch.shape[0]
ids = {int(b): i for i, b in enumerate(bus[:, 0])}
gb = np.array([ids[int(x)] for x in gen[:, 0]])
typ = np.array(["Pmax"] * ng + ["Pmin"] * ng + ["Qmax"] * ng + ["Qmin"] * ng + ["Vmax"] * nb + ["Vmin"] * nb + ["Sf"] * nl + ["St"] * nl)
group = {"Pmax": "P", "Pmin": "P", "Qmax": "Q", "Qmin": "Q", "Vmax": "V", "Vmin": "V", "Sf": "S", "St": "S"}
slack_g = np.where(bus[gb, 1] == 3)[0]
clamped = np.zeros(typ.size, bool)
clamped[: 2 * ng] = True
clamped[slack_g] = False
clamped[ng + slack_g] = False
clamped[4 * ng + gb] = True
clamped[4 * ng + nb + gb] = True
active = lambda S: np.array([(np.abs(r["mu"]) > 1e-4).astype(np.int8) for r in S])
a_train, a_all = active(train), active(all_s)
freq = a_train.mean(0)
switching = (freq > 0) & (freq < 1) & ~clamped
always = (freq == 1) & ~clamped
load = np.array([r["pd"].sum() for r in all_s]) / bus[:, PD].sum()
mu_all = np.abs(np.array([r["mu"] for r in all_s]))
slack_all = np.array([r["slk"] for r in all_s])
pid = np.array([r["pid"] for r in path])
step = np.array([r["t"] for r in path])
a_path = a_all[n:]
n_points = int(step.max()) + 1
paths = {}
for k in np.unique(pid):
    idx = np.where(pid == k)[0]
    idx = idx[np.argsort(step[idx])]
    if len(idx) == n_points:
        paths[int(k)] = idx


def violations(models):
    arrays = []
    for m in models:
        v = np.full((len(all_s), typ.size), np.nan)
        for i, x in enumerate(ev[m]["rec"]):
            if x is not None:
                v[i] = -x[0]
        arrays.append(v)
    return np.nanmean(arrays, axis=0)


def episodes(constraints):
    """Single-crossing episodes: (path, constraint, crossing index, ends active)."""
    out = []
    for k, idx in paths.items():
        for c in np.where(constraints)[0]:
            a = a_path[idx, c]
            ch = np.where(np.diff(a) != 0)[0]
            if len(ch) == 1:
                out.append((k, int(c), int(ch[0]), bool(a[-1] == 1)))
    return out


def demean(y, keys):
    y = y.astype(float).copy()
    for u in np.unique(keys):
        s = keys == u
        y[s] -= y[s].mean()
    return y


def fit(y, x_cols, fixed, cluster):
    x = np.column_stack([demean(c, fixed) for c in x_cols])
    r = sm.OLS(demean(y, fixed), x).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    b, se = float(r.params[0]), float(r.bse[0])
    return dict(slope=round(b, 4), ci=[round(b - 1.96 * se, 4), round(b + 1.96 * se, 4)], p=float(f"{r.pvalues[0]:.3g}"),
                n=int(len(y)), clusters=int(np.unique(cluster).size))


def active_rows(v, eps):
    rows = []
    for e, (k, c, b, ends_active) in enumerate(eps):
        idx = paths[k]
        for j in range(n_points):
            if a_path[idx[j], c] == 1:
                depth = (j - b - 1) if ends_active else (b - j)
                rows.append((k, c, e, depth, v[n + idx[j], c], load[n + idx[j]], mu_all[n + idx[j], c]))
    rows = np.array(rows)
    return rows[np.isfinite(rows[:, 4])] if len(rows) else rows


def analyze(models):
    v = violations(models)
    eps = episodes(switching)
    R = active_rows(v, eps)
    k_, c_, e_, d_, y_, l_, m_ = R.T
    logd = np.log1p(d_)
    out = {}
    logy = np.log10(y_ + 1e-4)
    out["R1_path_clustered"] = fit(logy, [logd, l_], e_, k_)
    out["R2_floor_1e-5"] = fit(np.log10(y_ + 1e-5), [logd, l_], e_, k_)
    out["R2_floor_1e-3"] = fit(np.log10(y_ + 1e-3), [logd, l_], e_, k_)
    out["R2_linear_pct_of_range"] = fit(100 * y_, [logd, l_], e_, k_)
    out["R2_any_violation_share"] = fit((y_ > 1e-3).astype(float), [logd, l_], e_, k_)
    out["R3_across_paths_constraint_fe"] = fit(logy, [logd, l_], c_, k_)
    near, deep = y_[d_ <= 2], y_[d_ >= 13]
    out["raw_near_deep_pct"] = [round(100 * float(near.mean()), 3), round(100 * float(deep.mean()), 3)] if len(deep) else None

    # R4: real episode timing applied to always-active constraints on the same path.
    rows = []
    placebo_ids = np.where(always)[0]
    timing = sorted({(k, b, ea) for k, _, b, ea in eps})
    for t_id, (k, b, ea) in enumerate(timing):
        idx = paths[k]
        for c in placebo_ids:
            for j in range(n_points):
                depth = (j - b - 1) if ea else (b - j)
                if depth >= 0 and np.isfinite(v[n + idx[j], c]):
                    rows.append((k, t_id * 100000 + c, depth, v[n + idx[j], c], load[n + idx[j]]))
    if rows:
        P = np.array(rows)
        out["R4_placebo_always_active"] = fit(np.log10(P[:, 3] + 1e-4), [np.log1p(P[:, 2]), P[:, 4]], P[:, 1], P[:, 0])
        out["R4_placebo_constraints"] = int(placebo_ids.size)
    else:
        out["R4_placebo_always_active"] = None
        out["R4_placebo_constraints"] = int(placebo_ids.size)

    # R5: no paths. Same constraint across random test instances.
    rows = []
    for c in np.where(switching)[0]:
        s = np.where(a_all[:n, c] == 1)[0]
        if s.size >= 10:
            for i in s:
                if np.isfinite(v[i, c]):
                    rows.append((c, v[i, c], load[i], mu_all[i, c]))
    if rows:
        T = np.array(rows)
        out["R5_test_set_multiplier"] = fit(np.log10(T[:, 1] + 1e-4), [np.log10(T[:, 3] + 1e-6), T[:, 2]], T[:, 0], T[:, 0])
    else:
        out["R5_test_set_multiplier"] = None

    # R6: slope by constraint type, clustered by path.
    out["R6_by_type"] = {}
    groups = np.array([group[typ[int(c)]] for c in c_])
    for g in np.unique(groups):
        s = groups == g
        if np.unique(e_[s]).size >= 10:
            out["R6_by_type"][g] = {**fit(logy[s], [logd[s], l_[s]], e_[s], k_[s]), "episodes": int(np.unique(e_[s]).size)}

    # R7: approach steepness against post-activation growth, per episode.
    approach, growth = [], []
    for e, (k, c, b, ea) in enumerate(eps):
        idx = paths[k]
        inactive = np.array([j for j in range(n_points) if a_path[idx[j], c] == 0])
        act = R[R[:, 2] == e]
        if inactive.size >= 5 and act.shape[0] >= 5 and np.ptp(act[:, 3]) > 0:
            approach.append(abs(np.polyfit(inactive, slack_all[n + idx[inactive], c], 1)[0]))
            growth.append(np.polyfit(np.log1p(act[:, 3]), np.log10(act[:, 4] + 1e-4), 1)[0])
    if len(approach) >= 10:
        rho, p = spearmanr(approach, growth)
        out["R7_underbending"] = dict(rho=round(float(rho), 3), p=float(f"{p:.3g}"), episodes=len(approach),
                                      share_positive_growth=round(float(np.mean(np.array(growth) > 0)), 3))
    else:
        out["R7_underbending"] = None
    out["episodes"] = len(eps)
    return out


families = {"mlp256": [m for m in ev if m.startswith("mlp256")], "mlp64": [m for m in ev if m.startswith("mlp64")], "ridge": ["ridge"]}
results = {fam: analyze(models) for fam, models in families.items()}
with open(f"robust_{name}.json", "w") as fh:
    json.dump(results, fh, indent=1)
for fam, r in results.items():
    print(f"\n=== {fam}: {r['episodes']} episodes; raw near/deep % of range {r['raw_near_deep_pct']}")
    for key, val in r.items():
        if key.startswith("R"):
            print(key, val)
