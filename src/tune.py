"""Tune surrogates on a validation split of the training data only.

Usage, from the folder holding data_<case>.pkl:
    python tune.py 300_ieee 1800 --workers 3

Reads only the first <ntrain> random instances, so the test set and load paths stay
unseen. Hyperparameters are chosen by recovered reactive violation on validation data,
the selection most favorable to each family on the harm metric. Writes tuned_<case>.json.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from common import PD, PG, QD, QG, QMAX, QMIN, VG, VM, VMAX, VMIN, PMAX, PMIN, PF_OPT, load_case
from pypower.api import runpf

# Early stopping on these targets halted the pilot MLPs near 60 epochs, leaving them underfit.
# Each MLP family keeps one early-stopped baseline so that effect stays visible.
_FULL = {"early_stopping": False, "max_iter": 1500, "tol": 0.0}
_EARLY = {"early_stopping": True, "max_iter": 600, "n_iter_no_change": 30}


def _mlp_grid(hidden):
    grid = [{"hidden_layer_sizes": hidden, **_EARLY}]
    grid += [{"hidden_layer_sizes": hidden, "alpha": a, "learning_rate_init": lr, **_FULL}
             for a in (1e-5, 1e-3, 1e-1) for lr in (1e-3, 3e-4)]
    return grid


GRID = {
    "ridge": [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0, 100.0)],
    "mlp64": _mlp_grid((64,)),
    "mlp256": _mlp_grid((256, 256)),
}
NONCONVERGENCE_LIMIT = 0.01
_case = None
_gb = None


def _init_worker(case_path):
    global _case, _gb
    _case = load_case(case_path)
    ids = {int(b): i for i, b in enumerate(_case["bus"][:, 0])}
    _gb = np.array([ids[int(x)] for x in _case["gen"][:, 0]])


def _recover(args):
    """Return total reactive violation in MVAr after AC power flow, or None."""
    pd, qd, y = args
    case = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in _case.items()}
    ng = case["gen"].shape[0]
    case["bus"][:, PD], case["bus"][:, QD] = pd, qd
    case["gen"][:, PG], case["gen"][:, VG] = y[:ng], y[ng:]
    case["bus"][_gb, VM] = y[ng:]
    try:
        result, _ = runpf(case, PF_OPT)
    except Exception:
        return None
    if not result["success"]:
        return None
    gen = result["gen"]
    return float(np.maximum(gen[:, QG] - gen[:, QMAX], 0).sum() + np.maximum(gen[:, QMIN] - gen[:, QG], 0).sum())


def _build(family, params, seed):
    if family == "ridge":
        return Ridge(**params)
    return MLPRegressor(random_state=seed, **params)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("ntrain", type=int)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pglib-dir", default=os.environ.get("PGLIB_DIR", "pglib-opf"))
    args = ap.parse_args()

    case_path = os.path.join(args.pglib_dir, f"pglib_opf_case{args.case}.m")
    case = load_case(case_path)
    with open(f"data_{args.case}.pkl", "rb") as fh:
        rand = [r for r in pickle.load(fh) if r and r["ok"] and r["kind"] == "rand"][: args.ntrain]
    if len(rand) < args.ntrain:
        raise SystemExit(f"Only {len(rand)} training instances available")

    gen, bus = case["gen"], case["bus"]
    ids = {int(b): i for i, b in enumerate(bus[:, 0])}
    gb = np.array([ids[int(x)] for x in gen[:, 0]])
    ng = gen.shape[0]
    pr = np.maximum(gen[:, PMAX] - gen[:, PMIN], 1.0)
    lb = np.r_[gen[:, PMIN], bus[gb, VMIN]]
    ub = np.r_[gen[:, PMAX], bus[gb, VMAX]]
    loads = np.where((bus[:, PD] != 0) | (bus[:, QD] != 0))[0]
    X = np.array([np.r_[r["pd"][loads], r["qd"][loads]] for r in rand])
    Y = np.array([np.r_[r["pg"], r["vg"]] for r in rand])

    n_val = int(round(args.ntrain * args.val_fraction))
    fit, val = np.arange(args.ntrain - n_val), np.arange(args.ntrain - n_val, args.ntrain)
    xs, ys = StandardScaler().fit(X[fit]), StandardScaler().fit(Y[fit])
    x_fit, y_fit, x_val = xs.transform(X[fit]), ys.transform(Y[fit]), xs.transform(X[val])

    records, best = [], {}
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(case_path,)) as pool:
        for family, grid in GRID.items():
            for params in grid:
                t0 = time.time()
                model = _build(family, params, args.seed).fit(x_fit, y_fit)
                pred = np.clip(ys.inverse_transform(model.predict(x_val)), lb, ub)
                q = pool.map(_recover, [(rand[i]["pd"], rand[i]["qd"], y) for i, y in zip(val, pred)], chunksize=8)
                solved = [v for v in q if v is not None]
                rec = dict(family=family, params=params, val_q_mvar=float(np.mean(solved)) if solved else None,
                           nonconverged=float(1 - len(solved) / len(q)),
                           val_mae_pg_pct=float(100 * (np.abs(pred[:, :ng] - Y[val][:, :ng]) / pr).mean()),
                           val_mae_vg_pu=float(np.abs(pred[:, ng:] - Y[val][:, ng:]).mean()), seconds=round(time.time() - t0, 1))
                records.append(rec)
                print(json.dumps(rec), flush=True)
            eligible = [r for r in records if r["family"] == family and r["val_q_mvar"] is not None
                        and r["nonconverged"] <= NONCONVERGENCE_LIMIT]
            pool_ = eligible or [r for r in records if r["family"] == family and r["val_q_mvar"] is not None]
            best[family] = min(pool_, key=lambda r: r["val_q_mvar"])["params"]

    with open(f"tuned_{args.case}.json", "w") as fh:
        json.dump(dict(best=best, n_fit=len(fit), n_val=len(val), records=records), fh, indent=1)
    print("Selected:", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
