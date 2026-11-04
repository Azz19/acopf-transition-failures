"""Generate solved AC-OPF instances and complete load paths with Ipopt.

Output matches gen.py, so evalm.py and claims.py can read it unchanged. Paths with
any unsolved point are replaced, and those replacements are counted in the metadata.
Reruns resume from the checkpoint with the same arguments.

Example:
    python gen_ipopt.py 300_ieee --random 3000 --paths 50 --points 41 --workers 3
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import platform
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np

import ipopt_backend
from common import PD, PG, QD, QG, VG, VM, LAM_P, active_vector, load_case, slacks, solve

_case = None


def _init_worker(case_path):
    """Load the case and select Ipopt once in each worker process."""
    global _case
    _case = load_case(case_path)
    ipopt_backend.install()


def _draw(rng):
    """Pilot sampling: global U(0.85, 1.15) times per-load U(0.9, 1.1)."""
    pd0, qd0 = _case["bus"][:, PD], _case["bus"][:, QD]
    b = rng.uniform(0.85, 1.15)
    return pd0 * b * rng.uniform(0.9, 1.1, pd0.size), qd0 * b * rng.uniform(0.9, 1.1, pd0.size)


def _solve_record(kind, pid, t, pd, qd):
    try:
        r = solve(_case, pd, qd)
    except Exception as exc:
        return dict(kind=kind, pid=pid, t=t, ok=False, error=repr(exc))
    if not r["success"]:
        return dict(kind=kind, pid=pid, t=t, ok=False)
    act, mu = active_vector(r)
    return dict(kind=kind, pid=pid, t=t, ok=True, pd=pd, qd=qd,
                pg=r["gen"][:, PG], qg=r["gen"][:, QG], vg=r["gen"][:, VG],
                vm=r["bus"][:, VM], f=r["f"], act=act, mu=mu,
                slk=slacks(r, _case["baseMVA"]), lam=r["bus"][:, LAM_P])


def _random_task(args):
    seed, attempt = args
    rng = np.random.default_rng([seed, 0, attempt])
    pd, qd = _draw(rng)
    return _solve_record("rand", -1, attempt, pd, qd)


def _path_task(args):
    """Solve one straight-line path, stopping at the first unsolved point."""
    seed, attempt, n_points = args
    rng = np.random.default_rng([seed, 1, attempt])
    pa, qa = _draw(rng)
    pb, qb = _draw(rng)
    records = []
    for j, t in enumerate(np.linspace(0, 1, n_points)):
        rec = _solve_record("path", attempt, j, pa + t * (pb - pa), qa + t * (qb - qa))
        if not rec["ok"]:
            return attempt, None, j + 1
        records.append(rec)
    return attempt, records, n_points


def _save(path, obj):
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", help="PGLib case suffix, for example 300_ieee")
    ap.add_argument("--random", type=int, default=3000, help="solved random instances wanted")
    ap.add_argument("--paths", type=int, default=50, help="complete paths wanted")
    ap.add_argument("--points", type=int, default=41, help="points per path; the pilot used 41")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-attempt-factor", type=float, default=4.0, help="stop after this many times the target attempts")
    ap.add_argument("--pglib-dir", default=os.environ.get("PGLIB_DIR", "pglib-opf"))
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    case_path = os.path.join(args.pglib_dir, f"pglib_opf_case{args.case}.m")
    if not os.path.exists(case_path):
        raise SystemExit(f"Case file not found: {case_path}")
    os.makedirs(args.out_dir, exist_ok=True)
    final = os.path.join(args.out_dir, f"data_{args.case}.pkl")
    ckpt = os.path.join(args.out_dir, f"data_{args.case}.ckpt")
    config = dict(case=args.case, random=args.random, paths=args.paths, points=args.points, seed=args.seed)

    state = dict(config=config, rand=[], paths=[], rand_attempts=0, path_attempts=0, path_fail_at=[], elapsed=0.0)
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as fh:
            saved = pickle.load(fh)
        if saved["config"] != config:
            raise SystemExit(f"{ckpt} used different arguments. Move it, or rerun with {saved['config']}.")
        state = saved
        print(f"Resuming: {sum(r['ok'] for r in state['rand'])} random, {len(state['paths'])} paths", flush=True)

    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(case_path,)) as pool:
        # Random instances, in attempt order so the training split is reproducible.
        max_rand = int(args.random * args.max_attempt_factor)
        while sum(r["ok"] for r in state["rand"]) < args.random and state["rand_attempts"] < max_rand:
            remaining = args.random - sum(r["ok"] for r in state["rand"])
            step = min(200, max(remaining, args.workers))
            batch = range(state["rand_attempts"], min(state["rand_attempts"] + step, max_rand))
            state["rand"] += pool.map(_random_task, [(args.seed, a) for a in batch], chunksize=4)
            state["rand_attempts"] = batch.stop
            _save(ckpt, state)
            ok = sum(r["ok"] for r in state["rand"])
            print(f"random solved {ok}/{args.random}, attempts {state['rand_attempts']}, "
                  f"{state['elapsed'] + time.time() - t0:.0f}s", flush=True)

        # Whole paths are the unit of work; failed paths are replaced and counted.
        max_path = int(args.paths * args.max_attempt_factor)
        while len(state["paths"]) < args.paths and state["path_attempts"] < max_path:
            need = args.paths - len(state["paths"])
            batch = range(state["path_attempts"], min(state["path_attempts"] + max(need, args.workers), max_path))
            for attempt, records, stopped in pool.imap(_path_task, [(args.seed, a, args.points) for a in batch]):
                if records is None:
                    state["path_fail_at"].append(stopped)
                else:
                    state["paths"].append(records)
            state["path_attempts"] = batch.stop
            _save(ckpt, state)
            print(f"paths complete {len(state['paths'])}/{args.paths}, attempts {state['path_attempts']}, "
                  f"{state['elapsed'] + time.time() - t0:.0f}s", flush=True)

    state["elapsed"] += time.time() - t0
    _save(ckpt, state)

    # Random instances first, then paths: the order evalm.py and claims.py expect.
    rand_ok = [r for r in state["rand"] if r["ok"]][: args.random]
    path_records = []
    for new_pid, records in enumerate(state["paths"][: args.paths]):
        for rec in records:
            rec["pid"] = new_pid
            path_records.append(rec)
    _save(final, rand_ok + path_records)

    import cyipopt
    meta = dict(config=config, solver="Ipopt via cyipopt", ipopt_version=list(cyipopt.IPOPT_VERSION),
                options=ipopt_backend.IPOPT_OPTIONS, python=platform.python_version(),
                random_solved=len(rand_ok), random_attempts=state["rand_attempts"],
                paths_complete=min(len(state["paths"]), args.paths), path_attempts=state["path_attempts"],
                failed_path_stop_points=state["path_fail_at"], elapsed_seconds=round(state["elapsed"], 1))
    with open(os.path.join(args.out_dir, f"meta_{args.case}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "options"}), flush=True)
    if len(rand_ok) < args.random or meta["paths_complete"] < args.paths:
        print("WARNING: targets were not reached. Check the counts above before using this data.", flush=True)


if __name__ == "__main__":
    main()
