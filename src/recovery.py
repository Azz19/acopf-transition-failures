"""AC power flow recovery in worker processes, separate from PyTorch imports."""
import numpy as np
from pypower.api import runpf

from common import PD, PG, QD, RATE_A, VG, VM, VA, PF_OPT, load_case, slacks

_case, _gb = None, None


def init(case_path):
    """Load the case once per worker."""
    global _case, _gb
    _case = load_case(case_path)
    ids = {int(b): i for i, b in enumerate(_case["bus"][:, 0])}
    _gb = np.array([ids[int(x)] for x in _case["gen"][:, 0]])


def _run(pd, qd, pg, vg):
    case = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in _case.items()}
    case["bus"][:, PD], case["bus"][:, QD] = pd, qd
    case["gen"][:, PG], case["gen"][:, VG] = pg, vg
    case["bus"][_gb, VM] = vg
    try:
        result, _ = runpf(case, PF_OPT)
    except Exception:
        return None
    return result if result["success"] else None


def recover(args):
    """Return (negative normalized slacks, generation cost), matching evalm.py, or None."""
    pd, qd, y = args
    ng = _case["gen"].shape[0]
    result = _run(pd, qd, y[:ng], y[ng:])
    if result is None:
        return None
    gc = _case["gencost"]
    pg = result["gen"][:, PG]
    cost = float(np.sum(gc[:, 4] * pg ** 2 + gc[:, 5] * pg + gc[:, 6]))
    return np.minimum(slacks(result, _case["baseMVA"]), 0).astype(np.float32), cost


def angles(args):
    """Return bus voltage angles in degrees at a solved operating point."""
    pd, qd, pg, vg = args
    result = _run(pd, qd, pg, vg)
    return None if result is None else result["bus"][:, VA]
