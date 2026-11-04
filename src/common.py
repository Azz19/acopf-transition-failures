"""Pilot utilities: PGLib parser, AC-OPF solve via PYPOWER (PIPS), active-set and margin extraction."""
import re, numpy as np
from pypower.api import runopf, runpf, ppoption

def load_case(path):
    txt = open(path).read()
    out = {"version": "2"}
    out["baseMVA"] = float(re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", txt).group(1))
    for k in ["bus", "gen", "branch", "gencost"]:
        body = re.search(r"mpc\.%s\s*=\s*\[(.*?)\];" % k, txt, re.S).group(1)
        rows = []
        for line in body.splitlines():
            line = line.split("%")[0].strip().rstrip(";")
            if line:
                rows.append([float(x) for x in line.split()])
        out[k] = np.array(rows)
    return out

OPF_OPT = ppoption(VERBOSE=0, OUT_ALL=0, OPF_VIOLATION=1e-6, PDIPM_FEASTOL=1e-7, PDIPM_GRADTOL=1e-7, PDIPM_COMPTOL=1e-7)
PF_OPT = ppoption(VERBOSE=0, OUT_ALL=0, PF_MAX_IT=20)

# PYPOWER column indices
PD, QD, VM, VA, VMAX, VMIN, LAM_P, MU_VMAX, MU_VMIN = 2, 3, 7, 8, 11, 12, 13, 15, 16
PG, QG, QMAX, QMIN, VG, GST, PMAX, PMIN = 1, 2, 3, 4, 5, 7, 8, 9
MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN = 21, 22, 23, 24
RATE_A, PF, QF, PT, QT, MU_SF, MU_ST = 5, 13, 14, 15, 16, 17, 18

def solve(case, pd, qd):
    c = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in case.items()}
    c["bus"][:, PD] = pd; c["bus"][:, QD] = qd
    r = runopf(c, OPF_OPT)
    return r

def active_vector(r, mu_tol=1e-4):
    """Binary active set from multipliers, ordered Pmax,Pmin,Qmax,Qmin,Vmax,Vmin,Sf,St."""
    g, b, l = r["gen"], r["bus"], r["branch"]
    mu = np.concatenate([g[:, MU_PMAX], g[:, MU_PMIN], g[:, MU_QMAX], g[:, MU_QMIN],
                         b[:, MU_VMAX], b[:, MU_VMIN], l[:, MU_SF], l[:, MU_ST]])
    return (np.abs(mu) > mu_tol).astype(np.int8), mu

def slacks(r, base):
    """Normalized slack of each inequality, same order as active_vector. >=0 feasible."""
    g, b, l = r["gen"], r["bus"], r["branch"]
    pr = np.maximum(g[:, PMAX] - g[:, PMIN], 1.0); qr = np.maximum(g[:, QMAX] - g[:, QMIN], 1.0)
    vr = np.maximum(b[:, VMAX] - b[:, VMIN], 1e-3)
    rate = l[:, RATE_A].copy(); rate[rate <= 0] = 1e9
    sf = np.hypot(l[:, PF], l[:, QF]); st = np.hypot(l[:, PT], l[:, QT])
    return np.concatenate([(g[:, PMAX]-g[:, PG])/pr, (g[:, PG]-g[:, PMIN])/pr,
                           (g[:, QMAX]-g[:, QG])/qr, (g[:, QG]-g[:, QMIN])/qr,
                           (b[:, VMAX]-b[:, VM])/vr, (b[:, VM]-b[:, VMIN])/vr,
                           (rate-sf)/rate, (rate-st)/rate])
