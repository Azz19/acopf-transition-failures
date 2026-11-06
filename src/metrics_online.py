"""Accuracy and reactive violation over in-service generators only.

Offline generators keep stale voltage labels and zero output, while predictions are clipped to
their limits, so all-generator accuracy can be misleading on cases such as 200_activ.

Usage, from the folder holding the data: python metrics_online.py 200_activ 1800 [tag]
"""
import os
import pickle
import sys

import numpy as np

from common import GST, PMAX, PMIN, QMAX, QMIN, load_case

name, ntrain = sys.argv[1], int(sys.argv[2])
tag = f"_{sys.argv[3]}" if len(sys.argv) > 3 else ""
case = load_case(os.path.join(os.environ.get("PGLIB_DIR", "pglib-opf"), f"pglib_opf_case{name}.m"))
res = [r for r in pickle.load(open(f"data_{name}.pkl", "rb")) if r and r["ok"]]
rand = [r for r in res if r["kind"] == "rand"]
path = [r for r in res if r["kind"] == "path"]
test = rand[ntrain:]
n = len(test)
ev = pickle.load(open(f"eval_{name}{tag}.pkl", "rb"))
gen = case["gen"]
ng = gen.shape[0]
online = gen[:, GST] > 0
y = np.array([np.r_[r["pg"], r["vg"]] for r in test + path])
pr = np.maximum(gen[:, PMAX] - gen[:, PMIN], 1.0)
qr = np.maximum(gen[:, QMAX] - gen[:, QMIN], 1.0)
print(f"{name}: {int(online.sum())} of {ng} generators in service; test instances {n}")
rows = {}
for m, e in ev.items():
    yh = e["yh"]
    q = [(-x[0][2 * ng : 3 * ng] * qr - x[0][3 * ng : 4 * ng] * qr)[online].sum() for x in e["rec"][:n] if x is not None]
    row = (100 * (np.abs(yh[:n, :ng] - y[:n, :ng])[:, online] / pr[online]).mean(), np.abs(yh[:n, ng:] - y[:n, ng:])[:, online].mean(), float(np.mean(q)))
    rows[m] = row
for fam in sorted({m.split("_s")[0] for m in rows}):
    vals = np.array([v for m, v in rows.items() if m.split("_s")[0] == fam])
    print(f"{fam:8s} Pg MAE {vals[:, 0].mean():.3f}%  Vg MAE {vals[:, 1].mean():.5f} pu  Q violation {vals[:, 2].mean():.3f} MVAr  seeds {len(vals)}")
