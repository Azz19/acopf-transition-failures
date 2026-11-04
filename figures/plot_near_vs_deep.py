"""Near-versus-deep violation figure, read from the frozen claims results."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fam = [("mlp256", "MLP 256x2"), ("mlp64", "MLP 64"), ("ridge", "Ridge")]
rows = []
for case, lab in [("118_ieee", "118-bus"), ("57_ieee", "57-bus")]:
    c = json.load(open(f"results/claims_{case}.json"))["C2"]
    rows += [(f"{lab}, {n}", c[k]["near"] * 100, c[k]["deep"] * 100) for k, n in fam]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for i, (n, a, b) in enumerate(rows):
    y = len(rows) - i
    ax.plot([a, b], [y, y], color="#b8b8ad", lw=2, zorder=1)
    ax.scatter(a, y, color="#9a9a90", s=40, zorder=2, label="At most 2 steps after activation" if i == 0 else None)
    ax.scatter(b, y, color="#2f6fd0", s=40, zorder=2, label="At least 13 steps after activation" if i == 0 else None)
ax.set_yticks([len(rows) - i for i in range(len(rows))], [r[0] for r in rows])
ax.set_xscale("log"); ax.set_xlabel("Mean recovered violation, % of limit range (log scale)")
ax.grid(axis="x", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1), ncol=2)
fig.tight_layout(); fig.savefig("figures/near_vs_deep.png", dpi=200)
