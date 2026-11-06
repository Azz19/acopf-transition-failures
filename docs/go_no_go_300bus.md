# 300-bus go/no-go rule

Approved Nov 5, 2026, before `claims.py` was run on the 300-bus data.

**Go** only if, on 300 buses, H2's depth slope is at least 0.1 with a 95% CI excluding 0 in both tuned MLP families (`mlp64`, `mlp256`), with at least 50 clean episodes. Ridge is reported but does not count.

- **Inconclusive:** fewer than 50 clean episodes. Add paths without changing the test.
- **No-go for this GM deadline:** with at least 50 episodes, the slope is below 0.1 or the CI includes 0 in either MLP family.

**Test:** `claims.py`, field `C2.slope_depth`, with `se` from episode-clustered errors and CI = slope ± 1.96 se. It uses the fixed-effects regression on the active side only, with load held constant.

**Data:** `gen_ipopt.py 300_ieee --random 2400 --paths 50 --points 41 --load-max 1.05 --seed 2027`; 1,800 training instances.

**Models:** `tune.py` selections on training-only validation data, then `evalm.py` with three seeds per MLP family.

**Known limits:** the depth test was designed after viewing 118-bus data. The load factor is capped at 1.05 for this system, and 21 of the 71 path attempts did not count (20 failed and 1 extra was completed).

## Extension to the Lagrangian MLP and GNN

Set Nov 5, 2026, before either family was trained on the 300-bus data. H2 extends to a family if its depth slope is at least 0.1 with a 95% CI excluding 0 and at least 50 clean episodes. This does not change the go decision. A failure limits what the paper can claim about those families and must be reported.
