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

## Second independent system: 200_activ

Set Nov 6, 2026, before full data generation, tuning or scoring on this system. Only a 30-instance solver test with a separate seed has been run.

**H2 replicates** if the depth slope is at least 0.1 with a 95% CI excluding 0 in both tuned scikit-learn MLP families (`mlp64`, `mlp256`), with at least 50 clean episodes. The test, recovery, thresholds and episode definition match the 300-bus rule. Ridge and the full-state MLP are reported as exploratory.

- **Inconclusive:** fewer than 50 clean episodes. Add paths without changing the test.
- **Not replicated:** with at least 50 episodes, the slope is below 0.1 or the CI includes 0 in either MLP family. The paper must then limit its H2 claim to the 300-bus case and report this result.

**Data:** `gen_ipopt.py 200_activ --random 2400 --paths 50 --points 41 --seed 2027`, using the default load range (global factor 0.85 to 1.15); 1,800 training instances.

## Mechanism and correction criteria

Set Nov 7, 2026, before `mechanism.py` or `triage.py` ran on the 300- or 200-bus data. These scripts were developed on the 118-bus pilot, so 118-bus results are exploratory. "Both MLP families" means the tuned scikit-learn `mlp64` and `mlp256` used in the locked H2 tests.

- **Under-bending, measured directly:** supported on a system if, in both MLP families, the median bend ratio (learned derivative jump at activation divided by the optimal jump, Q and V episodes) is below 1 with a path-bootstrap 95% CI entirely below 1, and the median pre-activation tracking ratio lies between 0.5 and 1.5.
- **AC sensitivity:** error amplification explains the growth only if the depth slope of log sensitivity, over all active-side points, is positive with a CI excluding 0 and is at least half the depth slope of the log linearized violation. Otherwise the growth is attributed to whichever of control-error size and alignment has a positive slope with a CI excluding 0, or reported as unexplained.
- **Depth versus load:** the depth slope stays positive with a CI excluding 0 when comparing the same constraint within the same 1%-of-load bin across paths, with five load-pattern components as controls, in both MLP families.
- **Selective re-solve:** active-set-aware selection has operational value on a system if, at a 20% re-solve budget, it leaves less remaining reactive violation than both random and total-load selection, with instance-bootstrap 95% CIs excluding 0, for the most accurate family. The practical tolerance is 5 MVAr per generator, 0.005 pu voltage and 2% of range or rating. Re-solved instances are assumed to reach the stored optimum; a timed warm-start sample reports cost and success.
