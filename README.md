# Active-Constraint Transition Failures in Learned AC-OPF

Experiment design and exploratory pilot for a planned IEEE PES General Meeting 2027 paper, "Beyond Nominal Accuracy: Active-Constraint Transition Failures in Learned AC Optimal Power Flow."

**Status: pilot only.** Results come from 118- and 57-bus systems with untuned scikit-learn models. They justify the design and are not publishable findings.

## Contents

| Path | Purpose |
| --- | --- |
| `docs/experiment_and_paper_design.md` | Hypotheses, metrics, experiments, paper outline, reviewer defenses, schedule |
| `src/common.py` | PGLib parser, AC-OPF solve, active-set and slack extraction |
| `src/gen.py` | Solved random and load-path instances |
| `src/evalm.py` | Trains surrogates and recovers operating points by AC power flow |
| `src/analyze.py` | Stratified audit, kink regression, rare-activation gap |
| `src/followup.py` | Exploratory 118-bus follow-up |
| `src/claims.py` | Claims frozen from 118-bus, applied unchanged to new systems |
| `results/` | Pilot result files |
| `figures/` | Near-versus-deep figure and its script |

## Reproduce

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/power-grid-lib/pglib-opf.git
export PGLIB_DIR=$PWD/pglib-opf
cd src

python gen.py 118_ieee 2400 20 41 7
python evalm.py 118_ieee 1800
python analyze.py 118_ieee 1800
python followup.py
python claims.py 118_ieee 1800

python gen.py 57_ieee 1500 20 41 11
python evalm.py 57_ieee 800
python claims.py 57_ieee 800
```

Arguments for `gen.py` are case, random instances, paths, points per path and seed. Arguments for `evalm.py`, `analyze.py` and `claims.py` are case and training instances. Solved instances are not committed, so `gen.py` must run first.

## Known limits

- `analyze.py` fails on the 57-bus case, so `results/` has no such file for that system.
- The 57-bus split moved from 1,100 to 800 training instances after the first run left 48 test instances.
- The depth test was defined after looking at 118-bus data, so only new systems can confirm it.
- `claims.py` computes a differential-voltage claim (C4), which the pilot did not support.
