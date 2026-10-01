#!/bin/bash
# Follow-up stage of the suite, started after the first pass:
#  1. exp3: the plans with a joint margin were infeasible because the margin was also applied to the straight-limb stops
#     of elbow and knee, on which the hanging body rests (fixed in nlp.py: joint_margin_lo_mask); they are re-solved, and
#     the scenario-robust variants get a larger CPU limit;
#  2. exp4 part 2t: the costate finite-difference check with tight tolerances and warm-started re-solves (the default
#     tolerance check is dominated by re-solve noise).
# Waits for the solver lane of run_suite.sh to finish so that the machine is not oversubscribed.
#   nohup bash scripts/exp/run_followup.sh > logs/followup.log 2>&1 &
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/suite}
WS=${WS:-10}
step() { echo "=== [$(date '+%F %T')] $*" | tee -a $OUT/STATUS; }

until grep -q "solver lane: done" $OUT/STATUS; do sleep 600; done
step "followup: exp3 re-solve (joint margins at the straight-limb stops fixed; robust variants with 7200 s)"
$PY scripts/exp/exp3_margins.py --grid results/grid --comp results/comp_t18 --out $OUT/exp3 --workers $WS \
    --redo B_joint2deg,B_joint5deg,B_all_small,B_all_large,C_robust2ms,D_small_robust,D_large_robust --max-cpu-robust 7200 >> logs/suite_exp3.log 2>&1
step "followup: exp4.2t (tight finite-difference costate check)"
$PY scripts/exp/exp4_audit.py --parts 2t --grid results/grid --data results/dual_field --out $OUT/exp4 --workers $WS >> logs/suite_exp4.log 2>&1
step "followup: done"
