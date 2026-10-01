#!/bin/bash
# Experiment suite on hades. Two lanes run in parallel: the solver lane (experiments 1, 2, 3, 9, 8) and the learning lane
# (experiments 4, field training on the GPU, 5, 6, 7). Every stage skips work that is already in its output CSV, so the
# script can be re-run after an interruption.
#   bash scripts/exp/run_suite.sh > logs/suite.log 2>&1 &
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/suite}
WS=${WS:-12}          # solver lane workers
WL=${WL:-10}          # learning lane workers
mkdir -p logs $OUT
step() { echo "=== [$(date '+%F %T')] $*" | tee -a $OUT/STATUS; }

$PY scripts/exp/exp0_config.py --out $OUT > logs/suite_exp0.log 2>&1
cp results/dual_field/rows.csv $OUT/dual_field_rows_frozen.csv 2>/dev/null

solver_lane() {
  step "solver lane: exp1 (objective weights)"
  $PY scripts/exp/exp1_objective.py --grid results/grid --out $OUT/exp1 --workers $WS > logs/suite_exp1.log 2>&1
  step "solver lane: exp9 (physics interventions)"
  $PY scripts/exp/exp9_physics.py --grid results/grid --comp results/comp_t18 --out $OUT/exp9 --workers $WS > logs/suite_exp9.log 2>&1
  step "solver lane: exp2 (replay decomposition)"
  $PY scripts/exp/exp2_replay.py --grid results/grid --comp results/comp_t18 --out $OUT/exp2 --workers $WS > logs/suite_exp2.log 2>&1
  step "solver lane: exp3 (plan margins)"
  $PY scripts/exp/exp3_margins.py --grid results/grid --comp results/comp_t18 --out $OUT/exp3 --workers $WS > logs/suite_exp3.log 2>&1
  step "solver lane: exp8 (generalisation / transfer)"
  $PY scripts/exp/exp8_general.py --grid results/grid --data results/dual_field --out $OUT/exp8 --workers $WS > logs/suite_exp8.log 2>&1
  step "solver lane: done"
}

learning_lane() {
  step "learning lane: exp4 (label audit)"
  $PY scripts/exp/exp4_audit.py --grid results/grid --data results/dual_field --models results/dfl/it0/models.pt --out $OUT/exp4 --workers $WL > logs/suite_exp4.log 2>&1
  step "learning lane: training (GPU)"
  $PY scripts/exp/train_fields.py --data results/dual_field --out $OUT/models/full --seeds 0 1 2 > logs/suite_train.log 2>&1
  $PY scripts/exp/exp4_audit.py --parts 3 --data results/dual_field --models $OUT/models/full/models_seed0.pt --out $OUT/exp4_newmodels >> logs/suite_exp4.log 2>&1
  step "learning lane: exp5 (candidate ranking)"
  $PY scripts/exp/exp5_ranking.py --grid results/grid --models $OUT/models/full --out $OUT/exp5 --workers $WL > logs/suite_exp5.log 2>&1
  step "learning lane: exp6 (main comparison)"
  $PY scripts/exp/exp6_compare.py --grid results/grid --models $OUT/models/full --out $OUT/exp6 --workers $WL > logs/suite_exp6.log 2>&1
  step "learning lane: exp7 (data scaling / visited states)"
  $PY scripts/exp/exp7_data.py --grid results/grid --data results/dual_field --out $OUT/exp7 --models-full $OUT/models/full --workers $WL > logs/suite_exp7.log 2>&1
  step "learning lane: done"
}

solver_lane &
learning_lane &
wait
step "suite finished"
