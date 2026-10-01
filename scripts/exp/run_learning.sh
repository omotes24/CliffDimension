#!/bin/bash
# Learning lane of the suite on its own (training on the GPU -> experiments 4.3, 5, 6, 7). Used to re-run the lane after a
# failure; every stage resumes from its output CSV, so finished closed-loop trials are not repeated.
#   nohup bash scripts/exp/run_learning.sh > logs/learning.log 2>&1 &
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/suite}
WL=${WL:-8}
mkdir -p logs $OUT
step() { echo "=== [$(date '+%F %T')] $*" | tee -a $OUT/STATUS; }

# pre-flight: every package the lane imports (the first run lost the GMM fit to a missing scikit-learn)
if ! $PY -c "import sklearn, torch, scipy, casadi, pandas" 2>/dev/null; then
  step "learning lane: missing python packages, run: $PY -m pip install -r requirements.txt"; exit 1
fi

step "learning lane: training (GPU)"
$PY scripts/exp/train_fields.py --data results/dual_field --out $OUT/models/full --seeds 0 1 2 > logs/suite_train.log 2>&1
if ! ls $OUT/models/full/models_seed*.pt > /dev/null 2>&1; then
  step "learning lane: training produced no models, see logs/suite_train.log"; exit 1
fi
$PY scripts/exp/exp4_audit.py --parts 3 --data results/dual_field --models $OUT/models/full/models_seed0.pt --out $OUT/exp4_newmodels >> logs/suite_exp4.log 2>&1
step "learning lane: exp5 scoring (candidates already simulated and solved)"
if [ -f $OUT/exp5/candidates.csv ]; then
  $PY scripts/exp/exp5_ranking.py --score-only --models $OUT/models/full --out $OUT/exp5 >> logs/suite_exp5.log 2>&1
else
  $PY scripts/exp/exp5_ranking.py --grid results/grid --models $OUT/models/full --out $OUT/exp5 --workers $WL >> logs/suite_exp5.log 2>&1
fi
step "learning lane: exp6 (main comparison, learned controllers)"
$PY scripts/exp/exp6_compare.py --grid results/grid --models $OUT/models/full --out $OUT/exp6 --workers $WL >> logs/suite_exp6.log 2>&1
step "learning lane: exp7 (data scaling / visited states)"
$PY scripts/exp/exp7_data.py --grid results/grid --data results/dual_field --out $OUT/exp7 --models-full $OUT/models/full --workers $WL >> logs/suite_exp7.log 2>&1
step "learning lane: done"
