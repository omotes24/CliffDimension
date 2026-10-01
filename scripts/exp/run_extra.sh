#!/bin/bash
# Extra stage: (1) the rest of experiment 8 -- the closed loop on the new bodies now runs through the worker pool (it ran
# one trial at a time before) and the envelope-theorem transfer; (2) horizon ablation of the main comparison: experiment 5
# shows that at the 0.3 s horizon of the MPC the candidate values differ by less than the value error, while at 0.8 s the
# Sobolev value picks the best candidate in ~80 % of the cases, so VMPC / SMPC are repeated with H = 40 (0.8 s).
#   nohup bash scripts/exp/run_extra.sh > logs/extra.log 2>&1 &
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/suite}
WX=${WX:-6}
step() { echo "=== [$(date '+%F %T')] $*" | tee -a $OUT/STATUS; }

step "extra: exp8 (closed loop on new bodies, envelope transfer)"
$PY scripts/exp/exp8_general.py --grid results/grid --data results/dual_field --out $OUT/exp8 --workers $WX --parts b,c >> logs/suite_exp8.log 2>&1
step "extra: exp6 horizon ablation (H = 40, 0.8 s)"
$PY scripts/exp/exp6_compare.py --grid results/grid --models $OUT/models/full --out $OUT/exp6_h40 --workers $WX --methods VMPC,SMPC \
    --mpc-H 40 --mpc-max-cpu 60 --n-perturb 5 --learn-seeds 1 --recovery-knots 0 > logs/suite_exp6_h40.log 2>&1
step "extra: done"
