#!/bin/bash
# Dual-field learning loop on hades: fit -> evaluate -> DAgger states -> oracle labels -> refit ...
#   bash scripts/dfl_pipeline.sh <n_iter> [holdout_m]
# Uses results/dual_field (table rows.csv + solution pickles) as the growing dataset.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-.venv/bin/python}
NIT=${1:-2}
HOLD=${2:-69}
REFS=${REFS:-"results/grid/sol_ref_T1[6789]_m*_phi*.pkl"}
DATA=${DATA:-results/dual_field}
W=${W:-10}
mkdir -p logs
step() { echo "=== [$(date '+%F %T')] $*"; }
for it in $(seq 0 $NIT); do
  step "iteration $it: fit + closed-loop evaluation (holdout m=$HOLD)"
  $PY scripts/dfl_experiment.py --data $DATA --refs "$REFS" --holdout-m $HOLD --out results/dfl/it$it --epochs ${EPOCHS:-6000} \
      --episodes ${EPISODES:-6} > logs/dfl_it$it.log 2>&1
  tail -8 logs/dfl_it$it.log
  if [ $it -lt $NIT ]; then
    step "iteration $it: DAgger rollouts -> visited states"
    $PY scripts/dfl_dagger.py --models results/dfl/it$it/models.pt --refs "$REFS" --out $DATA/states_it$((it+1)).json \
        --episodes 3 --stride 0.25 --max-states ${NSTATES:-300} --seed $it > logs/dfl_dagger_it$it.log 2>&1
    tail -2 logs/dfl_dagger_it$it.log
    step "iteration $it: oracle labels for the visited states"
    $PY scripts/dual_field_data.py --states $DATA/states_it$((it+1)).json --out $DATA --workers $W --max-cpu 400 > logs/dfl_labels_it$((it+1)).log 2>&1
    tail -1 logs/dfl_labels_it$((it+1)).log
  fi
done
step "done"
