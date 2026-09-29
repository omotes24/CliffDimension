#!/bin/bash
# Stage-1 sensitivity analysis (plan Sec. 13.1 / 3.2): one factor at a time around the reference setting.
# Reduced set: T=10 s, m=60 kg, 8 release phases. Run after the main grid.
cd "$(dirname "$0")/.."
PH="0 0.25 0.5 0.75 1.0 1.25 1.5 1.75"
W=${WORKERS:-2}
run() { tag=$1; shift; python3 scripts/run_grid.py --out results/sens --T 10 --m 60 --phis $PH --workers 1 --tag "$tag" --params "$1" ; }
run eps010  '{"eps": 0.10}' &
run eps040  '{"eps": 0.40}' &
wait
run cap070  '{"body_kw": {"cap_scale": 0.7}}' &
run cap130  '{"body_kw": {"cap_scale": 1.3}}' &
wait
run dc030   '{"delta_catch": 0.03}' &
run dc100   '{"delta_catch": 0.10}' &
wait
run vrel3   '{"v_rel_max": 3.0}' &
run vrel5   '{"v_rel_max": 5.0}' &
wait
run mu060   '{"body_kw": {"mu_out": 0.6}}' &
run mu150   '{"body_kw": {"mu_out": 1.5}}' &
wait
echo "sensitivity done"
