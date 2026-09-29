#!/bin/bash
# Stage-1 sensitivity analysis (plan Sec. 13.1 / 3.2): one factor at a time around the reference setting.
# Reduced set: T=20 s (one-way 10 s), m=60 kg, release phases {0.125, 0.25, 0.375, 0.625, 0.75}. Sequential.
cd "$(dirname "$0")/.."
PH=${PHASES:-"0.125 0.25 0.375 0.625 0.75"}
run() { tag=$1; shift; python3 scripts/run_grid.py --out results/sens --T 20 --m 60 --phis $PH --workers 1 --tag "$tag" --params "$1" ; }
run eps040  '{"eps": 0.40}'
run cap070  '{"body_kw": {"cap_scale": 0.7}}'
run dc030   '{"delta_catch": 0.03}'
run vrel3   '{"v_rel_max": 3.0}'
run mu060   '{"body_kw": {"mu_out": 0.6}}'
run cap130  '{"body_kw": {"cap_scale": 1.3}}'
run eps010  '{"eps": 0.10}'
run dc100   '{"delta_catch": 0.10}'
run vrel5   '{"v_rel_max": 5.0}'
run mu150   '{"body_kw": {"mu_out": 1.5}}'
echo "sensitivity done"
