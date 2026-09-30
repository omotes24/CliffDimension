#!/bin/bash
# Full recomputation pipeline (corrected forearm-face collision model). Designed to run unattended on hades:
#   bash scripts/hades_pipeline.sh > logs/pipeline.log 2>&1 &
# Every step skips work that already exists, so the script can be re-run after an interruption.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-.venv/bin/python}
NW=${NW:-24}
mkdir -p logs results/grid results/figs
step() { echo "=== [$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- 0. archive results of the old (v2) model ----
if [ ! -f results/.model_v3 ]; then
  step "0. archiving v2 results (no forearm-face collision) under results/legacy_v2"
  mkdir -p results/legacy_v2
  for d in grid build sens sens16 compliant duals duals_fd robust windows windows_robust bodygrid spatial figs; do
    [ -d results/$d ] && mv results/$d results/legacy_v2/$d
  done
  mkdir -p results/grid results/figs
  echo "model v3: forearm clearance points + shoulder hyperextension 35 deg" > results/.model_v3
fi

# ---------------------------------------------------------------- 1. dense grid (T x m x 16 phases) --------
step "1. dense grid (warm-started from the v2 solutions where available)"
KATSUMI_MAX_CPU=2400 $PY scripts/run_grid.py --out results/grid --T 16 17 18 19 20 21 22 23 24 --m 60 63 66 69 72 75 \
    --nphi 16 --workers $NW --tag ref --init-from-grid results/legacy_v2/grid > logs/grid.log 2>&1

# ---------------------------------------------------------------- 2. lower envelope ------------------------
step "2a. envelope: multi-start + continuation for the 12 representative conditions"
$PY scripts/envelope_grid.py --grid results/grid --tag ref --T 16 19 20 24 --m 60 66 75 --cold 3 --passes 4 --workers $NW \
    --log results/grid/envelope_ref.json > logs/envelope.log 2>&1
step "2b. continuation passes for all conditions"
$PY scripts/envelope_grid.py --grid results/grid --tag ref --T 16 17 18 19 20 21 22 23 24 --m 60 63 66 69 72 75 --cold 0 --passes 2 \
    --workers $NW --log results/grid/envelope_all.json > logs/envelope_all.log 2>&1
$PY scripts/analyze_grid.py --grid results/grid --tag ref --out results/figs > logs/analyze_grid.log 2>&1

# ---------------------------------------------------------------- 3. planar studies (parallel groups) -------
step "3. planar studies"
mkdir -p results/sens16 results/compliant results/duals results/duals_fd results/robust results/windows results/build results/bodygrid
# 3a. one-factor variants, 16 phases (T=20, m=60), incl. the shoulder hyperextension range
(
run() { KATSUMI_MAX_CPU=1500 $PY scripts/run_grid.py --out results/sens16 --T 20 --m 60 --nphi 16 --workers 1 --tag "$1" --params "$2" \
        --init-from-grid results/grid > logs/sens16_$1.log 2>&1; }
run mu060  '{"body_kw": {"mu_out": 0.6}}' &
run mu150  '{"body_kw": {"mu_out": 1.5}}' &
run cap070 '{"body_kw": {"cap_scale": 0.7}}' &
run cap130 '{"body_kw": {"cap_scale": 1.3}}' &
run eps010 '{"eps": 0.10}' &
run eps040 '{"eps": 0.40}' &
run dc030  '{"delta_catch": 0.03}' &
run dc100  '{"delta_catch": 0.10}' &
run vrel3  '{"v_rel_max": 3.0}' &
run vrel5  '{"v_rel_max": 5.0}' &
run shx015 '{"rel_hi_face_neg": [2.6, 0.26, 0.35, 2.6]}' &
run shx052 '{"rel_hi_face_neg": [2.6, 0.90, 0.35, 2.6]}' &
wait
) &
# 3b. compliant catch inside the optimisation (T=20, m=60 & 66), then its own envelope
(
KATSUMI_MAX_CPU=3000 $PY scripts/run_grid.py --out results/compliant --T 20 --m 60 66 --nphi 16 --workers 8 --tag comp \
    --params '{"catch_model": "compliant"}' --split-phases --init-from-grid results/grid > logs/compliant.log 2>&1
$PY scripts/envelope_grid.py --grid results/compliant --tag comp --T 20 --m 60 66 --cold 2 --passes 3 --workers 8 \
    --log results/compliant/envelope_comp.json > logs/envelope_comp.log 2>&1
) &
# 3c. duals / shadow prices (+ finite-difference validation)
(
$PY scripts/run_duals.py --grid results/grid --tag ref --T 20 --m 60 66 --out results/duals --workers 4 > logs/duals.log 2>&1
$PY scripts/run_duals.py --grid results/grid --tag ref --T 20 --m 66 --phis 0.25 0.6875 --out results/duals_fd --workers 2 --fd > logs/duals_fd.log 2>&1
) &
wait
step "3d. robust release timing + windows + body sweeps"
# 3d. robust release-timing scenarios (open / closed loop)
(
for dw in 0.005 0.01 0.02 0.04; do
  tag=rob$(python3 -c "print('%03d' % int(round($dw*1000)))")
  KATSUMI_MAX_CPU=14000 $PY scripts/run_grid.py --out results/robust --T 20 --m 66 --phis 0.25 0.6875 --workers 2 --tag $tag \
     --params "{\"robust_deltas\": [-$dw, 0.0, $dw]}" --split-phases --init-from-grid results/grid > logs/$tag.log 2>&1 &
  KATSUMI_MAX_CPU=14000 $PY scripts/run_grid.py --out results/robust --T 20 --m 66 --phis 0.25 0.6875 --workers 2 --tag ${tag}cl \
     --params "{\"robust_deltas\": [-$dw, 0.0, $dw], \"robust_shared_flight\": false}" --split-phases --init-from-grid results/grid > logs/${tag}cl.log 2>&1 &
done
wait
mkdir -p results/windows_robust
for t in rob005 rob010 rob020 rob040 rob005cl rob010cl rob020cl rob040cl; do
  $PY scripts/run_windows.py --grid results/robust --tag $t --out results/windows_robust --fine 0.06 --dt-fine 0.0005 --coarse 0.10 --suffix _fine --workers 2 > logs/windows_$t.log 2>&1 &
done
wait
) &
# 3e. fine release windows of all best-phase plans (0.5 ms) + the 5 ms reference
(
$PY scripts/run_windows.py --grid results/grid --tag ref --out results/windows --best-only --fine 0.04 --dt-fine 0.0005 --coarse 0.10 --suffix _fine --workers 4 > logs/windows_fine.log 2>&1
$PY scripts/run_windows.py --grid results/grid --tag ref --out results/windows --best-only --workers 4 > logs/windows.log 2>&1
) &
# 3f. body-size sweeps (stature / arm length, 16 phases, T=20 m=66) and the (H, c, m) body grid at the two valley phases
(
for H in 1.60 1.65 1.70 1.80 1.85; do
  tag=stature$(python3 -c "print(int(round($H*100)))")
  KATSUMI_MAX_CPU=1500 $PY scripts/run_grid.py --out results/build --T 20 --m 66 --nphi 16 --workers 1 --tag $tag \
      --params "{\"body_kw\": {\"stature\": $H}}" --init-from-grid results/grid > logs/build_$tag.log 2>&1 &
done
KATSUMI_MAX_CPU=1500 $PY scripts/run_grid.py --out results/build --T 20 --m 66 --nphi 16 --workers 1 --tag arm095 --params '{"body_kw": {"arm_scale": 0.95}}' --init-from-grid results/grid > logs/build_arm095.log 2>&1 &
KATSUMI_MAX_CPU=1500 $PY scripts/run_grid.py --out results/build --T 20 --m 66 --nphi 16 --workers 1 --tag arm105 --params '{"body_kw": {"arm_scale": 1.05}}' --init-from-grid results/grid > logs/build_arm105.log 2>&1 &
wait
run_body() {
  H=$1; C=$2
  tag=H$(python3 -c "print(int(round($H*100)))")_c$(python3 -c "print(int(round($C*100)))")
  KATSUMI_MAX_CPU=1500 $PY scripts/run_grid.py --out results/bodygrid --T 20 --m 50 55 60 65 70 75 80 85 90 --phis 0.25 0.6875 --workers 2 --tag $tag \
      --params "{\"body_kw\": {\"stature\": $H, \"cap_scale\": $C}}" --split-phases --init-from-grid results/grid > logs/bodygrid_$tag.log 2>&1
}
export -f run_body; export PY
for H in 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95; do for C in 0.7 0.85 1.0 1.15 1.3; do echo "$H $C"; done; done \
  | xargs -P 8 -L 1 bash -c 'run_body $0 $1'
) &
wait

# ---------------------------------------------------------------- 4. spatial (twisting, two-hand) model ------
step "4. spatial model: strategies x 8 phases (T=20, m=66)"
mkdir -p results/spatial
PH8="0 0.125 0.25 0.375 0.5 0.625 0.75 0.875"
run3d() { $PY scripts/run_grid3d.py --T 20 --m 66 --phis $PH8 --grip $1 --release $2 --catch $3 --tag $4 --out results/spatial \
          --init results/grid --init3d results/spatial --workers 8 --maxit 1500 --max-cpu 20000 > logs/spatial_$4.log 2>&1; }
run3d 0.39 none none wide_sim &
run3d 0.10 none none narrow_sim &
run3d 0.39 L L wide_stag &
wait
run3d 0.10 L L narrow_stag &
run3d 0.39 none L wide_catchstag &
run3d 0.25 none none mid_sim &
wait
step "4b. spatial body sweep at the two valley phases (best strategy = wide_sim unless overridden)"
BEST=${BEST3D:-wide_sim}
for m in 55 66 80; do for H in 1.60 1.75 1.90; do for C in 0.8 1.0 1.2; do
  tag=${BEST}_m${m}_H$(python3 -c "print(int(round($H*100)))")_c$(python3 -c "print(int(round($C*100)))")
  echo "$m $H $C $tag"
done; done; done | xargs -P 12 -L 1 bash -c '$PY scripts/run_grid3d.py --T 20 --m $0 --phis 0.25 0.6875 --grip 0.39 --release none --catch none --tag $3 --out results/spatial_body --init results/grid --init3d results/spatial --body "{\"stature\": $1, \"cap_scale\": $2}" --workers 2 --maxit 1500 --max-cpu 20000 > logs/$3.log 2>&1'

# ---------------------------------------------------------------- 5. analyses (figures / csv / json) --------
step "5. analyses"
$PY scripts/analyze_grid.py --grid results/grid --tag ref --out results/figs > logs/analyze_grid.log 2>&1
$PY scripts/analyze_envelope.py --grid results/grid --log results/grid/envelope_ref.json --out results/figs > logs/analyze_envelope.log 2>&1
$PY scripts/analyze_ranking.py --grid results/grid --out results/figs > logs/analyze_ranking.log 2>&1
$PY scripts/analyze_duals.py --duals results/duals --fd results/duals_fd --out results/figs > logs/analyze_duals.log 2>&1
$PY scripts/analyze_robust.py --out results/figs > logs/analyze_robust.log 2>&1
$PY scripts/analyze_windows_fine.py --out results/figs > logs/analyze_windows_fine.log 2>&1
$PY scripts/analyze_sensitivity.py --grid results/grid --sens results/sens16 --out results/figs > logs/analyze_sensitivity.log 2>&1
step "pipeline finished"
