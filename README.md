# CliffDimension

SASUKE *Cliff Dimension* backward jump (moving cliff A → moving cliff B, 3 cm ledges):
physics, trajectory optimisation, and a 3-D MuJoCo environment for learning-based comparisons.

```
katsumi/
  device.py            device kinematics (plan eq. 2 / 4), ledge geometry (3 cm, 5 cm, 15 cm face, R5 edge)
  planar/              Stage 1 — planar 5-link reduced model
    anthro.py          de Leva (1996) segment parameters scaled to body mass m
    model.py           CasADi dynamics (hands = floating base), pinned / free / constraint force
    nlp.py             multi-phase Hermite–Simpson collocation: wait → swing → flight → impact → 2 s hold
    simulate.py        RK4 re-integration (audit), release-window analysis (plan §13.2, compliant catch)
    plots.py           stick-figure / load plots
  mujoco/              Stage 2 — 3-D humanoid environment
    build_model.py     31 hinge DoF humanoid + finger channels, cliffs as mocap bodies (rounded tip edge)
    grasp.py           finite-capacity grasp (soft connect constraint + admissible-wrench check, slip, release)
    env.py             control 20 ms / physics 1 ms, RK4 in flight, implicit Euler when constrained; plan §7 logic
    search.py          CMA-ES trajectory search (spline joint targets + release/close timing), planar init
scripts/
  run_grid.py          Stage-1 release-phase grid (required grip capacity)          -> results/grid
  analyze_grid.py      figures / tables (H1, H2, heatmaps, start-phase strategy)    -> results/figs
  run_windows.py       release-window analysis                                     -> results/windows
  run_sensitivity.sh   one-factor sensitivity (eps, capability, delta_c, v_rel, mu) -> results/sens
  polish_grid.py       re-solve local-optimum outliers from their phase neighbours (warm start)
  extra_figs.py        device / windows / sensitivity / 3-D search / body-size figures  -> results/figs
  paper_numbers.py     LaTeX macros + tables for paper/ (numbers.tex, tab_*.tex)
  make_outputs.sh      whole reporting pipeline: analysis -> figures -> tables -> PDF -> deliverables
  check_solution.py    audit of one solution (HS residuals, re-integration, metrics, plot)
  search_3d.py         3-D CMA-ES search (glacus)                                  -> results/search3d
  render_episode.py    offscreen rendering (MUJOCO_GL=osmesa or egl)
tests/                 unit / acceptance tests (device, planar dynamics, MuJoCo env: plan §13.3)
paper/                 MIRU2025 LaTeX manuscript (platex + dvipdfmx; latexmk -r latexmkrc main.tex)
```

## Setup

```bash
pip install -r requirements.txt          # numpy scipy casadi mujoco matplotlib pandas cma pytest
python -m pytest tests -q                # ~2 min
```

## Stage 1 — reduced model

```bash
python scripts/solve_one.py --T 20 --m 66 --phi0 0.0                  # one case (T = full period 20 s)
python scripts/run_grid.py --T 16 17 18 19 20 21 22 23 24 --m 60 63 66 69 72 75 --nphi 16 --workers 20 --tag ref
python scripts/polish_grid.py --tag ref --workers 20 --passes 2       # warm-start re-solve of outliers
python scripts/analyze_grid.py --tag ref                              # figures in results/figs
python scripts/run_windows.py --tag ref --best-only                    # release windows (54 best cases)
bash scripts/run_sensitivity.sh                                        # sensitivity grid
# body-size sweep (T = 20 s, m = 66 kg): stature 1.60–1.85 m, arm length ±5 %
for H in 1.60 1.65 1.70 1.80 1.85; do python scripts/run_grid.py --T 20 --m 66 --nphi 16 --tag stature${H/./} \
       --params "{\"body_kw\": {\"stature\": $H}}" --out results/build; done
python scripts/run_grid.py --T 20 --m 66 --nphi 16 --tag arm095 --params '{"body_kw": {"arm_scale": 0.95}}' --out results/build
python scripts/run_grid.py --T 20 --m 66 --nphi 16 --tag arm105 --params '{"body_kw": {"arm_scale": 1.05}}' --out results/build
bash scripts/make_outputs.sh                                           # figures, tables, PDF
```
Objective: minimise the peak grasp utilisation `U_peak` (epigraph) with the effort integral as
regulariser; `U_peak × 1300 N` is the two-hand grip capacity required to execute the release phase.
Every solution is re-integrated with RK4 at 1 ms; the CSV rows carry the HS residuals and the
re-integration errors (audit columns `res_*`, `ver_*`).

## Stage 2 — 3-D environment and search (glacus)

```bash
# smoke test
python -c "from katsumi.mujoco.env import CliffEnv; e=CliffEnv(); e.reset(); print(e.obs_dim, e.n_act)"
# CMA-ES search initialised from a planar solution (use as many workers as cores)
python scripts/search_3d.py --T 20 --m 66 --planar results/grid/sol_ref_T20_m60_phi0.250.pkl \
       --gens 300 --pop 32 --workers 32 --out results/search3d/T10_m66
MUJOCO_GL=egl python scripts/render_episode.py --search results/search3d/T10_m66.pkl --out results/figs/strip.png
```
Action = 31 normalised joint targets + 2 finger commands; observation = plan eq. (13) (126 dims).
`env.log` holds per-control-step hand forces (max, 10 ms average, impulse), torques, joint work,
slip, contact modes, and the wall-contact impulse. Curriculum knobs: `GraspParams.f_cap_hand`,
`hook_tol`, `ModelParams.cap_scale`, `EnvParams.illegal_support`.

RL baselines (SAC / MBPO / DreamerV3) are to be run against `CliffEnv` with the same
termination / load definitions; the environment exposes `reset(T, m_body, phi0)` for conditioning.

## Dual field learning (planar, compiled environment)

```bash
# 1. dual labels: remaining-problem solves (value, costate, time-to-release, capacity, prices) along / around references
python scripts/dual_field_data.py --refs "results/grid/sol_ref_T1[6789]_m*_phi*.pkl" --out results/dual_field --stride 15 --levels 0 1 2 --workers 12
# 2. fit the dual field (Sobolev), value-only ablation, behaviour cloning; closed-loop evaluation
python scripts/dfl_experiment.py --data results/dual_field --refs "results/grid/sol_ref_T1[6789]_m*_phi*.pkl" --holdout-m 69 --out results/dfl/it0
python scripts/dfl_experiment.py --eval-only results/dfl/it0/models.pt --refs "results/grid/sol_ref_T18_m*_phi0.250.pkl" \
       --ctrls DFL,BC --H 15 --replan 2 --w-tau 20 --kp-track 1 --kd-track 0.1 --out results/dfl/eval     # executor: 2 ms, mid-point sampling
python scripts/dfl_experiment.py --eval-only none --refs "results/grid/sol_ref_T18_m*_phi0.250.pkl" --ctrls ORACLE --out results/dfl/oracle
# 3. executor decomposition: replay of oracle solutions (control period / feed-forward sampling / catch model)
python scripts/replay_table.py --refs "results/grid/sol_ref_T18_m*_phi0.250.pkl" --control-dt 0.02 0.002 --lead zero half --reach 0.25
# 4. RL baselines, analysis, paper numbers, manuscript
python scripts/train_planar_rl.py --shaping dual --steps 3000000 --out results/rl_planar/ppo_dual
python scripts/analyze_learning.py --rl results/rl_planar --dfl results/dfl --out results/figs
python scripts/paper_numbers_dfl.py --data results/dual_field --dfl results/dfl --rl results/rl_planar --grid results/grid
bash scripts/package_paper.sh outputs          # PDF + self-contained LaTeX source
```
Environment (`katsumi/planar/env.py`, `fastsim.py`): CasADi code-generated physics of the optimiser, soft joint stops,
finger-hook capacity on both cliffs, sliding hook on B, bracing contact with B's wall panel while hooked, first- or
zero-order hold of the torque commands. Controllers (`katsumi/learn/`): dual-field MPC (`swing_mpc.py`), oracle-in-the-loop
MPC (`oracle_mpc.py`), landing reflex shared by all methods (`reflex.py`).

## Experiment suite of the paper (experiments 0–9)

Frozen evaluation rules: `scripts/exp/exp0_config.py` (`F_ref` = 1300 N is the display unit, `F_max` = c·`F_ref` the capacity
allowed in closed loop, headline c = 2.0; success = rest start → release before the deadline → catch → 2 s hold).
```
bash scripts/exp/run_suite.sh        # solver lane (exp1, exp9, exp2, exp3) and learning lane (exp4 → training → exp5 → exp6 → exp7) on CPU + GPU
bash scripts/exp/run_extra.sh        # exp8 (generalisation, envelope transfer) and the 0.8 s horizon ablation of exp6
bash scripts/exp/run_followup.sh     # exp3 re-solve, exp4 tight finite-difference check of the costates
python scripts/exp/exp9_polish.py --grid results/grid --tag ref --workers 22 --passes 12    # lattice-neighbour continuation of the 864-case grid
python scripts/exp/exp1_objective.py --grid results/grid --out results/suite/exp1           # objective weights, from the polished grid
MUJOCO_GL=osmesa python scripts/fig_3d.py --ref results/grid/sol_ref_T18_m66_phi0.250.pkl --out results/figs   # 3D figures
python scripts/paper_final.py --parts all      # every figure, table and number of the manuscript (paper/numbers_final.tex)
bash scripts/package_paper.sh outputs          # PDF + self-contained LaTeX source
```
| experiment | script | question |
|---|---|---|
| 1 | `exp1_objective.py` | does the objective weight change the required capacity? |
| 2 | `exp2_replay.py` | why does the replay of a plan fail (control period, release time, mesh, start phase)? |
| 3 | `exp3_margins.py` | do constraint margins or release-time scenarios make plans executable? |
| 4 | `exp4_audit.py` | are the dual labels correct (dependence on the last command, finite differences, trajectory labels)? |
| 5 | `exp5_ranking.py` | does the learned field rank candidate actions like the oracle? |
| 6 | `exp6_compare.py` | closed loop: dual-field MPC against the same MPC without dual labels, and the other controllers |
| 7 | `exp7_data.py` | amount of data against how it is collected (reference neighbourhood / visited states) |
| 8 | `exp8_general.py` | extrapolation in mass, period and stature; first-order transfer by the envelope theorem |
| 9 | `exp9_physics.py`, `exp9_polish.py` | which phase sets the required capacity; the complete 864-case grid |

`results/suite/` is working data and is not tracked; `results/suite_final/` is the tracked snapshot of its tables
(CSV / JSON, the summary of the grid before and after the continuation, the continuation log). `scripts/paper_final.py`
reads `results/suite` and falls back to `results/suite_final`.

## Conventions

`T` is the FULL device period (one cycle out and back); the one-way travel time is `T/2`.
Device phase `phi = (t mod T)/T` in [0, 1): `phi < 0.5` outbound (A descends, B moves away), `phi >= 0.5` return.
The plan's one-way times 8–12 s correspond to `T = 16–24 s`; competition settings 9.5 / 10 s are `T = 19 / 20 s`.

## Frames and units

Plan frame: +x A→B, +y up, z lateral. MuJoCo frame: x = plan x, y = plan z (lateral), z = plan y (up).
All quantities SI; angles in radians inside Python (degrees only in the MJCF text).
