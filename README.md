# CliffDimension — Katsumi

Physics / trajectory-optimisation study of the SASUKE *Cliff Dimension* backward jump
(moving cliff A → moving cliff B, 3 cm ledges).

* `katsumi/device.py` — device kinematics (eq. 2 triangle wave, eq. 4 smoothed wave, 3 cm ledge geometry)
* `katsumi/planar/` — Stage 1: planar 5-link reduced model, multi-phase Hermite–Simpson collocation (CasADi/IPOPT)
* `katsumi/mujoco/` — Stage 2: 3-D humanoid MuJoCo environment (extended `humanoid.xml`, finite-capacity grasp)
* `tests/` — unit tests (device end-points/period, free-flight CoM, angular momentum, pinned/free consistency)
* `scripts/` — experiment runners
* `paper/` — MIRU2025 LaTeX manuscript

```bash
pip install -r requirements.txt
python -m pytest tests -q
python scripts/solve_one.py --T 10 --m 66
```
