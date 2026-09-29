"""Model-free SAC baseline (plan Sec. 12.1) with Stable-Baselines3 — intended for glacus.

pip install gymnasium stable-baselines3 torch
python scripts/train_sac.py --steps 2000000 --seed 0 --out results/sac/seed0 [--fixed-T 10 --fixed-m 66]

Uses the plan's learning reward (env.py: 100 S_end - 100 F_end - w_U dU_peak - w_E dE - 0.01 dt/t_ref),
gamma = 0.999, batch 256, lr 3e-4, MLP 3x256 (plan Sec. 10.2). Evaluation with the frozen policy is done
by scripts/eval_policy.py on the condition grid with 8 phase bins x n trials.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/sac/run")
    ap.add_argument("--fixed-T", type=float, default=None)
    ap.add_argument("--fixed-m", type=float, default=None)
    ap.add_argument("--n-envs", type=int, default=8)
    a = ap.parse_args()
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from katsumi.mujoco.gym_wrapper import CliffGym
    fixed = None
    if a.fixed_T is not None and a.fixed_m is not None:
        fixed = dict(T=a.fixed_T, m=a.fixed_m, phi0=0.0)

    def make(rank):
        def _f():
            return Monitor(CliffGym(fixed=fixed, seed=a.seed * 1000 + rank))
        return _f

    os.makedirs(a.out, exist_ok=True)
    venv = SubprocVecEnv([make(i) for i in range(a.n_envs)])
    model = SAC("MlpPolicy", venv, learning_rate=3e-4, batch_size=256, gamma=0.999, buffer_size=1_000_000,
                policy_kwargs=dict(net_arch=[256, 256, 256]), seed=a.seed, verbose=1, tensorboard_log=a.out)
    model.learn(total_timesteps=a.steps, log_interval=10)
    model.save(os.path.join(a.out, "sac_final"))


if __name__ == "__main__":
    main()
