"""RL baselines on the planar Cliff-Dimension environment (swing + release; landing reflex after the release).

    python scripts/train_planar_rl.py --algo ppo --shaping none --steps 3000000 --n-envs 16 --out results/rl_planar/ppo_none
    shaping in {none, energy, dual, dual+energy, margin+energy}

The agent acts while hooked on A (4 torques + release trigger); after the release the landing reflex (reference
flight / hold tracking, same for every method) takes over, so that the learning problem is the pumping + release
timing. Success = held B for 2 s. Episodes start hanging still at the swing-start phase of the reference solution
(--phi0) or at a random phase (--phi0 -1). Logs episode outcomes to <out>/episodes.csv.
"""
import argparse, csv, glob, json, os, pickle, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
from katsumi.planar.env import PlanarCliffEnv, prices_from_duals, prices_from_rows
from katsumi.learn.reflex import LandingReflex, ReflexEnv


class GymReflexEnv(gym.Env):
    """Gymnasium-compatible wrapper around ReflexEnv with outcome logging."""
    metadata = {"render_modes": []}

    def __init__(self, env_kw, ref_glob, phi0, log_path=None, rsi=0.0, rsi_refs=None, seed=0):
        super().__init__()
        self.base = PlanarCliffEnv(**env_kw)
        self.wrapped = ReflexEnv(self.base, LandingReflex.from_glob(ref_glob))
        self.observation_space, self.action_space = self.base.observation_space, self.base.action_space
        self.phi0 = phi0
        self.log_path = log_path
        self.n_ep = 0
        # reference-state initialisation (DeepMimic-style): with probability `rsi` start from a random knot of a
        # random oracle swing (same body / period as the environment)
        self.rsi = rsi
        self.rsi_pool = []
        if rsi > 0:
            for f in sorted(glob.glob(rsi_refs or ref_glob)):
                r = pickle.load(open(f, "rb"))
                if r.get("ok") and abs(r["T"] - env_kw.get("T", r["T"])) < 1e-9 and abs(r["m"] - env_kw.get("m", r["m"])) < 1e-9:
                    self.rsi_pool.append(r)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        opts = dict(options or {})
        if self.rsi > 0 and self.rsi_pool and self.rng.uniform() < self.rsi and not opts:
            r = self.rsi_pool[int(self.rng.integers(len(self.rsi_pool)))]
            N = r["S_X"].shape[1] - 1
            k = int(self.rng.integers(0, N - 5))
            opts = dict(t0=r["t_s0"] + r["d_s"] * k / N, state=(r["S_X"][:5, k], r["S_X"][5:, k]))
        elif self.phi0 is not None and self.phi0 >= 0 and "phi0" not in opts:
            opts["phi0"] = self.phi0
        return self.wrapped.reset(seed=seed, options=opts)

    def step(self, action):
        obs, r, term, trunc, info = self.wrapped.step(action)
        if term or trunc:
            self.n_ep += 1
            res = self.base.result
            info["episode_result"] = res
            if self.log_path:
                new = not os.path.exists(self.log_path)
                with open(self.log_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["time", "success", "reason", "t", "U_peak", "release_time", "phi_release", "d_min", "T", "m"])
                    w.writerow([time.time(), int(res["success"]), res["reason"], round(res["t"], 3), round(res["U_peak"], 4),
                                res["release_time"], res["phi_release"], res["d_min"] if np.isfinite(res["d_min"]) else "", res["T"], res["m"]])
        return obs, r, term, trunc, info


def make_env_fn(env_kw, ref_glob, phi0, log_path, seed, rsi=0.0, rsi_refs=None):
    def _f():
        kw = dict(env_kw); kw["seed"] = seed
        return GymReflexEnv(kw, ref_glob, phi0, log_path, rsi=rsi, rsi_refs=rsi_refs, seed=seed)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--shaping", default="none")
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--T", type=float, default=18.0)
    ap.add_argument("--m", type=float, default=66.0)
    ap.add_argument("--phi0", type=float, default=None, help="start phase; default = swing start of the reference; -1 = random")
    ap.add_argument("--refs", default="results/grid/sol_ref_T18_m66_phi0.250.pkl")
    ap.add_argument("--duals", default=None, help="run_duals JSON for the constraint prices (dual shaping)")
    ap.add_argument("--prices-rows", default=None, help="dual-field rows.csv for the constraint prices (dual shaping)")
    ap.add_argument("--price-scale", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--control-dt", type=float, default=0.02)
    ap.add_argument("--log-std-init", type=float, default=-1.0, help="PPO initial exploration std = exp(.) (0.37)")
    ap.add_argument("--rsi", type=float, default=0.0, help="probability of a reference-state initialisation per episode")
    ap.add_argument("--rsi-refs", default=None, help="glob of oracle swings for RSI (default: --refs)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    refs = sorted(glob.glob(a.refs))
    r0 = pickle.load(open(refs[0], "rb"))
    phi0 = a.phi0 if a.phi0 is not None else (r0["t_s0"] % r0["T"]) / r0["T"]
    prices = prices_from_duals(a.duals) if a.duals else (prices_from_rows(a.prices_rows) if a.prices_rows else None)
    env_kw = dict(T=a.T, m=a.m, shaping=a.shaping, prices=prices, price_scale=a.price_scale, control_dt=a.control_dt)
    json.dump(dict(vars(a), phi0=phi0, prices=prices), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import CheckpointCallback
    fns = [make_env_fn(env_kw, a.refs, phi0, os.path.join(a.out, "episodes.csv"), a.seed * 1000 + i, a.rsi, a.rsi_refs) for i in range(a.n_envs)]
    venv = VecMonitor(SubprocVecEnv(fns) if a.n_envs > 1 else DummyVecEnv(fns))
    if a.algo == "ppo":
        model = PPO("MlpPolicy", venv, n_steps=512, batch_size=2048, n_epochs=10, learning_rate=3e-4, gamma=0.995, gae_lambda=0.97,
                    ent_coef=0.003, clip_range=0.2, policy_kwargs=dict(net_arch=[256, 256], log_std_init=a.log_std_init), verbose=1,
                    seed=a.seed, device=a.device)
    else:
        model = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=1_000_000, batch_size=512, gamma=0.995, train_freq=1,
                    gradient_steps=1, policy_kwargs=dict(net_arch=[256, 256]), verbose=1, seed=a.seed, device=a.device)
    cb = CheckpointCallback(save_freq=max(200_000 // a.n_envs, 1), save_path=a.out, name_prefix="model")
    model.learn(total_timesteps=a.steps, callback=cb, log_interval=10)
    model.save(os.path.join(a.out, "final"))


if __name__ == "__main__":
    main()
