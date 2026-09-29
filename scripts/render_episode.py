"""Render snapshots / a strip of frames from the 3-D environment (OSMesa or EGL offscreen).

usage:
  MUJOCO_GL=osmesa python scripts/render_episode.py --search results/search3d/run.pkl --out results/figs/strip3d.png
  MUJOCO_GL=osmesa python scripts/render_episode.py --hang --out results/figs/hang3d.png
"""
import argparse, os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco
from katsumi.mujoco.env import CliffEnv, EnvParams
from katsumi.mujoco.build_model import ModelParams
from katsumi.mujoco.search import SplinePolicy, SearchParams, rollout


def make_camera(lookat=(1.1, 0.0, -0.2), distance=4.2, azimuth=125, elevation=-8):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def render_state(env, renderer, cam):
    opt = mujoco.MjvOption()
    renderer.update_scene(env.data, camera=cam, scene_option=opt)
    return renderer.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=str, default="")
    ap.add_argument("--hang", action="store_true")
    ap.add_argument("--out", type=str, default="results/figs/render.png")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    a = ap.parse_args()
    from PIL import Image
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.hang or not a.search:
        env = CliffEnv(EnvParams(T=10.0, phi0=0.0))
        env.reset()
        r = mujoco.Renderer(env.model, a.h, a.w)
        img = render_state(env, r, make_camera())
        Image.fromarray(img).save(a.out)
        print("saved", a.out)
        return
    d = pickle.load(open(a.search, "rb"))
    ep = EnvParams(model=ModelParams(m=d["m"]), T=d["T"], phi0=d["phi0"])
    env = CliffEnv(ep)
    sp = SearchParams(**d["sp"])
    env.reset(T=d["T"], m_body=d["m"], phi0=d["phi0"])
    pol = SplinePolicy(env, sp)
    x = d["search"]["x_best"]
    frames = []
    env.reset(T=d["T"], m_body=d["m"], phi0=d["phi0"])
    r = mujoco.Renderer(env.model, a.h, a.w)
    cam = make_camera()
    t = 0.0
    while True:
        act = pol.action(x, t)
        obs, rew, term, trunc, info = env.step(act)
        t = env.t_elapsed
        frames.append(render_state(env, r, cam))
        if term or trunc:
            break
    idx = np.linspace(0, len(frames) - 1, a.n).round().astype(int)
    strip = np.concatenate([frames[i] for i in idx], axis=1)
    Image.fromarray(strip).save(a.out)
    print("saved", a.out, "frames", len(frames), "result", env.result)


if __name__ == "__main__":
    main()
