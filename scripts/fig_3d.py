"""Three-dimensional views for the manuscript, rendered offscreen with MuJoCo (the 31-joint humanoid and the two cliffs
of katsumi/mujoco, posed kinematically from the planar model):

  device3d.pdf      the device with the athlete hanging on cliff A, with the dimensions and the coordinate frame
  oracle3d.pdf      the optimal solution (swing / release / flight / catch / hold) as a strip of 3-D frames
  replay3d.pdf      the same solution executed in the compiled environment (2 ms executor + landing reflex)
  failures3d.pdf    typical first failures of the closed-loop trials (replays with perturbed starts)

The planar model lumps left and right limbs, so both sides of the humanoid take the same sagittal joint angles. The
planar model treats the half twist of the flight as free: the frames show the body facing cliff A until the middle of
the flight and facing cliff B afterwards; the positions of all joints in the sagittal plane are those of the model.

usage: MUJOCO_GL=osmesa python scripts/fig_3d.py --ref results/grid/sol_ref_T18_m66_phi0.250.pkl --out results/figs
"""
import argparse, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
from katsumi import device
from katsumi.mujoco.build_model import build_xml, ModelParams
from katsumi.planar.anthro import make_body
from katsumi.planar.model import NQ, NTH


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Scene:
    def __init__(self, m=66.0, stature=1.75, w=820, h=600):
        xml = build_xml(ModelParams(m=m, stature=stature))
        # visual-only trunk and neck (the simulation model has bar-shaped trunk geoms; inertias are given explicitly)
        vis = ('<geom name="vis_trunk" type="capsule" fromto="0 0 -0.02 0 0 0.30" size="0.115" contype="0" conaffinity="0"/>'
               '<geom name="vis_neck" type="capsule" fromto="0 0 0.30 0 0 0.44" size="0.05" contype="0" conaffinity="0"/>')
        xml = xml.replace('<site name="chest"', vis + '<site name="chest"', 1)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.body = make_body(m, stature=stature)
        self.w, self.h = w, h
        self.renderer = mujoco.Renderer(self.model, h, w)
        mo = self.model
        self.mocapA = mo.body_mocapid[mo.body("cliffA").id]; self.mocapB = mo.body_mocapid[mo.body("cliffB").id]
        mo.geom_rgba[mo.geom("water").id, 3] = 0.0
        for i in range(mo.ngeom):                         # print-friendly colours
            nm = mo.geom(i).name
            if nm.startswith("cliff"):
                solid = nm.endswith("_ledge") or nm.endswith("_edge") or nm.endswith("_tipface")
                mo.geom_rgba[i] = (0.30, 0.34, 0.40, 1.0) if solid else (0.62, 0.68, 0.74, 0.45)
            elif nm != "water":
                mo.geom_rgba[i] = (0.86, 0.55, 0.30, 1.0)
        self.root = mo.joint("root").qposadr[0]
        self.j = {n: mo.joint(n).qposadr[0] for n in
                  [f"{k}_{s}" for k in ("shoulder_flex", "elbow", "hip_flex", "knee", "ankle_pf", "wrist_flex") for s in ("left", "right")]}
        self.grip = [mo.site("grip_left").id, mo.site("grip_right").id]

    def pose(self, q, t, T, eps, facing):
        """q = (hand x, hand y, theta_1..5) of the planar model at absolute time t; facing 'neg' (chest to A) / 'pos' (chest to B)."""
        mo, d = self.model, self.data
        dv = device.device_state(t, T, eps)
        d.mocap_pos[self.mocapA] = (dv["pA"][0], 0.0, dv["pA"][1]); d.mocap_pos[self.mocapB] = (dv["pB"][0], 0.0, dv["pB"][1])
        th = np.asarray(q[2:7], float)
        dirs = np.stack([np.sin(th), -np.cos(th)], 1)             # link directions, proximal (hand side) -> distal
        s = -1.0 if facing == "neg" else 1.0

        def psi(v):                                               # angle from "down" towards the chest side
            return np.arctan2(s * v[0], -v[1])
        up = -dirs[2]
        beta = np.arctan2(s * up[0], up[1])
        a_sh = wrap(psi(-dirs[1]) + beta)
        if a_sh < -np.pi / 3:                                     # the shoulder joint range of the humanoid is (-60, 180) deg
            a_sh += 2 * np.pi
        a_el = wrap(psi(-dirs[0]) - psi(-dirs[1]))
        a_hip = wrap(psi(dirs[3]) + beta)
        a_kn = wrap(psi(dirs[3]) - psi(dirs[4]))
        d.qpos[:] = 0.0
        yaw = np.pi if facing == "neg" else 0.0
        qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]); qp = np.array([np.cos(beta / 2), 0, np.sin(beta / 2), 0])
        quat = np.zeros(4); mujoco.mju_mulQuat(quat, qy, qp)
        d.qpos[self.root:self.root + 3] = 0.0; d.qpos[self.root + 3:self.root + 7] = quat
        for sd in ("left", "right"):
            d.qpos[self.j[f"shoulder_flex_{sd}"]] = a_sh; d.qpos[self.j[f"elbow_{sd}"]] = a_el
            d.qpos[self.j[f"hip_flex_{sd}"]] = a_hip; d.qpos[self.j[f"knee_{sd}"]] = a_kn
            d.qpos[self.j[f"ankle_pf_{sd}"]] = 0.45
        mujoco.mj_kinematics(mo, d)
        g = 0.5 * (d.site_xpos[self.grip[0]] + d.site_xpos[self.grip[1]])
        d.qpos[self.root:self.root + 3] = np.array([q[0], 0.0, q[1]]) - g
        mujoco.mj_forward(mo, d)
        return dv

    def camera(self, lookat=(0.85, 0.0, -0.42), distance=5.1, azimuth=76.0, elevation=-8.0):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat; cam.distance = distance; cam.azimuth = azimuth; cam.elevation = elevation
        return cam

    def render(self, cam, white=True):
        r = self.renderer
        r.update_scene(self.data, camera=cam)
        img = r.render().copy()
        if white:
            r.enable_segmentation_rendering(); r.update_scene(self.data, camera=cam)
            seg = r.render()[:, :, 0]; r.disable_segmentation_rendering()
            img[seg < 0] = 255
        return img

    def project(self, cam, pts):
        """Pixel coordinates (u from the left, v from the top) of world points for a free camera."""
        az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        pos = np.array(cam.lookat) - cam.distance * fwd
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); upv = np.cross(right, fwd)
        f = 0.5 * self.h / np.tan(np.radians(self.model.vis.global_.fovy) / 2)
        out = []
        for p in np.atleast_2d(pts):
            v = np.asarray(p, float) - pos
            zc = v @ fwd
            out.append((self.w / 2 + f * (v @ right) / zc, self.h / 2 - f * (v @ upv) / zc))
        return np.array(out)


# ------------------------------------------------------------------------------------------------- frames of a solution
def oracle_states(r, times):
    T, eps = r["T"], r["params"]["eps"]
    out = []
    for t in times:
        if t <= r["t_l"]:
            X, t0, dur, ph = r["S_X"], r["t_s0"], r["d_s"], "A"
        elif t < r["t_c"]:
            X, t0, dur, ph = r["F_X"], r["t_l"], r["d_f"], "F"
        else:
            X, t0, dur, ph = r["H_X"], r["t_h0"], r["params"]["T_hold"], "C"
        N = X.shape[1] - 1; tt = t0 + np.linspace(0, dur, N + 1)
        x = np.array([np.interp(t, tt, row) for row in X])
        dv = device.device_state(t, T, eps)
        q = np.concatenate([dv["pA"], x[:NTH]]) if ph == "A" else (np.concatenate([dv["pB"], x[:NTH]]) if ph == "C" else x[:NQ])
        out.append((t, q, ph))
    return out


def facing_of(ph, t, r):
    if ph == "A":
        return "neg"
    if ph == "C":
        return "pos"
    return "neg" if t < 0.5 * (r["t_l"] + r["t_c"]) else "pos"


def strip(sc, frames, r, labels, fn, cam=None, ncol=None, title=None, sub=None, dpi=220, crop=(0.0, 1.0, 0.13, 0.93)):
    """frames: list of (t, q, phase[, facing]); one rendered panel per frame."""
    cam = cam or sc.camera()
    n = len(frames); ncol = ncol or n; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.1, 7.1 / ncol * 0.585 * nrow + (0.30 if ncol <= 4 else 0.36) * nrow + (0.25 if title else 0)))
    axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k >= n:
            continue
        fr = frames[k]; t, q, ph = fr[:3]
        fc = fr[3] if len(fr) > 3 else facing_of(ph, t, r)
        sc.pose(q, t, r["T"], r["params"]["eps"], fc)
        img = sc.render(cam)
        x0, x1, y0, y1 = int(crop[0] * sc.w), int(crop[1] * sc.w), int(crop[2] * sc.h), int(crop[3] * sc.h)
        ax.imshow(img[y0:y1, x0:x1])
        ax.set_title(labels[k], fontsize=7.5 if ncol <= 4 else 6.0, pad=2)
        if sub is not None and sub[k]:
            ax.text(0.5, -0.02, sub[k], transform=ax.transAxes, ha="center", va="top", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9, y=0.995)
    fig.subplots_adjust(left=0.005, right=0.995, top=(0.93 if title else 0.95) if ncol <= 4 else 0.88, bottom=0.01, wspace=0.03, hspace=0.20 if ncol <= 4 else 0.42)
    fig.savefig(fn, dpi=dpi); fig.savefig(os.path.splitext(fn)[0] + ".png", dpi=dpi); plt.close(fig)
    print("saved", fn)


def device_figure(sc, r, out):
    """Overview of the device with the athlete hanging on cliff A at the swing start, and a close-up of the ledge."""
    T, eps = r["T"], r["params"]["eps"]
    big = Scene(m=r["m"], w=1200, h=860)
    t = 0.0                                                   # A at its top, B at its closest position
    q = np.concatenate([device.device_state(t, T, eps)["pA"], np.zeros(NTH)])
    dv = big.pose(q, t, T, eps, "neg")
    cam = big.camera(lookat=(1.15, 0.0, -0.10), distance=5.9, azimuth=108.0, elevation=-24.0)
    img = big.render(cam)
    fig = plt.figure(figsize=(7.1, 3.3))
    ax = fig.add_axes([0.0, 0.0, 0.655, 1.0]); ax.imshow(img); ax.axis("off")
    pA = np.array([dv["pA"][0], 0, dv["pA"][1]]); pB = np.array([dv["pB"][0], 0, dv["pB"][1]])
    P = lambda p: big.project(cam, p)[0]

    def arrow(a, b, txt, off=(0, 0), both=True, col="#B03A2E", ha="center", va="center"):
        ua, ub = P(a), P(b)
        ax.annotate("", xy=ub, xytext=ua, arrowprops=dict(arrowstyle="<->" if both else "->", color=col, lw=1.3, shrinkA=0, shrinkB=0))
        mid = 0.5 * (ua + ub) + np.array(off)
        ax.text(mid[0], mid[1], txt, color=col, fontsize=7.5, ha=ha, va=va, bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))

    yn = -0.78                                               # annotations on the near side of the ledges (ledge half width 0.6 m)
    arrow([-0.02, yn, 0.90], [-0.02, yn, 0.0], "A の上下動 $h(t)$\n0.90 → 0 m", off=(-70, -8))
    arrow([1.80, yn, 0.0], [2.70, yn, 0.0], "B の左右動（0.90 m）", off=(34, 22))
    arrow([0.0, yn, 0.0], [1.80, yn, 0.0], "先端間距離 $x(t)$\n1.80 → 2.70 m", off=(0, 24), col="#1F618D")
    o = np.array([-0.95, yn, -1.25])
    for vec, name, dxy in (([0.45, 0, 0], "$x$", (6, 6)), ([0, 0, 0.45], "$y$", (6, 2))):
        ua, ub = P(o), P(o + np.array(vec))
        ax.annotate("", xy=ub, xytext=ua, arrowprops=dict(arrowstyle="->", color="k", lw=1.4))
        ax.text(ub[0] + dxy[0], ub[1] + dxy[1], name, fontsize=9)
    for p, name in ((pA + [-0.03, 0.75, 0.80], "クリフ A"), (pB + [0.03, 0.75, 0.80], "クリフ B")):
        u = P(p); ax.text(u[0], u[1] - 10, name, fontsize=9, ha="center", weight="bold")
    for a_, b_ in (([2.70, -0.6, -0.2], [2.70, 0.6, -0.2]), ([2.70, -0.6, -0.2], [2.70, -0.6, 0.8]), ([2.70, 0.6, -0.2], [2.70, 0.6, 0.8]), ([2.70, -0.6, 0.8], [2.70, 0.6, 0.8])):
        ua, ub = P(a_), P(b_); ax.plot([ua[0], ub[0]], [ua[1], ub[1]], ls="--", lw=0.8, color="#777777")
    u = P([2.70, 0.0, 0.8]); ax.text(u[0] + 30, u[1] - 12, "破線：B の最遠位置", fontsize=7, ha="center", color="#555555")
    # close-up of the ledge and the hands, seen from below on the athlete's side
    cam2 = big.camera(lookat=(0.0, 0.0, dv["pA"][1] - 0.05), distance=0.95, azimuth=128.0, elevation=12.0)
    img2 = big.render(cam2)
    ax2 = fig.add_axes([0.66, 0.06, 0.335, 0.86]); ax2.imshow(img2[:, 170:1030]); ax2.axis("off")
    ax2.set_title("突起の拡大（奥行き 3 cm，高さ 5 cm）", fontsize=7.5, pad=2)
    fig.savefig(os.path.join(out, "device3d.pdf"), dpi=260); fig.savefig(os.path.join(out, "device3d.png"), dpi=200); plt.close(fig)
    print("saved device3d")


def env_log(r, sig_th=0.0, sig_thd=0.0, seed=0, env_kw=None):
    """Execute a reference in the compiled environment; returns ([(t, q, mode, U)], result)."""
    from katsumi.exp.trials import make_env, RefTracker, start_options
    from katsumi.exp.common import TRACK_KW
    from katsumi.planar.model import NTAU
    env = make_env(r, env_kw)
    pol = RefTracker(r, env.control_dt, kp=TRACK_KW["kp"], kd=TRACK_KW["kd"], lead=None, dt_release=0.0, exact=True)
    env.reset(options=start_options(r, "rest", sig_th, sig_thd, seed))
    log = [(env.t, env.q.copy(), env.mode, 0.0)]; done = False; steps = 0
    while not done and steps < int(40.0 / env.control_dt):
        a = pol(env.env) if env.mode == "A" else np.zeros(NTAU + 1)
        obs, rew, term, trunc, inf = env.step(a); done = term or trunc; steps += 1
        log.append((env.t, env.q.copy(), env.mode, env.U))
    return log, (env.env.result or dict(reason="max_steps"))


def pick(log, times):
    ts = np.array([l[0] for l in log])
    return [log[int(np.argmin(np.abs(ts - t)))] for t in times]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="results/grid/sol_ref_T18_m66_phi0.250.pkl"); ap.add_argument("--out", default="results/figs")
    ap.add_argument("--only", default="device,oracle,replay,failures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = pickle.load(open(a.ref, "rb")); r["ref_file"] = a.ref
    sc = Scene(m=r["m"])
    only = set(a.only.split(","))
    ts0, tl, tc, th0 = r["t_s0"], r["t_l"], r["t_c"], r["t_h0"]
    rel = [0.0, 0.45, 0.62, 0.78, 0.90]
    times = [ts0 + f * r["d_s"] for f in rel] + [tl, tl + 0.30 * r["d_f"], tl + 0.75 * r["d_f"], th0 + 0.02, th0 + 0.35, th0 + 1.0, th0 + 2.0]
    names = ["振り開始（静止）", "振り 45%", "振り 62%", "振り 78%", "振り 90%", "離手", "飛行（前半）", "飛行（後半）", "捕捉", "保持 +0.35 s", "保持 +1.0 s", "保持 +2.0 s"]
    if "device" in only:
        device_figure(sc, r, a.out)
    if "oracle" in only:
        fr = oracle_states(r, times)
        labels = [f"{n}\n$t$={t - ts0:.2f} s" for n, (t, _, _) in zip(names, fr)]
        strip(sc, fr, r, labels, os.path.join(a.out, "oracle3d.pdf"), ncol=4)
    if "replay" in only:
        log, res = env_log(r)
        print("replay:", res.get("reason"), "U_peak", res.get("U_peak"))
        t_rel = next((l[0] for l in log if l[2] != "A"), tl); t_cat = next((l[0] for l in log if l[2] == "C"), tc)
        tt = [ts0 + f * (t_rel - ts0) for f in rel] + [t_rel + 0.004, t_rel + 0.30 * (t_cat - t_rel), t_rel + 0.75 * (t_cat - t_rel), t_cat + 0.02, t_cat + 0.35, t_cat + 1.0, min(t_cat + 2.0, log[-1][0])]
        fr = pick(log, tt)
        frames = [(t, q, {"A": "A", "F": "F", "C": "C"}[md], ("neg" if md == "A" else ("pos" if md == "C" else ("neg" if t < 0.5 * (t_rel + t_cat) else "pos")))) for t, q, md, U in fr]
        labels = [f"{n}\n$t$={t - ts0:.2f} s，$U$={U:.2f}" for n, (t, _, _, U) in zip(names, fr)]
        strip(sc, frames, r, labels, os.path.join(a.out, "replay3d.pdf"), ncol=4)
        pickle.dump(dict(reason=res.get("reason"), U_peak=res.get("U_peak")), open(os.path.join(a.out, "replay3d_info.pkl"), "wb"))
    if "failures" in only:
        rows = []
        for seed in range(1, 40):
            for st, sd in ((0.01, 0.1), (0.03, 0.2)):
                log, res = env_log(r, st, sd, seed)
                rows.append((res.get("reason"), seed, st, sd, log))
            if len({x[0] for x in rows}) >= 5:
                break
        seen = {}
        for reason, seed, st, sd, log in rows:
            if reason not in seen and reason != "held B":
                seen[reason] = (seed, st, sd, log)
        print("failure examples:", {k: v[:3] for k, v in seen.items()})
        JP = {"hit wall": "壁に接触", "slipped off A": "A から滑落", "missed B": "B を掴み損ね", "hit B's face": "B の前面に衝突", "lost hook": "保持中に指が外れる",
              "grip capacity exceeded": "把持容量超過", "no release": "離手せず", "max_steps": "時間切れ"}
        frames, labels = [], []
        for reason, (seed, st, sd, log) in list(seen.items())[:4]:
            tend = log[-1][0]
            for dt_, nm in ((0.35, "−0.35 s"), (0.12, "−0.12 s"), (0.0, "失敗時")):
                t, q, md, U = pick(log, [tend - dt_])[0]
                t_rel = next((l[0] for l in log if l[2] != "A"), np.inf); t_cat = next((l[0] for l in log if l[2] == "C"), np.inf)
                fc = "neg" if md == "A" else ("pos" if md == "C" else ("neg" if t < t_rel + 0.25 else "pos"))
                frames.append((t, q, md, fc)); labels.append(f"{JP.get(reason, reason)}\n{nm}（$t$={t - ts0:.2f} s）")
        if frames:
            strip(sc, frames, r, labels, os.path.join(a.out, "failures3d.pdf"), ncol=6)


if __name__ == "__main__":
    main()
