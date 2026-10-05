"""Standalone stills of the backward jump for slides and posters (MuJoCo offscreen, the humanoid posed from the planar
solution exactly as in scripts/fig_3d.py):

  jump_01_... jump_10_...   key moments of one solution (hang, swing, release, flight, catch, hold), one fixed camera
  flight_close_*.png        close-ups of the release, the flight and the catch
  strobe_flight.png         multiple exposure of the flight (release -> catch) in one image
  strobe_swing.png          multiple exposure of the last forward swing up to the release
  strobe_jump.png           multiple exposure of the whole jump (last swing, flight, catch, swing-through on B)
  strobe_flight_*.png       the multiple exposure of the flight from other camera positions
  contact_sheet.png         all key moments on one sheet

The planar model lumps left and right limbs and treats the half twist of the flight as free: the body is drawn facing
cliff A until the middle of the flight and facing cliff B afterwards (the same convention as the manuscript).

usage: MUJOCO_GL=osmesa python scripts/render_jump_images.py --ref results/grid/sol_ref_T18_m66_phi0.250.pkl --out results/images
"""
import argparse, os, pickle, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco
from katsumi import device
from fig_3d import Scene, oracle_states, facing_of


class Stills(Scene):
    def __init__(self, r, w=1920, h=1080):
        super().__init__(m=r["m"], w=w, h=h)
        self.r = r
        mo = self.model
        self.human = np.array([i for i in range(mo.ngeom) if not mo.geom(i).name.startswith("cliff") and mo.geom(i).name != "water"])
        mo.geom_rgba[self.human] = (0.88, 0.47, 0.20, 1.0)                 # stills: deeper body colour, softer lights (no wash-out on white)
        mo.light_diffuse[:] *= 0.62; mo.light_specular[:] *= 0.5
        mo.site_rgba[:, 3] = 0.0                                           # no marker dots (grip sites, ledge markers)
        mo.vis.headlight.diffuse[:] = 0.42; mo.vis.headlight.ambient[:] = 0.38
        self.rgba0 = mo.geom_rgba.copy()

    def put(self, t):
        (t_, q, ph), = oracle_states(self.r, [t])
        return self.pose(q, t, self.r["T"], self.r["params"]["eps"], facing_of(ph, t, self.r))

    def device_at(self, tA, tB):
        """Place cliff A as at time tA and cliff B as at time tB (the multiple exposures show both at the moments of contact)."""
        T, eps = self.r["T"], self.r["params"]["eps"]
        a = device.device_state(tA, T, eps); b = device.device_state(tB, T, eps)
        self.data.mocap_pos[self.mocapA] = (a["pA"][0], 0.0, a["pA"][1]); self.data.mocap_pos[self.mocapB] = (b["pB"][0], 0.0, b["pB"][1])
        mujoco.mj_forward(self.model, self.data)

    def shot(self, cam):
        """(image with white background, mask of the humanoid pixels)."""
        r = self.renderer
        r.update_scene(self.data, camera=cam); img = r.render().copy()
        r.enable_segmentation_rendering(); r.update_scene(self.data, camera=cam); seg = r.render().copy(); r.disable_segmentation_rendering()
        gid, typ = seg[:, :, 0], seg[:, :, 1]
        img[gid < 0] = 255
        mask = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(gid, self.human)
        return img, mask

    def background(self, cam):
        mo = self.model
        mo.geom_rgba[self.human, 3] = 0.0
        img, _ = self.shot(cam)
        mo.geom_rgba[:] = self.rgba0
        return img

    def strobe(self, cam, times, tA, tB, alpha=(0.30, 0.80)):
        """Multiple exposure: earlier poses are faded, the last one is opaque. Cliff A as at tA, cliff B as at tB."""
        self.put(times[-1]); self.device_at(tA, tB)
        out = self.background(cam).astype(float)
        n = len(times)
        for k, t in enumerate(times):
            self.put(t); self.device_at(tA, tB)
            img, mask = self.shot(cam)
            a = 1.0 if k == n - 1 else alpha[0] + (alpha[1] - alpha[0]) * k / max(n - 2, 1)
            out[mask] = (1 - a) * out[mask] + a * img[mask]
        return out.round().astype(np.uint8)


def autocrop(img, margin=0.07, aspect=None):
    """Crop to the non-white content with a margin; aspect = width / height keeps a fixed format (padding with white)."""
    nz = np.argwhere((img < 250).any(axis=2))
    (y0, x0), (y1, x1) = nz.min(0), nz.max(0) + 1
    h, w = y1 - y0, x1 - x0; m = int(margin * max(h, w))
    H, W = h + 2 * m, w + 2 * m
    if aspect:
        if W / H < aspect: W = int(round(H * aspect))
        else: H = int(round(W / aspect))
    out = np.full((H, W, 3), 255, np.uint8)
    oy, ox = (H - h) // 2, (W - w) // 2
    out[oy:oy + h, ox:ox + w] = img[y0:y1, x0:x1]
    return out


def save(img, fn):
    Image.fromarray(img).save(fn, optimize=True); print("saved", fn, img.shape[1], "x", img.shape[0])


def font(sz):
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="results/grid/sol_ref_T18_m66_phi0.250.pkl"); ap.add_argument("--out", default="results/images")
    ap.add_argument("--w", type=int, default=1600); ap.add_argument("--h", type=int, default=1200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = pickle.load(open(a.ref, "rb"))
    sc = Stills(r, a.w, a.h)
    ts0, tl, tc, th0, ds, df = r["t_s0"], r["t_l"], r["t_c"], r["t_h0"], r["d_s"], r["d_f"]
    main_cam = sc.camera(lookat=(0.80, 0.0, -0.33), distance=5.25, azimuth=74.0, elevation=-7.0)
    # ---- key moments, one fixed camera (the frames can be shown as a sequence)
    keys = [("hang", "ぶら下がり（振り開始）", ts0), ("backswing", "振り（後方）", ts0 + 0.78 * ds), ("foreswing", "振り（前方，離手の直前）", ts0 + 0.93 * ds),
            ("release", "離手", tl), ("flight1", "飛行（前半）", tl + 0.25 * df), ("flight2", "飛行（中間）", tl + 0.50 * df),
            ("flight3", "飛行（後半）", tl + 0.80 * df), ("catch", "捕捉", th0 + 0.01), ("hold1", "保持（捕捉後 0.35 s）", th0 + 0.35), ("hold2", "保持（捕捉後 2.0 s）", th0 + 2.0)]
    thumbs = []
    for i, (tag, jp, t) in enumerate(keys, 1):
        sc.put(t)
        img, _ = sc.shot(main_cam)
        save(img, os.path.join(a.out, f"jump_{i:02d}_{tag}.png"))
        thumbs.append((img, f"{i:02d} {jp}（t = {t - ts0:.2f} s）"))
    # ---- close-ups of the flight
    close = sc.camera(lookat=(1.12, 0.0, 0.06), distance=3.75, azimuth=72.0, elevation=-6.0)
    for i, (tag, t) in enumerate((("release", tl), ("flight1", tl + 0.25 * df), ("flight2", tl + 0.50 * df), ("flight3", tl + 0.80 * df), ("catch", th0 + 0.01)), 1):
        sc.put(t); img, _ = sc.shot(close)
        save(img, os.path.join(a.out, f"flight_close_{i}_{tag}.png"))
    # ---- multiple exposures (cliff A as at the release, cliff B as at the catch)
    fl = [tl, tl + 0.30 * df, tl + 0.62 * df, th0 + 0.01]
    save(autocrop(sc.strobe(close, fl, tl, tc, alpha=(0.42, 0.80)), aspect=4 / 3), os.path.join(a.out, "strobe_flight.png"))
    sw = [ts0 + f * ds for f in (0.80, 0.86, 0.91, 0.955)] + [tl]
    save(autocrop(sc.strobe(main_cam, sw, tl, tc, alpha=(0.35, 0.80)), aspect=4 / 3), os.path.join(a.out, "strobe_swing.png"))
    full = [ts0 + 0.89 * ds, ts0 + 0.95 * ds, tl, tl + 0.30 * df, tl + 0.62 * df, th0 + 0.01, th0 + 0.35]
    save(autocrop(sc.strobe(main_cam, full, tl, tc, alpha=(0.38, 0.85)), aspect=4 / 3), os.path.join(a.out, "strobe_jump.png"))
    # ---- the flight from other camera positions
    views = dict(side=sc.camera(lookat=(1.12, 0.0, -0.15), distance=3.9, azimuth=90.0, elevation=0.0),
                 quarter=sc.camera(lookat=(1.15, 0.0, -0.15), distance=4.0, azimuth=56.0, elevation=-10.0),
                 low=sc.camera(lookat=(1.12, 0.0, 0.0), distance=3.9, azimuth=66.0, elevation=18.0),
                 above=sc.camera(lookat=(1.12, 0.0, -0.15), distance=4.3, azimuth=80.0, elevation=-38.0))
    for nm, cam in views.items():
        save(autocrop(sc.strobe(cam, fl, tl, tc, alpha=(0.42, 0.80)), aspect=4 / 3), os.path.join(a.out, f"strobe_flight_{nm}.png"))
    # ---- contact sheet
    cols = 5; rows = int(np.ceil(len(thumbs) / cols)); tw = 640; th_ = int(tw * a.h / a.w); cap = 44
    sheet = Image.new("RGB", (cols * tw, rows * (th_ + cap)), "white"); dr = ImageDraw.Draw(sheet); f = font(22)
    for k, (img, txt) in enumerate(thumbs):
        x, y = (k % cols) * tw, (k // cols) * (th_ + cap)
        sheet.paste(Image.fromarray(img).resize((tw, th_), Image.LANCZOS), (x, y + cap))
        dr.text((x + 14, y + 10), txt, fill=(40, 40, 40), font=f)
    sheet.save(os.path.join(a.out, "contact_sheet.png"), optimize=True); print("saved contact_sheet")


if __name__ == "__main__":
    main()
