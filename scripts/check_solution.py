"""Audit a saved solution: HS residuals at quarter points, RK4 re-integration, metrics, plot."""
import pickle, numpy as np, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from katsumi.planar.anthro import make_body
from katsumi.planar.model import PlanarChain, NTH, NQ
from katsumi import device
from katsumi.planar.simulate import _hs_interp, Resim
from katsumi.planar.plots import plot_solution


def hs_residuals(r, body):
    ch = PlanarChain(body)
    T, eps = r["T"], r["params"]["eps"]
    out = {}
    for ph, dur, t0 in (("S", r["d_s"], r["t_s0"]), ("F", r["d_f"], r["t_l"]),
                        ("H", r["params"]["T_hold"], r["t_h0"])):
        X, A = r[ph + "_X"], r[ph + "_A"]
        N = X.shape[1] - 1; dt = dur / N
        na = A.shape[0]
        res = []
        for k in range(N):
            x0, x1 = X[:, k], X[:, k + 1]
            f0 = np.concatenate([X[na:, k], A[:, k]]); f1 = np.concatenate([X[na:, k + 1], A[:, k + 1]])
            for s in (0.25, 0.5, 0.75):
                h00 = 2*s**3-3*s**2+1; h10 = s**3-2*s**2+s; h01 = -2*s**3+3*s**2; h11 = s**3-s**2
                xs = h00*x0 + h10*dt*f0 + h01*x1 + h11*dt*f1
                dh00 = 6*s**2-6*s; dh10 = 3*s**2-4*s+1; dh01 = -6*s**2+6*s; dh11 = 3*s**2-2*s
                xds = (dh00*x0 + dh10*dt*f0 + dh01*x1 + dh11*dt*f1) / dt
                t = t0 + dt * (k + s)
                dv = device.device_state(t, T, eps)
                u = _hs_interp(r[ph + "_U"], r[ph + "_Um"], dur, dt * (k + s)) * body.tau_cap
                if ph in ("S", "H"):
                    pp, vv, aa = (dv["pA"], dv["vA"], dv["aA"]) if ph == "S" else (dv["pB"], dv["vB"], dv["aB"])
                    q = np.concatenate([pp, xs[:NTH]]); qd = np.concatenate([vv, xs[NTH:]])
                    thdd, _ = ch.pinned(q, qd, u, aa)
                    f = np.concatenate([xs[NTH:], thdd])
                else:
                    f = np.concatenate([xs[NQ:], ch.qdd_free(xs[:NQ], xs[NQ:], u)])
                res.append(np.abs(xds - f)[na:].max())   # acceleration residual [rad/s^2 or m/s^2]
        res = np.array(res)
        out[ph] = dict(max=float(res.max()), p95=float(np.percentile(res, 95)), median=float(np.median(res)))
    return out


if __name__ == "__main__":
    r = pickle.load(open(sys.argv[1], "rb"))
    body = make_body(r["m"])
    np.set_printoptions(precision=4, suppress=True)
    print("HS residuals:", hs_residuals(r, body))
    rs = Resim(body, r)
    print("verify:", {k: round(float(v), 5) for k, v in rs.verify().items()})
    for k, v in rs.metrics().items(): print(f"  {k}: {v}")
    if len(sys.argv) > 2:
        plot_solution(r, body, sys.argv[2])
