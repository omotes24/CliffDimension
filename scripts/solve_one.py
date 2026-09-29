"""Solve a single (T, m, phi0) case of the reduced model and print the key quantities."""
import argparse, time, pickle, sys
import numpy as np

sys.path.insert(0, ".")
from katsumi.planar.anthro import make_body
from katsumi.planar.nlp import PlanarNLP, ReducedParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--m", type=float, default=66.0)
    ap.add_argument("--phi0", type=float, default=0.0)
    ap.add_argument("--d_w", type=float, default=0.0)
    ap.add_argument("--d_s", type=float, default=3.0)
    ap.add_argument("--print_level", type=int, default=5)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()
    body = make_body(a.m)
    t = time.time()
    nlp = PlanarNLP(body, a.T, a.phi0, ReducedParams())
    print(f"build {time.time()-t:.1f}s")
    nlp.set_initial(d_w=a.d_w, d_s=a.d_s)
    t = time.time()
    r = nlp.solve(print_level=a.print_level)
    print(f"solve {time.time()-t:.1f}s  ok={r['ok']} status={r['status']} iters={r['iters']}")
    print(f"d_w={r['d_w']:.3f} d_s={r['d_s']:.3f} d_f={r['d_f']:.3f} d_c={r['d_c']:.3f}")
    print(f"phi_l={r['phi_l']:.3f} phi_c={r['phi_c']:.3f} U_peak={r['U_peak']:.3f} E_eff={r['effort']:.3f}")
    qF = r["qF"]; qdF = r["qdF"]
    print("release hand pos", qF[:2, 0], "vel", qdF[:2, 0])
    print("catch  hand pos", qF[:2, -1], "vel", qdF[:2, -1])
    if a.out:
        with open(a.out, "wb") as f:
            pickle.dump(r, f)


if __name__ == "__main__":
    main()
