"""Sensitivity table: required capacity (BW) at T = 20 s (one-way 10 s), m = 60 kg for each perturbed factor vs reference.

usage: python scripts/analyze_sensitivity.py --grid results/grid --sens results/sens --out results/figs
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABELS = {
    "ref": "基準（$\\epsilon=0.20$ s，能力$\\times1.0$，$\\Delta_c=50$ ms，$v_{\\rm rel}\\le4$ m/s，$\\mu_{\\rm out}=1.0$）",
    "eps010": "折り返し $\\epsilon=0.10$ s", "eps040": "折り返し $\\epsilon=0.40$ s",
    "cap070": "関節能力 $\\times0.7$", "cap130": "関節能力 $\\times1.3$",
    "dc030": "捕捉均し時間 $\\Delta_c=30$ ms", "dc100": "捕捉均し時間 $\\Delta_c=100$ ms",
    "vrel3": "手先相対速度上限 3 m/s", "vrel5": "手先相対速度上限 5 m/s",
    "mu060": "摩擦・引掛り $\\mu_{\\rm out}=0.6$", "mu150": "摩擦・引掛り $\\mu_{\\rm out}=1.5$",
}


def load_rows(d, tag, T=20.0, m=60.0):
    files = glob.glob(os.path.join(d, f"rows_{tag}_T{T:g}_m{m:g}*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["phi_l"] = df["phi_l"].round(4)
    df = df.sort_values("ok", ascending=False).drop_duplicates(["T", "m", "phi_l"], keep="first")
    df = df[df["ok"] == 1]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--sens", default="results/sens")
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    ref = load_rows(a.grid, "ref")
    rows = []
    tags = [t for t in LABELS if t != "ref"]
    phases = None
    for tag in tags:
        df = load_rows(a.sens, tag)
        if df is None or len(df) == 0:
            continue
        phases = sorted(df["phi_l"].unique()) if phases is None else phases
        rsub = ref[ref["phi_l"].round(3).isin([round(p, 3) for p in phases])] if ref is not None else None
        rows.append(dict(factor=LABELS[tag], tag=tag, n=len(df),
                         min_cap_BW=df["req_cap_BW"].min(), mean_cap_BW=df["req_cap_BW"].mean(),
                         best_phi=df.loc[df["req_cap_BW"].idxmin(), "phi_l"],
                         mean_E=df["effort"].mean(), mean_tau=df["d_f"].mean(),
                         ref_min_cap_BW=None if rsub is None else rsub["req_cap_BW"].min(),
                         ref_mean_cap_BW=None if rsub is None else rsub["req_cap_BW"].mean()))
    if ref is not None and phases is not None:
        rsub = ref[ref["phi_l"].round(3).isin([round(p, 3) for p in phases])]
        rows.insert(0, dict(factor=LABELS["ref"], tag="ref", n=len(rsub), min_cap_BW=rsub["req_cap_BW"].min(),
                            mean_cap_BW=rsub["req_cap_BW"].mean(), best_phi=rsub.loc[rsub["req_cap_BW"].idxmin(), "phi_l"],
                            mean_E=rsub["effort"].mean(), mean_tau=rsub["d_f"].mean(),
                            ref_min_cap_BW=rsub["req_cap_BW"].min(), ref_mean_cap_BW=rsub["req_cap_BW"].mean()))
    s = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    s.to_csv(os.path.join(a.out, "sensitivity.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s[["factor", "n", "min_cap_BW", "mean_cap_BW", "best_phi", "mean_E", "mean_tau"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
