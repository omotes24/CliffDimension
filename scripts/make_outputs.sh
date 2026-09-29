#!/bin/bash
# Final pipeline: figures / tables / paper / deliverables. Run after the grid (and optionally the
# windows + sensitivity runs) have finished.
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-/mnt/user-data/outputs}
mkdir -p results/figs "$OUT"

echo "== grid analysis"
python3 scripts/analyze_grid.py --grid results/grid --tag ref --out results/figs

echo "== representative trajectory figure (T=20, m=60, best phase)"
python3 - <<'EOF'
import glob, pickle, pandas as pd, sys
sys.path.insert(0, ".")
from katsumi.planar.anthro import make_body
from katsumi.planar.plots import plot_solution, plot_solution_compact
df = pd.read_csv("results/figs/grid_ref_all.csv"); df = df[df.ok == 1]
d = df[(df["T"] == 20.0) & (df["m"] == 60.0)]
if len(d) == 0:
    d = df
b = d.loc[d["req_cap_BW"].idxmin()]
pkl = f"results/grid/sol_ref_T{b['T']:g}_m{b['m']:g}_phi{b['phi_l']:.3f}.pkl"
r = pickle.load(open(pkl, "rb"))
plot_solution(r, make_body(r["m"]), "results/figs/traj_best_full.png",
              title=f"T={r['T']:g} s, m={r['m']:g} kg, release phase {r['phi_l']:.3f}: tau={r['d_f']:.2f} s, required capacity {b['req_cap_BW']:.2f} BW")
plot_solution_compact(r, make_body(r["m"]), "results/figs/traj_best.png",
              title=f"T = {r['T']:g} s, m = {r['m']:g} kg, release phase {r['phi_l']:.3f}, τ = {r['d_f']:.2f} s, required capacity {b['req_cap_BW']:.2f} BW")
print("trajectory figure from", pkl)
EOF

echo "== sensitivity table"
python3 scripts/analyze_sensitivity.py --grid results/grid --sens results/sens --out results/figs || true

echo "== extra figures (device, windows, sensitivity, 3-D search, body size)"
python3 scripts/extra_figs.py --out results/figs

echo "== paper numbers / tables"
python3 scripts/paper_numbers.py --grid results/grid --tag ref --windows results/windows/windows_ref.csv \
        --sens results/figs/sensitivity.csv --audit results/figs/audit.json --out paper > results/figs/paper_numbers.txt

echo "== paper build"
( cd paper && latexmk -r latexmkrc main.tex > build.log 2>&1 && latexmk -r latexmkrc main.tex >> build.log 2>&1 ) || { grep -A3 "^!" paper/main.log | head; exit 1; }
pdfinfo paper/main.pdf | grep Pages

echo "== deliverables"
cp paper/main.pdf "$OUT/CliffDimension_MIRU2025_draft.pdf"
cp paper/main.tex "$OUT/CliffDimension_MIRU2025_draft.tex"
cp paper/figs/Cliff_Front_Section_R5.tex paper/figs/Cliff_Front_Section_R5.pdf "$OUT/" 2>/dev/null || true
mkdir -p "$OUT/figures" "$OUT/data"
cp results/figs/*.png results/figs/*.pdf "$OUT/figures/" 2>/dev/null || true
cp results/figs/grid_ref_all.csv results/figs/summary_h1_h2.csv results/figs/audit.json paper/best_per_condition.csv "$OUT/data/" 2>/dev/null || true
cp results/windows/windows_ref.csv results/figs/sensitivity.csv "$OUT/data/" 2>/dev/null || true
# self-contained LaTeX source (figures copied next to main.tex, paths rewritten)
PS="$OUT/paper_src"; rm -rf "$PS"; mkdir -p "$PS/figs"
cp paper/main.tex paper/numbers.tex paper/tab_*.tex paper/miru2025j.cls paper/miru2025j.bst paper/latexmkrc paper/miru_logo_color.png "$PS/"
for f in $(grep -o "\.\./results/figs/[A-Za-z0-9_./-]*" paper/main.tex | sort -u); do cp "paper/$f" "$PS/figs/" 2>/dev/null || true; done
cp paper/figs/Cliff_Front_Section_R5.tex paper/figs/Cliff_Front_Section_R5.pdf "$PS/figs/" 2>/dev/null || true
sed -i 's#\.\./results/figs/#figs/#g' "$PS/main.tex"
( cd "$PS" && latexmk -r latexmkrc main.tex > /dev/null 2>&1 && latexmk -c > /dev/null 2>&1 && rm -f main.dvi ) || echo "WARNING: standalone paper_src did not compile"
ls "$PS/main.pdf" > /dev/null 2>&1 && rm -f "$PS/main.pdf"
git add -A >/dev/null 2>&1 || true
git bundle create "$OUT/CliffDimension.bundle" --all > /dev/null 2>&1 || true
rm -f "$OUT/CliffDimension_src.zip"
git archive --format=zip -o "$OUT/CliffDimension_src.zip" HEAD
echo "done -> $OUT"
