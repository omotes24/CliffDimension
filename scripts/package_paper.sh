#!/bin/bash
# Build the manuscript and package the deliverables: PDF + self-contained LaTeX source (figures copied next to the
# .tex files, paths rewritten) + zip.   usage: bash scripts/package_paper.sh [OUT_DIR]
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/mnt/user-data/outputs}
mkdir -p "$OUT"
( cd paper && latexmk -r latexmkrc main.tex > build.log 2>&1 && latexmk -r latexmkrc main.tex >> build.log 2>&1 ) || { grep -A3 "^!" paper/main.log | head; exit 1; }
pdfinfo paper/main.pdf | grep Pages
cp paper/main.pdf "$OUT/CliffDimension_MIRU2025_draft.pdf"
PS="$OUT/paper_src"; rm -rf "$PS"; mkdir -p "$PS/figs"
cp paper/main.tex paper/sec_*.tex paper/numbers.tex paper/numbers_dfl.tex paper/tab_*.tex paper/miru2025j.cls paper/latexmkrc "$PS/" 2>/dev/null || true
cp paper/miru2025j.bst paper/miru_logo_color.png "$PS/" 2>/dev/null || true
for f in $(cat paper/main.tex paper/sec_*.tex | grep -o "\.\./results/figs/[A-Za-z0-9_./-]*" | sort -u); do cp "paper/$f" "$PS/figs/" 2>/dev/null || true; done
cp paper/figs/Cliff_Front_Section_R5.tex paper/figs/Cliff_Front_Section_R5.pdf "$PS/figs/" 2>/dev/null || true
sed -i 's#\.\./results/figs/#figs/#g' "$PS"/main.tex "$PS"/sec_*.tex
( cd "$PS" && latexmk -r latexmkrc main.tex > /dev/null 2>&1 && latexmk -c > /dev/null 2>&1 && rm -f main.dvi ) || echo "WARNING: standalone paper_src did not compile"
[ -f "$PS/main.pdf" ] && rm -f "$PS/main.pdf"
rm -f "$OUT/CliffDimension_paper_src.zip"
( cd "$OUT" && zip -qr CliffDimension_paper_src.zip paper_src )
echo "done -> $OUT (CliffDimension_MIRU2025_draft.pdf, paper_src/, CliffDimension_paper_src.zip)"
