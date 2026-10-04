#!/bin/bash
# Build the manuscript and package the deliverables: PDF + self-contained LaTeX source (figures copied next to the
# .tex files, paths rewritten) + zip.   usage: bash scripts/package_paper.sh [OUT_DIR]
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/mnt/user-data/outputs}
mkdir -p "$OUT"
( cd paper && latexmk -r latexmkrc main.tex > build.log 2>&1 ) || { grep -A3 "^!" paper/main.log | head; exit 1; }
pdfinfo paper/main.pdf | grep Pages
cp paper/main.pdf "$OUT/CliffDimension_MIRU2025_draft.pdf"
PS="$OUT/paper_src"; rm -rf "$PS"; mkdir -p "$PS/figs"
# every .tex file reachable from main.tex through \input (unused drafts stay out of the package)
USED=$(python3 - <<'PY'
import re, os
seen, todo = [], ['main.tex']
while todo:
    f = todo.pop()
    if f in seen or not os.path.exists('paper/' + f):
        continue
    seen.append(f)
    for m in re.findall(r'\\input\{([^}]+)\}', open('paper/' + f).read()):
        todo.append(m if m.endswith('.tex') else m + '.tex')
print(' '.join('paper/' + f for f in seen))
PY
)
cp $USED paper/miru2025j.cls paper/latexmkrc "$PS/"
cp paper/miru2025j.bst paper/miru_logo_color.png "$PS/" 2>/dev/null || true
for f in $(cat $USED | grep -o "\.\./results/figs/[A-Za-z0-9_./-]*" | sort -u); do cp "paper/$f" "$PS/figs/" 2>/dev/null || true; done
cp paper/figs/Cliff_Front_Section_R5.tex paper/figs/Cliff_Front_Section_R5.pdf "$PS/figs/" 2>/dev/null || true
sed -i 's#\.\./results/figs/#figs/#g' "$PS"/*.tex
( cd "$PS" && latexmk -r latexmkrc main.tex > /dev/null 2>&1 && latexmk -c > /dev/null 2>&1 && rm -f main.dvi ) || echo "WARNING: standalone paper_src did not compile"
[ -f "$PS/main.pdf" ] && rm -f "$PS/main.pdf"
rm -f "$OUT/CliffDimension_paper_src.zip"
( cd "$OUT" && zip -qr CliffDimension_paper_src.zip paper_src )
echo "done -> $OUT (CliffDimension_MIRU2025_draft.pdf, paper_src/, CliffDimension_paper_src.zip)"
