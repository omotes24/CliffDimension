"""Shared plotting style of the manuscript figures: one categorical order, thin marks, recessive axes, text in ink."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots (fixed order; a method / condition keeps its colour in every figure)
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", green="#008300", violet="#4a3aa7", red="#e34948")
CAT = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"], C["violet"], C["red"]]
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e3e2dc", "#9a988f"
SEQ = "Blues"                      # sequential: one hue, light -> dark
# status colours are reserved for good / serious outcomes
GOOD, BAD, NEUTRAL = "#1a7f37", "#c0392b", "#b8b6ad"

# methods of the closed-loop comparison keep one colour everywhere
METHOD_COL = dict(REPLAY=MUTED, BC=C["yellow"], TRACK=C["magenta"], VMPC=C["orange"], SMPC=C["blue"], ORACLE=C["aqua"], **{"ORACLE+": C["green"]})
METHOD_JP = dict(REPLAY="再生", BC="行動模倣", TRACK="追従 MPC", VMPC="価値 MPC（値のみ）", SMPC="双対場 MPC（Sobolev）", ORACLE="オラクル MPC", **{"ORACLE+": "オラクル MPC+"})


def setup():
    matplotlib.rcParams.update({
        "font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "lines.markersize": 5, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "mathtext.fontset": "dejavusans",
    })


def save(fig, out, name):
    import os
    fig.savefig(os.path.join(out, name + ".pdf")); fig.savefig(os.path.join(out, name + ".png"), dpi=170); plt.close(fig)
    print("saved", name)


def bar(ax, x, h, color, width=0.6, **kw):
    """Thin bar with a small surface gap to its neighbours."""
    return ax.bar(x, h, width=width, color=color, edgecolor="white", linewidth=1.0, **kw)
