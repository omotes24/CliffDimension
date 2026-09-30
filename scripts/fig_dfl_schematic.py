"""Schematic of dual-field learning (paper figure)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAPGothic", "DejaVu Sans"]
INK, INK2 = "#0b0b0b", "#52514e"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#f2f2f0"


def box(ax, x, y, w, h, title, lines, color, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc="white", ec=color, lw=1.6))
    ax.text(x + 0.03, y + h - 0.05, title, fontsize=fs + 1, weight="bold", color=color, va="top")
    ax.text(x + 0.03, y + h - 0.20, "\n".join(lines), fontsize=fs, color=INK, va="top", linespacing=1.35)


def arrow(ax, p, q, label=None, color=INK2, rad=0.0, fs=7.5, dy=0.03):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=color, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, label, fontsize=fs, color=color, ha="center", va="bottom")


def main(out="results/figs"):
    os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 3.5); ax.set_ylim(-0.30, 1.45); ax.axis("off")
    # top row: the contrast
    ax.add_patch(FancyBboxPatch((0.02, 1.02), 1.62, 0.40, boxstyle="round,pad=0.02", fc=GREY, ec="none"))
    ax.text(0.06, 1.37, "従来の世界モデル（主変数）", fontsize=9, weight="bold", color=INK2, va="top")
    ax.text(0.06, 1.24, r"$s'=f_\theta(s,a)$ を学び，想像上の経験で方策を探索する" "\n" "疎な成功／失敗が唯一の教師", fontsize=8, color=INK, va="top", linespacing=1.4)
    ax.add_patch(FancyBboxPatch((1.86, 1.02), 1.62, 0.40, boxstyle="round,pad=0.02", fc="#eaf2fc", ec="none"))
    ax.text(1.90, 1.37, "双対場（提案）", fontsize=9, weight="bold", color=BLUE, va="top")
    ax.text(1.90, 1.24, r"$V(x,t,\theta_b),\ \mathbf{p}=\partial V/\partial x,\ \lambda_k$ を学ぶ" "\n" "「ここから先どの物理法則がどれだけ効くか」", fontsize=8, color=INK, va="top", linespacing=1.4)
    # bottom row: pipeline
    box(ax, 0.02, 0.10, 0.95, 0.78, "厳密オラクル", ["多相軌道最適化（IPOPT）", "残り最適化問題を任意の", "状態・時刻・身体から解く", "KKT 乗数 → 双対ラベル", r"$V^*,\ \mathbf{p}^*=\partial J^*/\partial x_0,$", r"$\tau^*,\ U^*,\ \Lambda_k,\ \partial J^*/\partial\ln p$"], ORANGE)
    box(ax, 1.20, 0.10, 1.05, 0.78, "双対場ネットワーク", ["Sobolev 学習：", r"$\|V_\vartheta-V^*\|^2+\alpha\|\nabla_x V_\vartheta-\mathbf{p}^*\|^2$", r"$+\|U_\vartheta-U^*\|^2+\|\tau_\vartheta-\tau^*\|^2$", "身体パラメータを入力に含む", "アンサンブル：不確かさ", "（データの外で悲観的）"], BLUE)
    box(ax, 2.48, 0.10, 1.00, 0.78, "行動の導出（方策学習なし）", ["既知の剛体力学 × 双対場", r"Pontryagin：$\arg\min_u H(x,u,\mathbf{p}_\vartheta)$", "短地平線 MPC：0.5 s 積分＋", r"終端 $V_\vartheta$（厳密微分）", r"離手：$\tau_\vartheta\leq \Delta t/2$", r"価格 $\bar\Lambda_k$ → 密な報酬"], GREEN)
    arrow(ax, (0.97, 0.49), (1.20, 0.49), "ラベル", ORANGE)
    arrow(ax, (2.25, 0.49), (2.48, 0.49), r"$V,\nabla V,\tau,U$", BLUE)
    arrow(ax, (2.98, 0.10), (0.50, 0.10), None, INK2, rad=-0.28)
    ax.text(1.75, -0.13, "DAgger：制御器が訪れた状態をオラクルが再ラベル（データの穴を埋める）", fontsize=8, color=INK2, ha="center", va="top")
    ax.text(1.75, -0.24, r"転移：$\partial V^*/\partial\theta_b=\sum_k\lambda_k\,\partial g_k/\partial\theta_b$（包絡線定理）", fontsize=8.5, color=INK2, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "dfl_schematic.pdf")); fig.savefig(os.path.join(out, "dfl_schematic.png"), dpi=200)
    print("written")


if __name__ == "__main__":
    main()
