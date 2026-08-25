# -*- coding: utf-8 -*-
"""
Fig3_GNN_Scatter: SCI 风格散点图
数据来自用户提供
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ===================== 1. 全局SCI绘图参数 =====================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9

# ===================== 2. 原始数据 =====================
datasets = ["ChEMBL", "Davis", "KIBA", "BindingDB"]
mean_similarity = [0.1221, 0.1367, 0.2500, 0.3600]
delta_R2 = [0.0579 - 0.2620,
            0.3998 - 0.2586,
            0.2233 - 0.3222,
            0.0995 - 0.2335]

color_list = ["#ff7f0e", "#d62728", "#2ca02c", "#1f77b4"]

# ===================== 3. 创建画布绘图 =====================
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

scatter = ax.scatter(
    mean_similarity, delta_R2,
    c=color_list,
    s=110,
    edgecolors="white",
    linewidth=0.8,
    zorder=3
)

ax.axhline(y=0, color="#666666", linestyle="--", lw=1.2, label=r"$\Delta R^2=0$")
ax.axvline(x=0.13, color="#c82423", linestyle=":", lw=1.2, label="Threshold = 0.13")
ax.axvspan(xmin=0.10, xmax=0.13, alpha=0.12, color="#888888", zorder=1)

# ===================== 4. 每个点文本标注 =====================
text_offset_x = 0.004
text_offset_y = 0.007
for x, y, name in zip(mean_similarity, delta_R2, datasets):
    ax.text(x + text_offset_x, y + text_offset_y, name, fontsize=9)

# ===================== 5. 坐标轴设置 =====================
ax.set_xlabel("Mean Tanimoto Similarity of Dataset")
ax.set_ylabel(r"$\Delta R^2$ (GNN $\minus$ Best non-GNN Model)")
ax.set_xlim(0.10, 0.38)
ax.set_ylim(-0.26, 0.14)
ax.grid(color="#e0e0e0", linestyle="--", linewidth=0.6, zorder=0)

# ===================== 6. 图例 =====================
legend_elements = [
    Line2D([], [], marker='o', color=c, linestyle="", markersize=8, label=name)
    for c, name in zip(color_list, datasets)
]
ax.legend(handles=legend_elements, loc="upper right", framealpha=0.95)

plt.tight_layout()

# ===================== 7. 保存图片 =====================
out_tiff = FIG_DIR / "Fig3_GNN_Scatter.tiff"
out_png = FIG_DIR / "Fig3_GNN_Scatter.png"
plt.savefig(out_tiff, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ 已保存: {out_tiff}")
print(f"✅ 已保存: {out_png}")
