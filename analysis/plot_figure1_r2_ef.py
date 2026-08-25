# -*- coding: utf-8 -*-
"""
Figure 1: R² vs EF@1% across 4 datasets (GNN vs Non-GNN)
2行2列散点回归分析图，从 xlsx 读取数据
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ===================== 全局参数 =====================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#d0d0d0"
plt.rcParams["grid.alpha"] = 0.7
plt.rcParams["axes.facecolor"] = "#ffffff"

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# 颜色
COLOR_GNN = "#d63027"
COLOR_NONGNN = "#2b6cb0"
COLOR_FIT = "#555555"

# ===================== 从 xlsx 读取真实数据 =====================
xlsx_path = BASE_DIR / "ChEMBL_Davis模型性能对比（现象）.xlsx"
xl = pd.ExcelFile(str(xlsx_path))

# Sheet 名映射
sheet_map = {
    "ChEMBL数据集": "ChEMBL",
    "Davis数据集": "Davis",
    "BingdingBD数据集": "BindingDB",
    "KIBA数据集": "KIBA",
}

# 用户提供的 Pearson r 值
USER_R = {
    "ChEMBL": 0.8131,
    "Davis": -0.9789,
    "BindingDB": 0.6519,
    "KIBA": 0.2272,
}

# 每个数据集的坐标轴范围
AXIS_LIMITS = {
    "ChEMBL":    {"xlim": (-0.20, 0.25), "ylim": (0, 6)},
    "Davis":     {"xlim": (0.14, 0.46),  "ylim": (-0.5, 5)},
    "BindingDB": {"xlim": (0.00, 0.28),  "ylim": (1.5, 5.2)},
    "KIBA":      {"xlim": (-0.15, 0.30), "ylim": (3.5, 4.8)},
}

# 存储处理后的数据
DATASETS = {}

for sheet_name, ds_name in sheet_map.items():
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    
    points = []
    for _, row in df.iterrows():
        model_name = str(row['模型'])
        r2 = float(row['R的平方'])
        ef1 = float(row['EF@1%'])
        
        # 判断是否为 GNN
        is_gnn = 'GraphDTA' in model_name
        
        # 简化模型名
        if 'MLP' in model_name:
            short_name = 'MLP'
        elif 'Transformer' in model_name and 'Reptile' not in model_name:
            short_name = 'Transformer'
        elif 'GAT_GCN' in model_name:
            short_name = 'GAT_GCN'
        elif 'GCNNet' in model_name:
            short_name = 'GCNNet'
        elif 'GATNet' in model_name:
            short_name = 'GATNet'
        elif 'GINConnvNet' in model_name or 'GINConvNet' in model_name:
            short_name = 'GINConvNet'
        else:
            short_name = model_name
        
        points.append((short_name, r2, ef1, is_gnn))
    
    DATASETS[ds_name] = {
        "points": points,
        "pearson_r": USER_R.get(ds_name, None),
    }

DS_ORDER = ["ChEMBL", "Davis", "BindingDB", "KIBA"]

# 打印读取的数据
for ds_name in DS_ORDER:
    print(f"\n{ds_name}:")
    for p in DATASETS[ds_name]["points"]:
        print(f"  {p[0]}: R²={p[1]:.4f}, EF@1%={p[2]:.2f}, GNN={p[3]}")

# ===================== 主绘图 =====================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9), dpi=300)
fig.suptitle("Figure 1: R² vs EF@1% across 4 datasets (GNN vs Non-GNN)",
             fontsize=12, y=0.98, fontweight='normal')
axes = axes.flatten()

for idx, ds_name in enumerate(DS_ORDER):
    ax = axes[idx]
    ds_info = DATASETS[ds_name]
    points = ds_info["points"]
    user_r = ds_info.get("pearson_r", None)

    xs = np.array([p[1] for p in points], dtype=float)
    ys = np.array([p[2] for p in points], dtype=float)
    names = [p[0] for p in points]

    # 1. 散点（处理完全重叠的点：轻微抖动使其可见）
    coord_count = {}
    for i, (name, x, y, gnn) in enumerate(points):
        key = (round(x, 4), round(y, 4))
        coord_count[key] = coord_count.get(key, 0) + 1

    # 记录每个点实际绘制的位置（含抖动）
    drawn_positions = []
    drawn_count = {}
    for i, (name, x, y, gnn) in enumerate(points):
        key = (round(x, 4), round(y, 4))
        if coord_count[key] > 1:
            drawn_count[key] = drawn_count.get(key, 0)
            jitter_x = x + (drawn_count[key] - (coord_count[key] - 1) / 2) * 0.010
            drawn_count[key] += 1
        else:
            jitter_x = x
        drawn_positions.append((jitter_x, y))

        if gnn:
            ax.scatter(jitter_x, y, marker='o', c=COLOR_GNN, s=55, zorder=5,
                       edgecolors='white', linewidths=0.7)
        else:
            ax.scatter(jitter_x, y, marker='s', c=COLOR_NONGNN, s=55, zorder=5,
                       edgecolors='white', linewidths=0.7)

    # 2. 线性拟合 + 95% CI（均值置信区间）
    mask = np.isfinite(xs) & np.isfinite(ys)
    xdata, ydata = xs[mask], ys[mask]
    slope, intercept, r_value, p_value, std_err = stats.linregress(xdata, ydata)
    r_display = user_r if user_r is not None else r_value

    # CI 只画在数据范围内（不外推到坐标轴边缘，避免两端无限变宽）
    x_data_min = np.min(xdata)
    x_data_max = np.max(xdata)
    x_fit = np.linspace(x_data_min, x_data_max, 100)
    y_fit = slope * x_fit + intercept

    # 残差标准误: s_e = std_err_slope * sqrt(SS_x)
    n_p = len(xdata)
    x_mean = np.mean(xdata)
    ss_x = np.sum((xdata - x_mean) ** 2)
    s_e = std_err * np.sqrt(ss_x)  # 残差标准误

    # 均值CI: SE_fit = s_e * sqrt(1/n + (x - x_mean)^2 / SS_x)
    se_fit = s_e * np.sqrt(1.0 / n_p + (x_fit - x_mean) ** 2 / (ss_x + 1e-12))
    t_95 = stats.t.ppf(0.975, n_p - 2)
    y_upper = y_fit + t_95 * se_fit
    y_lower = y_fit - t_95 * se_fit

    ax.fill_between(x_fit, y_lower, y_upper, color="#cccccc",
                    alpha=0.35, zorder=1, label="95% CI")
    ax.plot(x_fit, y_fit, color=COLOR_FIT, linestyle='--', linewidth=1.5,
            zorder=2, label=f"Fit (r={r_display:.2f})")

    # 3. 标注每个点的模型名（使用抖动后的位置）
    for i, (name, x, y, gnn) in enumerate(points):
        plot_x, plot_y = drawn_positions[i]
        offset_x, offset_y = 0.010, 0.15
        ha = "left"

        if ds_name == "Davis":
            if name == "MLP":
                offset_x, offset_y = -0.005, -0.25
                ha = "center"
            elif name == "Transformer":
                offset_x, offset_y = 0.015, 0.08
                ha = "left"
            elif name == "GINConvNet":
                offset_x, offset_y = 0.010, -0.18
                ha = "left"
            elif name == "GAT_GCN":
                offset_x, offset_y = 0.010, 0.18
                ha = "left"
            elif name == "GCNNet":
                offset_x, offset_y = -0.025, -0.18
                ha = "right"
            elif name == "GATNet":
                offset_x, offset_y = 0.010, -0.25
                ha = "left"
        elif ds_name == "ChEMBL":
            if name == "GAT_GCN":
                offset_x, offset_y = 0.010, -0.18
            elif name == "GCNNet":
                offset_x, offset_y = -0.028, 0.08
                ha = "right"
            elif name == "GINConvNet":
                offset_x, offset_y = 0.010, 0.12
            elif name == "GATNet":
                offset_x, offset_y = 0.010, -0.20
            elif name == "Transformer":
                offset_x, offset_y = -0.028, 0.12
                ha = "right"
            elif name == "MLP":
                offset_x, offset_y = 0.010, 0.15
        elif ds_name == "BindingDB":
            if name == "GATNet":
                offset_x, offset_y = 0.010, -0.18
            elif name == "GINConvNet":
                offset_x, offset_y = 0.010, 0.12
            elif name == "GCNNet":
                offset_x, offset_y = 0.010, -0.18
            elif name == "GAT_GCN":
                offset_x, offset_y = -0.028, 0.12
                ha = "right"
            elif name == "Transformer":
                offset_x, offset_y = -0.028, 0.08
                ha = "right"
            elif name == "MLP":
                offset_x, offset_y = 0.010, 0.08
        elif ds_name == "KIBA":
            if name == "GATNet":
                offset_x, offset_y = 0.010, -0.08
            elif name == "GINConvNet":
                offset_x, offset_y = 0.010, 0.05
            elif name == "GCNNet":
                offset_x, offset_y = 0.010, 0.05
            elif name == "GAT_GCN":
                offset_x, offset_y = -0.028, 0.02
                ha = "right"
            elif name == "Transformer":
                offset_x, offset_y = -0.028, 0.02
                ha = "right"
            elif name == "MLP":
                offset_x, offset_y = 0.010, -0.08

        # 防止超出边界
        ylim = AXIS_LIMITS[ds_name]["ylim"]
        if y + offset_y * 2.5 > ylim[1]:
            offset_y = -0.25
        if y + offset_y * 2.5 < ylim[0]:
            offset_y = 0.25

        ax.annotate(name, (plot_x, plot_y),
                    xytext=(plot_x + offset_x, plot_y + offset_y),
                    ha=ha, fontsize=8, alpha=0.92,
                    annotation_clip=False, zorder=7,
                    arrowprops=None)

    # 4. 坐标轴
    ax.set_xlabel("$R^2$")
    ax.set_ylabel("EF@1%")
    ax.set_xlim(*AXIS_LIMITS[ds_name]["xlim"])
    ax.set_ylim(*AXIS_LIMITS[ds_name]["ylim"])
    ax.set_title(f"{ds_name}  (n={len(points)})", fontsize=10, pad=6)

    # 5. 每个子图单独图例
    from matplotlib.lines import Line2D
    leg_handles = [
        Line2D([], [], color="#cccccc", lw=6, alpha=0.35, label="95% CI"),
        Line2D([], [], color=COLOR_FIT, linestyle='--', lw=1.5,
               label=f"Fit (r={r_display:.2f})"),
        Line2D([], [], marker='o', color='w', markerfacecolor=COLOR_GNN,
               markersize=8, label='GNN', markeredgecolor='white', markeredgewidth=0.5),
        Line2D([], [], marker='s', color='w', markerfacecolor=COLOR_NONGNN,
               markersize=8, label='Non-GNN', markeredgecolor='white', markeredgewidth=0.5),
    ]
    ax.legend(handles=leg_handles, loc='upper right', framealpha=0.9, borderpad=0.4,
              handletextpad=0.5)

# 布局调整
plt.subplots_adjust(top=0.94, bottom=0.06, left=0.07, right=0.97,
                    hspace=0.24, wspace=0.18)

out_png = FIG_DIR / "Figure1_R2_EF1_scatter.png"
out_tiff = FIG_DIR / "Figure1_R2_EF1_scatter.tiff"
plt.savefig(str(out_png), dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(str(out_tiff), dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
plt.close()

print(f"\n✅ PNG  已保存: {out_png}")
print(f"✅ TIFF 已保存: {out_tiff}")
