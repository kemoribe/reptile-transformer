# -*- coding: utf-8 -*-
"""
ECE 概率校准对比图 — 2D 柱状图（蓝绿色系）
4个子图：ChEMBL、Davis、BindingDB、KIBA
- 黑色虚线 = 理想校准 (ECE=0)
- 绿色柱 = Reptile-Transformer（本文模型），突出显示
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATASETS = ['ChEMBL', 'Davis', 'BindingDB', 'KIBA']
DS_SUFFIX = {'ChEMBL': 'chembl', 'Davis': 'davis', 'BindingDB': 'bindingdb', 'KIBA': 'kiba'}

MODELS = [
    ('MLP',                 'MLP'),
    ('Transformer',         'Transformer'),
    ('GAT_GCN',             'GAT_GCN'),
    ('GCNNet',              'GCNNet'),
    ('GATNet',              'GATNet'),
    ('GINConvNet',          'GINConvNet'),
]

# 蓝绿色系配色
BAR_COLORS = ['#66BB6A', '#81C784', '#4DB6AC', '#4FC3F7', '#26A69A', '#29B6F6']
BAR_EDGES = ['#388E3C', '#388E3C', '#00695C', '#0277BD', '#00695C', '#01579B']


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    text = None
    for enc in ('utf-8', 'gbk', 'latin-1'):
        try:
            with open(p, 'r', encoding=enc) as f:
                text = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        return None
    return json.loads(text)


def get_metric(metrics, key):
    if metrics is None:
        return np.nan
    if 'test_metrics' in metrics:
        metrics = metrics['test_metrics']
    val = metrics.get(key)
    if val is None:
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def collect_ece():
    data = {ds: [] for ds in DATASETS}
    for ds in DATASETS:
        ds_suffix = DS_SUFFIX[ds]
        print(f"\n[{ds}]")
        for model_key, model_label in MODELS:
            paths = []
            if model_key == 'MLP':
                paths = [
                    BASE_DIR / f"baseline_output_{ds_suffix}_morgan_descriptors" / "final_results.json",
                    BASE_DIR / f"baseline_output_{ds_suffix}" / "final_results.json",
                    BASE_DIR / "baseline_output" / "final_results.json",
                ]
            elif model_key == 'Transformer':
                paths = [BASE_DIR / f"transformer_ablation_output_{ds_suffix}" / "morgan_descriptors" / "final_results.json"]
            else:
                paths = [BASE_DIR / "GrapthDTA" / f"result_{model_key}_{ds_suffix}.json"]

            ece = np.nan
            for p in paths:
                metrics = load_json(p)
                if metrics is not None:
                    ece = get_metric(metrics, 'ECE')
                    if not np.isnan(ece):
                        print(f"  {model_label}: ECE={ece:.4f} ({p.name})")
                        break
            if np.isnan(ece):
                print(f"  {model_label:20s}: 未找到结果")
            data[ds].append(ece)
    return data


def plot_ece_bar(data):
    """画 4 子图 2D 柱状图"""
    print("\n" + "=" * 60)
    print("生成 ECE 2D 柱状图 (蓝绿色系)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    model_labels = [m[1] for m in MODELS]
    n_models = len(MODELS)
    x = np.arange(n_models)
    bar_width = 0.65

    for ds_idx, ds_name in enumerate(DATASETS):
        ax = axes[ds_idx]
        ece_vals = data[ds_name]

        valid_vals = [v if not np.isnan(v) else 0 for v in ece_vals]

        # 画柱子
        bars = ax.bar(x, valid_vals, bar_width,
                      color=BAR_COLORS, edgecolor=BAR_EDGES,
                      linewidth=1.2, alpha=0.9, zorder=3)

        # 标注数值
        for i, val in enumerate(ece_vals):
            if not np.isnan(val) and val > 0:
                ax.text(x[i], val + 0.003, f'{val:.4f}',
                        ha='center', va='bottom',
                        fontsize=9, fontweight='normal',
                        color='#1a3a5c')

        # 理想校准线 (ECE=0)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2.5, alpha=0.8,
                   label='理想校准 (ECE=0)', zorder=2)

        ax.set_xlabel('模型', fontsize=12, fontweight='bold')
        ax.set_ylabel('ECE (越低越好)', fontsize=12, fontweight='bold')
        ax.set_title(ds_name, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9, rotation=25, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)
        ax.set_axisbelow(True)

        # Y轴范围
        valid_only = [v for v in ece_vals if not np.isnan(v)]
        if valid_only:
            ymax = max(valid_only)
            ax.set_ylim(-ymax * 0.08, ymax * 1.25)

        if ds_idx == 0:
            ax.legend(fontsize=10, loc='upper right')

    fig.suptitle('概率校准误差 (ECE) 对比 — 越低越好', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    out_path = FIG_DIR / 'fig_ece_calibration.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 已保存: {out_path}")


def main():
    print("=" * 60)
    print("ECE 2D 概率校准对比图 (蓝绿色系)")
    print("=" * 60)
    data = collect_ece()

    print("\n数据汇总:")
    print(f"{'模型':<25}", end='')
    for ds in DATASETS:
        print(f"  {ds:>10}", end='')
    print()
    for i, (mk, ml) in enumerate(MODELS):
        print(f"{ml:<25}", end='')
        for ds in DATASETS:
            v = data[ds][i]
            print(f"  {v:>10.4f}" if not np.isnan(v) else f"  {'N/A':>10}", end='')
        print()

    plot_ece_bar(data)
    print(f"\n图表已保存至: {FIG_DIR}")


if __name__ == '__main__':
    main()
