# -*- coding: utf-8 -*-
"""
Figure S5: λ 敏感性热力图（依赖 supplement_s9_lambda_sensitivity.py 的 lambda_results.csv）
用法: python supplement_s5_lambda_heatmap.py
输出: supplement_output/paper_revision/figures/FigureS5_lambda_heatmap.{png,tiff}
      supplement_output/paper_revision/TableS9_lambda.csv
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(r'd:\lht')
CSV = ROOT / 'supplement_output' / 'lambda_sensitivity' / 'lambda_results.csv'
FIG = ROOT / 'supplement_output' / 'paper_revision' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
TEAL_CMAP = LinearSegmentedColormap.from_list('teal', ['#FFFFFF', '#B2DFDB', '#006D77'])

df = pd.read_csv(CSV)
df['lambda1'] = df['lambda1'].round(4)
df['lambda2'] = df['lambda2'].round(4)
l1s = sorted(df['lambda1'].unique())
l2s = sorted(df['lambda2'].unique())
METRICS = [('R2', 'R²'), ('EF1', 'EF@1%'), ('ECE', 'ECE')]

# 透视为 λ1×λ2 网格
grids = {}
for mk, _ in METRICS:
    g = np.full((len(l1s), len(l2s)), np.nan)
    for _, r in df.iterrows():
        i = l1s.index(r['lambda1']); j = l2s.index(r['lambda2'])
        if mk in r and pd.notna(r[mk]):
            g[i, j] = float(r[mk])
    grids[mk] = g

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=300)
for ax, (mk, mlab) in zip(axes, METRICS):
    g = grids[mk]
    vmin, vmax = np.nanmin(g), np.nanmax(g)
    if mk == 'ECE':                       # ECE 越低越好 → 反转配色
        im = ax.imshow(g, cmap=TEAL_CMAP.reversed(), aspect='auto')
    else:
        im = ax.imshow(g, cmap=TEAL_CMAP, aspect='auto')
    ax.set_xticks(range(len(l2s))); ax.set_xticklabels([f'{v:g}' for v in l2s])
    ax.set_yticks(range(len(l1s))); ax.set_yticklabels([f'{v:g}' for v in l1s])
    ax.set_xlabel(r'$\lambda_2$ (consistency)')
    ax.set_ylabel(r'$\lambda_1$ (contrastive)')
    ax.set_title(mlab, fontsize=11)
    for i in range(len(l1s)):
        for j in range(len(l2s)):
            if np.isfinite(g[i, j]):
                ax.text(j, i, f'{g[i, j]:.3f}', ha='center', va='center',
                        fontsize=9, color='white' if im.norm(g[i, j]) > 0.55 else '#12333A')
            else:
                ax.text(j, i, '—', ha='center', va='center', color='#999')
# 标记基准点 (λ1=0.05, λ2=0.10)
if 0.05 in l1s and 0.10 in l2s:
    bi, bj = l1s.index(0.05), l2s.index(0.10)
    for ax in axes:
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                   edgecolor='#C82423', lw=2))
fig.suptitle('λ sensitivity (Reptile+Transformer; red box = published baseline)',
             fontsize=11, y=1.02)
fig.tight_layout()
for ext in ('png', 'tiff'):
    kw = {'dpi': 300, 'bbox_inches': 'tight', 'facecolor': 'white'}
    if ext == 'tiff':
        kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(FIG / f'FigureS5_lambda_heatmap.{ext}', **kw)
plt.close(fig)

# Table S9 CSV
tab = df.copy()
tab.to_csv(ROOT / 'supplement_output' / 'paper_revision' / 'TableS9_lambda.csv',
           index=False, encoding='utf-8-sig')
print('Figure S5 与 TableS9_lambda.csv 已保存')
print(tab.to_string(index=False))
