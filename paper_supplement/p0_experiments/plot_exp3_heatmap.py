# -*- coding: utf-8 -*-
"""Figure S11: MMseqs2 参数敏感性热图（跨簇比例%）。12 行(数据集×覆盖率) × 6 列(同一性)。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

OUT = Path(r'd:\lht\修改supplement_output\p0_experiments')
df = pd.read_csv(OUT / 'param_sensitivity' / 'mmseqs_grid.csv')

datasets = ['chembl', 'davis', 'kiba', 'bindingdb']
ds_label = {'chembl': 'ChEMBL', 'davis': 'Davis', 'kiba': 'KIBA', 'bindingdb': 'BindingDB'}
covs = [0.7, 0.8, 0.9]
ids = [30, 40, 50, 60, 70, 80]

mat = np.full((len(datasets) * len(covs), len(ids)), np.nan)
ylabels = []
for i, ds in enumerate(datasets):
    for j, cov in enumerate(covs):
        r = i * 3 + j
        ylabels.append(f'{ds_label[ds]}  c={cov}')
        for k, t in enumerate(ids):
            v = df[(df.dataset == ds) & (df.coverage == cov) &
                   (df.identity_pct == t)]['cross_cluster_ratio_pct']
            mat[r, k] = v.iloc[0]

cmap = LinearSegmentedColormap.from_list('bg', ['#E0F3F1', '#6FBFB3', '#0B6E6E', '#073B3B'])

fig, ax = plt.subplots(figsize=(7.2, 4.6))
im = ax.imshow(mat, cmap=cmap, vmin=25, vmax=100, aspect='auto')
ax.set_xticks(range(len(ids)))
ax.set_xticklabels([f'{t}%' for t in ids])
ax.set_yticks(range(len(ylabels)))
ax.set_yticklabels(ylabels, fontsize=9)
ax.set_xlabel('Sequence-identity threshold', fontsize=10)
ax.set_ylabel('Dataset / bidirectional coverage', fontsize=10)
for r in range(mat.shape[0]):
    for c in range(mat.shape[1]):
        v = mat[r, c]
        ax.text(c, r, f'{v:.1f}', ha='center', va='center', fontsize=8,
                color='white' if v > 68 else '#073B3B')
for x in [1.5, 3.5, 5.5]:
    ax.axvline(x, color='white', lw=1.4)
for x in [2.5, 5.5, 8.5]:
    ax.axhline(x, color='#073B3B', lw=1.2)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label('Cross-cluster test targets (%)', fontsize=9)
ax.set_title('Homology-audit parameter sensitivity (MMseqs2 easy-cluster)', fontsize=10)
plt.tight_layout()
for ext, dpi in [('png', 300), ('tiff', 300)]:
    fig.savefig(OUT / f'FigureS11_audit_param_sensitivity.{ext}', dpi=dpi, bbox_inches='tight')
print('saved')
