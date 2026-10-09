# -*- coding: utf-8 -*-
"""P1 figures (three separate plots, blue-green palette, 300 dpi PNG+TIFF).

Fig 1: DTA (dirty) vs PPI (clean) cross-cluster ratio @ 40% identity
Fig 2: PPI binding-spectrum mirror audit, hidden redundancy KS >= 0.7
Fig 3: model rank changes Standard -> Audited-40% (Spearman, parallel arrows)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 1.1,
    'axes.edgecolor': '#33424A',
    'axes.labelcolor': '#1F2D34',
    'xtick.color': '#33424A',
    'ytick.color': '#33424A',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUT = Path(r'd:\lht\审计验证与跨领域')
FIG = OUT / 'p1_figures'
FIG.mkdir(exist_ok=True)

# ---- palette (light blue -> green) ----
BLUE_DEEP   = '#2E7DA8'
BLUE_MID    = '#5BA8C4'
BLUE_LIGHT  = '#93C7D9'
TEAL_DEEP   = '#1F9E89'
TEAL_MID    = '#57BFA8'
TEAL_LIGHT  = '#A9DBC9'
GREEN_DEEP  = '#15836B'
INK         = '#1F2D34'
GREY        = '#8FA3AB'

# ============================================================
# Fig 1: cross-cluster ratio bars
# ============================================================
datasets = ['Davis', 'KIBA', 'BindingDB', 'SKEMPI\nPr/PI']
ratios = [41.5, 41.2, 39.7, 98.55]
counts = ['27/65', '14/34', '31/78', '68/69']
colors = [BLUE_MID, BLUE_DEEP, BLUE_LIGHT, TEAL_DEEP]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.arange(len(datasets))
bars = ax.bar(x, ratios, width=0.62, color=colors, edgecolor='white', linewidth=1.2,
              zorder=3)
for xi, r, c in zip(x, ratios, counts):
    ax.text(xi, r + 2.0, f'{r:.1f}%', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=INK, zorder=4)
    ax.text(xi, r / 2, c, ha='center', va='center', fontsize=10.5,
            color='white', fontweight='bold', zorder=4)

# group brackets below the tick labels (blended: data-x, axes-y)
from matplotlib.transforms import blended_transform_factory
bt = blended_transform_factory(ax.transData, ax.transAxes)
y_b = -0.19
for xa, xb in ((-0.48, 2.48), (2.52, 3.48)):
    ax.plot([xa, xb], [y_b, y_b], color=INK, lw=1.2, transform=bt, clip_on=False)
    ax.plot([xa, xa], [y_b, y_b - 0.025], color=INK, lw=1.2, transform=bt, clip_on=False)
    ax.plot([xb, xb], [y_b, y_b - 0.025], color=INK, lw=1.2, transform=bt, clip_on=False)
ax.text(1.0, y_b - 0.045, 'DTA benchmarks — homology "dirty"',
        transform=bt, ha='center', va='top', fontsize=10.5,
        color=BLUE_DEEP, fontweight='bold')
ax.text(3.0, y_b - 0.045, 'PPI hold-out —\nhomology clean',
        transform=bt, ha='center', va='top', fontsize=10.5,
        color=GREEN_DEEP, fontweight='bold')

ax.axhline(50, color=GREY, ls='--', lw=1.0, zorder=1)
ax.text(-0.5, 51.5, '50%', color=GREY, fontsize=9, ha='left', va='bottom')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_xlim(-0.6, 3.6)
ax.set_ylim(0, 109)
ax.set_ylabel('Cross-cluster test targets (%) @ 40% identity', fontsize=11.5)
ax.set_title('Homology audit: DTA vs PPI target-disjoint splits', fontsize=12.5,
             fontweight='bold', pad=10)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, color='#DCE6EA', lw=0.8, zorder=0)
ax.set_axisbelow(True)
fig.subplots_adjust(bottom=0.24)
for ext in ('png', 'tif'):
    fig.savefig(FIG / f'fig1_cross_cluster_dta_vs_ppi.{ext}')
plt.close(fig)

# ============================================================
# Fig 2: binding-spectrum mirror audit histogram
# ============================================================
ks = pd.read_csv(OUT / 'binding_spectrum_mirror_results.csv')
labels = pd.read_csv(OUT / 'skempi_prpi' / 'labels.csv')
audit = pd.read_csv(OUT / 'audit_results' / 'skempi_prpi_expanded' / 'per_target_audit.csv')
chain_seqs = json.load(open(OUT / 'pdb_chain_sequences.json'))
seq2tid = {r.target_sequence: r.target_id for r in audit.itertuples()}
flags = {r.target_id: bool(r.same_cluster_40) for r in audit.itertuples()}

def is_same(complex_id):
    p = complex_id.split('_')
    pdb, chs = p[0], p[1:]
    for ch in chs:
        tid = seq2tid.get(chain_seqs.get(f'{pdb}_{ch}', ''))
        if tid and flags.get(tid):
            return True
    return False

ks['same40'] = ks['test_complex'].apply(is_same)
cross = ks.loc[~ks['same40'], 'max_ks_similarity'].values
n_cross = len(cross)
n_high = int((cross >= 0.7).sum())
pct = 100 * n_high / n_cross

bins = np.linspace(0, 1, 21)
counts_b, edges = np.histogram(cross, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2
bar_colors = [TEAL_DEEP if c >= 0.7 else BLUE_LIGHT for c in centers]

fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.bar(centers, counts_b, width=np.diff(edges) * 0.92, color=bar_colors,
       edgecolor='white', linewidth=0.8, zorder=3)
ax.axvline(0.7, color='#D2691E', ls='--', lw=1.6, zorder=4)
ax.text(0.7, 1.025, 'mirror-redundancy threshold: KS similarity = 0.70',
        transform=blended_transform_factory(ax.transData, ax.transAxes),
        color='#B85C12', fontsize=9.3, va='bottom', ha='center')

# rug / jitter points (larger markers, user preference)
rng = np.random.RandomState(0)
jitter = 0.18 + rng.rand(n_cross) * 0.55
pt_colors = [TEAL_DEEP if v >= 0.7 else BLUE_MID for v in cross]
ax.scatter(cross, jitter, s=42, c=pt_colors, alpha=0.75, edgecolor='white',
           linewidth=0.6, zorder=5)

ax.annotate(f'{n_high}/{n_cross} = {pct:.1f}% hidden\nbinding-spectrum redundancy',
            xy=(0.85, 4.2), xytext=(0.24, 0.74),
            xycoords='data', textcoords='axes fraction',
            fontsize=10.5, color=GREEN_DEEP, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color=GREEN_DEEP, lw=1.4,
                            connectionstyle='arc3,rad=-0.22'),
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=TEAL_MID, lw=1.0))

ax.set_xlabel('Max KS similarity to training complexes (ΔΔG spectra)', fontsize=11.5)
ax.set_ylabel('Number of test complexes', fontsize=11.5)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, max(counts_b) + 1.4)
fig.text(0.5, 0.965, 'PPI binding-spectrum mirror audit (59 cross-cluster complexes, 40% cut)',
         ha='center', fontsize=12, fontweight='bold', color=INK)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, color='#DCE6EA', lw=0.8, zorder=0)
ax.set_axisbelow(True)
fig.subplots_adjust(top=0.81, bottom=0.15, left=0.09, right=0.98)
for ext in ('png', 'tif'):
    fig.savefig(FIG / f'fig2_ppi_mirror_redundancy.{ext}')
plt.close(fig)

# ============================================================
# Fig 3: parallel rank arrows, Standard -> Audited-40%
# ============================================================
m = pd.read_csv(OUT / 'p1_out' / 'p1_model_results.csv')
m = m[m['model'] != 'Mean'].copy()  # constant predictions -> Spearman undefined
piv_m = m.pivot_table(index='model', columns='eval_set', values='Spearman', aggfunc='mean')
piv_s = m.pivot_table(index='model', columns='eval_set', values='Spearman', aggfunc='std')
rank_std = piv_m['Standard'].rank(ascending=False, method='min').astype(int)
rank_aud = piv_m['Audited-40%'].rank(ascending=False, method='min').astype(int)

order = ['ESM2-site+MLP', 'Linear', 'ESM2-site+TF', 'RF', 'ESM2-mean+MLP', 'Zero']
model_color = {
    'ESM2-site+MLP': TEAL_DEEP,
    'Linear':       BLUE_DEEP,
    'ESM2-site+TF': TEAL_MID,
    'RF':           BLUE_MID,
    'ESM2-mean+MLP': TEAL_LIGHT,
    'Zero':         GREY,
}
pretty = {
    'ESM2-site+MLP': 'ESM2-site + MLP',
    'Linear': 'Linear (mutation feats)',
    'ESM2-site+TF': 'ESM2-site + Transformer',
    'RF': 'Random Forest',
    'ESM2-mean+MLP': 'ESM2-mean + MLP',
    'Zero': 'Zero baseline',
}

fig, ax = plt.subplots(figsize=(8.8, 5.4))
xL, xR = 0.05, 0.95
n_rank = 6
ys = {r: n_rank - r for r in range(1, n_rank + 1)}  # rank 1 at top
X_LEFT_TEXT, X_ARROW_L = 0.10, 0.47
X_ARROW_R, X_RIGHT_TEXT = 0.53, 0.90

# headers
ax.text(xL, n_rank - 0.25, 'Standard evaluation', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color=INK)
ax.text(xR, n_rank - 0.25, 'Audited evaluation (40%)', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color=GREEN_DEEP)

for model in order:
    r1, r2 = int(rank_std[model]), int(rank_aud[model])
    y1, y2 = ys[r1], ys[r2]
    col = model_color[model]
    reversal = (r1 != r2)
    lw = 3.4 if reversal else 2.2
    rad = 0.0
    if model == 'ESM2-site+TF':
        rad = 0.18
    elif model == 'RF':
        rad = -0.18
    arrow = FancyArrowPatch((X_ARROW_L, y1), (X_ARROW_R, y2),
                            arrowstyle='-|>', mutation_scale=18,
                            lw=lw, color=col, alpha=1.0,
                            connectionstyle=f'arc3,rad={rad}', zorder=3)
    ax.add_patch(arrow)
    # nodes
    ax.scatter([xL, xR], [y1, y2], s=140 if reversal else 95,
               color=col, edgecolor='white', linewidth=1.4, zorder=5)
    ax.text(xL, y1, str(r1), ha='center', va='center', fontsize=9,
            color='white', fontweight='bold', zorder=6)
    ax.text(xR, y2, str(r2), ha='center', va='center', fontsize=9,
            color='white', fontweight='bold', zorder=6)
    # left label block: name + standard Spearman
    v1, v2 = piv_m.loc[model, 'Standard'], piv_m.loc[model, 'Audited-40%']
    s1 = piv_s.loc[model, 'Standard']; s2 = piv_s.loc[model, 'Audited-40%']
    ax.text(X_LEFT_TEXT, y1 + 0.15, pretty[model], ha='left', va='center',
            fontsize=10, color=INK, fontweight='bold')
    if model != 'Zero':
        ax.text(X_LEFT_TEXT, y1 - 0.20, f'ρ = {v1:.3f} ± {s1:.3f}',
                ha='left', va='center', fontsize=9, color='#4A5B63')
    # right value block: audited Spearman
    if model != 'Zero':
        ax.text(X_RIGHT_TEXT, y2, f'ρ = {v2:.3f} ± {s2:.3f}',
                ha='right', va='center', fontsize=9, color='#4A5B63')

# reversal legend
ax.plot([], [], color='#1F2D34', lw=3.4, label='rank reversal (bold arrow)')
ax.plot([], [], color='#1F2D34', lw=2.2, alpha=0.6, label='rank preserved')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False,
          fontsize=9.5)

# annotate the leakage-beneficiary model (rank 5 -> 5, rho -24%)
ax.annotate('identity-memory probe:\nρ drops 24% after audit',
            xy=(xR - 0.03, ys[5]), xytext=(0.46, -0.62),
            fontsize=9.2, color=GREEN_DEEP, fontweight='bold', ha='left',
            arrowprops=dict(arrowstyle='->', color=GREEN_DEEP, lw=1.2,
                            connectionstyle='arc3,rad=0.25'),
            bbox=dict(boxstyle='round,pad=0.35', fc='#F2FAF7', ec=TEAL_MID, lw=0.9),
            zorder=7)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-1.0, n_rank + 0.25)
ax.axis('off')
ax.set_title('Model ranking under standard vs homology-audited PPI evaluation\n'
             '(Spearman ρ; rank 1 = best; 3 seeds, mean ± std)',
             fontsize=12.5, fontweight='bold', pad=12)
for ext in ('png', 'tif'):
    fig.savefig(FIG / f'fig3_rank_reversal_arrows.{ext}')
plt.close(fig)

print('Figures written to', FIG)
for f in sorted(FIG.iterdir()):
    print(' ', f.name, f.stat().st_size // 1024, 'KB')
