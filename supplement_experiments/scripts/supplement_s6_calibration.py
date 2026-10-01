# -*- coding: utf-8 -*-
"""
Figure S6: 校准可靠性曲线（regression reliability, 按数据集分面）
数据源:
  1) 已存 predictions.npz（y_true/y_pred）:
     ChEMBL-MLP        baseline_output/predictions.npz
     ChEMBL-Transformer(MSE变体) transformer_morgan_mse_output/predictions.npz
     KIBA-Transformer(全特征消融) transformer_ablation_output_kiba/full/predictions.npz
  2) GNN 现算: GrapthDTA model_{M}_{ds}.model（chembl权重0字节自动跳过）
用法:
  python supplement_s6_calibration.py
输出: supplement_output/paper_revision/figures/FigureS6_reliability.{png,tiff}
      supplement_output/paper_revision/TableS6b_calibration_ece.csv
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'd:\lht')
OUT = ROOT / 'supplement_output' / 'paper_revision' / 'figures'
PRED_CACHE = ROOT / 'supplement_output' / 'paper_revision' / 'gnn_preds'
OUT.mkdir(parents=True, exist_ok=True)
PRED_CACHE.mkdir(parents=True, exist_ok=True)

DSS = ['chembl', 'davis', 'kiba', 'bindingdb']
DS_CN = {'chembl': 'ChEMBL', 'davis': 'Davis', 'kiba': 'KIBA', 'bindingdb': 'BindingDB'}
GNN = ['GCNNet', 'GATNet', 'GAT_GCN', 'GINConvNet']

NPZ_SOURCES = {
    ('chembl', 'MLP'): ROOT / 'baseline_output' / 'predictions.npz',
    ('chembl', 'Transformer(MSE variant)'): ROOT / 'transformer_morgan_mse_output' / 'predictions.npz',
    ('kiba', 'Transformer(full-feat ablation)'): ROOT / 'transformer_ablation_output_kiba' / 'full' / 'predictions.npz',
}


# ---------- 1) 非 GNN：读现成 npz ----------
series = {}          # (ds, label) -> (y_true, y_pred)
for (ds, lab), fp in NPZ_SOURCES.items():
    if fp.exists():
        d = np.load(fp)
        series[(ds, lab)] = (d['y_true'].astype(float), d['y_pred'].astype(float))
        print(f'[npz] {ds} {lab}: n={len(d["y_true"])}')
    else:
        print(f'[miss] {fp}')

# ---------- 2) GNN：加载 processed 数据现算 ----------
sys.path.insert(0, str(ROOT / 'GrapthDTA'))
os.chdir(ROOT / 'GrapthDTA')          # TestbedDataset 相对路径 root='data'
from utils import TestbedDataset       # noqa: E402
from torch_geometric.loader import DataLoader
from models.gcn import GCNNet
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.ginconv import GINConvNet
MODEL_CLS = {'GCNNet': GCNNet, 'GATNet': GATNet, 'GAT_GCN': GAT_GCN, 'GINConvNet': GINConvNet}
import torch

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def gnn_predict(ds, m):
    cache = PRED_CACHE / f'{m}_{ds}.npz'
    if cache.exists():
        d = np.load(cache)
        return d['y_true'], d['y_pred']
    w = ROOT / 'GrapthDTA' / f'model_{m}_{ds}.model'
    if not w.exists() or w.stat().st_size == 0:
        print(f'[skip] {m}-{ds}: 权重缺失')
        return None
    state = torch.load(w, map_location='cpu', weights_only=False)
    sd = None
    for k in ('model_state_dict', 'state_dict', 'model'):
        if isinstance(state, dict) and k in state and isinstance(state[k], dict):
            sd = state[k]
            break
    if sd is None and isinstance(state, dict) and all(torch.is_tensor(v) for v in state.values()):
        sd = state
    model = MODEL_CLS[m]()
    model.load_state_dict(sd, strict=True)
    model = model.to(dev).eval()
    loader = DataLoader(TestbedDataset(root='data', dataset=f'{ds}_test'),
                        batch_size=1024, shuffle=False)
    ps, ys = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(dev)
            ps.append(model(b).view(-1).float().cpu().numpy())
            ys.append(b.y.view(-1).float().cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    np.savez(cache, y_true=y_true, y_pred=y_pred)
    print(f'[infer] {m}-{ds}: n={len(y_true)} 已缓存')
    return y_true, y_pred


for ds in DSS:
    for m in GNN:
        r = gnn_predict(ds, m)
        if r is not None:
            series[(ds, f'GraphDTA {m}')] = r
os.chdir(ROOT)

# ---------- 3) 可靠性曲线（min-max缩放 + 10等宽bin） ----------
def reliability(yt, yp, n_bins=10):
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    rng = hi - lo + 1e-9
    yt_n, yp_n = (yt - lo) / rng, (yp - lo) / rng
    edges = np.linspace(0, 1, n_bins + 1)
    xs, es, ws = [], [], []
    for i in range(n_bins):
        m = (yp_n >= edges[i]) & (yp_n < edges[i + 1] + (i == n_bins - 1) * 1e-9)
        if m.sum() == 0:
            continue
        xs.append(yp_n[m].mean()); es.append(yt_n[m].mean()); ws.append(m.mean())
    ece = float(np.sum([w * abs(x - e) for x, e, w in zip(xs, es, ws)]))
    return np.array(xs), np.array(es), ece


fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.4), dpi=300, sharey=True)
PALETTE = ['#006D77', '#2E9E8F', '#83C5BE', '#1B5E20', '#4C8C8C', '#76B5AD',
           '#B2DFDB', '#5F9EA0', '#8FBF9F', '#2E7D6B', '#C0625B']
rows = []
for ax, ds in zip(axes, DSS):
    keys = [k for k in series if k[0] == ds]
    for i, k in enumerate(sorted(keys, key=lambda x: x[1])):
        yt, yp = series[k]
        xs, es, ece = reliability(yt, yp)
        ax.plot(xs, es, marker='o', ms=4, lw=1.4, color=PALETTE[i % len(PALETTE)],
                label=f'{k[1]} (ECE′={ece:.3f}, n={len(yt)})')
        rows.append([DS_CN[ds], k[1], round(ece, 4), len(yt)])
    ax.plot([0, 1], [0, 1], ls='--', color='#B0B0B0', lw=1, zorder=0)
    ax.set_title(DS_CN[ds], fontsize=11)
    ax.set_xlabel('mean predicted (scaled)')
    ax.grid(color='#E9F1F0', ls='--', lw=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axes[0].set_ylabel('mean observed (scaled)')
axes[0].legend(fontsize=6.5, loc='upper left', framealpha=0.9)
fig.suptitle('Regression reliability curves (min-max scaled per dataset; diagonal = perfect)',
             fontsize=11, y=1.02)
fig.tight_layout()
for ext in ('png', 'tiff'):
    kw = {'dpi': 300, 'bbox_inches': 'tight', 'facecolor': 'white'}
    if ext == 'tiff':
        kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(OUT / f'FigureS6_reliability.{ext}', **kw)
plt.close(fig)

pd.DataFrame(rows, columns=['Dataset', 'Model', 'ECE_prime(scaled)', 'n_test']).to_csv(
    ROOT / 'supplement_output' / 'paper_revision' / 'TableS6b_calibration_ece.csv',
    index=False, encoding='utf-8-sig')
print('Figure S6 与 TableS6b 已保存；GNN 预测缓存在', PRED_CACHE)
