# -*- coding: utf-8 -*-
"""
补充实验1：四个 GraphDTA GNN（GCNNet / GATNet / GAT_GCN / GINConvNet）
在“低相似度样本”上的 Grad-CAM 原子归因 + Edge Occlusion 边重要性。

低相似度样本定义（靶点冷启动场景）：
  1) 蛋白轴（主）：测试靶点与训练集最近序列一致性 < 40%，
     即 MMseqs2 40% 阈值下的跨簇靶点（novel target）；
  2) 药物轴（辅）：测试药物对训练集药物的最大 ECFP4 Tanimoto 尽量低
     （Davis 训练/测试共用 68 个药物的固定面板，该轴无区分度，自动跳过）；
  3) 标签为该测试靶点内亲和力 top20% 的活性化合物（与逐靶点 EF 定义一致），
     且四个模型预测排名一致靠前。

用法:
  python supplement_gnn_explain.py kiba
  python supplement_gnn_explain.py davis
输出: d:\\lht\\supplement_output\\gnn_explain\\<dataset>\\
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r'd:\lht')
GDT = BASE / 'GrapthDTA'
CLUSTER_DIR = BASE / 'supplement_output' / 'protein_cluster'
sys.path.insert(0, str(GDT))
from models.gcn import GCNNet          # noqa: E402
from models.gat import GATNet          # noqa: E402
from models.gat_gcn import GAT_GCN     # noqa: E402
from models.ginconv import GINConvNet  # noqa: E402

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
RADIUS, NBITS = 2, 2048
N_SELECT = 6
PROT_IDENT_CUT = 40.0   # 最近训练靶点一致性 < 40% 视为低相似度（跨40%簇）

TEAL_CMAP = LinearSegmentedColormap.from_list(
    'teal', ['#f2faf8', '#c6e7e2', '#7fc6bf', '#3d9f9b', '#0f6e6e', '#064c4d'])

MODELS = {
    'GCNNet': (GCNNet, 'conv3'),
    'GATNet': (GATNet, 'gcn2'),
    'GAT_GCN': (GAT_GCN, 'conv2'),
    'GINConvNet': (GINConvNet, 'bn5'),
}

# ---------------- 与 GrapthDTA/create_data.py 完全一致的建图逻辑 ----------------
def one_of_k_encoding(x, allowable_set):
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), list(range(11))) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11))) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11))) +
                    [atom.GetIsAromatic()])


seq_voc = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}


def seq_cat(prot):
    x = np.zeros(1000, dtype=np.int64)
    for i, ch in enumerate(prot[:1000]):
        if ch in seq_dict:
            x[i] = seq_dict[ch]
    return x


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    features = [atom_features(a) / sum(atom_features(a)) for a in mol.GetAtoms()]
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    g = nx.Graph(edges).to_directed()
    edge_index = [[e1, e2] for e1, e2 in g.edges]
    return mol, np.array(features, dtype=np.float32), edge_index


def make_data(smile, seq_enc, affinity=0.0):
    g = smile_to_graph(smile)
    if g is None:
        return None, None
    _, feats, edge_index = g
    ei = (torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index
          else torch.zeros((2, 1), dtype=torch.long))
    d = Data(x=torch.tensor(feats), edge_index=ei,
             y=torch.tensor([affinity], dtype=torch.float))
    d.target = torch.LongTensor(np.array([seq_enc]))
    d.batch = torch.zeros(feats.shape[0], dtype=torch.long)
    return d, g


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=NBITS) if m is not None else None


# ---------------- 低相似度样本筛选 ----------------
def build_pool(dataset, out_dir):
    tr = pd.read_csv(GDT / 'data' / f'{dataset}_train.csv')
    te = pd.read_csv(GDT / 'data' / f'{dataset}_test.csv')

    # 蛋白最近一致性（与聚类脚本相同的 sorted(set) 编号）
    pid = pd.read_csv(CLUSTER_DIR / dataset / 'per_test_target_identity.csv')
    test_seqs = sorted(set(te['target_sequence'].dropna()))
    seq2tid = {s: f'test_{i:04d}' for i, s in enumerate(test_seqs)}
    ident_map = dict(zip(pid['target_id'], pid['nearest_train_identity_pct']))
    te['prot_identity'] = te['target_sequence'].map(
        lambda s: ident_map.get(seq2tid[s], np.nan))

    # 药物最大 Tanimoto
    tr_smi = pd.unique(tr['compound_iso_smiles'])
    tr_fps = [f for f in (fp(s) for s in tr_smi) if f is not None]
    te_smi = pd.unique(te['compound_iso_smiles'])
    drug_max = {}
    for s in te_smi:
        f = fp(s)
        drug_max[s] = float(np.max(DataStructs.BulkTanimotoSimilarity(f, tr_fps))) if f else np.nan
    te['drug_max_tanimoto'] = te['compound_iso_smiles'].map(drug_max)
    te.to_csv(out_dir / 'test_low_similarity_table.csv', index=False, encoding='utf-8-sig')

    # 逐靶点 top20% 活性
    def _top20(v):
        thr = np.sort(v.values)[::-1][max(1, int(len(v) * 0.2)) - 1]
        return v >= thr

    te['is_active'] = te.groupby('target_sequence')['affinity'].transform(_top20)

    pool = te[(te['prot_identity'] < PROT_IDENT_CUT) & te['is_active']].copy()
    print(f'[{dataset}] 低相似度跨簇靶点活性配对候选: {len(pool)} '
          f'(测试靶点数 {te.target_sequence.nunique()}, '
          f'药物Tanimoto 中位 {te.drug_max_tanimoto.median():.3f})')
    return pool


def load_models(dataset):
    ms = {}
    for name, (cls, _) in MODELS.items():
        m = cls().to(DEVICE)
        sd = torch.load(GDT / f'model_{name}_{dataset}.model', map_location=DEVICE)
        m.load_state_dict(sd)
        m.eval()
        ms[name] = m
    return ms


@torch.no_grad()
def predict_all(models, data):
    data = data.to(DEVICE)
    return {n: float(m(data).reshape(-1)[0].item()) for n, m in models.items()}


def gradcam_atoms(model, hook_name, data):
    data = data.to(DEVICE)
    acts, grads = {}, {}

    def fwd(_m, _i, out):
        acts['h'] = out

    def bwd(_m, _gi, go):
        grads['h'] = go[0]

    module = dict(model.named_modules())[hook_name]
    h1 = module.register_forward_hook(fwd)
    h2 = module.register_full_backward_hook(bwd)
    model.zero_grad(set_to_none=True)
    out = model(data).reshape(-1)[0]
    out.backward()
    H, G = acts['h'].detach(), grads['h'].detach()
    # 钩子位于 ReLU 之前：用 ReLU(H) 作为特征、G 仅在 H>0 单元非零，
    # 两者逐元素等价于对 ReLU 后激活做 Grad-CAM，避免死单元负值将 CAM 全部抵消
    cam = torch.relu((H.clamp(min=0) * G.mean(0)).sum(1)).detach().cpu().numpy()
    h1.remove()
    h2.remove()
    if cam.max() > 1e-12:
        cam = cam / cam.max()
    else:
        cam = np.zeros_like(cam)
    return cam, float(out.item())


@torch.no_grad()
def edge_occlusion(model, data, bond_pairs):
    data = data.to(DEVICE)
    y_full = float(model(data).reshape(-1)[0].item())
    ei = data.edge_index
    imp = np.zeros(len(bond_pairs))
    edges = [tuple(x) for x in ei.cpu().numpy().T]
    for b, (i, j) in enumerate(bond_pairs):
        drop = {(i, j), (j, i)}
        keep_idx = [k for k, e in enumerate(edges) if e not in drop]
        if not keep_idx:
            keep_idx = [0]
        d2 = Data(x=data.x, edge_index=ei[:, keep_idx], y=data.y).to(DEVICE)
        d2.target = data.target
        d2.batch = data.batch
        imp[b] = y_full - float(model(d2).reshape(-1)[0].item())
    return imp, y_full


def atom_weights_from_edges(mol, edge_imp):
    pos = np.clip(edge_imp, 0, None)
    w = np.zeros(mol.GetNumAtoms())
    cnt = np.zeros(mol.GetNumAtoms())
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        w[i] += pos[b.GetIdx()]; w[j] += pos[b.GetIdx()]
        cnt[i] += 1; cnt[j] += 1
    cnt[cnt == 0] = 1
    return w / cnt


# ---------------- 可视化 ----------------
ATOM_COLORS = {'C': (0.25, 0.25, 0.25), 'N': (0.05, 0.25, 0.55),
               'O': (0.65, 0.10, 0.10), 'S': (0.60, 0.45, 0.05),
               'F': (0.0, 0.45, 0.35), 'Cl': (0.0, 0.45, 0.15),
               'Br': (0.45, 0.15, 0.0), 'P': (0.55, 0.10, 0.55)}


def _mol_xy(mol):
    m2 = Chem.Mol(mol)
    AllChem.Compute2DCoords(m2)
    conf = m2.GetConformer()
    n = m2.GetNumAtoms()
    P = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                  for i in range(n)])
    return m2, P


def _draw_bonds(ax, mol, P):
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        p, q = P[i], P[j]
        d = q - p
        L = np.hypot(*d) + 1e-9
        perp = np.array([-d[1], d[0]]) / L
        bt = b.GetBondType()
        if bt == Chem.BondType.AROMATIC:
            ax.plot([p[0], q[0]], [p[1], q[1]], color='0.25', lw=1.6, zorder=1)
            off = 0.10
            ax.plot([p[0] + perp[0] * off, q[0] + perp[0] * off],
                    [p[1] + perp[1] * off, q[1] + perp[1] * off],
                    color='0.25', lw=1.0, ls=(0, (3, 2)), zorder=1)
        elif bt == Chem.BondType.DOUBLE:
            for off in (-0.08, 0.08):
                ax.plot([p[0] + perp[0] * off, q[0] + perp[0] * off],
                        [p[1] + perp[1] * off, q[1] + perp[1] * off],
                        color='0.25', lw=1.4, zorder=1)
        elif bt == Chem.BondType.TRIPLE:
            ax.plot([p[0], q[0]], [p[1], q[1]], color='0.25', lw=1.4, zorder=1)
            for off in (-0.13, 0.13):
                ax.plot([p[0] + perp[0] * off, q[0] + perp[0] * off],
                        [p[1] + perp[1] * off, q[1] + perp[1] * off],
                        color='0.25', lw=1.2, zorder=1)
        else:
            ax.plot([p[0], q[0]], [p[1], q[1]], color='0.25', lw=1.6, zorder=1)


def draw_atom_heatmap(mol, weights, ax):
    mol, P = _mol_xy(mol)
    w = np.array(weights, dtype=float)
    if w.max() > 1e-12:
        w = w / w.max()
    _draw_bonds(ax, mol, P)
    # 热晕
    ax.scatter(P[:, 0], P[:, 1], s=900, c=np.clip(w, 0, 1), cmap=TEAL_CMAP,
               vmin=0, vmax=1, alpha=0.55, edgecolors='none', zorder=2)
    # 原子底色 + 元素符号
    ax.scatter(P[:, 0], P[:, 1], s=150, facecolors='white',
               edgecolors='0.3', linewidths=0.8, zorder=3)
    for i, a in enumerate(mol.GetAtoms()):
        sym = a.GetSymbol()
        col = ATOM_COLORS.get(sym, (0.1, 0.1, 0.1))
        ax.text(P[i, 0], P[i, 1], sym, ha='center', va='center',
                fontsize=6.5, color=col, zorder=4, fontfamily='DejaVu Sans')
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.margins(0.18)
    ax.axis('off')


def save_atom_map(mol, weights, title, stem):
    fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=300)
    draw_atom_heatmap(mol, weights, ax)
    ax.set_title(title, fontsize=11)
    fig.patch.set_facecolor('white')
    fig.savefig(str(stem) + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(str(stem) + '.tiff', dpi=300, bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def draw_edge_mol(mol, edge_imp, ax=None):
    pos = np.clip(edge_imp, 0, None)
    rng = pos.max() if pos.max() > 1e-9 else 1.0
    bond_colors = {}
    for b in mol.GetBonds():
        v = edge_imp[b.GetIdx()]
        if v > 1e-9:
            bond_colors[b.GetIdx()] = TEAL_CMAP(min(1.0, v / rng))
        elif v < -1e-9:
            bond_colors[b.GetIdx()] = (0.75, 0.30, 0.30, 0.6)
    aw = atom_weights_from_edges(mol, edge_imp)
    atom_colors = {a.GetIdx(): TEAL_CMAP(min(1.0, aw[a.GetIdx()] / (aw.max() + 1e-12)))
                   for a in mol.GetAtoms()}
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(420, 360)
    drawer.drawOptions().bondLineWidth = 2
    drawer.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())),
                        highlightBonds=list(bond_colors.keys()),
                        highlightAtomColors=atom_colors,
                        highlightBondColors=bond_colors)
    drawer.FinishDrawing()
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert('RGB')
    if ax is not None:
        ax.imshow(img)
    return img


def save_edge_map(mol, edge_imp, title, stem):
    fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=300)
    draw_edge_mol(mol, edge_imp, ax)
    ax.axis('off')
    ax.set_title(title, fontsize=11)
    fig.patch.set_facecolor('white')
    fig.savefig(str(stem) + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(str(stem) + '.tiff', dpi=300, bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    neg = np.clip(edge_imp, None, 0).sum() / (np.abs(edge_imp).sum() + 1e-12)
    return float(neg)


def panel_figure(mol, cams, occs, preds, tag, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.8), dpi=300)
    for j, name in enumerate(MODELS):
        draw_atom_heatmap(mol, cams[name], axes[0, j])
        axes[0, j].set_title(f'{name}  Grad-CAM (y={preds[name]:.2f})', fontsize=10)
        draw_edge_mol(mol, occs[name], axes[1, j])
        axes[1, j].set_title(f'{name}  Edge importance', fontsize=10)
        axes[1, j].axis('off')
    fig.suptitle(tag.replace('_', ' '), fontsize=12)
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(out_dir / f'panel_{tag}.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / f'panel_{tag}.tiff', dpi=300, bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def main(dataset):
    t_start = time.time()
    out_dir = BASE / 'supplement_output' / 'gnn_explain' / dataset
    maps = out_dir / 'maps'
    maps.mkdir(parents=True, exist_ok=True)

    print(f'==== {dataset} | device={DEVICE} ====')
    pool = build_pool(dataset, out_dir)
    models = load_models(dataset)

    # 候选去重（同 smiles+靶点只算一次），分子尺寸约束
    cand = pool.drop_duplicates(['compound_iso_smiles', 'target_sequence']).copy()
    rows = []
    seq_cache = {}
    t0 = time.time()
    for _, r in cand.iterrows():
        mol = Chem.MolFromSmiles(r['compound_iso_smiles'])
        if mol is None or not (10 <= mol.GetNumHeavyAtoms() <= 48):
            continue
        seq = r['target_sequence']
        if seq not in seq_cache:
            seq_cache[seq] = seq_cat(seq)
        d, _ = make_data(r['compound_iso_smiles'], seq_cache[seq], r['affinity'])
        pr = predict_all(models, d)
        rows.append({'smiles': r['compound_iso_smiles'], 'seq': seq,
                     'affinity': r['affinity'],
                     'prot_identity': r['prot_identity'],
                     'drug_sim': r['drug_max_tanimoto'],
                     'n_ha': mol.GetNumHeavyAtoms(), **pr})
    cdf = pd.DataFrame(rows)
    print(f'[{dataset}] 有效候选 {len(cdf)}，预测耗时 {time.time()-t0:.0f}s')
    if cdf.empty:
        raise RuntimeError('候选为空')

    for m in MODELS:
        cdf[m + '_r'] = cdf[m].rank(pct=True)
    cdf['mean_rank'] = cdf[[m + '_r' for m in MODELS]].mean(axis=1)
    cdf['scaffold'] = cdf['smiles'].map(
        lambda s: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s),
                                                      includeChirality=False))
    # 药物轴有区分度时优先低药物相似度；Davis（全部~1.0）跳过
    drug_discriminative = cdf['drug_sim'].quantile(0.25) < 0.9
    cdf = cdf.sort_values(
        ['mean_rank'] + (['drug_sim'] if drug_discriminative else []),
        ascending=[False] + ([True] if drug_discriminative else [])).reset_index(drop=True)

    chosen, used_scaff, used_seq = [], set(), set()
    for _, r in cdf.iterrows():
        if r['scaffold'] in used_scaff or r['seq'] in used_seq:
            continue
        used_scaff.add(r['scaffold']); used_seq.add(r['seq'])
        chosen.append(r)
        if len(chosen) >= N_SELECT:
            break
    print(f'[{dataset}] 入选 {len(chosen)} 个低相似度样本，开始归因 ...')

    records, summary = [], []
    from scipy.stats import spearmanr
    for si, r in enumerate(chosen):
        smi, seq = r['smiles'], r['seq']
        mol, _, _ = smile_to_graph(smi)
        data, _ = make_data(smi, seq_cache[seq], r['affinity'])
        bond_pairs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        tag = (f"{dataset}_sample{si+1}_protID{r.prot_identity:.0f}"
               f"_drugSim{r.drug_sim:.2f}_aff{r.affinity:.1f}")
        with open(out_dir / f'{tag}.smi', 'w') as f:
            f.write(smi)

        cams, occs, preds = {}, {}, {}
        for name, (_, hook) in MODELS.items():
            cam, yp = gradcam_atoms(models[name], hook, data)
            occ, yf = edge_occlusion(models[name], data, bond_pairs)
            cams[name], occs[name], preds[name] = cam, occ, yf
            save_atom_map(mol, cam, f'{name} Grad-CAM', maps / f'{tag}__{name}__gradcam')
            neg = save_edge_map(mol, occ, f'{name} edge importance',
                                maps / f'{tag}__{name}__edge')

            ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
            hetero = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() not in ('C', 'H')]
            aw = atom_weights_from_edges(mol, occ)
            rho = (float(spearmanr(cam, aw).statistic)
                   if cam.std() > 1e-9 and aw.std() > 1e-9 else np.nan)
            summary.append({
                'sample': tag, 'model': name, 'pred': yf,
                'cam_ring_fraction': float(cam[ring].sum() / (cam.sum() + 1e-12)),
                'cam_heteroatom_fraction': float(cam[hetero].sum() / (cam.sum() + 1e-12)),
                'edge_negative_fraction': neg, 'gradcam_edge_spearman': rho})
            records.append({'sample': tag, 'model': name, 'smiles': smi,
                            'prot_identity': r.prot_identity,
                            'drug_max_tanimoto': r.drug_sim, 'affinity': r.affinity,
                            'pred': yf})
        panel_figure(mol, cams, occs, preds, tag, out_dir)
        print(f'  [{si+1}/{len(chosen)}] {tag}: ' +
              ', '.join(f'{k}={v:.2f}' for k, v in preds.items()))

    sel_cols = ['smiles', 'affinity', 'prot_identity', 'drug_sim', 'n_ha'] + list(MODELS)
    pd.DataFrame(chosen)[sel_cols].to_csv(out_dir / 'selected_samples.csv',
                                          index=False, encoding='utf-8-sig')
    pd.DataFrame(records).to_csv(out_dir / 'selected_predictions_long.csv',
                                 index=False, encoding='utf-8-sig')
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_dir / 'attribution_summary.csv', index=False, encoding='utf-8-sig')
    agg = sdf.groupby('model').mean(numeric_only=True).round(4)
    print('\n各模型归因汇总（低相似度样本均值）:')
    print(agg.to_string())
    agg_json = agg.reset_index().to_dict(orient='records')
    with open(out_dir / 'attribution_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'dataset': dataset,
                   'prot_identity_cut_pct': PROT_IDENT_CUT,
                   'drug_axis_used': bool(drug_discriminative),
                   'n_selected': len(chosen),
                   'per_model': agg_json}, f, ensure_ascii=False, indent=2)
    print(f'[{dataset}] 完成，用时 {(time.time()-t_start)/60:.1f} min -> {out_dir}')


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else 'kiba'
    main(ds)
