# -*- coding: utf-8 -*-
"""
Figure S4: Transformer(Reptile+Transformer) 归因可视化 —— 药物侧原子/子结构重要性
重要说明（与审稿人沟通口径）:
  主模型架构中药物与蛋白在交叉注意力前均已池化为单向量
  (reptile_transformer_model.py:240-241 unsqueeze(1)，注意力为1×1)，
  因此"配体原子×蛋白残基"注意力图在该架构中不存在、逐残基归因不可辨识。
  本图提供可辨识的一半：药物侧 Morgan 位梯度重要性 → 映射回原子环境并高亮；
  蛋白侧以注释框说明。若需逐残基图需改用 token 级交叉注意力重训（另立项）。
输入（自动）:
  - 权重: reptile_output/best_model.pt (ChEMBL Reptile+Transformer)
  - 特征: reptile_output/precomputed_features.npz (全量531247行, 训练时真实输入)
  - 行对应: 3_all_data_chembl_targets_preprocessed/combined_activities.csv 全量逐行对应,
            用 npz 内 splits=='test' 取测试集(31132行)全局索引
用法:
  python supplement_s4_attribution.py            # 3个案例
  python supplement_s4_attribution.py --n_cases 2
输出: supplement_output/paper_revision/figures/FigureS4_drug_attribution.{png,tiff}
"""
import sys, os, json, gc
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = Path(r'd:\lht')
OUT = ROOT / 'supplement_output' / 'paper_revision' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

DESC_NAMES = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
              'RingCount', 'TPSA', 'FractionCsp3', 'HeavyAtomCount', 'NumAromaticRings']

# ---------- 参数 ----------
n_cases = 3
if '--n_cases' in sys.argv:
    n_cases = int(sys.argv[sys.argv.index('--n_cases') + 1])

CKPT = ROOT / 'reptile_output' / 'best_model.pt'
FEAT = ROOT / 'reptile_output' / 'precomputed_features.npz'
SCALER = ROOT / 'reptile_output' / 'target_scaler.json'
CSV = ROOT / '3_all_data_chembl_targets_preprocessed' / 'combined_activities.csv'
SMILES_COL, TARGET_COL, AFF_COL = 'smiles', 'target_name', 'paffinity'

# ---------- 加载模型 ----------
from reptile_transformer_model import ReptileTransformer   # noqa: E402
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ReptileTransformer().to(dev)
state = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = None
if isinstance(state, dict):
    for k in ('model_state_dict', 'state_dict', 'model'):
        if k in state and isinstance(state[k], dict):
            sd = state[k]
            break
    if sd is None and all(torch.is_tensor(v) for v in state.values()):
        sd = state
if sd is None:
    raise RuntimeError(f'无法识别 checkpoint 结构: {type(state)}, keys={list(state)[:10] if isinstance(state, dict) else "-"}')
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'权重加载: missing={len(missing)} unexpected={len(unexpected)}')
model.eval()

# ---------- 加载预计算特征（全量池，用 splits 取 test 全局索引） ----------
print('加载 precomputed_features.npz（压缩包，首次约1-2分钟）...')
z = np.load(FEAT)
print('  keys:', z.files)
assert 'splits' in z.files, f'npz 缺少 splits，实际 keys={z.files}'
splits_all = z['splits']
y_all = z['y'] if 'y' in z.files else z['y_norm']
test_idx = np.where(splits_all == 'test')[0]
print(f'  全量{len(splits_all)}行, test={len(test_idx)}行')

# ---------- CSV 哈希连接定位 smiles（npz 与 csv 行序不同） ----------
df = pd.read_csv(CSV)
assert len(df) == len(splits_all), f'csv行数{len(df)} != npz行数{len(splits_all)}'
tn_all = z['target_names']
yv = y_all.astype(np.float64)
# 建图: (target, round(y,4)) -> csv 行索引
df['_key'] = list(zip(df[TARGET_COL].astype(str), np.round(df[AFF_COL].astype(float), 4)))
key2csv = {k: i for i, k in enumerate(df['_key'])}
gi_csv = np.full(len(splits_all), -1, dtype=np.int64)
miss = 0
for i in range(len(splits_all)):
    k = (str(tn_all[i]), round(float(yv[i]), 4))
    j = key2csv.get(k, -1)
    if j < 0:
        miss += 1
        continue
    gi_csv[i] = j
hit = (gi_csv >= 0).sum()
print(f'  哈希连接: {hit}/{len(splits_all)} 命中 (缺失 {miss})')
assert hit >= 0.99 * len(splits_all), f'连接命中率过低 {hit/len(splits_all):.3f}'

# ---------- 选案例候选：test 集每靶点活性最高 ----------
test_mask = splits_all == 'test'
test_gi = np.where(test_mask)[0]
test_rows = pd.DataFrame({
    '_gi': test_gi,
    TARGET_COL: tn_all[test_gi].astype(str),
    AFF_COL: yv[test_gi].astype(float),
})
# 通过哈希映射拿 smiles
test_rows[SMILES_COL] = df.iloc[gi_csv[test_gi]][SMILES_COL].values
pool = (test_rows.sort_values(AFF_COL, ascending=False)
                 .groupby(TARGET_COL, sort=False).head(3)
                 .sort_values(AFF_COL, ascending=False).head(n_cases * 15))
pool = pool.reset_index(drop=True)
print(f'  候选池 {len(pool)} 行，top3 靶点: {pool[TARGET_COL].head(3).tolist()}')

# ---------- 读取大数组 → 只保留候选行 → 立即释放 ----------
KEY = {}
for want, cands in {'morgan': ['morgan', 'morgan_fp'], 'maccs': ['maccs', 'maccs_fp'],
                    'desc': ['descriptors', 'desc'], 'protein': ['protein', 'protein_feat']}.items():
    KEY[want] = next((c for c in cands if c in z.files), None)
    assert KEY[want], f'npz 中找不到 {want}，实际 keys={z.files}'
gi = pool['_gi'].values
morgan_sel = z[KEY['morgan']][gi]
maccs_sel = z[KEY['maccs']][gi]
desc_sel = z[KEY['desc']][gi]
prot_sel = z[KEY['protein']][gi]
del z, splits_all, y_all
gc.collect()
print(f'  shapes: morgan{morgan_sel.shape} maccs{maccs_sel.shape} '
      f'desc{desc_sel.shape} protein{prot_sel.shape}')

# 过滤 RDKit 无法解析 / 指纹全零的候选
keep = []
for i, smi in enumerate(pool[SMILES_COL]):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None and mol.GetNumAtoms() > 3 and (morgan_sel[i] > 0).any():
        keep.append(i)
    if len(keep) >= n_cases:
        break
assert keep, '候选池中无有效分子'
cases = pool.iloc[keep].reset_index(drop=True)
morgan_sel, maccs_sel = morgan_sel[keep], maccs_sel[keep]
desc_sel, prot_sel = desc_sel[keep], prot_sel[keep]

# ---------- 反归一化（pred 为按靶点标准化尺度） ----------
scaler = json.load(open(SCALER, encoding='utf-8'))
g_std = float(scaler.get('global_std', 1.0))
tmeans = scaler.get('target_means', {})
g_mean = float(scaler.get('global_mean', 0.0))
def denorm(v, target):
    return float(v) * g_std + float(tmeans.get(target, tmeans.get(str(target), g_mean)))

# ---------- 逐案例归因 ----------
def atom_importance(mol, w_bits):
    """把 Morgan 位梯度映射到原子（bitInfo 半径2）"""
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bi)
    atom_w = {}
    for b, w in w_bits.items():
        if b in bi:
            for (at, rad) in bi[b]:
                atom_w[at] = max(atom_w.get(at, 0.0), abs(w) / (rad + 1))
    return atom_w

panels = []
for row_i, r in cases.iterrows():
    x_m = torch.tensor(morgan_sel[row_i], dtype=torch.float32, device=dev).unsqueeze(0).requires_grad_(True)
    x_k = torch.tensor(maccs_sel[row_i], dtype=torch.float32, device=dev).unsqueeze(0).requires_grad_(True)
    x_d = torch.tensor(desc_sel[row_i], dtype=torch.float32, device=dev).unsqueeze(0).requires_grad_(True)
    x_p = torch.tensor(prot_sel[row_i], dtype=torch.float32, device=dev).unsqueeze(0)
    preds_n, _, _ = model(x_m, x_k, x_d, x_p)
    pred_scaled = preds_n[0]
    g = torch.autograd.grad(pred_scaled, [x_m, x_k, x_d], retain_graph=False)
    gw_m = g[0].squeeze().cpu().numpy()
    gw_d = g[2].squeeze().cpu().numpy()
    with torch.no_grad():
        pred = denorm(pred_scaled.item(), r[TARGET_COL])
    # Morgan bit 重要性（仅取激活位）
    act_bits = np.where(morgan_sel[row_i] > 0)[0]
    w_bits = {int(b): float(gw_m[b]) for b in act_bits if abs(gw_m[b]) > 1e-9}
    top_bits = sorted(w_bits.items(), key=lambda kv: -abs(kv[1]))[:20]
    panels.append({'smiles': r[SMILES_COL], 'pred': pred, 'w_bits': dict(top_bits),
                   'desc': {DESC_NAMES[i]: float(gw_d[i]) for i in range(len(DESC_NAMES))},
                   'aff': float(r[AFF_COL]), 'target': str(r[TARGET_COL])})
    print(f"  case {r[TARGET_COL]}: pred={pred:.2f} aff={r[AFF_COL]:.2f} top_bits={len(top_bits)}")

# ---------- 绘图 ----------
fig, axes = plt.subplots(2, n_cases, figsize=(5.2 * n_cases, 7.6), dpi=300,
                         gridspec_kw={'height_ratios': [3, 1.2]})
if n_cases == 1:
    axes = axes.reshape(2, 1)
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('teal', ['#DAF2EE', '#2E9E8F', '#006D77'])
for j, p in enumerate(panels):
    mol = Chem.MolFromSmiles(p['smiles'])
    AllChem.Compute2DCoords(mol)
    aw = atom_importance(mol, p['w_bits'])
    mx = max(aw.values()) if aw else 1.0
    # 高亮原子（渐变teal）与键
    hit, cols = list(aw.keys()), []
    for a in hit:
        cols.append(cmap(norm(aw[a] / mx)))
    d2d = Draw.MolDraw2DCairo(700, 620)
    if hit:
        Draw.PrepareAndDrawMolecule(d2d, mol, highlightAtoms=hit,
                                    highlightAtomColors={a: c[:3] for a, c in zip(hit, cols)},
                                    highlightBondColors={})
    else:
        Draw.PrepareAndDrawMolecule(d2d, mol)
    d2d.FinishDrawing()
    import PIL.Image as PILImage
    import io
    img = PILImage.open(io.BytesIO(d2d.GetDrawingText()))
    axes[0, j].imshow(img); axes[0, j].axis('off')
    axes[0, j].set_title(f"{p['target'][-8:]}\npred={p['pred']:.2f}  (exp {p['aff']:.2f})", fontsize=10.5)
    # 描述符梯度条形图
    dd = pd.Series(p['desc']).sort_values(key=np.abs)
    axes[1, j].barh(range(len(dd)), dd.values,
                    color=['#006D77' if v > 0 else '#C0625B' for v in dd.values])
    axes[1, j].set_yticks(range(len(dd))); axes[1, j].set_yticklabels(dd.index, fontsize=7)
    axes[1, j].axvline(0, color='#888', lw=0.7)
    axes[1, j].tick_params(labelsize=7)
    axes[1, j].set_title('descriptor gradient (drug side)', fontsize=8.5)
fig.suptitle('Reptile+Transformer drug-side attribution (Morgan-bit gradient mapped to atoms)\n'
             'Note: cross-attention operates on pooled vectors (1x1); per-residue attribution is not identifiable in this architecture',
             fontsize=10, y=1.0)
fig.tight_layout()
for ext in ('png', 'tiff'):
    kw = {'dpi': 300, 'bbox_inches': 'tight', 'facecolor': 'white'}
    if ext == 'tiff':
        kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(OUT / f'FigureS4_drug_attribution.{ext}', **kw)
plt.close(fig)
print('Figure S4 已保存 →', OUT)
