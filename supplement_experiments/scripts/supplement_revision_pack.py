# -*- coding: utf-8 -*-
"""
论文返修补充材料 P0 打包：
  Table S2 跨簇比例            Table S3 阈值敏感性(τ=0.10/0.13/0.16)
  Table S4 EF@1% 完整矩阵      Table S5 EF@5% 完整矩阵
  Table S6 EF均值与平均排名    Table S7 分组均值(GNN vs 非GNN, 含排除Davis)
  Table S10 GNN归因汇总        Table S11 数据集划分统计(含逐split平均Tanimoto)
  Table S12 超参数与随机种子
  Figure S1 跨簇比例 vs EF@1%   Figure S2 Grad-CAM (GIN vs GATNet, 3分子)
  Figure S3 Edge occlusion 对比 Figure S7 模型选择决策树
  Figure S8 阈值扫描扩展图
输出：d:\\lht\\supplement_output\\paper_revision\\
"""
import sys, os, json, shutil, random
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = Path(r'd:\lht')
OUT = ROOT / 'supplement_output' / 'paper_revision'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
TEAL_D, TEAL_L, TEAL_BG = '#006D77', '#83C5BE', '#F2F8F7'

# ================================================================ 数据
DSS = ['chembl', 'davis', 'kiba', 'bindingdb']
DS_CN = {'chembl': 'ChEMBL', 'davis': 'Davis', 'kiba': 'KIBA', 'bindingdb': 'BindingDB'}
MODELS = ['Reptile+Transformer', 'Transformer', 'MLP',
          'GraphDTA GAT_GCN', 'GraphDTA GCNNet', 'GraphDTA GATNet', 'GraphDTA GINConnvNet']
SHORT = {'Reptile+Transformer': 'Reptile', 'Transformer': 'Transformer', 'MLP': 'MLP',
         'GraphDTA GAT_GCN': 'GAT_GCN', 'GraphDTA GCNNet': 'GCNNet',
         'GraphDTA GATNet': 'GATNet', 'GraphDTA GINConnvNet': 'GINConvNet'}
GNN = ['GraphDTA GAT_GCN', 'GraphDTA GCNNet', 'GraphDTA GATNet', 'GraphDTA GINConnvNet']
NON = ['Reptile+Transformer', 'Transformer', 'MLP']

long = pd.read_csv(ROOT / 'supplement_output' / 'ef7x4_long.csv')
long['dataset'] = long['dataset'].str.lower()
EF1 = long.pivot(index='model', columns='dataset', values='EF1')
EF5 = long.pivot(index='model', columns='dataset', values='EF5')
R2M = long.pivot(index='model', columns='dataset', values='R2')

CROSS = pd.read_csv(ROOT / 'supplement_output' / 'protein_cluster' / 'cross_cluster_summary.csv')
TANI = {'chembl': 0.1221, 'davis': 0.1367, 'kiba': 0.2500, 'bindingdb': 0.3600}

# ================================================================ Table S11 计算
def split_stats(ds, split):
    fp = ROOT / 'GrapthDTA' / 'data' / f'{ds}_{split}.csv'
    df = pd.read_csv(fp)
    n_drug = df['compound_iso_smiles'].nunique()
    n_prot = df['target_sequence'].nunique()
    return len(df), n_drug, n_prot


def mean_tanimoto(smiles_list, cap=30000, sample=20000, seed=42):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import BulkTanimotoSimilarity
    RDLogger.DisableLog('rdApp.*')
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    if len(fps) > cap:
        random.seed(seed)
        fps = random.sample(fps, sample)
    n = len(fps)
    if n < 2:
        return np.nan
    tot = 0.0
    for i in range(n):
        sims = BulkTanimotoSimilarity(fps[i], fps)
        tot += sum(sims[:i]) + sum(sims[i + 1:])
    return tot / (n * (n - 1))


S11_ROWS = []
print('计算 Table S11 (逐split Tanimoto) ...')
for ds in DSS:
    for split in ['train', 'val', 'test']:
        n, nd, np_ = split_stats(ds, split)
        uni = pd.read_csv(ROOT / 'GrapthDTA' / 'data' / f'{ds}_{split}.csv',
                          usecols=['compound_iso_smiles'])['compound_iso_smiles'].dropna().unique().tolist()
        if split == 'train':
            tan = TANI[ds]                # 训练组沿用 analyze_tanimoto.py 论文口径（与Fig3/Fig7一致）
            note = '（沿用analyze_tanimoto.py）'
        else:
            tan = mean_tanimoto(uni)
            note = ''
        S11_ROWS.append([DS_CN[ds], split, n, nd, np_, round(tan, 4), note])
        print(f'  {ds}-{split}: pairs={n} drugs={nd} targets={np_} tanimoto={tan:.4f}')
S11_DF = pd.DataFrame(S11_ROWS, columns=['Dataset', 'Split', 'Pairs', 'Unique drugs',
                                         'Unique targets', 'Mean Tanimoto (ECFP4)', '备注'])

# ================================================================ 表格构造
def ef_matrix(EF):
    m = EF.copy()
    m = m.loc[MODELS]
    m.columns = [DS_CN[c] for c in m.columns]
    m['Mean'] = m.mean(axis=1)
    return m.round(3)


def avg_rank(EF):
    ranks = EF.loc[MODELS, DSS].T.rank(ascending=False)   # 每数据集内 1=最优
    return ranks.mean(axis=0)[MODELS]


S6_DF = pd.DataFrame({
    'Model': [SHORT[m] for m in MODELS],
    'EF1_mean': EF1.loc[MODELS].mean(axis=1).values,
    'EF1_avgRank': avg_rank(EF1).values,
    'EF5_mean': EF5.loc[MODELS].mean(axis=1).values,
    'EF5_avgRank': avg_rank(EF5).values,
})
S6_DF['EF1-EF5'] = S6_DF['EF1_mean'] - S6_DF['EF5_mean']
for c in ['EF1_mean', 'EF5_mean', 'EF1-EF5']:
    S6_DF[c] = S6_DF[c].round(3)
S6_DF['EF1_avgRank'] = S6_DF['EF1_avgRank'].round(2)
S6_DF['EF5_avgRank'] = S6_DF['EF5_avgRank'].round(2)

rows = []
for tag, sel in [('全部4数据集', DSS), ('排除Davis', [d for d in DSS if d != 'davis'])]:
    g1 = EF1.loc[GNN, sel].mean(axis=1).mean()
    n1 = EF1.loc[NON, sel].mean(axis=1).mean()
    g5 = EF5.loc[GNN, sel].mean(axis=1).mean()
    n5 = EF5.loc[NON, sel].mean(axis=1).mean()
    rows.append([tag, round(g1, 3), round(n1, 3), round(n1 - g1, 3),
                 round(g5, 3), round(n5, 3), round(n5 - g5, 3)])
S7_A = pd.DataFrame(rows, columns=['范围', 'GNN EF1均值', '非GNN EF1均值', 'ΔEF1',
                                   'GNN EF5均值', '非GNN EF5均值', 'ΔEF5'])
rows = []
for ds in DSS:
    g1 = EF1.loc[GNN, ds].mean(); n1 = EF1.loc[NON, ds].mean()
    g5 = EF5.loc[GNN, ds].mean(); n5 = EF5.loc[NON, ds].mean()
    rows.append([DS_CN[ds], round(g1, 3), round(n1, 3), round(n1 - g1, 3),
                 round(g5, 3), round(n5, 3), round(n5 - g5, 3)])
S7_B = pd.DataFrame(rows, columns=['数据集', 'GNN EF1均值', '非GNN EF1均值', 'ΔEF1',
                                   'GNN EF5均值', '非GNN EF5均值', 'ΔEF5'])

# S2
rows = []
for ds in DSS:
    sub = CROSS[CROSS['dataset'] == ds].sort_values('identity_threshold_pct')
    r = [DS_CN[ds], int(sub['n_test_targets'].iloc[0])]
    for _, rr in sub.iterrows():
        r += [int(rr['n_test_targets_same_cluster']), int(rr['n_test_targets_cross_cluster']),
              float(rr['cross_cluster_ratio_pct'])]
    rows.append(r)
S2_DF = pd.DataFrame(rows, columns=['数据集', '测试靶点数',
                                    '同簇数(40%)', '跨簇数(40%)', '跨簇比例%(40%)',
                                    '同簇数(60%)', '跨簇数(60%)', '跨簇比例%(60%)',
                                    '同簇数(80%)', '跨簇数(80%)', '跨簇比例%(80%)'])

# S3
best_ef1 = {d: max(MODELS, key=lambda m: EF1.loc[m, d]) for d in DSS}
best_r2 = {d: max(MODELS, key=lambda m: R2M.loc[m, d]) for d in DSS}


def s3_block(tau):
    lines = []
    for regime, sel in ((f'低相似度 (<{tau:.2f})', [d for d in DSS if TANI[d] < tau]),
                        (f'高相似度 (>={tau:.2f})', [d for d in DSS if TANI[d] >= tau])):
        if not sel:
            lines.append([f'{tau:.2f}', regime, '无', '—', '—', '—'])
            continue
        r2t = '；'.join(f'{DS_CN[d]}→{SHORT[best_r2[d]]}({R2M.loc[best_r2[d], d]:.3f})' for d in sel)
        eft = '；'.join(f'{DS_CN[d]}→{SHORT[best_ef1[d]]}({EF1.loc[best_ef1[d], d]:.2f})' for d in sel)
        ag = [d for d in sel if best_r2[d] == best_ef1[d]]
        agt = f'{len(ag)}/{len(sel)}一致' + (f'（{"、".join(DS_CN[d] for d in ag)}）' if ag else '（无）')
        lines.append([f'{tau:.2f}', regime,
                      '、'.join(f'{DS_CN[d]}({TANI[d]:.3f})' for d in sel), r2t, eft, agt])
    return lines


S3_ROWS = []
for tau in (0.10, 0.13, 0.16):
    S3_ROWS += s3_block(tau)
S3_DF = pd.DataFrame(S3_ROWS, columns=['τ', '区间', '落入数据集(平均Tanimoto)',
                                       'R²准则推荐', 'EF@1%准则推荐', '两准则一致性'])

# S10
att_rows = []
for ds in ['davis', 'kiba']:
    j = json.load(open(ROOT / 'supplement_output' / 'gnn_explain' / ds / 'attribution_summary.json',
                       encoding='utf-8'))
    for pm in j['per_model']:
        att_rows.append([DS_CN[ds], SHORT_G := pm['model'], round(pm['cam_ring_fraction'], 3),
                         round(pm['cam_heteroatom_fraction'], 3),
                         round(pm['edge_negative_fraction'], 3),
                         round(pm['gradcam_edge_spearman'], 3)])
S10_DF = pd.DataFrame(att_rows, columns=['Dataset', 'Model', 'CAM环原子占比', 'CAM杂原子占比',
                                         '负贡献边占比*', 'GradCAM-Edge Spearman'])

# S12
S12_DF = pd.DataFrame([
    ['GNN (GraphDTA×4)', 'GrapthDTA/training.py', 'Adam', 5e-4, 1024, '≤1000(早停patience=50)',
     'ReduceLROnPlateau(0.5, patience=20)', '未固定随机种子'],
    ['Reptile+Transformer(主模型)', 'train_best_ems2_can_revise.py', 'Adam(分组)', 0.001,
     '(按配置)', '(按配置)', 'ESM2微调层lr=0.1×主lr; weight_decay=1e-4', '未固定随机种子'],
    ['Reptile内循环', 'run_reptile_transformer.py / reptile_training.py', '—', '—', 512, 200,
     'inner_steps=3; 梯度累积=1', '未固定随机种子'],
    ['λ 权重(Reptile模型)', 'reptile_training.py:96-97', '—', '—', '—', '—',
     'λ1 contrastive=0.05; λ2 consistency=0.1', '—'],
    ['Transformer基线', 'run_transformer_baseline.py:284-298', 'AdamW', 1.5e-4, '(默认)', '(默认)',
     'λ1 contrastive=0.03; λ2 consistency=0.05; λ3 ranking=0.5; λ4 calibration=0.1; warmup后启用', '未固定随机种子'],
    ['MLP基线', 'run_baseline_mlp.py:340-346', 'AdamW', 0.001, '(默认)', '(默认)',
     'weight_decay=1e-4; 余弦退火', '未固定随机种子'],
], columns=['模型', '脚本', '优化器', '学习率', 'Batch', 'Epochs', '关键配置', '随机种子'])

# ================================================================ 写Excel
wb = Workbook()
thin = Side(style='thin', color='B0BEC5')
BD = Border(left=thin, right=thin, top=thin, bottom=thin)
HF = PatternFill('solid', fgColor='006D77')
SF = PatternFill('solid', fgColor='E3F0EE')
CT = Alignment(horizontal='center', vertical='center', wrap_text=True)
LT = Alignment(horizontal='left', vertical='center', wrap_text=True)


def put(ws, df, widths, note_rows=()):
    ws.append(list(df.columns))
    for _, r in df.iterrows():
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else v for v in r])
    for j, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=len(df.columns)):
        for c in row:
            c.border = BD
            if c.row == 1:
                c.fill = HF; c.font = Font(name='Arial', bold=True, color='FFFFFF', size=10.5); c.alignment = CT
            else:
                c.font = Font(name='Arial', size=10.5); c.alignment = CT
    r0 = ws.max_row + 1
    for k, n in enumerate(note_rows):
        ws.merge_cells(start_row=r0 + k, start_column=1, end_row=r0 + k, end_column=len(df.columns))
        cc = ws.cell(r0 + k, 1, n)
        cc.font = Font(name='Arial', size=9, italic=True, color='555555'); cc.alignment = LT
    return ws


ws = wb.active; ws.title = 'S2_cross_cluster'
put(ws, S2_DF, [12] + [11] * 10,
    ['注：MMseqs2 easy-cluster，覆盖度≥0.8；跨簇比例 = 测试靶点中与训练集全部靶点不同簇的比例。'
     '跨簇比例随一致性阈值升高而升高（80%时KIBA达100%），支持靶点冷启动的远同源外推设定。'])
ws = wb.create_sheet('S3_threshold_tau')
put(ws, S3_DF, [8, 16, 34, 44, 44, 16],
    ['注：τ 为训练集药物平均ECFP4 Tanimoto（ChEMBL 0.122/Davis 0.137/KIBA 0.250/BindingDB 0.360）。'
     '各数据集R²最优与EF@1%最优仅在KIBA一致（Reptile+Transformer）；低相似度场景拟合优者（Reptile/GAT_GCN）富集反而弱。',
     '注：Davis每靶点仅66化合物，GNN的EF@1%小样本失真（GINConvNet 4.34名义最高），实际筛选推荐MLP/Transformer(4.20)。'])
ws = wb.create_sheet('S4_EF1_full')
put(ws, ef_matrix(EF1).reset_index().rename(columns={'model': 'Model'}), [20] + [10] * 5,
    ['注：EF@1%（Enrichment Factor，逐靶点计算后平均），加粗列均值；Davis的GNN值受每靶点66化合物小样本影响。'])
ws = wb.create_sheet('S5_EF5_full')
put(ws, ef_matrix(EF5).reset_index().rename(columns={'model': 'Model'}), [20] + [10] * 5,
    ['注：EF@5%，逐靶点计算后平均。'])
ws = wb.create_sheet('S6_mean_rank')
put(ws, S6_DF, [16, 11, 12, 11, 12, 10],
    ['注：avgRank 为该模型在4个数据集内的排名（1=最优）取平均，越小越好；EF1−EF5 反映头部与头5%排序落差。'])
ws = wb.create_sheet('S7_group_mean')
put(ws, S7_A, [14, 12, 13, 9, 12, 13, 9])
r = ws.max_row + 2
ws.cell(r, 1, '—— 逐数据集 ——').font = Font(bold=True, size=10.5)
for j, col in enumerate(S7_B.columns, 1):
    ws.cell(r + 1, j, col).fill = SF
    ws.cell(r + 1, j).font = Font(bold=True, size=10.5); ws.cell(r + 1, j).alignment = CT
for i, (_, rr) in enumerate(S7_B.iterrows()):
    for j, v in enumerate(rr, 1):
        c = ws.cell(r + 2 + i, j, v); c.font = Font(size=10.5); c.alignment = CT; c.border = BD
ws.merge_cells(start_row=r + 7, start_column=1, end_row=r + 7, end_column=7)
ws.cell(r + 7, 1, '注：Δ=非GNN−GNN；正值表示非GNN占优。排除Davis后GNN组EF1均值升至3.19（Davis小样本拖累明显），'
                  '但仍低于非GNN的3.60；唯一反超点为KIBA(ΔEF1=−0.08)。').font = Font(size=9, italic=True, color='555555')
ws = wb.create_sheet('S10_gnn_attribution')
put(ws, S10_DF, [11, 13, 14, 14, 14, 20],
    ['注：6个低相似度样本（蛋白一致性0%跨簇靶点×逐靶点top20%活性）的均值；CAM环原子/杂原子占比为Grad-CAM权重落入环/杂原子的比例。',
     '*负贡献边占比：负值表示遮挡该键后预测平均升高（即该键对预测为负贡献）的边所占比例；Davis上GATNet(−0.77)反映其小药物面板下的预测噪声。'])
ws = wb.create_sheet('S11_dataset_stats')
put(ws, S11_DF, [11, 8, 10, 12, 14, 20, 24],
    ['注：数据划分采用靶点冷启动（train/test 蛋白序列零重叠）；Tanimoto 为该split唯一药物的ECFP4平均成对相似度'
     '（>30000分子时按seed=42抽样20000计算）。train组沿用 analyze_tanimoto.py 论文口径（与Fig3/Fig7阈值分析一致），'
     'val/test 由 GrapthDTA/data 口径现算，两组分子集口径不完全相同。'])
ws = wb.create_sheet('S12_hyperparams')
put(ws, S12_DF, [24, 34, 10, 9, 9, 18, 42, 14],
    ['注：全项目训练脚本未固定随机种子（torch/np/random 均未设置），DeepDTA 复现基线除外；'
     '“(默认)”表示脚本采用函数默认值。'])

xlsx = OUT / 'Supplementary_Tables_P0.xlsx'
wb.save(xlsx)
print('Excel →', xlsx)

# ================================================================ Figure S1
for ext in ('png', 'tiff'):
    src = ROOT / 'supplement_output' / f'fig1_crosscluster_vs_ef1.{ext}'
    shutil.copy(src, FIG / f'FigureS1_crosscluster_vs_EF1.{ext}')
print('Figure S1 已复制')

# ================================================================ Figure S2 / S3 拼图
MAPS = ROOT / 'supplement_output' / 'gnn_explain'
GM = ['GINConvNet', 'GATNet', 'GAT_GCN', 'GCNNet']
SHORT_M = lambda m: m
S2_SEL = [('kiba', 'kiba_sample3_protID0_drugSim1.00_aff14.2'),
          ('kiba', 'kiba_sample1_protID0_drugSim1.00_aff13.9'),
          ('davis', 'davis_sample1_protID0_drugSim1.00_aff8.9')]
S3_SEL = [('davis', 'davis_sample1_protID0_drugSim1.00_aff8.9'),
          ('kiba', 'kiba_sample3_protID0_drugSim1.00_aff14.2')]


def grid_figure(items, ncols, figsize, title_fn, out_stem, row_label_fn=None):
    import matplotlib.image as mpimg
    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=300)
    axes = np.atleast_2d(axes)
    for k, it in enumerate(items):
        ax = axes[k // ncols, k % ncols]
        img = mpimg.imread(str(it['path']))
        ax.imshow(img); ax.axis('off')
        ax.set_title(title_fn(it), fontsize=10)
        if row_label_fn and k % ncols == 0:
            ax.text(-0.08, 0.5, row_label_fn(it), transform=ax.transAxes,
                    rotation=90, va='center', ha='right', fontsize=10, fontweight='bold')
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')
    fig.patch.set_facecolor('white')
    for ext in ('png', 'tiff'):
        kw = {'dpi': 300, 'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == 'tiff':
            kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(FIG / f'{out_stem}.{ext}', **kw)
    plt.close(fig)


items2 = [{'ds': ds, 'tag': tag, 'model': m, 'path': MAPS / ds / 'maps' / f'{tag}__{m}__gradcam.png'}
          for ds, tag in S2_SEL for m in ['GINConvNet', 'GATNet']]
grid_figure(items2, 2, (7.6, 10.5),
            lambda it: f"{SHORT_M(it['model'])}   ({DS_CN[it['ds']]})",
            'FigureS2_gradcam_GIN_vs_GATNet',
            row_label_fn=lambda it: f"{it['tag'].split('_prot')[0]} aff={it['tag'].split('_aff')[1]}")
print('Figure S2 已保存')

items3 = [{'ds': ds, 'tag': tag, 'model': m, 'path': MAPS / ds / 'maps' / f'{tag}__{m}__{kind}.png'}
          for ds, tag in S3_SEL for kind in ['gradcam', 'edge'] for m in GM]
grid_figure(items3, 4, (15, 12.5),
            lambda it: f"{SHORT_M(it['model'])}  {'Grad-CAM' if it['path'].stem.endswith('gradcam') else 'Edge occlusion'}",
            'FigureS3_edge_vs_gradcam',
            row_label_fn=lambda it: f"{it['tag'].split('_prot')[0]} aff={it['tag'].split('_aff')[1]}\n"
                                    f"{'Grad-CAM' if it['path'].stem.endswith('gradcam') else 'Edge occl.'}")
print('Figure S3 已保存')

# ================================================================ Figure S7 决策树
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'Arial']
fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=300)
ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(-9, 62)


def box(x, y, w, h, text, fc=TEAL_BG, ec=TEAL_D, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6',
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color='#12333A', linespacing=1.5)


def arrow(x1, y1, x2, y2, label='', lx=0, ly=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=13, color='#4F7C82', lw=1.2))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=8.5,
                color='#2F5D63', ha='center', style='italic')


box(32, 52, 36, 8, '为新靶点面板选择 DTA 模型', fc=TEAL_D, ec=TEAL_D, fs=12, bold=True)
ax.texts[-1].set_color('white')
box(2, 34, 28, 11, '优先概率校准\n(可靠的不确定度)\n→ Reptile+Transformer\nECE 0.005–0.026，EF≈2 保守', fs=8.8)
box(36, 34, 28, 11, '优先虚拟筛选富集 EF@1%\n→ 按药物空间相似度分流', fs=9.5, bold=True)
box(70, 34, 28, 11, '需要子结构可解释性\n/ 仅限图模型\n→ GINConvNet\n(GNN 中 EF 最稳: 3.45/4.37)', fs=8.8)
box(24, 12, 24, 12, '低相似度场景\ntrain平均Tanimoto<0.13\n(ChEMBL 0.122 / Davis 0.137)\n→ MLP / Transformer\nEF1=4.63 / 4.44 (ChEMBL)', fs=8.5)
box(52, 12, 24, 12, '高相似度场景\nTanimoto≥0.13\n(KIBA 0.250 / BindingDB 0.360)\n→ Reptile+Transformer(KIBA双优) /\nTransformer(BDB EF1=4.68)', fs=8.5)
box(36, 0, 28, 7, '参考基线：随机富集 EF=1；\n所有模型 EF@1% > EF@5%（头部排序更好）', fc='white', ec='#9AA5A8', fs=8.3)
arrow(50, 52, 50, 45.2)
arrow(44, 45, 16, 45.2); arrow(56, 45, 84, 45.2)
arrow(50, 34, 36.5, 24.4, '低', lx=-2, ly=2)
arrow(50, 34, 64, 24.4, '高', lx=2, ly=2)
ax.text(50, -6.5, '依据：Table S2–S7（4数据集×7模型；τ=0.10/0.13/0.16 敏感性一致）',
        ha='center', fontsize=9, color='#456', style='italic')
for ext in ('png', 'tiff'):
    kw = {'dpi': 300, 'bbox_inches': 'tight', 'facecolor': 'white'}
    if ext == 'tiff':
        kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(FIG / f'FigureS7_model_selection_tree.{ext}', **kw)
plt.close(fig)
print('Figure S7 已保存')

# ================================================================ Figure S8 阈值扫描
taus = np.arange(0.08, 0.401, 0.01)
win = {(d, c): [] for d in DSS for c in ('R2', 'EF1')}
agree = []
for t in taus:
    ag = 0
    for d in DSS:
        wr = max(MODELS, key=lambda m: R2M.loc[m, d])
        wf = max(MODELS, key=lambda m: EF1.loc[m, d])
        win[(d, 'R2')].append(SHORT[wr]); win[(d, 'EF1')].append(SHORT[wf])
        ag += (wr == wf)
    agree.append(ag / 4)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=300,
                         gridspec_kw={'width_ratios': [1, 1.9]})
axa, axb = axes
axa.plot(taus, agree, color=TEAL_D, lw=2, marker='o', ms=4.5, mfc=TEAL_L, mec=TEAL_D)
for t in (0.10, 0.13, 0.16):
    axa.axvline(t, color='#C82423', ls=':', lw=1.1)
axa.set_xlabel('Tanimoto threshold τ'); axa.set_ylabel('Agreement fraction')
axa.set_title('(a) R²-winner vs EF@1%-winner agreement', fontsize=10.5)
axa.set_ylim(-0.05, 1.05); axa.grid(color='#E4ECEB', ls='--', lw=0.6)

MCOLOR = {'Reptile': '#1B5E20', 'Transformer': '#4C8C8C', 'MLP': '#76B5AD',
          'GAT_GCN': '#8FBF9F', 'GCNNet': '#5F9EA0', 'GATNet': '#B2DFDB', 'GINConvNet': '#2E7D6B'}
ytick_lab, yy = [], 0
for d in DSS:
    for crit in ('R2', 'EF1'):
        seq = win[(d, crit)]
        for i in range(len(taus) - 1):
            axb.axhspan(yy - 0.33, yy + 0.33, xmin=i / (len(taus) - 1),
                        xmax=(i + 1) / (len(taus) - 1), color=MCOLOR[seq[i]], alpha=0.85)
        # 恒定段内居中标注（段长≥4个τ点才标注，避免拥挤）
        seg_start = 0
        for i in range(1, len(seq) + 1):
            if i == len(seq) or seq[i] != seq[seg_start]:
                if i - seg_start >= 4:
                    cx = (taus[seg_start] + taus[i - 1]) / 2
                    axb.text(cx, yy, seq[seg_start], fontsize=7.5, ha='center', va='center',
                             color='white', fontweight='bold')
                seg_start = i
        ytick_lab.append(f'{DS_CN[d]}·{"R²" if crit == "R2" else "EF@1%"}')
        yy -= 1
for t in (0.10, 0.13, 0.16):
    axb.axvline(t, color='#C82423', ls=':', lw=1.1)
axb.set_yticks(range(0, -8, -1)); axb.set_yticklabels(ytick_lab, fontsize=8.5)
axb.set_xlabel('Tanimoto threshold τ'); axb.set_title('(b) Winning model vs τ', fontsize=10.5)
axb.set_xlim(taus[0], taus[-1]); axb.set_ylim(-7.8, 0.8)
axb.spines[['top', 'right', 'left']].set_visible(False)
axb.tick_params(left=False)
handles = [plt.Line2D([], [], marker='s', ls='', ms=9, color=c, label=m) for m, c in MCOLOR.items()]
axb.legend(handles=handles, ncol=4, fontsize=7.5, loc='upper left',
           bbox_to_anchor=(-0.02, -0.16), frameon=False)
fig.tight_layout()
for ext in ('png', 'tiff'):
    kw = {'dpi': 300, 'bbox_inches': 'tight', 'facecolor': 'white'}
    if ext == 'tiff':
        kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(FIG / f'FigureS8_threshold_sweep.{ext}', **kw)
plt.close(fig)
print('Figure S8 已保存')
print('\n全部 P0 产物完成 →', OUT)
