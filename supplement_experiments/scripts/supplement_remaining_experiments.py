# -*- coding: utf-8 -*-
"""
补做实验合集（无需训练）
============================================
2  McNemar / Fisher 精确检验 — R²-winner vs EF@1%-winner 一致性
   （winner 身份以论文 Table S3 为唯一权威，不再从数值矩阵重算）
3  Target mirroring 检测 / binding profile 相似性审计（4 数据集，
   按 same_cluster_40 分层 + Fisher 精确检验 + 散点图）
6  理论框架 conceptual figure（横轴 τ / 纵轴 ΔR²，三区标注，
   ΔR² 取 Figure 6 权威值，τ 取训练集口径）
8  配对检验 p 值矩阵热力图（matplotlib 实现，无 seaborn 依赖）
9  Table 4 / S4 / S5 补 ±std（E5 5 seeds）
14 Family-remote 二分类验证（判定指标、阈值、误分类率）

输出目录: d:\lht\修改supplement_output\remaining_experiments\
"""
import json
import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r'd:\lht')
OUT = BASE / '修改supplement_output' / 'remaining_experiments'
OUT.mkdir(parents=True, exist_ok=True)
GDTA = BASE / 'GrapthDTA' / 'data'
CLUSTER = BASE / '修改supplement_output' / 'protein_cluster'
E5_DIR = BASE / '修改supplement_output' / 'revision2_experiments' / 'e5_repeats'
E5_CSV = E5_DIR / 'e5_repeats_metrics.csv'

R2 = 'R\u00b2'
DELTA = '\u0394'

# ========== 论文权威数据（Table S3 / Figure 6） ==========
# τ：训练集药物平均 ECFP4 Tanimoto（Table S3 注）
TAU = {'ChEMBL': 0.122, 'Davis': 0.137, 'KIBA': 0.250, 'BindingDB': 0.360}

# Table S3 权威 winner（论文原文照录）
R2_WINNER = {'ChEMBL': ('Reptile+Transformer', 0.202),
             'Davis': ('GAT_GCN', 0.400),
             'KIBA': ('Reptile+Transformer', 0.322),
             'BindingDB': ('Reptile+Transformer', 0.234)}
EF1_WINNER = {'ChEMBL': ('MLP', 4.63),
              'Davis': ('GINConvNet', 4.34),
              'KIBA': ('Reptile+Transformer', 4.72),
              'BindingDB': ('Transformer', 4.68)}

# Figure 6 (plot_gnn_vs_nongnn.py DATASETS_INFO) 权威 ΔR² = best GNN − best non-GNN
DELTA_R2 = {'ChEMBL': -0.2041, 'Davis': 0.1412, 'KIBA': -0.0989, 'BindingDB': -0.1340}

# S4 EF@1% 完整矩阵（xlsx 照录，供 ±std 对照）
EF1_MATRIX = {
    'Reptile+Transformer': {'ChEMBL': 1.98, 'Davis': 1.75, 'KIBA': 4.72, 'BindingDB': 1.85},
    'Transformer':         {'ChEMBL': 4.44, 'Davis': 4.20, 'KIBA': 4.05, 'BindingDB': 4.68},
    'MLP':                 {'ChEMBL': 4.63, 'Davis': 4.20, 'KIBA': 3.75, 'BindingDB': 3.89},
    'GAT_GCN':             {'ChEMBL': 2.11, 'Davis': 0.428, 'KIBA': 4.49, 'BindingDB': 3.02},
    'GCNNet':              {'ChEMBL': 2.41, 'Davis': 0.432, 'KIBA': 4.28, 'BindingDB': 3.37},
    'GATNet':              {'ChEMBL': 1.59, 'Davis': 0.469, 'KIBA': 3.88, 'BindingDB': 2.46},
    'GINConvNet':          {'ChEMBL': 2.84, 'Davis': 4.34, 'KIBA': 4.37, 'BindingDB': 3.45},
}

# Table 4（KIBA 消融表，论文原值）
TABLE4 = {
    'MLP (full features)':                {'R2': 0.2565, 'RMSE': 1.0140, 'AUPR': 0.4821, 'ECE': 0.0227},
    'Transformer (MSE-only)':             {'R2': 0.1800, 'RMSE': 1.0649, 'AUPR': 0.5237, 'ECE': 0.3844},
    'Transformer (multi-objective loss)': {'R2': 0.2308, 'RMSE': 0.7418, 'AUPR': 0.5000, 'ECE': 0.1581},
    'Reptile-Transformer (full pipeline)': {'R2': 0.3222, 'RMSE': 0.6963, 'AUPR': 0.6635, 'ECE': 0.0045},
}

# 蓝绿色系（用户偏好）
C_BLUE = '#0277BD'
C_BLUE_L = '#4FC3F7'
C_BLUE_SOFT = '#BBDEFB'
C_TEAL = '#00897B'
C_TEAL_SOFT = '#B2DFDB'
C_GREEN = '#1B5E20'
C_GREEN_L = '#66BB6A'
C_GREEN_SOFT = '#C8E6C9'
C_GRAY = '#78909C'


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


# ============================================================
# 实验 2：McNemar / Fisher 精确检验（winner 以 Table S3 为准）
# ============================================================
def exp2_mcnemar_fisher():
    datasets = ['ChEMBL', 'Davis', 'KIBA', 'BindingDB']
    models = list(EF1_MATRIX.keys())
    r2_winner = {ds: R2_WINNER[ds][0] for ds in datasets}
    ef_winner = {ds: EF1_WINNER[ds][0] for ds in datasets}

    # 聚合 2×2 列联表（28 观察 = 4 数据集 × 7 模型）
    a = b = c = d = 0
    for ds in datasets:
        for m in models:
            is_r2 = (m == r2_winner[ds])
            is_ef = (m == ef_winner[ds])
            if is_r2 and is_ef:
                a += 1
            elif is_r2 and not is_ef:
                b += 1
            elif not is_r2 and is_ef:
                c += 1
            else:
                d += 1

    table = np.array([[a, b], [c, d]])
    _, p_fisher = stats.fisher_exact(table, alternative='two-sided')
    _, p_fisher_greater = stats.fisher_exact(table, alternative='greater')

    # 二项检验：winner 一致次数 vs 随机期望 1/7
    p_match_null = 1.0 / len(models)
    n_agree = sum(1 for ds in datasets if r2_winner[ds] == ef_winner[ds])
    p_binom_two = stats.binomtest(n_agree, n=len(datasets), p=p_match_null, alternative='two-sided').pvalue
    # 单侧：H1 = 一致性显著高于随机（若解耦为真，则应不显著）
    p_binom_greater = stats.binomtest(n_agree, n=len(datasets), p=p_match_null, alternative='greater').pvalue

    # 家族层面（GNN vs non-GNN）
    families = {'Reptile+Transformer': 'non-GNN', 'Transformer': 'non-GNN', 'MLP': 'non-GNN',
                'GAT_GCN': 'GNN', 'GCNNet': 'GNN', 'GATNet': 'GNN', 'GINConvNet': 'GNN'}
    n_agree_family = sum(1 for ds in datasets if families[r2_winner[ds]] == families[ef_winner[ds]])
    p_binom_family = stats.binomtest(n_agree_family, n=len(datasets), p=0.5, alternative='two-sided').pvalue

    # McNemar（以 Reptile+Transformer 为焦点模型）
    y_r2 = [1 if r2_winner[ds] == 'Reptile+Transformer' else 0 for ds in datasets]
    y_ef = [1 if ef_winner[ds] == 'Reptile+Transformer' else 0 for ds in datasets]
    n11 = sum(1 for i in range(4) if y_r2[i] == 1 and y_ef[i] == 1)
    n10 = sum(1 for i in range(4) if y_r2[i] == 1 and y_ef[i] == 0)
    n01 = sum(1 for i in range(4) if y_r2[i] == 0 and y_ef[i] == 1)
    n00 = sum(1 for i in range(4) if y_r2[i] == 0 and y_ef[i] == 0)
    if n10 + n01 > 0:
        # 精确二项 McNemar（小样本推荐）
        p_mcnemar = stats.binomtest(min(n10, n01), n=n10 + n01, p=0.5, alternative='two-sided').pvalue
        mcnemar_chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    else:
        p_mcnemar = 1.0
        mcnemar_chi2 = 0.0

    # 置换检验（对 winner 身份向量做置换，估计一致性分布）
    rng = np.random.default_rng(42)
    n_perm = 100000
    r2_vec = np.array([r2_winner[ds] for ds in datasets])
    ef_vec = np.array([ef_winner[ds] for ds in datasets])
    obs = int((r2_vec == ef_vec).sum())
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(ef_vec)
        if int((r2_vec == perm).sum()) >= obs:
            cnt += 1
    p_perm = (cnt + 1) / (n_perm + 1)

    report = (
        "实验 2：R²-winner vs EF@1%-winner 一致性统计检验\n"
        "数据源：论文 Table S3（winner 身份照录，非重算）\n"
        "=" * 70 + "\n"
        f"{'数据集':<10}\t{'R²-winner':<24}\t{'EF@1%-winner':<24}\t一致?\n"
    )
    for ds in datasets:
        match = 'Y' if r2_winner[ds] == ef_winner[ds] else 'N'
        report += (f"{ds:<10}\t{r2_winner[ds]:<16}({R2_WINNER[ds][1]:.3f})\t"
                   f"{ef_winner[ds]:<16}({EF1_WINNER[ds][1]:.2f})\t{match}\n")

    report += (
        f"\n一致性: {n_agree}/4（仅 KIBA）\n"
        f"\n[检验 1] 聚合 2×2 列联表（28 观察 = 4 数据集 × 7 模型）：\n"
        f"              EF-winner=True  EF-winner=False\n"
        f"R²-winner=True       {a}              {b}\n"
        f"R²-winner=False      {c}             {d}\n"
        f"Fisher 精确检验（双侧）:            p = {p_fisher:.4g}\n"
        f"Fisher 精确检验（单侧,关联>期望）:  p = {p_fisher_greater:.4g}\n"
        f"\n[检验 2] 二项检验（H0: P(一致)=1/7={p_match_null:.3f}，7 模型均匀随机）:\n"
        f"  一致次数 = {n_agree}/4\n"
        f"  双侧 p = {p_binom_two:.4g}\n"
        f"  单侧(>随机) p = {p_binom_greater:.4g}\n"
        f"\n[检验 3] 家族层面二项检验（H0: P(GNN/non-GNN 家族一致)=0.5）:\n"
        f"  家族一致次数 = {n_agree_family}/4  双侧 p = {p_binom_family:.4g}\n"
        f"\n[检验 4] McNemar 精确检验（焦点模型 Reptile+Transformer）：\n"
        f"  配对 2×2: (R2=1,EF=1)={n11}, (1,0)={n10}, (0,1)={n01}, (0,0)={n00}\n"
        f"  chi2(校正) = {mcnemar_chi2:.3f}, 精确二项 p = {p_mcnemar:.4g}\n"
        f"\n[检验 5] 置换检验（10^5 次置换 EF-winner 向量）：\n"
        f"  观察一致 = {obs}/4, 置换 p = {p_perm:.4g}\n"
        f"\n结论：\n"
    )
    report += (
        f"  (1) 全部五种检验均不显著（最小 p = {min(p_fisher, p_binom_two, p_binom_family, p_mcnemar, p_perm):.3g}）。\n"
        f"  (2) 观察一致性 1/4 与随机期望（1/7≈0.57/4）在 N=4 下无法区分——\n"
        f"      这既不支持'两准则协同'，也不支持'显著解耦'；winner 层面的解耦\n"
        f"      证据应锚定于 Table 2 的逐靶点相关性分析（24 检验、Bonferroni\n"
        f"      校正后 Davis 三项存活），而非 winner 一致性计数。\n"
        f"  (3) 论文中应显式声明统计力限制：4 个数据集最多提供 4 个独立观察，\n"
        f"      任何 winner 层面的检验功效均不足（1-β << 0.8）。\n"
    )

    (OUT / 'exp2_mcnemar_fisher_report.txt').write_text(report, encoding='utf-8')
    print(report)
    return report


# ============================================================
# 实验 3：Target mirroring 检测 / binding profile 相似性审计（4 数据集）
# ============================================================
def _per_target_aff_map(df):
    """{target_seq: pd.Series(index=smiles, values=affinity)}"""
    out = {}
    for seq, g in df.groupby('target_sequence'):
        s = g.drop_duplicates('compound_iso_smiles').set_index('compound_iso_smiles')['affinity']
        out[seq] = s.astype(float)
    return out


def exp3_target_mirroring():
    ds_list = ['chembl', 'davis', 'kiba', 'bindingdb']
    all_rows = []
    for ds in ds_list:
        train_fp = GDTA / f'{ds}_train.csv'
        test_fp = GDTA / f'{ds}_test.csv'
        cl_fp = CLUSTER / ds / 'per_test_target_identity.csv'
        if not (train_fp.exists() and test_fp.exists() and cl_fp.exists()):
            print(f'[exp3] {ds} 缺文件，跳过')
            continue
        train_map = _per_target_aff_map(pd.read_csv(train_fp))
        test_map = _per_target_aff_map(pd.read_csv(test_fp))
        test_seqs = sorted(test_map.keys())          # test_i 映射规则：排序后第 i 个
        cl = pd.read_csv(cl_fp)
        assert len(cl) == len(test_seqs), f'{ds}: 聚类表 {len(cl)} != 测试靶点 {len(test_seqs)}'

        train_seqs = list(train_map.keys())
        for i, tseq in enumerate(test_seqs):
            t_ser = test_map[tseq]
            best_rho, best_shared, best_j = np.nan, 0, -1
            for j, trseq in enumerate(train_seqs):
                tr_ser = train_map[trseq]
                shared = t_ser.index.intersection(tr_ser.index)
                if len(shared) < 5:
                    continue
                v1 = t_ser.loc[shared].values
                v2 = tr_ser.loc[shared].values
                if np.std(v1) == 0 or np.std(v2) == 0:
                    continue
                rho = stats.spearmanr(v1, v2).statistic
                if not np.isnan(rho) and (np.isnan(best_rho) or rho > best_rho):
                    best_rho, best_shared, best_j = rho, len(shared), j
            all_rows.append({
                'dataset': ds,
                'target_id': cl.iloc[i]['target_id'],
                'max_bp_spearman': best_rho,
                'n_shared_drugs': best_shared,
                'nearest_train_identity_pct': cl.iloc[i]['nearest_train_identity_pct'],
                'same_cluster_40': bool(cl.iloc[i]['same_cluster_40']),
            })
        print(f'[exp3] {ds}: {len(test_seqs)} 个测试靶点完成')

    bp = pd.DataFrame(all_rows)
    bp.to_csv(OUT / 'exp3_binding_profile_similarity.csv', index=False, encoding='utf-8-sig')

    # ---- 分层统计 + Fisher ----
    report = (
        "实验 3：Target Mirroring 检测 / Binding Profile 相似性审计\n"
        "数据源：GrapthDTA data/<ds>_{train,test}.csv；"
        "序列同源性：MMseqs2 per_test_target_identity.csv\n"
        "定义：max_bp_spearman = 测试靶点与全体训练靶点间最大 binding-profile Spearman ρ\n"
        + "=" * 70 + "\n"
    )
    fisher_rows = []
    for ds in ds_list:
        sub = bp[bp['dataset'] == ds].dropna(subset=['max_bp_spearman'])
        if sub.empty:
            continue
        same = sub[sub['same_cluster_40']]
        cross = sub[~sub['same_cluster_40']]
        report += (
            f"\n[{ds}] 测试靶点 {len(sub)}（同簇 {len(same)} / 跨簇 {len(cross)}）\n"
            f"  全体:   max-ρ mean={sub['max_bp_spearman'].mean():.3f} ± {sub['max_bp_spearman'].std():.3f}, "
            f"median={sub['max_bp_spearman'].median():.3f}, "
            f"ρ≥0.7: {(sub['max_bp_spearman'] >= 0.7).sum()}/{len(sub)}\n"
        )
        if len(same):
            report += (f"  同簇:   mean={same['max_bp_spearman'].mean():.3f}, "
                       f"ρ≥0.7: {(same['max_bp_spearman'] >= 0.7).sum()}/{len(same)}"
                       f" ({(same['max_bp_spearman'] >= 0.7).mean():.1%})\n")
        if len(cross):
            report += (f"  跨簇:   mean={cross['max_bp_spearman'].mean():.3f}, "
                       f"ρ≥0.7: {(cross['max_bp_spearman'] >= 0.7).sum()}/{len(cross)}"
                       f" ({(cross['max_bp_spearman'] >= 0.7).mean():.1%})\n")
        # Fisher: same_cluster_40 × (ρ≥0.7)
        a_ = int((same['max_bp_spearman'] >= 0.7).sum())
        b_ = int((same['max_bp_spearman'] < 0.7).sum())
        c_ = int((cross['max_bp_spearman'] >= 0.7).sum())
        d_ = int((cross['max_bp_spearman'] < 0.7).sum())
        if min(a_ + b_, c_ + d_) > 0:
            _, p_ = stats.fisher_exact([[a_, b_], [c_, d_]], alternative='two-sided')
        else:
            p_ = np.nan
        # Mann-Whitney: 同簇 vs 跨簇的 max-ρ 分布
        if len(same) >= 3 and len(cross) >= 3:
            u_, p_mw = stats.mannwhitneyu(same['max_bp_spearman'], cross['max_bp_spearman'],
                                          alternative='two-sided')
        else:
            p_mw = np.nan
        fisher_rows.append({'dataset': ds, 'same_hi': a_, 'same_lo': b_,
                            'cross_hi': c_, 'cross_lo': d_,
                            'fisher_p': p_, 'mw_p': p_mw})
        report += (f"  Fisher(同簇×ρ≥0.7) p={p_:.4g};  "
                   f"Mann-Whitney(同簇 vs 跨簇 max-ρ) p={p_mw:.4g}\n")

    # 全局合并
    valid = bp.dropna(subset=['max_bp_spearman'])
    a_ = int(((valid['same_cluster_40']) & (valid['max_bp_spearman'] >= 0.7)).sum())
    b_ = int(((valid['same_cluster_40']) & (valid['max_bp_spearman'] < 0.7)).sum())
    c_ = int(((~valid['same_cluster_40']) & (valid['max_bp_spearman'] >= 0.7)).sum())
    d_ = int(((~valid['same_cluster_40']) & (valid['max_bp_spearman'] < 0.7)).sum())
    _, p_all = stats.fisher_exact([[a_, b_], [c_, d_]], alternative='two-sided')
    same_all = valid[valid['same_cluster_40']]['max_bp_spearman']
    cross_all = valid[~valid['same_cluster_40']]['max_bp_spearman']
    _, p_mw_all = stats.mannwhitneyu(same_all, cross_all, alternative='two-sided')
    fisher_rows.append({'dataset': 'ALL', 'same_hi': a_, 'same_lo': b_,
                        'cross_hi': c_, 'cross_lo': d_,
                        'fisher_p': p_all, 'mw_p': p_mw_all})
    report += (
        f"\n[全局] N={len(valid)}（同簇 {a_ + b_} / 跨簇 {c_ + d_}）\n"
        f"  同簇 mean={same_all.mean():.3f} vs 跨簇 mean={cross_all.mean():.3f}\n"
        f"  Fisher p={p_all:.4g};  Mann-Whitney p={p_mw_all:.4g}\n"
        f"\n解读：\n"
        f"  若跨簇靶点仍普遍 ρ≥0.7，说明 binding-profile 泄漏独立于 40% 序列同源\n"
        f"  （kinase 家族保守 ATP 口袋 → 活性谱天然相关），支持论文关于\n"
        f"  'identity 阈值不足以阻断 mirroring' 的 limitation 讨论。\n"
    )
    pd.DataFrame(fisher_rows).to_csv(OUT / 'exp3_fisher_stratified.csv',
                                     index=False, encoding='utf-8-sig')
    (OUT / 'exp3_mirroring_audit.txt').write_text(report, encoding='utf-8')

    # ---- 散点图：x=nearest_train_identity, y=max_bp_rho，4 面板 ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, ds in zip(axes.flatten(), ds_list):
        sub = bp[bp['dataset'] == ds].dropna(subset=['max_bp_spearman'])
        same = sub[sub['same_cluster_40']]
        cross = sub[~sub['same_cluster_40']]
        ax.scatter(same['nearest_train_identity_pct'], same['max_bp_spearman'],
                   s=46, color=C_BLUE, alpha=0.75, edgecolors='white', linewidths=0.6,
                   label=f'in-cluster (n={len(same)})', zorder=3)
        ax.scatter(cross['nearest_train_identity_pct'], cross['max_bp_spearman'],
                   s=52, color=C_GREEN_L, alpha=0.85, edgecolors='white', linewidths=0.6,
                   marker='D', label=f'cross-cluster (n={len(cross)})', zorder=3)
        ax.axhline(0.7, color='#C62828', linestyle='--', linewidth=1.4, alpha=0.8)
        ax.axvline(40, color=C_GRAY, linestyle=':', linewidth=1.4, alpha=0.8)
        ax.text(0.985, 0.72, 'ρ = 0.7', transform=ax.get_yaxis_transform(),
                ha='right', va='bottom', fontsize=8, color='#C62828')
        ax.set_title(f'{ds.capitalize()}  (n={len(sub)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Nearest train identity (%)', fontsize=10)
        ax.set_ylabel('Max binding-profile Spearman ρ', fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_ylim(-0.15, 1.05)
    fig.suptitle('Binding-profile mirroring audit: sequence identity vs activity-profile similarity',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT / 'exp3_mirroring_scatter.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'exp3_mirroring_scatter.tiff', dpi=300, bbox_inches='tight')
    plt.close()

    print(report)
    return report, bp


# ============================================================
# 实验 6：理论框架 conceptual figure（权威 ΔR² + 训练集 τ）
# ============================================================
def exp6_conceptual_figure():
    datasets = ['ChEMBL', 'Davis', 'KIBA', 'BindingDB']
    tau = np.array([TAU[ds] for ds in datasets])
    dr2 = np.array([DELTA_R2[ds] for ds in datasets])
    # 各区 winner（叙述标签）
    zone_winner = {'ChEMBL': 'MLP wins', 'Davis': 'GNN wins',
                   'KIBA': 'Reptile wins', 'BindingDB': 'Reptile wins'}

    fig, ax = plt.subplots(figsize=(9.5, 6))

    # 三区背景（蓝-青-绿）
    x0, x1, x2, x3 = 0.08, 0.13, 0.20, 0.40
    ax.axvspan(x0, x1, alpha=0.16, color=C_BLUE_L, zorder=0)
    ax.axvspan(x1, x2, alpha=0.16, color=C_TEAL, zorder=0)
    ax.axvspan(x2, x3, alpha=0.14, color=C_GREEN_L, zorder=0)

    ax.axhline(0, color='gray', linewidth=1.1, alpha=0.6, zorder=1)
    for xx in (x1, x2):
        ax.axvline(xx, color='#5D4037', linewidth=1.6, linestyle=':', alpha=0.8, zorder=1)

    # 数据点（每数据集固定色，蓝绿色系内）
    ds_color = {'ChEMBL': C_BLUE, 'Davis': C_TEAL, 'KIBA': C_GREEN_L, 'BindingDB': C_GREEN}
    offsets = {'ChEMBL': (0, -26), 'Davis': (0, 14), 'KIBA': (0, 14), 'BindingDB': (0, -26)}
    for i, ds in enumerate(datasets):
        ax.scatter(tau[i], dr2[i], s=210, color=ds_color[ds], zorder=5,
                   edgecolors='white', linewidths=1.8)
        ax.annotate(f'{ds}\n{zone_winner[ds]}', (tau[i], dr2[i]),
                    textcoords='offset points', xytext=offsets[ds],
                    ha='center', fontsize=10, fontweight='bold', color=ds_color[ds], zorder=6)

    # 区域标题（MLP 区标签置底部，避免与 GNN 区标签框重叠）
    ytop, ybot = 0.225, -0.285
    ax.text((x0 + x1) / 2, ybot - 0.015, 'MLP advantage zone\n(scaffold-deficit)', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=C_BLUE,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C_BLUE_SOFT, alpha=0.95))
    ax.text((x1 + x2) / 2 - 0.012, ytop, 'GNN advantage zone\n(moderate overlap)', ha='center', va='top',
            fontsize=10, fontweight='bold', color=C_TEAL,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C_TEAL_SOFT, alpha=0.95))
    ax.text((x2 + x3) / 2, ytop, 'Meta-learning advantage zone\n(scaffold-redundant)', ha='center', va='top',
            fontsize=10, fontweight='bold', color=C_GREEN,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C_GREEN_SOFT, alpha=0.95))

    ax.text(x1 + 0.002, 0.008, 'τ ≈ 0.13', ha='left', fontsize=9, color='#5D4037')
    ax.text(x2 + 0.002, 0.008, 'τ ≈ 0.20', ha='left', fontsize=9, color='#5D4037')

    ax.set_xlabel('Average intra-panel Tanimoto similarity τ (ECFP4, training drugs)',
                  fontsize=12.5, fontweight='bold')
    ax.set_ylabel(f'{DELTA}{R2}  (best GNN − best non-GNN)', fontsize=12.5, fontweight='bold')
    ax.set_title('Conceptual framework: similarity-governed model applicability zones',
                 fontsize=13.5, fontweight='bold')
    ax.set_xlim(x0, x3)
    ax.set_ylim(ybot - 0.02, ytop + 0.02)
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / 'exp6_conceptual_framework.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'exp6_conceptual_framework.tiff', dpi=300, bbox_inches='tight')
    plt.close()

    report = (
        "实验 6：理论框架 conceptual figure 已生成（权威数据源）\n"
        f"  PNG:  {OUT / 'exp6_conceptual_framework.png'}\n"
        f"  TIFF: {OUT / 'exp6_conceptual_framework.tiff'}\n"
        "  数据源：τ = Table S3 训练集口径；ΔR² = Figure 6 (plot_gnn_vs_nongnn.py DATASETS_INFO)\n"
        f"  数据点：\n"
    )
    for ds in datasets:
        report += f"    {ds:10s} τ={TAU[ds]:.3f}  {DELTA}{R2}={DELTA_R2[ds]:+.4f}  -> {zone_winner[ds]}\n"
    report += (
        "  三区划分：τ<0.13 MLP 优势区；0.13≤τ≤0.20 GNN 优势区；τ>0.20 元学习优势区。\n"
        "  注：元学习优势区内 ΔR²<0 表示 GNN 劣于 non-GNN（Reptile 属 non-GNN 家族，\n"
        "  其增益来自元学习机制而非分子图）。\n"
    )
    (OUT / 'exp6_conceptual_framework.txt').write_text(report, encoding='utf-8')
    print(report)
    return report


# ============================================================
# 实验 8：配对检验 p 值矩阵热力图（matplotlib imshow，无 seaborn）
# ============================================================
def _draw_heatmap(ax, pivot_logp, pivot_raw, title):
    im = ax.imshow(pivot_logp.values, cmap='YlGnBu', vmin=0, vmax=3, aspect='auto')
    ax.set_xticks(range(len(pivot_logp.columns)))
    ax.set_xticklabels(pivot_logp.columns, rotation=30, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(pivot_logp.index)))
    ax.set_yticklabels(pivot_logp.index, fontsize=9)
    for i in range(pivot_logp.shape[0]):
        for j in range(pivot_logp.shape[1]):
            v = pivot_raw.values[i, j]
            if np.isnan(v):
                continue
            color = 'white' if pivot_logp.values[i, j] > 1.8 else 'black'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7.5, color=color)
    ax.set_title(title, fontsize=11, fontweight='bold')
    return im


def exp8_pvalue_heatmap():
    if not E5_CSV.exists():
        print(f"[exp8] 缺少 {E5_CSV}，跳过。")
        return None

    e5 = pd.read_csv(E5_CSV)
    e5 = e5[e5['status'] == 'ok']

    # 基线单次运行值（论文口径：MLP/Transformer/GNN 单跑）
    baseline_vals = {
        'chembl': {
            'MLP': {'R2': 0.262, 'EF@1%': 4.63, 'EF@5%': 4.07},
            'Transformer': {'R2': 0.180, 'EF@1%': 4.44, 'EF@5%': 3.80},
            'GAT_GCN': {'R2': 0.058, 'EF@1%': 2.11, 'EF@5%': 1.65},
            'GCNNet': {'R2': 0.045, 'EF@1%': 2.41, 'EF@5%': 1.97},
            'GATNet': {'R2': -0.153, 'EF@1%': 1.59, 'EF@5%': 1.42},
            'GINConvNet': {'R2': -0.089, 'EF@1%': 2.84, 'EF@5%': 2.25},
        },
        'davis': {
            'MLP': {'R2': 0.259, 'EF@1%': 4.20, 'EF@5%': 3.51},
            'Transformer': {'R2': 0.224, 'EF@1%': 4.20, 'EF@5%': 3.51},
            'GAT_GCN': {'R2': 0.394, 'EF@1%': 0.428, 'EF@5%': 0.735},
            'GCNNet': {'R2': 0.395, 'EF@1%': 0.432, 'EF@5%': 0.738},
            'GATNet': {'R2': 0.225, 'EF@1%': 0.469, 'EF@5%': 0.757},
            'GINConvNet': {'R2': 0.244, 'EF@1%': 4.34, 'EF@5%': 3.80},
        },
        'kiba': {
            'MLP': {'R2': 0.2565, 'EF@1%': 3.75, 'EF@5%': 3.57},
            'Transformer': {'R2': 0.1800, 'EF@1%': 4.05, 'EF@5%': 3.66},
            'GAT_GCN': {'R2': 0.223, 'EF@1%': 4.49, 'EF@5%': 3.30},
            'GCNNet': {'R2': 0.212, 'EF@1%': 4.28, 'EF@5%': 3.59},
            'GATNet': {'R2': -0.104, 'EF@1%': 3.88, 'EF@5%': 3.00},
            'GINConvNet': {'R2': 0.120, 'EF@1%': 4.37, 'EF@5%': 3.38},
        },
    }

    metrics = ['R2', 'EF@1%', 'EF@5%']
    baselines = ['MLP', 'Transformer', 'GAT_GCN', 'GCNNet', 'GATNet', 'GINConvNet']

    rows = []
    for ds in ['chembl', 'davis', 'kiba']:
        sub = e5[e5['dataset'] == ds]
        if sub.empty:
            continue
        for bl in baselines:
            for met in metrics:
                if met not in sub.columns or met not in baseline_vals[ds].get(bl, {}):
                    continue
                reptile_vals = sub[met].dropna().values
                if len(reptile_vals) < 2:
                    continue
                bl_val = baseline_vals[ds][bl][met]
                diffs = reptile_vals - bl_val
                if np.allclose(diffs, 0):
                    p = 1.0
                else:
                    # Wilcoxon 符号秩（5 seeds 对基线单值的配对差）；退化为 t 检验
                    try:
                        p = stats.wilcoxon(diffs).pvalue
                    except ValueError:
                        _, p = stats.ttest_1samp(diffs, 0)
                _, p_t = stats.ttest_1samp(diffs, 0)
                better = (diffs.mean() > 0)
                rows.append({'dataset': ds, 'baseline': bl, 'metric': met,
                             'reptile_mean': reptile_vals.mean(),
                             'reptile_std': reptile_vals.std(ddof=1),
                             'baseline_val': bl_val, 'diff_mean': diffs.mean(),
                             'p_wilcoxon': p, 'p_ttest': p_t,
                             'direction': 'Reptile better' if better else 'Reptile worse'})
    pval_df = pd.DataFrame(rows)
    pval_df['p_bh'] = bh_fdr(pval_df['p_wilcoxon'].values)
    pval_df.to_csv(OUT / 'exp8_pvalues.csv', index=False, encoding='utf-8-sig')

    # 热力图：每指标一个 panel（dataset × baseline），颜色=-log10(p)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    im = None
    for ax, met in zip(axes, metrics):
        sub = pval_df[pval_df['metric'] == met]
        pivot = sub.pivot(index='dataset', columns='baseline', values='p_wilcoxon')
        pivot = pivot.reindex(index=['chembl', 'davis', 'kiba'],
                              columns=[c for c in baselines if c in pivot.columns])
        logp = -np.log10(pivot.clip(lower=1e-4)).clip(upper=4)
        im = _draw_heatmap(ax, logp, pivot, f'{met}  (annot = raw p)')
        ax.set_xlabel('Baseline')
        if met == metrics[0]:
            ax.set_ylabel('Dataset')
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cbar.set_label('-log10(p), Wilcoxon signed-rank (5 seeds)', fontsize=10)
    fig.suptitle('Paired-test p-value matrix: Reptile-Transformer (5 seeds) vs single-run baselines',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / 'exp8_pvalue_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'exp8_pvalue_heatmap.tiff', dpi=300, bbox_inches='tight')
    plt.close()

    n_sig = int((pval_df['p_bh'] < 0.05).sum())
    report = (
        "实验 8：配对检验 p 值矩阵热力图已生成（Wilcoxon 符号秩 + BH-FDR）\n"
        f"  CSV:  {OUT / 'exp8_pvalues.csv'}\n"
        f"  PNG:  {OUT / 'exp8_pvalue_heatmap.png'}\n"
        f"  TIFF: {OUT / 'exp8_pvalue_heatmap.tiff'}\n"
        f"  总检验数: {len(pval_df)}（3 数据集 × 6 基线 × 3 指标）\n"
        f"  BH-FDR < 0.05: {n_sig}\n"
        "  注：5 seeds 与基线单次值之差做 Wilcoxon；n=5 时 Wilcoxon 最小可达\n"
        f"  p=0.0625（双侧），因此显著性判读辅以效应量 diff_mean 与方向列。\n"
        f"  显著且方向有利的检验数: "
        f"{int(((pval_df['p_bh'] < 0.05) & (pval_df['direction'] == 'Reptile better')).sum())}\n"
    )
    (OUT / 'exp8_report.txt').write_text(report, encoding='utf-8')
    print(report)
    return report, pval_df


# ============================================================
# 实验 9：Table 4 / S4 / S5 补 ±std
# ============================================================
def exp9_add_std():
    if not E5_CSV.exists():
        print(f"[exp9] 缺少 {E5_CSV}，跳过。")
        return None

    e5 = pd.read_csv(E5_CSV)
    e5 = e5[e5['status'] == 'ok']

    def fmt(ds, met, nd=4):
        sub = e5[e5['dataset'] == ds][met].dropna().values
        if len(sub) == 0:
            return 'n/a'
        return f"{sub.mean():.{nd}f} ± {sub.std(ddof=1):.{nd}f}"

    report = "实验 9：Table 4 / S4 / S5 补 ±std（Reptile-Transformer, 5 seeds）\n" + "=" * 70 + "\n"

    report += "\n[Table 4 — KIBA 消融] 最后一行 'Reptile-Transformer (full pipeline)' 补 ±std：\n"
    report += f"  {'指标':<8} {'论文原值':<12} {'5-seed 均值±std':<22}\n"
    t4map = {'R2': 'R2', 'RMSE': 'RMSE', 'AUPR': 'AUPR', 'ECE': 'ECE'}
    for label, orig in TABLE4['Reptile-Transformer (full pipeline)'].items():
        col = t4map[label]
        report += f"  {label:<8} {orig:<12.4f} {fmt('kiba', col):<22}\n"

    report += "\n[S4 EF@1% — Reptile+Transformer 行] 补 ±std：\n"
    for ds, name in [('chembl', 'ChEMBL'), ('davis', 'Davis'), ('kiba', 'KIBA')]:
        report += (f"  {name:<10} 论文值 {EF1_MATRIX['Reptile+Transformer'][name]:>5.2f}   "
                   f"5-seed {fmt(ds, 'EF@1%', 3)}\n")

    report += "\n[S5 EF@5% — Reptile+Transformer 行] 补 ±std：\n"
    ef5_s5 = {'ChEMBL': 1.91, 'Davis': 1.62, 'KIBA': 4.17}
    for ds, name in [('chembl', 'ChEMBL'), ('davis', 'Davis'), ('kiba', 'KIBA')]:
        report += (f"  {name:<10} 论文值 {ef5_s5[name]:>5.2f}   "
                   f"5-seed {fmt(ds, 'EF@5%', 3)}\n")

    report += "\n[全套指标 5-seed ±std 汇总]\n"
    for ds in ['chembl', 'davis', 'kiba']:
        report += f"  [{ds}]\n"
        for met in ['R2', 'RMSE', 'MAE', 'Pearson', 'Spearman', 'EF@1%', 'EF@5%', 'EF@10%', 'ECE', 'AUPR']:
            if met in e5.columns:
                report += f"    {met:<10}: {fmt(ds, met)}\n"

    report += (
        "\n注：\n"
        "  - ±std 仅对 Reptile-Transformer 有效（5 次独立重复，seed=11/22/33/44/55）。\n"
        "  - Davis 的 EF@1% 因每靶点仅 66 化合物（1% 截断 <1 个化合物），\n"
        "    5 seeds 全为 0 —— 与论文 S4 脚注（小样本失真）一致，建议正文标注 n/a。\n"
        "  - 基线模型（MLP/Transformer/GNN）为单次运行，表格中以 'single-run' 脚注说明。\n"
        "  - BindingDB 未纳入 E5 重复（训练代价 ~14h/run），建议维持单跑并脚注。\n"
    )

    (OUT / 'exp9_std_report.txt').write_text(report, encoding='utf-8')
    print(report)
    return report


# ============================================================
# 实验 14：Family-remote 二分类验证
# ============================================================
def exp14_family_remote():
    datasets = ['chembl', 'davis', 'kiba', 'bindingdb']
    all_rows = []
    for ds in datasets:
        fp = CLUSTER / ds / 'per_test_target_identity.csv'
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        df['dataset'] = ds
        all_rows.append(df[['dataset', 'target_id', 'nearest_train_identity_pct', 'same_cluster_40']])
    combo = pd.concat(all_rows, ignore_index=True)

    y_true = (~combo['same_cluster_40']).astype(int).values   # 1 = family-remote
    y_score = combo['nearest_train_identity_pct'].values

    def eval_at(thr):
        y_pred = (y_score < thr).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp_ = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        acc = (tp + tn) / len(y_true)
        fpr = fp_ / (fp_ + tn) if (fp_ + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        prec = tp / (tp + fp_) if (tp + fp_) > 0 else np.nan
        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        return dict(threshold=thr, TP=tp, FP=fp_, TN=tn, FN=fn,
                    Accuracy=acc, FPR=fpr, FNR=fnr, Precision=prec, Recall=rec)

    thr_rows = [eval_at(t) for t in [30.0, 35.0, 40.0, 45.0, 50.0, 60.0]]
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(OUT / 'exp14_threshold_scan.csv', index=False, encoding='utf-8-sig')
    main = eval_at(40.0)

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, -y_score)
    except Exception:
        auc = np.nan

    per_ds = []
    for ds in datasets:
        sub = combo[combo['dataset'] == ds]
        if sub.empty:
            continue
        yt = (~sub['same_cluster_40']).astype(int).values
        yp = (sub['nearest_train_identity_pct'].values < 40.0).astype(int)
        tp_ = int(((yt == 1) & (yp == 1)).sum())
        fp_ = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        acc_ = (tp_ + tn) / len(yt)
        fpr_ = fp_ / (fp_ + tn) if (fp_ + tn) > 0 else np.nan
        fnr_ = fn / (fn + tp_) if (fn + tp_) > 0 else np.nan
        per_ds.append({'dataset': ds, 'n_test': len(yt), 'n_remote': int((yt == 1).sum()),
                       'TP': tp_, 'FP': fp_, 'TN': tn, 'FN': fn,
                       'Accuracy': acc_, 'FPR': fpr_, 'FNR': fnr_})

    report = (
        "实验 14：Family-remote 二分类验证\n"
        "=" * 70 + "\n"
        "判定指标：测试靶点到训练集最近靶点的序列一致性 (nearest_train_identity_pct, %)\n"
        "Ground truth：MMseqs2 easy-cluster @ 40% min-seq-id 的 same_cluster_40 标签\n"
        "分类规则：identity < threshold -> family-remote（走 non-graph 分支）\n\n"
        f"总测试靶点数: {len(y_true)}（remote {int(y_true.sum())} / in-family {int((1 - y_true).sum())}）\n"
        f"AUC(identity 作为打分) = {auc:.3f}\n\n"
        "[主阈值 40%] 混淆矩阵（全局）:\n"
        f"              预测 remote   预测 in-family\n"
        f"真实 remote       {main['TP']:>4}          {main['FN']:>4}\n"
        f"真实 in-family    {main['FP']:>4}          {main['TN']:>4}\n"
        f"Accuracy = {main['Accuracy']:.3f}   Precision = {main['Precision']:.3f}   Recall = {main['Recall']:.3f}\n"
        f"FPR (in-family 误判为 remote) = {main['FPR']:.3f}\n"
        f"FNR (remote 误判为 in-family) = {main['FNR']:.3f}\n"
        f"误分类率合计 (FPR+FNR) = {main['FPR'] + main['FNR']:.3f}\n\n"
        "[阈值扫描]\n"
    )
    for r in thr_rows:
        report += (f"  thr={r['threshold']:>4.0f}%  Acc={r['Accuracy']:.3f}  FPR={r['FPR']:.3f}  "
                   f"FNR={r['FNR']:.3f}  Prec={r['Precision']:.3f}  Rec={r['Recall']:.3f}\n")

    report += "\n[逐数据集 @40%]\n"
    for r in per_ds:
        report += (f"  {r['dataset']:<10} n={r['n_test']:>3} remote={r['n_remote']:>3}  "
                   f"Acc={r['Accuracy']:.3f}  FPR={r['FPR']:.3f}  FNR={r['FNR']:.3f}\n")

    report += (
        "\n结论（供 Discussion 4.3 引用）：\n"
        f"  - 判定指标 = 最近训练靶点序列一致性；阈值 = 40%（与 MMseqs2 聚类口径一致）。\n"
        f"  - 全局 Accuracy={main['Accuracy']:.1%}，FPR={main['FPR']:.1%}，FNR={main['FNR']:.1%}，AUC={auc:.3f}。\n"
        f"  - 由于 same_cluster_40 的真值即由 40% 一致性定义，identity<40% 规则与真值\n"
        f"    几乎重合（残余误分类来自聚类的连通分量传递效应：两靶点 identity<40% 仍可\n"
        f"    经中间序列同簇）。部署建议：threshold=40% 为主，45-50% 为保守档（FNR↓）。\n"
    )

    (OUT / 'exp14_family_remote.txt').write_text(report, encoding='utf-8')
    pd.DataFrame(per_ds).to_csv(OUT / 'exp14_family_remote_per_dataset.csv',
                                index=False, encoding='utf-8-sig')
    print(report)
    return report


def main():
    print("=" * 70)
    print("补做实验合集（无需训练）")
    print("=" * 70)

    print("\n>>> 实验 2：McNemar / Fisher 精确检验")
    exp2_mcnemar_fisher()

    print("\n>>> 实验 3：Target mirroring 检测（4 数据集分层）")
    exp3_target_mirroring()

    print("\n>>> 实验 6：理论框架 conceptual figure")
    exp6_conceptual_figure()

    print("\n>>> 实验 8：配对检验 p 值矩阵热力图")
    exp8_pvalue_heatmap()

    print("\n>>> 实验 9：Table 4 / S4 / S5 补 ±std")
    exp9_add_std()

    print("\n>>> 实验 14：Family-remote 二分类验证")
    exp14_family_remote()

    print(f"\n全部结果已保存至: {OUT}")


if __name__ == '__main__':
    main()
