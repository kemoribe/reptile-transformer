# -*- coding: utf-8 -*-
"""
E5 统计分析：Bootstrap 95% 置信区间 + 配对 Wilcoxon 符号秩检验
=============================================================
输入: supplement_output/revision2_experiments/e5_repeats/（5 seeds × 4 datasets）

两类置信区间：
  (A) 重复级：对 5 次重复的全局指标做 bootstrap（B=10000，百分位法）；
  (B) 靶点级（更稳）：分层 bootstrap——每次在每个重复内对靶点有放回重采样，
      计算逐靶点 R²/EF@1% 均值，再跨重复平均（B=2000）。

Wilcoxon：逐靶点配对 Reptile vs 同数据集最强 GNN（按逐靶点 R² 均值数据驱动选择，
  预测来自 paper_revision/gnn_preds/*.npz，行序与 GrapthDTA/data/<ds>_test.csv 对齐）。
  每个种子各做一次（n=靶点数），BH-FDR 跨全部检验校正；ChEMBL 无 GNN 预测文件自动跳过。

用法: python supplement_e5_stats.py
输出: e5_repeats/{e5_bootstrap_ci.csv, e5_wilcoxon.csv, e5_stats_summary.txt}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from supplement_exp_common import stage_dir, DATASETS, ROOT  # noqa

GNN_DIR = ROOT / 'supplement_output' / 'paper_revision' / 'gnn_preds'
GNN_MODELS = ['GCNNet', 'GATNet', 'GAT_GCN', 'GINConvNet']
GNNS = {'GCNNet': 'GCNNet', 'GATNet': 'GATNet', 'GAT_GCN': 'GAT_GCN', 'GINConvNet': 'GINConvNet'}
GLOBAL_METRICS = ['R2', 'RMSE', 'MAE', 'Pearson', 'Spearman', 'EF@1%', 'EF@5%', 'ECE', 'AUPR']
TARGET_METRICS = ['R2', 'EF@1%']
BOOT_REPEATS = 10000
BOOT_TARGETS = 2000
RNG = np.random.default_rng(20260919)


def boot_ci(vals, n_boot, axis=0):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def r2_score(yt, yp):
    ss = np.sum((yt - yt.mean()) ** 2)
    return 1 - np.sum((yt - yp) ** 2) / ss if ss > 0 else np.nan


def ef_at(yt, yp, pct):
    """与 reptile_training._compute_metrics 完全一致的 EF 定义"""
    n = len(yt)
    if n < 10:
        return np.nan
    n_active = max(5, int(n * 0.2))
    thr = np.sort(yt)[::-1][min(n_active - 1, n - 1)]
    n_top = int(n * pct / 100)
    if n_top == 0:
        return np.nan
    n_active_top = np.sum(yt[np.argsort(yp)[::-1][:n_top]] >= thr)
    return (n_active_top / n_top) / (n_active / n)


def load_per_target(dataset, seed):
    fp = stage_dir('e5_repeats') / dataset / f'seed_{seed}' / 'per_target_results.json'
    if not fp.exists():
        return None
    d = json.loads(fp.read_text(encoding='utf-8'))
    tbl = pd.DataFrame([{'target': k, **{m: v.get(m) for m in TARGET_METRICS}}
                        for k, v in d.items()])
    # per_target 指标在 JSON 中可能是字符串
    for m in TARGET_METRICS:
        tbl[m] = pd.to_numeric(tbl[m], errors='coerce')
    return tbl


def gnn_per_target_table(dataset):
    """返回 {model: DataFrame(target_sequence -> R2, EF1)}"""
    csv_fp = ROOT / 'GrapthDTA' / 'data' / f'{dataset}_test.csv'
    tables = {}
    for m in GNN_MODELS:
        npz_fp = GNN_DIR / f'{m}_{dataset}.npz'
        if not npz_fp.exists() or not csv_fp.exists():
            continue
        z = np.load(npz_fp)
        df = pd.read_csv(csv_fp)
        if len(df) != len(z['y_true']):
            print(f'[warn] {m}_{dataset}: csv {len(df)} 行与 npz {len(z["y_true"])} 不一致，跳过')
            continue
        df = df[['target_sequence']].copy()
        df['yt'] = z['y_true']; df['yp'] = z['y_pred']
        rows = []
        for seq, g in df.groupby('target_sequence'):
            yt, yp = g['yt'].values, g['yp'].values
            rows.append({'sequence': seq, 'R2': r2_score(yt, yp),
                         'EF@1%': ef_at(yt, yp, 1) if len(g) >= 100 else np.nan})
        tables[m] = pd.DataFrame(rows)
    return tables


def seq_to_target_id(dataset):
    """测试集 target_name -> sequence（与预处理器同源）"""
    import os
    import data_preprocessing as _dp
    data_dir = ROOT / {
        'chembl': '3_all_data_chembl_targets_preprocessed', 'davis': '3_davis_preprocessed',
        'kiba': '3_kiba_preprocessed', 'bindingdb': 'bindingdb_preprocessed'}[dataset]
    os.environ['PREPROCESSED_DIR'] = str(data_dir)
    # 模块可能已在 import 时绑定默认 ChEMBL 目录，显式重绑
    _dp.PREPROCESSED_DIR = data_dir
    m = {}
    for info in _dp.get_all_targets()['test']:
        td = _dp.load_target_data(info)
        if td is not None and td.get('sequence'):
            m[td['target_name']] = td['sequence']
    return m


def bh_fdr(pvals):
    """Benjamini–Hochberg FDR 校正"""
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


def main():
    outroot = stage_dir('e5_repeats')
    met_csv = outroot / 'e5_repeats_metrics.csv'
    if not met_csv.exists():
        sys.exit(f'缺少 {met_csv}，请先运行 supplement_e5_repeats.py')
    df = pd.read_csv(met_csv)
    df = df[df['status'] == 'ok'].copy()
    seeds = sorted(df['seed'].unique())

    ci_rows, tgt_rows = [], []
    # (A) 全局指标重复级 bootstrap
    for ds, g in df.groupby('dataset'):
        for m in GLOBAL_METRICS:
            mean, lo, hi = boot_ci(g[m].values, BOOT_REPEATS)
            ci_rows.append({'dataset': ds, 'metric': m, 'level': 'repeat_global',
                            'n': len(g), 'mean': mean, 'ci_low': lo, 'ci_high': hi,
                            'std': float(np.nanstd(g[m].values, ddof=1)) if len(g) > 1 else np.nan})

    # (B) 逐靶点分层 bootstrap
    for ds in df['dataset'].unique():
        per_seed = {s: load_per_target(ds, s) for s in seeds}
        per_seed = {s: t for s, t in per_seed.items() if t is not None}
        if not per_seed:
            continue
        for m in TARGET_METRICS:
            mats = [t.set_index('target')[m].dropna() for t in per_seed.values()]
            common = set.intersection(*[set(x.index) for x in mats]) if mats else set()
            if not common:
                continue
            common = sorted(common)
            arr = np.stack([x.loc[common].values.astype(float) for x in mats])  # (R, T)
            R, T = arr.shape
            means = []
            for _ in range(BOOT_TARGETS):
                ti = RNG.integers(0, T, T)
                ri = RNG.integers(0, R, R)
                means.append(arr[ri[:, None], ti[None, :]].mean())
            lo, hi = np.percentile(means, [2.5, 97.5])
            tgt_rows.append({'dataset': ds, 'metric': f'per_target_{m}', 'level': 'target_stratified',
                             'n_repeats': R, 'n_targets': T,
                             'mean': float(arr.mean()), 'ci_low': lo, 'ci_high': hi, 'std': np.nan})
    ci = pd.DataFrame(ci_rows + tgt_rows)
    ci.to_csv(outroot / 'e5_bootstrap_ci.csv', index=False, encoding='utf-8-sig')

    # ===== Wilcoxon：Reptile(每种子) vs 最强 GNN，逐靶点配对 =====
    w_rows = []
    for ds in df['dataset'].unique():
        gnn_tables = gnn_per_target_table(ds)
        if not gnn_tables:
            print(f'[wilcoxon] {ds}: 无 GNN 预测文件，跳过')
            continue
        # 数据驱动选择逐靶点 R² 均值最高的 GNN
        best_m = max(gnn_tables, key=lambda k: gnn_tables[k]['R2'].mean())
        gnn = gnn_tables[best_m].set_index('sequence')
        id2seq = seq_to_target_id(ds)
        for s in seeds:
            pt = load_per_target(ds, s)
            if pt is None:
                continue
            pt = pt.copy(); pt['sequence'] = pt['target'].map(id2seq)
            for m in TARGET_METRICS:
                j = pt.dropna(subset=[m]).merge(
                    gnn[m].rename('gnn_v'), left_on='sequence', right_index=True, how='inner')
                j = j.dropna(subset=['gnn_v'])
                if m == 'EF@1%':
                    j = j[j['gnn_v'].notna()]
                if len(j) < 8:
                    w_rows.append({'dataset': ds, 'seed': s, 'metric': m, 'comparator': best_m,
                                   'n': len(j), 'reptile_mean': np.nan, 'gnn_mean': np.nan,
                                   'W': np.nan, 'p': np.nan, 'note': 'n<8'})
                    continue
                d = (j[m] - j['gnn_v']).values
                if np.allclose(d, 0):
                    p, W = np.nan, np.nan
                else:
                    W, p = stats.wilcoxon(j[m], j['gnn_v'], zero_method='wilcox',
                                          alternative='two-sided')
                w_rows.append({'dataset': ds, 'seed': s, 'metric': m, 'comparator': best_m,
                               'n': len(j), 'reptile_mean': float(j[m].mean()),
                               'gnn_mean': float(j['gnn_v'].mean()),
                               'W': float(W), 'p': float(p), 'note': ''})
    wdf = pd.DataFrame(w_rows)
    if not wdf.empty and wdf['p'].notna().any():
        mask = wdf['p'].notna()
        wdf.loc[mask, 'p_bh'] = bh_fdr(wdf.loc[mask, 'p'].values)
    wdf.to_csv(outroot / 'e5_wilcoxon.csv', index=False, encoding='utf-8-sig')

    # ===== 文本汇总 =====
    lines = ['E5 统计汇总 (5 seeds)', '=' * 60, '']
    for ds in sorted(df['dataset'].unique()):
        lines.append(f'[{ds}]')
        sub = ci[(ci['dataset'] == ds) & (ci['level'] == 'repeat_global')]
        for _, r in sub.iterrows():
            lines.append(f'  {r["metric"]:>9s}: {r["mean"]:.4f}  95%CI[{r["ci_low"]:.4f}, {r["ci_high"]:.4f}]  std={r["std"]:.4f}')
        sub2 = ci[(ci['dataset'] == ds) & (ci['level'] == 'target_stratified')]
        for _, r in sub2.iterrows():
            lines.append(f'  {r["metric"]:>15s}: {r["mean"]:.4f}  95%CI[{r["ci_low"]:.4f}, {r["ci_high"]:.4f}]  (T={r["n_targets"]})')
        wsub = wdf[wdf['dataset'] == ds] if not wdf.empty else pd.DataFrame()
        if not wsub.empty:
            for m in TARGET_METRICS:
                wm = wsub[wsub['metric'] == m].dropna(subset=['p_bh'])
                if not wm.empty:
                    comp = wm['comparator'].iloc[0]
                    sig = (wm['p_bh'] < 0.05).sum()
                    lines.append(f'  Wilcoxon {m} vs {comp}: {sig}/{len(wm)} 个种子 BH-FDR<0.05 '
                                 f'(p={", ".join(f"{v:.3g}" for v in wm["p_bh"])})')
        lines.append('')
    report = '\n'.join(lines)
    (outroot / 'e5_stats_summary.txt').write_text(report, encoding='utf-8')
    print(report)
    print(f'输出: {outroot}')


if __name__ == '__main__':
    main()
