# -*- coding: utf-8 -*-
"""
P0 实验4：审计 vs 标准（未审计）评估的定量对比 + 误选代价
数据: revision2_experiments/e8_homology_free_ablation/homology_free_ablation_results.csv
口径:
  标准评估 = 名义靶点冷启动测试集（完整 test，未经同源性审计；即冷启动文献惯用口径）
  审计评估 = 40% 同一性跨簇（同源自由）子集
对每个数据集、每个指标（R²↑ / EF@1%↑ / ECE↓）:
  - 两种口径下的模型排名
  - 排名一致性 Kendall τb、Spearman ρ
  - 误选代价 = 在审计子集上，标准口径所选 winner 相对审计口径 winner 的性能差
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

E8 = Path(r'd:\lht\修改supplement_output\revision2_experiments\e8_homology_free_ablation'
          r'\homology_free_ablation_results.csv')
OUT = Path(r'd:\lht\修改supplement_output\p0_experiments')

METRICS = {  # 列名前缀在 e8 中为 full_ / sub_
    'R2': ('higher', 'R²'),
    'EF@1%': ('higher', 'EF@1%'),
    'ECE': ('lower', 'ECE'),
}


def ranks(values, direction):
    # 名次 1 = 最优；higher: 大优；lower: 小优
    s = pd.Series(values)
    if direction == 'higher':
        return s.rank(ascending=False, method='min').astype(int)
    return s.rank(ascending=True, method='min').astype(int)


def main():
    df = pd.read_csv(E8)
    detail_rows, rank_rows, cost_rows = [], [], []

    for ds, sub in df.groupby('dataset'):
        sub = sub.reset_index(drop=True)
        models = sub['model'].tolist()
        n = len(models)
        for col, (direction, label) in METRICS.items():
            v_std = sub[f'full_{col}'].to_numpy(float)
            v_aud = sub[f'sub_{col}'].to_numpy(float)
            r_std = ranks(v_std, direction).to_numpy()
            r_aud = ranks(v_aud, direction).to_numpy()

            kt, _ = kendalltau(r_std, r_aud)
            sr, _ = spearmanr(r_std, r_aud)

            i_std = int(np.argmin(r_std))
            i_aud = int(np.argmin(r_aud))
            reversed_ = i_std != i_aud
            if direction == 'higher':
                cost = float(v_aud[i_aud] - v_aud[i_std])
                rel = cost / abs(v_aud[i_aud]) if v_aud[i_aud] != 0 else np.nan
            else:
                cost = float(v_aud[i_std] - v_aud[i_aud])
                rel = cost / abs(v_aud[i_aud]) if v_aud[i_aud] != 0 else np.nan

            rank_rows.append({
                'dataset': ds, 'metric': label, 'n_models': n,
                'kendall_tau': round(float(kt), 3),
                'spearman_rho': round(float(sr), 3),
                'standard_winner': models[i_std],
                'audited_winner': models[i_aud],
                'winner_reversed': reversed_,
                'misselection_cost': round(cost, 4),
                'relative_cost_pct': round(100 * rel, 1),
            })
            for j, m in enumerate(models):
                detail_rows.append({'dataset': ds, 'metric': label, 'model': m,
                                    'standard_value': round(v_std[j], 4),
                                    'audited_value': round(v_aud[j], 4),
                                    'standard_rank': int(r_std[j]),
                                    'audited_rank': int(r_aud[j])})
            cost_rows.append((ds, label, models[i_std], models[i_aud], cost))

    rk = pd.DataFrame(rank_rows)
    det = pd.DataFrame(detail_rows)
    rk.to_csv(OUT / 'p0_exp4_rank_comparison.csv', index=False, encoding='utf-8-sig')
    det.to_csv(OUT / 'p0_exp4_per_model_ranks.csv', index=False, encoding='utf-8-sig')

    lines = ['P0 实验4：审计 vs 标准（未审计名义冷启动）评估 —— 排名一致性与误选代价',
             '=' * 74,
             '标准评估 = 名义靶点冷启动完整测试集；审计评估 = 40% 同源自由跨簇子集',
             '模型: GCNNet/GATNet/GAT_GCN/GINConvNet/Reptile-Transformer (davis/kiba/bindingdb)',
             '']
    show = rk[['dataset', 'metric', 'kendall_tau', 'spearman_rho', 'standard_winner',
               'audited_winner', 'winner_reversed', 'misselection_cost',
               'relative_cost_pct']]
    lines.append(show.to_string(index=False))
    lines.append('')
    lines.append('逐数据集逐指标的模型名次:')
    for ds, s1 in det.groupby('dataset'):
        lines.append(f'-- {ds} --')
        for metric, s2 in s1.groupby('metric'):
            seq = ' > '.join(f"{r['model']}({r['audited_rank']}|std{r['standard_rank']})"
                             for _, r in s2.sort_values('audited_rank').iterrows())
            lines.append(f'  {metric:6s} 审计名次(标准名次): {seq}')
    lines.append('')
    # R2 汇总
    r2 = rk[rk.metric == 'R²']
    lines.append(f"R²: {int(r2.winner_reversed.sum())}/{len(r2)} 个数据集发生 winner 易主; "
                 f"Kendall τ 范围 {r2.kendall_tau.min():.2f}-{r2.kendall_tau.max():.2f}")
    rc = r2[r2.winner_reversed]
    if len(rc):
        lines.append('误选代价（审计子集 R² 绝对损失）:')
        for _, r in rc.iterrows():
            lines.append(f"  {r['dataset']}: 标准选 {r['standard_winner']} → "
                         f"ΔR²={r['misselection_cost']:.3f} ({r['relative_cost_pct']:.0f}% 相对损失)")
    lines.append('')
    lines.append('解读: 负/低 Kendall τ 与非零误选代价直接量化“不做审计的代价”——按未审计')
    lines.append('名义冷启动基准选出的 R² 最优模型，在真正同源自由的部署场景中并非最优。')

    (OUT / 'p0_exp4_report.txt').write_text('\n'.join(lines), encoding='utf-8')
    summary = {
        'R2_reversal_datasets': rc['dataset'].tolist(),
        'R2_kendall_tau': dict(zip(r2['dataset'], r2['kendall_tau'])),
        'R2_misselection_cost': dict(zip(rc['dataset'], rc['misselection_cost'])),
    }
    with open(OUT / 'p0_exp4_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
