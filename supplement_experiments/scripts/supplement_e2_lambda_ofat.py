# -*- coding: utf-8 -*-
"""
E2 — λ 单变量扰动（OFAT）跨数据集复现：KIBA + Davis
=====================================================
5 组配置（与 ChEMBL Table S9 完全一致）: base(0.05,0.10) / l1_low 0.025 / l1_high 0.10
/ l2_low 0.05 / l2_high 0.20。

设计说明：ChEMBL 的 S9 证据是在 60 epochs 截断训练下取得的（Table S9 已注明）。
为回答「KIBA/Davis 的变化幅度是否与 ChEMBL 一致」，三数据集必须同口径，
故 E2 默认 --epochs 60（batch/ablation 仍沿用各数据集论文配方）；
ChEMBL 直接复用 supplement_output/lambda_sensitivity/lambda_results.csv，
最后合并成 e2_lambda_ofat_merged.csv 并输出幅度/方向一致性结论。

用法:
  python supplement_e2_lambda_ofat.py --dry_run
  python supplement_e2_lambda_ofat.py                      # KIBA + Davis, 60 epochs
  python supplement_e2_lambda_ofat.py --datasets kiba
  python supplement_e2_lambda_ofat.py --epochs 220        # 全量口径（ChEMBL 需同步重跑）
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supplement_exp_common import LAMBDA_RUNS, stage_dir, run_stock, metric_row, OUTROOT  # noqa

S9_CHEMBL = OUTROOT.parent / 'lambda_sensitivity' / 'lambda_results.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['kiba', 'davis'])
    ap.add_argument('--epochs', type=int, default=60, help='与 ChEMBL S9 对齐的截断 epoch')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    outroot = stage_dir('e2_lambda_ofat')
    csv_path = outroot / 'e2_results.csv'
    rows = []
    for dataset in args.datasets:
        for tag, l1, l2 in LAMBDA_RUNS:
            run_dir = outroot / dataset / f'run_{tag}'
            if (run_dir / 'final_results.json').exists():
                print(f'[skip] {dataset}/{tag}')
            else:
                rc = run_stock(dataset, run_dir, epochs=args.epochs, inner_steps=3,
                               lambda1=l1, lambda2=l2, gpu=args.gpu, dry_run=args.dry_run,
                               log_suffix=f'E2 {dataset} {tag}')
                if rc != 0:
                    print(f'[fail] {dataset}/{tag} rc={rc}')
            row = metric_row('e2', run_dir, dataset=dataset, tag=tag,
                             lambda1=l1, lambda2=l2, epochs=args.epochs)
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')

    # ===== 合并 ChEMBL S9 =====
    merged_path = outroot / 'e2_lambda_ofat_merged.csv'
    fresh = pd.DataFrame(rows)
    if args.dry_run:
        print(f'\n[dry_run] 计划 {len(rows)} 个 run，跳过训练与汇总。正式运行后产出: {merged_path}')
        return
    # 只有真正跑出指标的 run 才进入合并（失败/MISSING 不污染汇总）
    fresh = fresh[fresh.get('status', pd.Series(dtype=object)) == 'ok'].copy()
    frames = [fresh]
    if S9_CHEMBL.exists():
        ch = pd.read_csv(S9_CHEMBL)
        ch = ch[ch.get('dataset', pd.Series(['chembl'] * len(ch))) == 'chembl'].copy()
        if 'epochs' in ch.columns:
            ch = ch[ch['epochs'].astype(str) == str(args.epochs)]
        ch = ch.rename(columns={'EF1': 'EF@1%'})
        ch['dataset'] = 'chembl'
        keep = ['dataset', 'tag', 'lambda1', 'lambda2', 'epochs', 'R2', 'EF@1%', 'ECE']
        frames.append(ch[[c for c in keep if c in ch.columns]])
    else:
        print(f'[warn] 未找到 ChEMBL S9: {S9_CHEMBL}（合并表将缺 ChEMBL 列）')

    merged = pd.concat([f for f in frames if not f.empty], ignore_index=True, sort=False)
    merged.to_csv(merged_path, index=False, encoding='utf-8-sig')

    # ===== 幅度与方向一致性 =====
    summary_path = outroot / 'e2_amplitude_summary.csv'
    summ = []
    required = {'tag', 'R2', 'EF@1%', 'ECE'}
    if not required.issubset(merged.columns):
        print(f'[warn] 合并表缺少指标列（可能所有 run 都未完成），跳过幅度汇总')
        print(f'\n合并表: {merged_path}')
        return
    for dataset, g in merged.groupby('dataset'):
        b = g[g['tag'] == 'base_l1_0.050_l2_0.10']
        if b.empty:
            continue
        b = b.iloc[0]
        if pd.isna(b['R2']) or pd.isna(b['EF@1%']) or pd.isna(b['ECE']):
            continue
        for _, r in g[g['tag'] != 'base_l1_0.050_l2_0.10'].iterrows():
            if pd.isna(r['R2']) or pd.isna(r['EF@1%']) or pd.isna(r['ECE']):
                continue
            summ.append({
                'dataset': dataset, 'tag': r['tag'],
                'dR2': r['R2'] - b['R2'],
                'dEF1': r['EF@1%'] - b['EF@1%'],
                'dECE': r['ECE'] - b['ECE'],
            })
    sdf = pd.DataFrame(summ)
    if not sdf.empty:
        # 方向一致性：每个扰动在各数据集上 Δ 的符号是否相同
        sign_tbl = sdf.groupby('tag').agg(
            sign_R2=('dR2', lambda x: '/'.join('+' if v >= 0 else '−' for v in x)),
            sign_EF1=('dEF1', lambda x: '/'.join('+' if v >= 0 else '−' for v in x)),
            mean_abs_dR2=('dR2', lambda x: x.abs().mean()),
            max_abs_dR2=('dR2', lambda x: x.abs().max()),
            mean_abs_dEF1=('dEF1', lambda x: x.abs().mean()),
        ).reset_index()
        sdf.to_csv(summary_path, index=False, encoding='utf-8-sig')
        sign_tbl.to_csv(outroot / 'e2_sign_agreement.csv', index=False, encoding='utf-8-sig')
        print('\n===== E2 各扰动相对 base 的变化 =====')
        print(sdf.to_string(index=False))
        print('\n===== 跨数据集方向/幅度 =====')
        print(sign_tbl.to_string(index=False))
    print(f'\n合并表: {merged_path}')


if __name__ == '__main__':
    main()
