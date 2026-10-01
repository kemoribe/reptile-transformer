# -*- coding: utf-8 -*-
"""
E5 — 主模型 5 次独立重复（种子 11/22/33/44/55）× 4 数据集
=========================================================
每数据集沿用其论文配方（见 supplement_exp_common.RECIPES），仅随机种子不同。
特征 npz 每数据集 canonical 预计算一次、硬链接复用，保证 5 次重复输入完全一致。
完成后运行: python supplement_e5_stats.py  （Bootstrap CI + 配对 Wilcoxon）

用法:
  python supplement_e5_repeats.py --dry_run
  python supplement_e5_repeats.py --datasets chembl davis kiba      # 先跑 3 个
  python supplement_e5_repeats.py --datasets bindingdb              # 4090 服务器排队
  python supplement_e5_repeats.py --seeds 11 22 --epochs 60 --smoke_epochs  # 自定义
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supplement_exp_common import SEEDS, DATASETS, stage_dir, run_stock, metric_row  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--epochs', type=int, default=None, help='默认用各数据集论文配方 epoch')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    outroot = stage_dir('e5_repeats')
    csv_path = outroot / 'e5_repeats_metrics.csv'
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            run_dir = outroot / dataset / f'seed_{seed}'
            if (run_dir / 'final_results.json').exists():
                print(f'[skip] {dataset}/seed{seed} 已完成')
            else:
                rc = run_stock(dataset, run_dir, epochs=args.epochs, inner_steps=3,
                               seed=seed, gpu=args.gpu, dry_run=args.dry_run,
                               log_suffix=f'E5 {dataset} seed{seed}')
                if rc != 0:
                    print(f'[fail] {dataset}/seed{seed} rc={rc}')
            row = metric_row('e5', run_dir, dataset=dataset, seed=seed)
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')

    df = pd.DataFrame(rows)
    if not df.empty:
        print('\n===== E5 重复结果 =====')
        cols = ['dataset', 'seed', 'R2', 'EF@1%', 'ECE', 'RMSE', 'walltime_min', 'status']
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
        ok = df[df['status'] == 'ok']
        if not ok.empty:
            summ = ok.groupby('dataset')[['R2', 'EF@1%', 'ECE', 'RMSE']].agg(['mean', 'std'])
            print('\n===== mean ± std =====')
            print(summ.to_string(float_format=lambda x: f'{x:.4f}'))
    print(f'\nCSV: {csv_path}')
    print('下一步: python supplement_e5_stats.py')


if __name__ == '__main__':
    main()
