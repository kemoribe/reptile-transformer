# -*- coding: utf-8 -*-
"""
E1 — inner_steps 消融（KIBA，1/3/5/10）
=========================================
配方固定为 KIBA 论文配方：morgan_descriptors(屏蔽MACCS) + batch 2048 + 220 epochs；
仅改变 inner_steps。基线 inner_steps=3（对应 Table 4 R²=0.3222）。

用法:
  python supplement_e1_inner_steps.py --dry_run
  python supplement_e1_inner_steps.py                 # 全量 220 epochs
  python supplement_e1_inner_steps.py --epochs 60     # 快速筛查（表注注明截断 epoch）
  python supplement_e1_inner_steps.py --steps 1 3     # 只跑指定档位
输出: supplement_output/revision2_experiments/e1_inner_steps/e1_results.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supplement_exp_common import (INNER_STEPS_GRID, stage_dir, run_stock, metric_row)  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='kiba')
    ap.add_argument('--epochs', type=int, default=None, help='默认使用数据集论文配方 epoch 数')
    ap.add_argument('--steps', type=int, nargs='+', default=INNER_STEPS_GRID)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    outroot = stage_dir('e1_inner_steps')
    csv_path = outroot / 'e1_results.csv'
    rows = []
    for steps in args.steps:
        tag = f'steps_{steps}'
        run_dir = outroot / args.dataset / tag
        if (run_dir / 'final_results.json').exists():
            print(f'[skip] {tag} 已完成')
        else:
            rc = run_stock(args.dataset, run_dir, epochs=args.epochs, inner_steps=steps,
                           gpu=args.gpu, dry_run=args.dry_run, log_suffix=f'E1 {tag}')
            if rc != 0:
                print(f'[fail] {tag} rc={rc}，继续后续档位')
        row = metric_row('e1', run_dir, dataset=args.dataset, inner_steps=steps,
                         is_baseline=(steps == 3))
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')

    df = pd.DataFrame(rows)
    if not df.empty and (df['status'] == 'ok').any():
        base = df[df['inner_steps'] == 3]
        if not base.empty and base.iloc[0]['status'] == 'ok':
            b = base.iloc[0]
            for k in ['R2', 'EF@1%', 'ECE', 'RMSE']:
                df[f'd_{k}'] = df[k] - b[k]
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print('\n===== E1 汇总 =====')
        cols = ['inner_steps', 'R2', 'd_R2', 'EF@1%', 'd_EF@1%', 'ECE', 'd_ECE', 'walltime_min', 'status']
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f'\nCSV: {csv_path}')


if __name__ == '__main__':
    main()
