# -*- coding: utf-8 -*-
"""
E3 — 四个损失（MSE / ListNet排序 / InfoNCE对比 / 一致性）梯度两两夹角诊断
=========================================================================
原理：reptile_training.py 中环境变量门控的 hook（REPTILE_GRAD_DIAG=1）在
指定 epoch 的前 8 个训练任务、每个任务第 0 个内步，对 4 个损失分别调用
torch.autograd.grad，再用 F.cosine_similarity 计算 6 对余弦（cos∈[-1,1]）。
不改变任何数值结果（额外 retain_graph 的只读求导，正常反传照旧）。

判读：cos>0.3 协同；|cos|≤0.3 近似正交（互不打架）；cos<−0.3 冲突。

用法:
  python supplement_e3_grad_conflict.py --dry_run
  python supplement_e3_grad_conflict.py                 # KIBA 30 epochs
  python supplement_e3_grad_conflict.py --epochs 10 --diag_epochs 1,2,3,5,10
输出: e3_grad_conflict/{grad_conflict_raw.csv, e3_pairwise_summary.csv}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supplement_exp_common import stage_dir, run_stock  # noqa

PAIRS = ['cos_MSE_Rank', 'cos_MSE_Contrast', 'cos_MSE_Consist',
         'cos_Rank_Contrast', 'cos_Rank_Consist', 'cos_Contrast_Consist']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='kiba')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--diag_epochs', default='1,2,3,10,20,30')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    outroot = stage_dir('e3_grad_conflict')
    run_dir = outroot / args.dataset
    raw_csv = run_dir / 'grad_conflict.csv'

    if not raw_csv.exists():
        rc = run_stock(args.dataset, run_dir, epochs=args.epochs, inner_steps=3,
                       grad_diag=True, grad_diag_epochs=args.diag_epochs,
                       gpu=args.gpu, dry_run=args.dry_run, log_suffix='E3 grad-conflict')
        if args.dry_run or rc != 0:
            print('[dry/fail] 结束')
            return
    else:
        print('[skip] 诊断记录已存在')

    df = pd.read_csv(raw_csv)
    # 每个 epoch 8 个任务先取均值，再跨 epoch 取均值（避免任务数不等造成加权偏差）
    per_epoch = df.groupby('epoch')[PAIRS].mean().reset_index()
    per_epoch.to_csv(outroot / 'e3_per_epoch_mean.csv', index=False, encoding='utf-8-sig')

    overall = per_epoch[PAIRS].agg(['mean', 'std', 'min']).T.reset_index().rename(
        columns={'index': 'pair'})

    def verdict(c):
        if c < -0.3:
            return '冲突(<−0.3)'
        if c <= 0.3:
            return '近似正交'
        return '协同(>0.3)'

    overall['verdict'] = overall['mean'].map(verdict)
    overall.to_csv(outroot / 'e3_pairwise_summary.csv', index=False, encoding='utf-8-sig')

    print('\n===== E3 梯度两两余弦（跨 epoch 均值 ± std） =====')
    print(overall.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\n===== 逐 epoch 均值 =====')
    print(per_epoch.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f'\n输出目录: {outroot}')


if __name__ == '__main__':
    main()
