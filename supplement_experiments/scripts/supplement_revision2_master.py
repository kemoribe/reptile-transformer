# -*- coding: utf-8 -*-
"""
Revision-2 补充实验总队列（单卡顺序执行，避免多进程抢 GPU）
=============================================================
顺序（短任务优先，全部可断点续跑——已完成的 run 自动跳过）：
  features → E3(梯度夹角) → E1(inner_steps) → E2(λ OFAT KIBA/Davis)
           → E4(ESM2 frozen/partial/full, KIBA) → E5(5 seeds × N datasets) → E5统计

用法（4090 服务器）:
  # 先全部烟雾测试（每个训练 2 epochs，约 10-15 分钟验证全链路）
  python supplement_revision2_master.py --smoke

  # 正式：先跑 chembl/davis/kiba（约 6-9 小时）
  python supplement_revision2_master.py --datasets chembl davis kiba

  # BindingDB 五轮（约 70 小时）单独排队
  python supplement_revision2_master.py --stages e5 e5stats --datasets bindingdb

GPU 利用率：AMP/FP16 + TF32 + cudnn.benchmark + KIBA batch2048 + ESM2 GPU预计算
+ 单卡顺序执行（无上下文切换/显存竞争）。多卡可分别开两个终端用 --gpu 0/1 跑不同 stage。
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from supplement_exp_common import ensure_features, RECIPES  # noqa


def call(desc, cmd):
    print('\n' + '#' * 90)
    print(f'# STAGE: {desc}')
    print('# CMD :', ' '.join(cmd))
    print('#' * 90)
    t0 = time.time()
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    r = subprocess.run(cmd, cwd=str(BASE), env=env)
    dt = (time.time() - t0) / 60
    print(f'# {desc} 结束 rc={r.returncode}，用时 {dt:.1f} 分钟')
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', default='features,e3,e1,e2,e4,e5,e5stats')
    ap.add_argument('--datasets', nargs='+', default=['chembl', 'davis', 'kiba'])
    ap.add_argument('--e2_datasets', nargs='+', default=['kiba', 'davis'])
    ap.add_argument('--e2_epochs', type=int, default=60)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    py = sys.executable
    ep = 2 if args.smoke else None

    # 烟雾产物全部写入 *_smoke 目录，与正式运行物理隔离
    if args.smoke:
        os.environ['REV2_STAGE_SUFFIX'] = '_smoke'
        print('⚠️ SMOKE 模式：所有输出目录带 _smoke 后缀，每个训练仅 2 epochs')

    # 依赖的数据集：E5 数据集 + e1/e3/e4 用 kiba + e2 用其数据集
    needed = set(args.datasets) | {'kiba'} | set(args.e2_datasets)
    if 'features' in stages:
        for ds in sorted(needed):
            ensure_features(ds, gpu=args.gpu)

    if 'e3' in stages:
        cmd = [py, 'supplement_e3_grad_conflict.py', '--gpu', args.gpu]
        if args.smoke:
            cmd += ['--epochs', '2', '--diag_epochs', '1,2']
        call('E3 梯度两两余弦 (KIBA)', cmd)

    if 'e1' in stages:
        cmd = [py, 'supplement_e1_inner_steps.py', '--gpu', args.gpu]
        if ep:
            cmd += ['--epochs', str(ep)]
        call('E1 inner_steps 1/3/5/10 (KIBA)', cmd)

    if 'e2' in stages:
        cmd = [py, 'supplement_e2_lambda_ofat.py', '--datasets', *args.e2_datasets,
               '--epochs', str(2 if args.smoke else args.e2_epochs), '--gpu', args.gpu]
        call('E2 λ OFAT (KIBA/Davis)', cmd)

    if 'e4' in stages:
        for mode in ('frozen', 'partial', 'full'):
            cmd = [py, 'supplement_e4_esm2_freeze.py', '--mode', mode, '--gpu', args.gpu]
            if args.smoke:
                cmd += ['--smoke']
            call(f'E4 ESM2 {mode} (KIBA)', cmd)

    if 'e5' in stages:
        cmd = [py, 'supplement_e5_repeats.py', '--datasets', *args.datasets, '--gpu', args.gpu]
        if ep:
            cmd += ['--epochs', str(ep)]
        call(f'E5 5-seed repeats ({",".join(args.datasets)})', cmd)

    if 'e5stats' in stages:
        call('E5 Bootstrap + Wilcoxon', [py, 'supplement_e5_stats.py'])

    print('\n全部队列结束。产物根目录: supplement_output/revision2_experiments/')


if __name__ == '__main__':
    main()
