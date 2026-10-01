# -*- coding: utf-8 -*-
"""
Revision-2 补充实验公共模块（E1–E5 共用）
=====================================================
单一事实源：每个数据集沿用其「论文已报告数字」对应的训练配方（用户 2026-09-19 确认）：
  chembl    : 全特征(含MACCS), batch=512,  epochs=200  (run_reptile_transformer.py 默认)
  davis     : 全特征(含MACCS), batch=512,  epochs=200
  kiba      : morgan_descriptors(屏蔽MACCS), batch=2048, epochs=220  (复现 Table4 R²=0.3222)
  bindingdb : 全特征(含MACCS), batch=512,  epochs=200

所有特征 npz 在每个数据集上只预计算一次（canonical cache），各 run 目录硬链接复用，
保证 E1–E5 输入特征逐字节一致、且不会误用历史上口径不明的 npz。
"""
import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
OUTROOT = ROOT / 'supplement_output' / 'revision2_experiments'
FEATROOT = OUTROOT / 'features'

ENTRY = ROOT / 'run_reptile_transformer.py'

# 每数据集论文配方
RECIPES = {
    'chembl':    dict(data_dir='3_all_data_chembl_targets_preprocessed', ablation='none',                batch=512,  epochs=200),
    'davis':     dict(data_dir='3_davis_preprocessed',                   ablation='none',                batch=512,  epochs=200),
    'kiba':      dict(data_dir='3_kiba_preprocessed',                    ablation='morgan_descriptors',  batch=2048, epochs=220),
    'bindingdb': dict(data_dir='bindingdb_preprocessed',                 ablation='none',                batch=512,  epochs=200),
}
DATASETS = ['chembl', 'davis', 'kiba', 'bindingdb']

# λ 基准与 OFAT 扰动（与 Table S9 完全一致）
L1_BASE, L2_BASE = 0.05, 0.10
LAMBDA_RUNS = [('base_l1_0.050_l2_0.10', 0.05, 0.10),
               ('l1_low_0.025', 0.025, 0.10),
               ('l1_high_0.100', 0.10, 0.10),
               ('l2_low_0.05', 0.05, 0.05),
               ('l2_high_0.20', 0.05, 0.20)]

SEEDS = [11, 22, 33, 44, 55]
INNER_STEPS_GRID = [1, 3, 5, 10]

METRIC_KEYS = ['R2', 'RMSE', 'MAE', 'Pearson', 'Spearman',
               'EF@1%', 'EF@5%', 'EF@10%', 'ECE', 'AUPR']


def data_path(dataset):
    return str(ROOT / RECIPES[dataset]['data_dir'])


def feat_cache_dir(dataset):
    d = FEATROOT / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def link_npz(src: Path, dst: Path):
    """同盘硬链接优先（秒成、零额外空间）；跨盘回退拷贝"""
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return 'exists'
        dst.unlink()
    try:
        os.link(src, dst)
        return 'hardlink'
    except OSError:
        shutil.copyfile(src, dst)
        return 'copy'


def ensure_features(dataset, gpu='0'):
    """返回该数据集 canonical 预计算特征 npz 路径；不存在则调用主脚本 --precompute_only 生成。"""
    cache_dir = feat_cache_dir(dataset)
    npz = cache_dir / 'precomputed_features.npz'
    if npz.exists() and npz.stat().st_size > 1_000_000:
        return npz
    print(f'[features] {dataset}: canonical 特征不存在，开始预计算（ESM2 走 GPU）...')
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['PREPROCESSED_DIR'] = data_path(dataset)
    env['ESM2_USE_GPU'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = gpu
    cmd = [sys.executable, str(ENTRY),
           '--data_dir', data_path(dataset),
           '--output_dir', str(cache_dir),
           '--precompute_only', '--gpu', gpu]
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0 or not npz.exists():
        raise RuntimeError(f'{dataset} 特征预计算失败 (returncode={r.returncode})')
    print(f'[features] {dataset}: 预计算完成 {npz.stat().st_size/1e9:.2f} GB')
    return npz


def stage_dir(stage):
    # master --smoke 时通过 REV2_STAGE_SUFFIX=_smoke 隔离烟雾产物，避免污染正式断点
    name = f'{stage}{os.environ.get("REV2_STAGE_SUFFIX", "")}'
    d = OUTROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_stock(dataset, outdir: Path, *, epochs=None, inner_steps=3, seed=None,
              lambda1=None, lambda2=None, grad_diag=False, grad_diag_epochs=None,
              gpu='0', dry_run=False, log_suffix=''):
    """调用 run_reptile_transformer.py（配方由 dataset 决定，逐字节复用 canonical npz）。"""
    recipe = RECIPES[dataset]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_dst = outdir / 'precomputed_features.npz'
    feat_npz = ensure_features(dataset, gpu=gpu)
    link_npz(feat_npz, npz_dst)

    ep = epochs if epochs is not None else recipe['epochs']
    cmd = [sys.executable, str(ENTRY),
           '--data_dir', data_path(dataset),
           '--output_dir', str(outdir),
           '--no_precompute',
           '--batch_size', str(recipe['batch']),
           '--epochs', str(ep),
           '--inner_steps', str(inner_steps),
           '--ablation', recipe['ablation'],
           '--gpu', gpu]
    if seed is not None:
        cmd += ['--seed', str(seed)]

    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['PREPROCESSED_DIR'] = data_path(dataset)
    env['ESM2_USE_GPU'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = gpu
    if lambda1 is not None:
        env['REPTILE_LAMBDA1'] = str(lambda1)
    if lambda2 is not None:
        env['REPTILE_LAMBDA2'] = str(lambda2)
    if grad_diag:
        env['REPTILE_GRAD_DIAG'] = '1'
        env['REPTILE_GRAD_DIAG_EPOCHS'] = grad_diag_epochs or '1,2,3,10,20'

    print('\n' + '=' * 80)
    print(f'[run] {log_suffix or outdir.name} | dataset={dataset} | epochs={ep} | '
          f'inner_steps={inner_steps} | batch={recipe["batch"]} | ablation={recipe["ablation"]} | '
          f'seed={seed} | λ1={lambda1} λ2={lambda2}')
    print('  ', ' '.join(cmd))
    print('=' * 80)
    if dry_run:
        return 0
    t0 = time.time()
    log_path = outdir / 'console.log'
    with open(log_path, 'a', encoding='utf-8') as lf:
        lf.write(f'\n===== launch {time.strftime("%Y-%m-%d %H:%M:%S")} =====\n')
        lf.write(' '.join(cmd) + '\n')
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    (outdir / 'walltime_sec.txt').write_text(str(elapsed), encoding='utf-8')
    print(f'[done] returncode={r.returncode} walltime={elapsed/60:.1f} min')
    return r.returncode


def _to_float(v):
    """final_results.json 中指标可能被存成字符串，统一转 float；无法转换则保留原值"""
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def read_metrics(outdir: Path):
    """从 final_results.json 提取 test_metrics（含配置）。"""
    fp = Path(outdir) / 'final_results.json'
    if not fp.exists():
        return None
    d = json.loads(fp.read_text(encoding='utf-8'))
    tm = d.get('test_metrics', {})
    out = {k: _to_float(tm.get(k)) for k in METRIC_KEYS}
    cfg = d.get('config', {})
    out['_config'] = cfg
    out['_best_val_r2'] = d.get('best_val_r2')
    wt = outdir / 'walltime_sec.txt'
    if wt.exists():
        out['_walltime_min'] = float(wt.read_text().strip()) / 60
    return out


def metric_row(prefix, outdir: Path, **extra):
    m = read_metrics(outdir)
    row = dict(extra)
    if m is None:
        row['status'] = 'MISSING'
        return row
    row.update({k: m[k] for k in METRIC_KEYS})
    row['status'] = 'ok'
    row['walltime_min'] = round(m.get('_walltime_min', float('nan')), 1)
    return row
