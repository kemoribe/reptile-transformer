# -*- coding: utf-8 -*-
"""
Table S9: λ 敏感性分析（Reptile+Transformer 主模型, OFAT 单因素法）
  λ1 = CONTRASTIVE_WEIGHT (基准0.05)   reptile_training.py:96
  λ2 = CONSISTENCY_WEIGHT  (基准0.10)  reptile_training.py:97
共 5 次训练: 基准 + λ1∈{0.025,0.1} + λ2∈{0.05,0.2}
每次为一次完整训练（ChEMBL 200 epochs 可能 3-6 小时/次），建议:
  1) 先 --dry_run 看计划
  2) 快速筛选用 --epochs 60（相对趋势可比，论文表注明截断epoch）
  3) 过夜跑全量: 不加 --epochs
用法:
  python supplement_s9_lambda_sensitivity.py --dry_run
  python supplement_s9_lambda_sensitivity.py --epochs 60
  python supplement_s9_lambda_sensitivity.py --dataset davis --epochs 60
输出: supplement_output/lambda_sensitivity/lambda_results.csv
"""
import sys, os, json, subprocess
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r'd:\lht')
OUTDIR = ROOT / 'supplement_output' / 'lambda_sensitivity'
OUTDIR.mkdir(parents=True, exist_ok=True)
PY = str(ROOT / '.venv' / 'Scripts' / 'python.exe')
ENTRY = ROOT / 'run_reptile_transformer.py'

DATA_DIR = {'chembl': '3_all_data_chembl_targets_preprocessed',
            'davis': '3_davis_preprocessed',
            'kiba': '3_kiba_preprocessed',
            'bindingdb': 'bindingdb_preprocessed'}

L1_BASE, L2_BASE = 0.05, 0.10
# (tag, λ1, λ2)
RUNS = [('base_l1_0.050_l2_0.10', 0.05, 0.10),
        ('l1_low_0.025', 0.025, 0.10),
        ('l1_high_0.100', 0.10, 0.10),
        ('l2_low_0.05', 0.05, 0.05),
        ('l2_high_0.20', 0.05, 0.20)]


def find_metrics(outdir):
    """从输出目录 final_results.json 提取 R2 / EF@1% / ECE（容错多种结构）"""
    best = {}
    for jf in sorted(outdir.rglob('*.json'), key=lambda p: ('final' not in p.name, p.name)):
        try:
            obj = json.loads(jf.read_text(encoding='utf-8'))
        except Exception:
            continue
        flat = {}
        def walk(o, pre=''):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f'{pre}{k}.')
            elif isinstance(o, (int, float)):
                flat[pre[:-1]] = o
        walk(obj)
        for k, v in flat.items():
            kl = k.lower()
            if ('r2' in kl) and 'mse' not in kl and 'R2' not in best:
                best['R2'] = v
            if ('ef@1' in kl or 'ef1' in kl) and 'EF1' not in best:
                best['EF1'] = v
            if 'ece' in kl and 'ECE' not in best:
                best['ECE'] = v
        if {'R2', 'EF1', 'ECE'} <= best.keys():
            break
    return best


def main():
    dataset = 'chembl'
    if '--dataset' in sys.argv:
        dataset = sys.argv[sys.argv.index('--dataset') + 1]
    epochs = None
    if '--epochs' in sys.argv:
        epochs = int(sys.argv[sys.argv.index('--epochs') + 1])
    dry = '--dry_run' in sys.argv

    print(f'数据集: {dataset} | epochs: {epochs or "脚本默认(200)"} | 共 {len(RUNS)} 次训练')
    print('方式: 原地运行 run_reptile_transformer.py, λ 经 REPTILE_LAMBDA1/2 环境变量注入\n')
    for tag, l1, l2 in RUNS:
        print(f'- {tag}: λ1={l1} λ2={l2}')
    if dry:
        print('\n[dry_run] 未执行训练。')
        return

    import pandas as pd
    results_csv = OUTDIR / 'lambda_results.csv'
    done = {}
    if results_csv.exists():
        df0 = pd.read_csv(results_csv)
        done = dict(zip(df0['tag'], df0.to_dict('records')))

    all_rows = [] if not results_csv.exists() else pd.read_csv(results_csv).to_dict('records')
    data_path = str(ROOT / DATA_DIR[dataset])
    # 特征与 λ 无关：只预计算一次（base 组），其余组用 NTFS 硬链接复用同一个 npz
    feat_cache = OUTDIR / 'run_base_l1_0.050_l2_0.10' / 'output' / 'precomputed_features.npz'
    for tag, l1, l2 in RUNS:
        if tag in done:
            print(f'[skip] {tag}')
            continue
        outdir = OUTDIR / f'run_{tag}' / 'output'
        outdir.mkdir(parents=True, exist_ok=True)
        feat_dst = outdir / 'precomputed_features.npz'
        # 完成判据：output/final_results.json
        if (outdir / 'final_results.json').exists():
            met = find_metrics(outdir)
            if {'R2', 'EF1', 'ECE'} <= met.keys():
                row = {'tag': tag, 'lambda1': l1, 'lambda2': l2, 'dataset': dataset,
                       'epochs': epochs or 'default', **met}
                all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(results_csv, index=False, encoding='utf-8-sig')
                print(f'[done-existing] {tag}: {met}')
                continue
        # 特征复用：缓存已生成且本组缺 npz 时，建硬链接（同盘秒成，不占额外空间）
        reuse_feat = False
        if tag != 'base_l1_0.050_l2_0.10' and feat_cache.exists():
            if feat_dst.exists() and feat_dst.stat().st_size != feat_cache.stat().st_size:
                feat_dst.unlink()
            if not feat_dst.exists():
                os.link(feat_cache, feat_dst)
                print(f'  [feat] 硬链接复用特征: {feat_dst.name} ({feat_cache.stat().st_size/1e9:.2f} GB)')
            reuse_feat = feat_dst.exists()
        cmd = [PY, str(ENTRY),
               '--data_dir', data_path,
               '--output_dir', str(outdir),
               '--gpu', '0']
        if epochs:
            cmd += ['--epochs', str(epochs)]
        if reuse_feat:
            cmd += ['--no_precompute']
        # 关键: PREPROCESSED_DIR 在 import data_preprocessing 之前生效；ESM2 走 GPU
        env = dict(os.environ)
        env['REPTILE_LAMBDA1'] = str(l1)
        env['REPTILE_LAMBDA2'] = str(l2)
        env['PREPROCESSED_DIR'] = data_path
        env['ESM2_USE_GPU'] = '1'
        print(f'\n[run] {tag} (λ1={l1}, λ2={l2}, GPU特征={"复用" if reuse_feat else "预计算"})\n  {cmd}')
        r = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if r.returncode != 0:
            print(f'[fail] {tag} 退出码 {r.returncode}，跳过汇总（可重跑，已完成的组会自动跳过）')
            continue
        met = find_metrics(outdir)
        row = {'tag': tag, 'lambda1': l1, 'lambda2': l2, 'dataset': dataset,
               'epochs': epochs or 'default', **met}
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(results_csv, index=False, encoding='utf-8-sig')
        print(f'[done] {tag}: {met}')

    print('\n== 汇总 ==')
    if all_rows:
        print(pd.DataFrame(all_rows).to_string(index=False))
    print(f'\n下一步: python supplement_s5_lambda_heatmap.py  生成 Figure S5')


if __name__ == '__main__':
    main()
