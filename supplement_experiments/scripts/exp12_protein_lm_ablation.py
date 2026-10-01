# -*- coding: utf-8 -*-
"""
实验 12：蛋白质语言模型消融 — ESM2(35M, 论文基线) vs ESM-1b(650M) vs ProtT5-XL(3B)
================================================================================
协议（与 data_preprocessing.ProteinFeatureExtractor 完全一致）：
  - 序列清洗：仅保留 20 种标准氨基酸；>1022 截断
  - 冻结 PLM，torch.no_grad() 推理，mean-pooling 取序列表征
  - 冻结随机投影 Linear(in_dim -> 128)（种子固定 0，保证两变体可复现；
    原 ESM2 管线的投影同样是随机初始化未训练层，协议一致）
  - 仅替换 canonical npz 的 protein 列，其余特征/标签/划分逐字节不变
训练：KIBA(morgan_descriptors, batch=2048, epochs=220) 与 Davis(batch=512, epochs=200)
  各变体 × seeds {11,22,33}，与 ESM2 基线（E5 repeats 同种子子集）配对比较。

用法：
  python exp12_protein_lm_ablation.py --stage extract          # 提取特征（GPU）
  python exp12_protein_lm_ablation.py --stage train            # 训练全部变体（GPU，可断点续跑）
  python exp12_protein_lm_ablation.py --stage report           # 汇总报告
"""
import argparse
import json
import os

# huggingface.co 直连 SSL 校验失败，走镜像（须在 import transformers 之前设置）
# hf-mirror 不支持 xet 传输协议，必须禁用否则大文件卡死在 0 字节
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r'd:\lht')
OUT = ROOT / '修改supplement_output' / 'remaining_experiments' / 'exp12_lm_ablation'
FEATROOT = ROOT / '修改supplement_output' / 'revision2_experiments' / 'features'
ENTRY = ROOT / 'run_reptile_transformer.py'
E5_CSV = ROOT / '修改supplement_output' / 'revision2_experiments' / 'e5_repeats' / 'e5_repeats_metrics.csv'
OUT.mkdir(parents=True, exist_ok=True)

DATA_DIRS = {'kiba': ROOT / '3_kiba_preprocessed', 'davis': ROOT / '3_davis_preprocessed'}
RECIPES = {'kiba': dict(ablation='morgan_descriptors', batch=2048, epochs=220),
           'davis': dict(ablation='none', batch=512, epochs=200)}
SEEDS = [11, 22, 33]
AA = set('ACDEFGHIKLMNPQRSTVWY')


# ----------------------------------------------------------------------
# 序列读取（与 data_preprocessing.load_target_data 相同的清洗规则）
# ----------------------------------------------------------------------
def clean_seq(seq: str) -> str:
    seq = ''.join(c for c in seq if c in AA)
    return seq[:1022]


def find_sequences(dataset: str, target_names):
    root = DATA_DIRS[dataset]
    # 建立 target_name -> 序列文件 的索引
    seq_map = {}
    if dataset == 'kiba':
        for p in root.rglob('*_processed_protein_sequence.txt'):
            name = p.name.replace('_processed_protein_sequence.txt', '')
            seq_map[name] = p
    else:  # davis: {TARGET}/sequence.fasta
        for p in root.rglob('sequence.fasta'):
            seq_map[p.parent.name] = p
    out = {}
    missing = []
    for t in target_names:
        p = seq_map.get(t)
        if p is None:
            missing.append(t)
            continue
        raw = p.read_text(encoding='utf-8', errors='ignore')
        seq = ''.join(l.strip() for l in raw.split('\n') if not l.startswith('>'))
        out[t] = clean_seq(seq)
    return out, missing


# ----------------------------------------------------------------------
# PLM 特征提取
# ----------------------------------------------------------------------
@__import__('torch').no_grad()
def extract_esm1b(seqs: dict, batch_size=8):
    """ESM-1b t33 650M，最后一层(33) mean-pool -> 1280 维"""
    import torch
    import esm
    print('[esm1b] 加载模型...')
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model = model.cuda().eval()
    bc = alphabet.get_batch_converter()
    LAYER = 33
    names = sorted(seqs, key=lambda k: len(seqs[k]))
    emb = {}
    for i in range(0, len(names), batch_size):
        chunk = names[i:i + batch_size]
        data = [(n, seqs[n]) for n in chunk]
        _, _, toks = bc(data)
        toks = toks.cuda()
        res = model(toks, repr_layers=[LAYER])
        rep = res['representations'][LAYER]
        for j, n in enumerate(chunk):
            L = len(seqs[n])
            emb[n] = rep[j, 1:L + 1].mean(0).float().cpu().numpy()
        if (i // batch_size) % 10 == 0:
            print(f'  [esm1b] {i + len(chunk)}/{len(names)}')
    del model
    __import__('torch').cuda.empty_cache()
    return emb, 1280


@__import__('torch').no_grad()
def extract_prott5(seqs: dict, batch_size=4):
    """ProtT5-XL-UniRef50 encoder (fp16)，last_hidden_state mean-pool -> 1024 维"""
    import torch
    from transformers import T5EncoderModel, T5Tokenizer
    print('[prott5] 加载模型 Rostlab/prot_t5_xl_half_uniref50-enc ...')
    tok = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc',
                                           torch_dtype=torch.float16).cuda().eval()
    names = sorted(seqs, key=lambda k: len(seqs[k]))
    emb = {}
    for i in range(0, len(names), batch_size):
        chunk = names[i:i + batch_size]
        texts = [' '.join(seqs[n]) for n in chunk]
        enc = tok(texts, add_special_tokens=True, padding='longest', return_tensors='pt')
        input_ids = enc['input_ids'].cuda()
        attn = enc['attention_mask'].cuda()
        out = model(input_ids=input_ids, attention_mask=attn).last_hidden_state
        for j, n in enumerate(chunk):
            L = int(attn[j].sum().item()) - 1  # 去掉 EOS
            emb[n] = out[j, :L].float().mean(0).cpu().numpy()
        if (i // batch_size) % 20 == 0:
            print(f'  [prott5] {i + len(chunk)}/{len(names)}')
    del model
    __import__('torch').cuda.empty_cache()
    return emb, 1024


def project_to_128(emb: dict, in_dim: int, seed=0):
    """冻结随机投影（与基线 ESM2 管线同协议），种子固定保证可复现"""
    import torch
    g = torch.Generator().manual_seed(seed)
    W = torch.empty(128, in_dim)
    b = torch.empty(128)
    # 与 nn.Linear 默认初始化一致：U(-1/sqrt(in), 1/sqrt(in))
    bound = 1.0 / np.sqrt(in_dim)
    W.uniform_(-bound, bound, generator=g)
    b.uniform_(-bound, bound, generator=g)
    return {k: (W @ torch.from_numpy(np.asarray(v, dtype=np.float32)) + b).numpy().astype(np.float32)
            for k, v in emb.items()}


def build_variant_npz(dataset: str, variant: str):
    canonical = FEATROOT / dataset / 'precomputed_features.npz'
    dst_dir = OUT / 'features' / dataset
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f'precomputed_features_{variant}.npz'
    cache_raw = dst_dir / f'raw_emb_{variant}.npz'
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f'[{dataset}/{variant}] 已存在，跳过')
        return dst

    z = np.load(canonical, allow_pickle=True)
    targets = sorted(set(z['target_names'].tolist()))
    seqs, missing = find_sequences(dataset, targets)
    if missing:
        raise RuntimeError(f'{dataset}: {len(missing)} 个靶点缺序列: {missing[:5]}')
    print(f'[{dataset}] 靶点 {len(targets)}，序列全部就绪，长度 '
          f'min/med/max = {min(map(len,seqs.values()))}/'
          f'{int(np.median([len(s) for s in seqs.values()]))}/{max(map(len,seqs.values()))}')

    if cache_raw.exists():
        raw = dict(np.load(cache_raw, allow_pickle=True))
        in_dim = int(raw['_dim'])
        emb = {k: raw[k] for k in raw.files if not k.startswith('_')}
        print(f'[{variant}] 复用缓存原始表征')
    else:
        if variant == 'esm1b':
            emb, in_dim = extract_esm1b(seqs)
        elif variant == 'prott5':
            emb, in_dim = extract_prott5(seqs)
        else:
            raise ValueError(variant)
        np.savez(cache_raw, _dim=in_dim, **emb)

    proj = project_to_128(emb, in_dim, seed=0)
    names = z['target_names']
    protein = np.stack([proj[t] for t in names]).astype(np.float32)

    arrays = {k: z[k] for k in z.files}
    arrays['protein'] = protein
    np.savez(dst, **arrays)
    print(f'[{dataset}/{variant}] 写出 {dst} ({dst.stat().st_size/1e9:.2f} GB), '
          f'protein shape={protein.shape}')
    return dst


# ----------------------------------------------------------------------
# 训练驱动
# ----------------------------------------------------------------------
def run_one(dataset, variant, seed, gpu='0'):
    outdir = OUT / f'{dataset}_{variant}_seed{seed}'
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / 'final_results.json').exists():
        print(f'[skip] {outdir.name} 已完成')
        return 0
    npz_src = OUT / 'features' / dataset / f'precomputed_features_{variant}.npz'
    npz_dst = outdir / 'precomputed_features.npz'
    if not npz_dst.exists():
        try:
            os.link(npz_src, npz_dst)
        except OSError:
            shutil.copyfile(npz_src, npz_dst)
    r = RECIPES[dataset]
    cmd = [sys.executable, str(ENTRY),
           '--data_dir', str(DATA_DIRS[dataset]),
           '--output_dir', str(outdir),
           '--no_precompute',
           '--batch_size', str(r['batch']),
           '--epochs', str(r['epochs']),
           '--inner_steps', '3',
           '--ablation', r['ablation'],
           '--seed', str(seed),
           '--gpu', gpu]
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['PREPROCESSED_DIR'] = str(DATA_DIRS[dataset])
    env['CUDA_VISIBLE_DEVICES'] = gpu
    print(f'\n[run] {dataset}/{variant}/seed{seed}  ' + ' '.join(cmd))
    t0 = time.time()
    with open(outdir / 'console.log', 'a', encoding='utf-8') as lf:
        lf.write(f'\n===== launch {time.strftime("%Y-%m-%d %H:%M:%S")} =====\n' + ' '.join(cmd) + '\n')
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    (outdir / 'walltime_sec.txt').write_text(str(time.time() - t0), encoding='utf-8')
    print(f'[done] {dataset}/{variant}/seed{seed} rc={rc} wall={(time.time()-t0)/60:.1f}min')
    return rc


# ----------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------
def collect():
    import pandas as pd
    rows = []
    for ds in RECIPES:
        for variant in ['esm1b', 'prott5']:
            for seed in SEEDS:
                fp = OUT / f'{ds}_{variant}_seed{seed}' / 'final_results.json'
                if not fp.exists():
                    continue
                d = json.loads(fp.read_text(encoding='utf-8'))
                tm = d.get('test_metrics', {})
                row = {'dataset': ds, 'variant': variant, 'seed': seed}
                for k in ['R2', 'RMSE', 'MAE', 'Pearson', 'Spearman',
                          'EF@1%', 'EF@5%', 'EF@10%', 'ECE', 'AUPR']:
                    v = tm.get(k)
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        v = np.nan
                    row[k] = v
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'exp12_variant_metrics.csv', index=False)

    # ESM2 基线（E5 repeats，同种子子集）
    e5 = pd.read_csv(E5_CSV)
    e5.columns = [c.strip() for c in e5.columns]
    base = e5[e5['seed'].isin(SEEDS)].copy()
    base['variant'] = 'esm2_t12_35m'
    base.to_csv(OUT / 'exp12_esm2_baseline_same_seeds.csv', index=False)

    metrics = ['R2', 'RMSE', 'MAE', 'Pearson', 'Spearman', 'EF@1%', 'EF@5%', 'EF@10%', 'ECE', 'AUPR']
    lines = ['实验 12：蛋白质语言模型消融（ESM2-t12-35M vs ESM-1b-650M vs ProtT5-XL-3B）',
             '协议：仅替换蛋白特征（mean-pool PLM 表征 + 冻结随机投影→128 维），',
             '      分子特征 / 标签 / 划分 / 超参逐字节不变；seeds = {11,22,33} 与 E5 基线同种子配对',
             '=' * 100]
    for ds in RECIPES:
        lines.append(f'\n### {ds.upper()}')
        sub = df[df['dataset'] == ds]
        bsub = base[base['dataset'] == ds]
        header = f"{'variant':<16}{'n':>3}" + ''.join(f'{m:>18}' for m in metrics)
        lines.append(header)
        if len(bsub):
            m = bsub[metrics].mean()
            s = bsub[metrics].std(ddof=1)
            lines.append(f"{'esm2_t12_35m':<16}{len(bsub):>3}" +
                         ''.join(f'{f"{m[k]:.4f}±{s[k]:.4f}":>18}' for k in metrics))
        for variant in ['esm1b', 'prott5']:
            v = sub[sub['variant'] == variant]
            if not len(v):
                lines.append(f'{variant:<16}{0:>3}  (未完成)')
                continue
            m = v[metrics].mean()
            s = v[metrics].std(ddof=1) if len(v) > 1 else v[metrics]*0
            lines.append(f"{variant:<16}{len(v):>3}" +
                         ''.join(f'{f"{m[k]:.4f}±{s[k]:.4f}":>18}' for k in metrics))
        # 配对差值（逐 seed ΔR2）
        for variant in ['esm1b', 'prott5']:
            v = sub[sub['variant'] == variant].set_index('seed')
            if not len(v) or not len(bsub):
                continue
            b = bsub.set_index('seed')
            common = v.index.intersection(b.index)
            if len(common):
                dR2 = (v.loc[common, 'R2'] - b.loc[common, 'R2'])
                dEF = (v.loc[common, 'EF@1%'] - b.loc[common, 'EF@1%'])
                lines.append(f'  Δ({variant}−esm2) 逐seed: '
                             f'ΔR2={dR2.mean():+.4f}±{dR2.std(ddof=1) if len(common)>1 else 0:.4f}, '
                             f'ΔEF@1%={dEF.mean():+.4f}±{dEF.std(ddof=1) if len(common)>1 else 0:.4f} (n={len(common)})')
    report = '\n'.join(lines)
    (OUT / 'exp12_report.txt').write_text(report, encoding='utf-8')
    print(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['extract', 'train', 'report'])
    ap.add_argument('--datasets', default='kiba,davis')
    ap.add_argument('--variants', default='esm1b,prott5')
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()

    if args.stage == 'extract':
        for ds in args.datasets.split(','):
            for v in args.variants.split(','):
                build_variant_npz(ds, v)
    elif args.stage == 'train':
        for ds in args.datasets.split(','):
            for v in args.variants.split(','):
                for seed in SEEDS:
                    run_one(ds, v, seed, gpu=args.gpu)
        collect()
    elif args.stage == 'report':
        collect()


if __name__ == '__main__':
    main()
