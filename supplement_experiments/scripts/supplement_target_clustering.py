# -*- coding: utf-8 -*-
"""
补充实验2：训练集 / 测试集靶点蛋白序列的 MMseqs2 同源聚类
（MMseqs2 release 18，Windows 原生构建；等价替代 CD-HIT）。

对四个数据集 chembl / davis / kiba / bindingdb：
  1. 收集 train、test 中去重后的靶点序列，写 FASTA；
  2. easy-cluster，序列一致性阈值 0.40 / 0.60 / 0.80，双向比对覆盖度 -c 0.8；
  3. 统计测试靶点中与任意训练靶点同簇的数量，报告“跨簇比例”
     = 未与任何训练靶点同簇的测试靶点数 / 测试靶点总数；
  4. easy-search 计算每个测试靶点到训练集的最近序列一致性（nearest identity）。

输出: d:\\lht\\supplement_output\\protein_cluster\\
  - <dataset>/sequences_train.fasta, sequences_test.fasta
  - <dataset>/clu_<t>_cluster.tsv            MMseqs2 簇结果 (rep<TAB>member)
  - <dataset>/per_test_target_identity.csv   每个测试靶点的最近训练相似度/同簇情况
  - cross_cluster_summary.csv / .json        汇总
"""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r'd:\lht')
DATA = BASE / 'GrapthDTA' / 'data'
OUT = BASE / 'supplement_output' / 'protein_cluster'
MMSEQS_DIR = BASE / 'tools' / 'mmseqs' / 'mmseqs' / 'bin'
MMSEQS = MMSEQS_DIR / 'mmseqs.exe'

DATASETS = ['chembl', 'davis', 'kiba', 'bindingdb']
THRESHOLDS = [0.40, 0.60, 0.80]
COVERAGE = 0.8

# cygwin busybox 的 bash 等工具必须在 PATH 上，easy-cluster 工作流才能执行
ENV = os.environ.copy()
ENV['PATH'] = f'{MMSEQS_DIR};{ENV["PATH"]}'


def write_fasta(seq2id, path):
    with open(path, 'w', newline='\n') as f:
        for seq, sid in seq2id.items():
            f.write(f'>{sid}\n')
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + '\n')


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=str(cwd), env=ENV, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if p.returncode != 0:
        print('CMD FAILED:', ' '.join(str(c) for c in cmd))
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise RuntimeError('mmseqs failed')
    return p


def cluster(dataset_dir, fasta, tag, ident):
    prefix = dataset_dir / f'clu_{tag}'
    tmp = dataset_dir / f'tmp_clu_{tag}'
    if not Path(str(prefix) + '_cluster.tsv').exists():
        run([str(MMSEQS), 'easy-cluster', str(fasta), str(prefix), str(tmp),
             '--min-seq-id', str(ident), '-c', str(COVERAGE),
             '--cov-mode', '0', '--cluster-mode', '0', '--threads', '8'],
            cwd=dataset_dir)
    shutil.rmtree(tmp, ignore_errors=True)
    clusters = {}
    with open(str(prefix) + '_cluster.tsv') as f:
        for line in f:
            rep, member = line.strip().split('\t')[:2]
            clusters.setdefault(rep, []).append(member)
    return clusters


def nearest_identity(dataset_dir):
    """每个测试靶点到训练集的最高序列一致性（-c 0.8 双向覆盖，默认 m8 输出）。"""
    out_m8 = dataset_dir / 'search_test_vs_train.m8'
    tmp = dataset_dir / 'tmp_search'
    if out_m8.exists():
        out_m8.unlink()
    run([str(MMSEQS), 'easy-search',
         str(dataset_dir / 'sequences_test.fasta'),
         str(dataset_dir / 'sequences_train.fasta'),
         str(out_m8), str(tmp),
         '--min-seq-id', '0.0', '-c', str(COVERAGE), '--cov-mode', '0',
         '--threads', '8'],
        cwd=dataset_dir)
    shutil.rmtree(tmp, ignore_errors=True)
    best = {}
    if out_m8.exists() and out_m8.stat().st_size > 0:
        # 默认 m8: query,target,pident(fraction 0-1),alnlen,mismatch,gapopen,
        #        qstart,qend,tstart,tend,evalue,bits
        for df in pd.read_csv(out_m8, sep='\t', header=None,
                              names=['query', 'target', 'pident', 'alnlen', 'mismatch',
                                     'gapopen', 'qstart', 'qend', 'tstart', 'tend',
                                     'evalue', 'bits'], chunksize=100000):
            df['pident'] = df['pident'].astype(float) * 100.0
            for q, sub in df.groupby('query'):
                v = float(sub['pident'].max())
                if q not in best or v > best[q]:
                    best[q] = v
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for ds in DATASETS:
        print('=' * 70)
        print(f'[{ds}] 读取 CSV ...')
        ddir = OUT / ds
        ddir.mkdir(exist_ok=True)
        tr = pd.read_csv(DATA / f'{ds}_train.csv', usecols=['target_sequence'])
        te = pd.read_csv(DATA / f'{ds}_test.csv', usecols=['target_sequence'])
        train_seqs = sorted(set(tr['target_sequence'].dropna()))
        test_seqs = sorted(set(te['target_sequence'].dropna()))
        train_map = {s: f'train_{i:04d}' for i, s in enumerate(train_seqs)}
        test_map = {s: f'test_{i:04d}' for i, s in enumerate(test_seqs)}
        write_fasta(train_map, ddir / 'sequences_train.fasta')
        write_fasta(test_map, ddir / 'sequences_test.fasta')
        all_fasta = ddir / 'sequences_all.fasta'
        write_fasta({**train_map, **test_map}, all_fasta)
        print(f'  训练靶点 {len(train_map)}，测试靶点 {len(test_map)}')

        print(f'  [{ds}] nearest identity search ...')
        best = nearest_identity(ddir)

        per_row = {sid: {'target_id': sid, 'split': 'test',
                         'nearest_train_identity_pct': best.get(sid, 0.0)}
                   for sid in test_map.values()}

        row_base = {'dataset': ds, 'n_train_targets': len(train_map),
                    'n_test_targets': len(test_map), 'coverage': COVERAGE}
        for t in THRESHOLDS:
            tag = f'{int(t*100)}'
            print(f'  [{ds}] clustering @ {t:.2f} ...')
            clusters = cluster(ddir, all_fasta, tag, t)
            train_ids, test_ids = set(train_map.values()), set(test_map.values())
            shared_clusters = 0
            test_shared = set()
            for rep, members in clusters.items():
                ms = set(members)
                has_tr = bool(ms & train_ids)
                has_te = bool(ms & test_ids)
                if has_tr and has_te:
                    shared_clusters += 1
                    test_shared |= (ms & test_ids)
            n_test = len(test_map)
            n_shared = len(test_shared)
            n_cross = n_test - n_shared
            # 在 per_row 上登记
            for sid in test_map.values():
                per_row.setdefault(sid, {'target_id': sid, 'split': 'test',
                                         'nearest_train_identity_pct': best.get(sid, 0.0)})
                per_row[sid][f'same_cluster_{int(t*100)}'] = sid in test_shared
            row = {**row_base, 'identity_threshold_pct': int(t * 100),
                   'n_clusters': len(clusters),
                   'n_shared_train_test_clusters': shared_clusters,
                   'n_test_targets_same_cluster': n_shared,
                   'n_test_targets_cross_cluster': n_cross,
                   'same_cluster_ratio_pct': round(100 * n_shared / n_test, 2),
                   'cross_cluster_ratio_pct': round(100 * n_cross / n_test, 2)}
            summary_rows.append(row)
            print(f"    簇数={len(clusters)} 同簇测试靶点={n_shared}/{n_test} "
                  f"跨簇比例={row['cross_cluster_ratio_pct']}%")

        pd.DataFrame(per_row.values()).sort_values('target_id').to_csv(
            ddir / 'per_test_target_identity.csv', index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / 'cross_cluster_summary.csv', index=False)
    with open(OUT / 'cross_cluster_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary.to_dict(orient='records'), f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 70)
    print('汇总（跨簇比例 = 未与任何训练靶点同簇的测试靶点占比）')
    print('=' * 70)
    print(summary.to_string(index=False))
    print('\n输出目录:', OUT)


if __name__ == '__main__':
    main()
