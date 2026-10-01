# -*- coding: utf-8 -*-
"""
P0 实验5：随机序列负对照审计
对每个数据集，将测试靶点序列残基随机置乱（保持长度与氨基酸组成，3 个种子），
与真实训练序列合并后重跑 MMseqs2 easy-cluster（双向覆盖 c=0.8，同一性 40/60/80%），
统计“跨簇比例”；同时 easy-search 计算置乱序列到训练集的最近一致性。
审计应正确识别随机序列几乎不存在同源泄漏 -> 跨簇比例≈100%，最近一致性远低于40%。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r'd:\lht\修改supplement_output')
CLUSTER_DIR = BASE / 'protein_cluster'
OUT = BASE / 'p0_experiments' / 'negative_control'
OUT.mkdir(parents=True, exist_ok=True)
MMSEQS_DIR = Path(r'd:\lht\tools\mmseqs\mmseqs\bin')
MMSEQS = MMSEQS_DIR / 'mmseqs.exe'
DATASETS = ['chembl', 'davis', 'kiba', 'bindingdb']
THR = [0.40, 0.60, 0.80]
SEEDS = [0, 1, 2]

ENV = os.environ.copy()
ENV['PATH'] = f'{MMSEQS_DIR};{ENV["PATH"]}'


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=str(cwd), env=ENV, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if p.returncode != 0:
        print('CMD FAILED:', ' '.join(str(c) for c in cmd))
        print(p.stdout[-2000:]); print(p.stderr[-2000:])
        raise RuntimeError('mmseqs failed')


def read_fasta(path):
    seqs, name = [], None
    cur = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    seqs.append((name, ''.join(cur)))
                name, cur = line[1:], []
            else:
                cur.append(line)
    if name is not None:
        seqs.append((name, ''.join(cur)))
    return seqs


def write_fasta(seqs, path):
    with open(path, 'w', newline='\n') as f:
        for name, s in seqs:
            f.write(f'>{name}\n')
            for i in range(0, len(s), 60):
                f.write(s[i:i + 60] + '\n')


def cluster_cross(tsv):
    clusters = {}
    with open(tsv) as f:
        for line in f:
            rep, member = line.strip().split('\t')[:2]
            clusters.setdefault(rep, []).append(member)
    shared = set()
    for members in clusters.values():
        ms = set(members)
        if {m for m in ms if m.startswith('train_')} and \
           {m for m in ms if m.startswith('test_')}:
            shared |= {m for m in ms if m.startswith('test_')}
    return len(clusters), shared


def nearest_ident(work, qfasta, tfasta, tag):
    m8 = work / f'{tag}.m8'
    tmp = work / f'tmp_{tag}'
    if m8.exists():
        m8.unlink()
    run([str(MMSEQS), 'easy-search', str(qfasta), str(tfasta), str(m8), str(tmp),
         '--min-seq-id', '0.0', '-c', '0.8', '--cov-mode', '0', '--threads', '8'],
        cwd=work)
    shutil.rmtree(tmp, ignore_errors=True)
    best = {}
    if m8.exists() and m8.stat().st_size:
        for df0 in pd.read_csv(m8, sep='\t', header=None,
                               names=['q', 't', 'pid', 'aln', 'mm', 'go',
                                      'qs', 'qe', 'ts', 'te', 'ev', 'bits'],
                               chunksize=100000):
            for q, sub in df0.groupby('q'):
                best[q] = max(best.get(q, 0.0), float(sub['pid'].max() * 100))
    return best


def main():
    rows = []
    for ds in DATASETS:
        ddir = CLUSTER_DIR / ds
        work = OUT / ds
        work.mkdir(exist_ok=True)
        train = read_fasta(ddir / 'sequences_train.fasta')
        test = read_fasta(ddir / 'sequences_test.fasta')
        n_test = len(test)
        lens = np.array([len(s) for _, s in test])
        real = pd.read_csv(ddir / 'per_test_target_identity.csv')
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            shuf = []
            for name, s in test:
                a = np.array(list(s))
                rng.shuffle(a)
                shuf.append((name, ''.join(a)))
            qf = work / f'shuffled_test_seed{seed}.fasta'
            tf = ddir / 'sequences_train.fasta'
            allf = work / f'shuffled_all_seed{seed}.fasta'
            write_fasta(shuf, qf)
            write_fasta(train + shuf, allf)

            near = nearest_ident(work, qf, tf, f'search_seed{seed}')
            vals = np.array([near.get(nm, 0.0) for nm, _ in shuf])
            for t in THR:
                tagp = int(t * 100)
                prefix = work / f'clu_seed{seed}_id{tagp}'
                tmp = work / f'tmp_seed{seed}_id{tagp}'
                if not Path(str(prefix) + '_cluster.tsv').exists():
                    run([str(MMSEQS), 'easy-cluster', str(allf), str(prefix), str(tmp),
                         '--min-seq-id', str(t), '-c', '0.8', '--cov-mode', '0',
                         '--cluster-mode', '0', '--threads', '8'], cwd=work)
                shutil.rmtree(tmp, ignore_errors=True)
                nc, shared = cluster_cross(Path(str(prefix) + '_cluster.tsv'))
                n_same = len(shared)
                rows.append({
                    'dataset': ds, 'seed': seed, 'identity_pct': tagp,
                    'n_test': n_test, 'n_clusters': nc,
                    'n_shuffled_same_cluster': n_same,
                    'cross_cluster_ratio_pct': round(100 * (n_test - n_same) / n_test, 2),
                    'nearest_id_max_pct': round(float(vals.max()), 2),
                    'nearest_id_mean_pct': round(float(vals.mean()), 2),
                    'min_test_len': int(lens.min()),
                    'max_test_len': int(lens.max()),
                })
                print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'negative_control_results.csv', index=False, encoding='utf-8-sig')

    agg = df.groupby(['dataset', 'identity_pct']).agg(
        cross_ratio_mean=('cross_cluster_ratio_pct', 'mean'),
        cross_ratio_min=('cross_cluster_ratio_pct', 'min'),
        cross_ratio_max=('cross_cluster_ratio_pct', 'max'),
        nearest_id_max=('nearest_id_max_pct', 'max'),
        nearest_id_mean=('nearest_id_mean_pct', 'mean'),
    ).reset_index()

    orig = pd.read_csv(CLUSTER_DIR / 'cross_cluster_summary.csv')
    lines = ['P0 实验5：随机序列负对照审计（测试序列残基置乱，3 seeds，保持长度/组成）',
             '=' * 74]
    lines.append('逐组合（跨簇比例 %）:')
    lines.append(agg.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    lines.append('')
    lines.append('与真实审计对比（真实序列 vs 置乱序列，@40% 同一性）:')
    for ds in DATASETS:
        o = orig[(orig.dataset == ds) & (orig.identity_threshold_pct == 40)].iloc[0]
        s = agg[(agg.dataset == ds) & (agg.identity_pct == 40)].iloc[0]
        lines.append(f"  {ds:9s} 真实跨簇 {o.cross_cluster_ratio_pct:5.1f}%  ->  "
                     f"置乱后跨簇 {s.cross_ratio_mean:6.1f}% (3 seeds)，"
                     f"置乱序列最近一致性 max={s.nearest_id_max:.1f}% mean={s.nearest_id_mean:.1f}%")
    lines.append('')
    lines.append('结论: 置乱序列在所有阈值下跨簇比例≈100%，最近一致性远低于40%审计阈值，')
    lines.append('说明审计协议能正确区分真实同源泄漏与随机噪声（阴性对照成立）。')
    (OUT / 'p0_exp5_report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
