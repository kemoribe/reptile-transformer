# -*- coding: utf-8 -*-
"""
P0 实验3a：MMseqs2 审计参数敏感性扫描
对 4 个数据集，在 覆盖率 c∈{0.7,0.8,0.9} × 同一性 t∈{30,40,50,60,70,80}%
共 18 个参数组合上重跑 easy-cluster（cov-mode 0 / cluster-mode 0，与原审计一致），
报告跨簇测试靶点比例，并校验 (c=0.8,t=40/60/80) 复现原审计结果。
"""
import os
import json
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r'd:\lht\修改supplement_output')
CLUSTER_DIR = BASE / 'protein_cluster'
OUT = BASE / 'p0_experiments' / 'param_sensitivity'
OUT.mkdir(parents=True, exist_ok=True)
MMSEQS_DIR = Path(r'd:\lht\tools\mmseqs\mmseqs\bin')
MMSEQS = MMSEQS_DIR / 'mmseqs.exe'

DATASETS = ['chembl', 'davis', 'kiba', 'bindingdb']
COVERAGES = [0.7, 0.8, 0.9]
IDENTITIES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

ENV = os.environ.copy()
ENV['PATH'] = f'{MMSEQS_DIR};{ENV["PATH"]}'


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=str(cwd), env=ENV, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if p.returncode != 0:
        print('CMD FAILED:', ' '.join(str(c) for c in cmd))
        print(p.stdout[-2000:]); print(p.stderr[-2000:])
        raise RuntimeError('mmseqs failed')


def cluster_all(fasta, prefix, tmp, ident, cov):
    if not Path(str(prefix) + '_cluster.tsv').exists():
        run([str(MMSEQS), 'easy-cluster', str(fasta), str(prefix), str(tmp),
             '--min-seq-id', str(ident), '-c', str(cov),
             '--cov-mode', '0', '--cluster-mode', '0', '--threads', '8'],
            cwd=prefix.parent)
    clusters = {}
    with open(str(prefix) + '_cluster.tsv') as f:
        for line in f:
            rep, member = line.strip().split('\t')[:2]
            clusters.setdefault(rep, []).append(member)
    return clusters


def count_cross(clusters):
    """按 train_/test_ ID 前缀统计与训练集同簇的测试靶点数。"""
    test_shared = set()
    n_clu_with_both = 0
    for rep, members in clusters.items():
        ms = set(members)
        tr = {m for m in ms if m.startswith('train_')}
        te = {m for m in ms if m.startswith('test_')}
        if tr and te:
            n_clu_with_both += 1
            test_shared |= te
    return n_clu_with_both, test_shared


def main():
    rows = []
    nseq = {}
    for ds in DATASETS:
        ddir = CLUSTER_DIR / ds
        fasta = ddir / 'sequences_all.fasta'
        n_tr = sum(1 for _ in open(ddir / 'sequences_train.fasta')) // 2 \
            if (ddir / 'sequences_train.fasta').exists() else None
        nseq[ds] = n_tr
        work = OUT / ds
        work.mkdir(exist_ok=True)
        for cov in COVERAGES:
            for ident in IDENTITIES:
                tag = f'clu_cov{int(cov*100)}_id{int(ident*100)}'
                prefix = work / tag
                tmp = work / f'tmp_{tag}'
                print(f'[{ds}] cov={cov} id={ident} ...', flush=True)
                clusters = cluster_all(fasta, prefix, tmp, ident, cov)
                shutil.rmtree(tmp, ignore_errors=True)
                n_both, test_shared = count_cross(clusters)
                # 测试靶点总数以 per_test_target_identity.csv 为准
                pid = pd.read_csv(ddir / 'per_test_target_identity.csv')
                n_test = len(pid)
                n_same = len(test_shared)
                n_cross = n_test - n_same
                rows.append({
                    'dataset': ds, 'coverage': cov,
                    'identity_pct': int(ident * 100),
                    'n_clusters': len(clusters),
                    'n_shared_clusters': n_both,
                    'n_test': n_test,
                    'n_test_same_cluster': n_same,
                    'n_test_cross_cluster': n_cross,
                    'cross_cluster_ratio_pct': round(100 * n_cross / n_test, 2),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'mmseqs_grid.csv', index=False, encoding='utf-8-sig')

    # ---- 复现校验：cov=0.8 × {40,60,80} 与原 cross_cluster_summary.csv 对比 ----
    old = pd.read_csv(CLUSTER_DIR / 'cross_cluster_summary.csv')
    check = []
    ok_all = True
    for ds in DATASETS:
        for t in [40, 60, 80]:
            a = df[(df.dataset == ds) & (df.coverage == 0.8) &
                   (df.identity_pct == t)].iloc[0]
            b = old[(old.dataset == ds) & (old.identity_threshold_pct == t)].iloc[0]
            same = int(a.n_test_cross_cluster) == int(b.n_test_targets_cross_cluster)
            ok_all &= same
            check.append({'dataset': ds, 'identity_pct': t,
                          'recalc_cross': int(a.n_test_cross_cluster),
                          'original_cross': int(b.n_test_targets_cross_cluster),
                          'recalc_ratio': a.cross_cluster_ratio_pct,
                          'original_ratio': b.cross_cluster_ratio_pct,
                          'match': bool(same)})
    chk = pd.DataFrame(check)
    chk.to_csv(OUT / 'reproduction_check.csv', index=False, encoding='utf-8-sig')

    # ---- 汇总 ----
    lines = ['P0 实验3a：MMseqs2 参数敏感性扫描（覆盖率×同一性，18组合/数据集）', '=' * 70]
    pivot = df.pivot_table(index=['dataset', 'coverage'], columns='identity_pct',
                           values='cross_cluster_ratio_pct')
    lines.append('跨簇比例 (%) 行=数据集/覆盖率, 列=同一性阈值%:')
    lines.append(pivot.to_string(float_format=lambda v: f'{v:6.2f}'))
    lines.append('')
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        r40 = sub[sub.identity_pct == 40]
        lines.append(
            f'{ds:9s} @40%同一性: 跨簇比例 {r40.cross_cluster_ratio_pct.min():.1f}-'
            f'{r40.cross_cluster_ratio_pct.max():.1f}% (cov 0.7→0.9); '
            f'全网格 {sub.cross_cluster_ratio_pct.min():.1f}-'
            f'{sub.cross_cluster_ratio_pct.max():.1f}%; '
            f'簇数 {sub.n_clusters.min()}-{sub.n_clusters.max()}')
    lines.append('')
    lines.append(f"复现校验 (cov=0.8, id=40/60/80 vs 原审计): "
                 f"{'全部一致 ✓' if ok_all else '存在不一致 ✗'}")
    lines.append(chk.to_string(index=False))
    report = '\n'.join(lines)
    (OUT / 'p0_exp3a_report.txt').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
