# -*- coding: utf-8 -*-
"""
P0 实验3b：聚类算法对照 —— MMseqs2 默认 set-cover/连通分量 (--cluster-mode 0)
           vs CD-HIT 式贪心增量 (--cluster-mode 2, greedy incremental, like CD-HIT)
固定双向覆盖 c=0.8，同一性 t∈{40,60,80}%，对比：
  - 簇数 / 跨簇比例
  - 逐测试靶点 same/cross 标签一致率 + Cohen's κ
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
OUT = BASE / 'p0_experiments' / 'param_sensitivity'
MMSEQS_DIR = Path(r'd:\lht\tools\mmseqs\mmseqs\bin')
MMSEQS = MMSEQS_DIR / 'mmseqs.exe'
DATASETS = ['chembl', 'davis', 'kiba', 'bindingdb']
THR = [0.40, 0.60, 0.80]
COV = 0.8

ENV = os.environ.copy()
ENV['PATH'] = f'{MMSEQS_DIR};{ENV["PATH"]}'


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=str(cwd), env=ENV, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if p.returncode != 0:
        print('CMD FAILED:', ' '.join(str(c) for c in cmd))
        print(p.stdout[-2000:]); print(p.stderr[-2000:])
        raise RuntimeError('mmseqs failed')


def labels_from_tsv(path):
    """返回 {test_id: same_with_train(bool)} 与簇数、同簇测试集合。"""
    clusters = {}
    with open(path) as f:
        for line in f:
            rep, member = line.strip().split('\t')[:2]
            clusters.setdefault(rep, []).append(member)
    shared = set()
    for members in clusters.values():
        ms = set(members)
        tr = {m for m in ms if m.startswith('train_')}
        te = {m for m in ms if m.startswith('test_')}
        if tr and te:
            shared |= te
    return clusters, shared


def kappa(lab0, lab1):
    keys = sorted(set(lab0) | set(lab1))
    y0 = np.array([lab0[k] for k in keys], dtype=int)
    y1 = np.array([lab1[k] for k in keys], dtype=int)
    n = len(keys)
    po = (y0 == y1).mean()
    p_e = ((y0.mean() * y1.mean()) + ((1 - y0.mean()) * (1 - y1.mean())))
    return float((po - p_e) / (1 - p_e)) if (1 - p_e) > 1e-12 else 1.0, int((y0 == y1).sum()), n


def main():
    rows = []
    for ds in DATASETS:
        ddir = CLUSTER_DIR / ds
        fasta = ddir / 'sequences_all.fasta'
        work = OUT / ds
        work.mkdir(exist_ok=True)
        n_test = len(pd.read_csv(ddir / 'per_test_target_identity.csv'))
        for t in THR:
            tagp = int(t * 100)
            # mode 0：原审计结果（复用既有文件）
            orig_tsv = ddir / f'clu_{tagp}_cluster.tsv'
            cl0, sh0 = labels_from_tsv(orig_tsv)
            # mode 2：CD-HIT 式贪心增量
            prefix = work / f'cdhit_clu_id{tagp}'
            tmp = work / f'tmp_cdhit_{tagp}'
            if not Path(str(prefix) + '_cluster.tsv').exists():
                run([str(MMSEQS), 'easy-cluster', str(fasta), str(prefix), str(tmp),
                     '--min-seq-id', str(t), '-c', str(COV),
                     '--cov-mode', '0', '--cluster-mode', '2', '--threads', '8'],
                    cwd=work)
            shutil.rmtree(tmp, ignore_errors=True)
            cl2, sh2 = labels_from_tsv(Path(str(prefix) + '_cluster.tsv'))

            tids = pd.read_csv(ddir / 'per_test_target_identity.csv')['target_id'].tolist()
            lab0 = {tid: (tid in sh0) for tid in tids}
            lab2 = {tid: (tid in sh2) for tid in tids}
            k, nagree, ntot = kappa(lab0, lab2)
            rows.append({
                'dataset': ds, 'identity_pct': tagp,
                'mmseqs_setcover_n_clusters': len(cl0),
                'cdhit_greedy_n_clusters': len(cl2),
                'mmseqs_cross_ratio_pct': round(100 * (n_test - len(sh0)) / n_test, 2),
                'cdhit_cross_ratio_pct': round(100 * (n_test - len(sh2)) / n_test, 2),
                'label_agreement_pct': round(100 * nagree / ntot, 2),
                'cohen_kappa': round(k, 4),
            })
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'cluster_algorithm_cdHit_comparison.csv', index=False, encoding='utf-8-sig')
    lines = ['P0 实验3b：聚类算法对照（MMseqs2 set-cover/连通分量 vs CD-HIT式贪心增量, c=0.8）',
             '=' * 74]
    lines.append(df.to_string(index=False))
    lines.append('')
    lines.append(f"跨簇比例最大绝对偏差: "
                 f"{(df.mmseqs_cross_ratio_pct - df.cdhit_cross_ratio_pct).abs().max():.2f} 个百分点; "
                 f"标签一致率: {df.label_agreement_pct.min():.1f}-{df.label_agreement_pct.max():.1f}%; "
                 f"Cohen κ: {df.cohen_kappa.min():.3f}-{df.cohen_kappa.max():.3f}")
    (OUT / 'p0_exp3b_report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
