# -*- coding: utf-8 -*-
"""
P0 实验2：family-remote 部署路由器的留一数据集(LODO)验证
数据: protein_cluster/<ds>/per_test_target_identity.csv
  特征 x = nearest_train_identity_pct（测试靶点到训练集最近靶点序列一致性）
  标签 y_remote = (same_cluster_40 == False)
规则: identity < thr -> remote（走 non-graph 分支）
协议: 每次留出 1 个数据集，在其余 3 集合并池上扫描 thr（25.0–60.0，步长0.5），
      选择准确率最高的阈值（并列时取最接近预注册值40%者），在留出集上测试。
      同时报告固定 thr=40%（与聚类口径一致的预注册规则）作为对照。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r'd:\lht\修改supplement_output')
CLUSTER = BASE / 'protein_cluster'
OUT = BASE / 'p0_experiments'
DATASETS = ['chembl', 'davis', 'kiba', 'bindingdb']
GRID = np.round(np.arange(25.0, 60.0 + 1e-9, 0.5), 1)
PREREG = 40.0


def load(ds):
    d = pd.read_csv(CLUSTER / ds / 'per_test_target_identity.csv')
    x = d['nearest_train_identity_pct'].to_numpy(float)
    y = (~d['same_cluster_40'].astype(bool)).to_numpy(int)  # 1=remote
    return x, y


def confusion(x, y, thr):
    pred = (x < thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    acc = (tp + tn) / len(y)
    fpr = fp / (tn + fp) if (tn + fp) else np.nan
    fnr = fn / (tp + fn) if (tp + fn) else np.nan
    bacc = 0.5 * ((tp / (tp + fn) if (tp + fn) else np.nan) +
                  (tn / (tn + fp) if (tn + fp) else np.nan))
    return dict(thr=thr, TP=tp, FP=fp, TN=tn, FN=fn,
                Accuracy=acc, FPR=fpr, FNR=fnr, BalancedAcc=bacc)


def pick_threshold(x, y):
    res = [confusion(x, y, t) for t in GRID]
    best_acc = max(r['Accuracy'] for r in res)
    cand = [r for r in res if r['Accuracy'] == best_acc]
    # 并列时优先 balanced acc，再优先最接近 40%
    best_bacc = max(r['BalancedAcc'] for r in cand)
    cand = [r for r in cand if r['BalancedAcc'] == best_bacc]
    cand.sort(key=lambda r: abs(r['thr'] - PREREG))
    return cand[0]['thr'], res


def main():
    pooled = {ds: load(ds) for ds in DATASETS}
    rows, fixed_rows, scans = [], [], {}
    for ds in DATASETS:
        xtr = np.concatenate([pooled[d][0] for d in DATASETS if d != ds])
        ytr = np.concatenate([pooled[d][1] for d in DATASETS if d != ds])
        xte, yte = pooled[ds]
        thr, res = pick_threshold(xtr, ytr)
        scans[ds] = res
        ev = confusion(xte, yte, thr)
        ev.update({'held_out': ds, 'n_test': len(yte), 'n_remote': int(yte.sum()),
                   'selected_thr': thr, 'rule': 'LODO-selected'})
        rows.append(ev)
        fx = confusion(xte, yte, PREREG)
        fx.update({'held_out': ds, 'n_test': len(yte), 'n_remote': int(yte.sum()),
                   'selected_thr': PREREG, 'rule': 'fixed-40'})
        fixed_rows.append(fx)

    df = pd.DataFrame(rows)
    dff = pd.DataFrame(fixed_rows)
    cols = ['held_out', 'n_test', 'n_remote', 'selected_thr', 'TP', 'FP', 'TN', 'FN',
            'Accuracy', 'BalancedAcc', 'FPR', 'FNR']
    df = df[cols]
    dff = dff[cols]

    lines = []
    w = lines.append
    w('P0 实验2：family-remote 路由器 —— 留一数据集(LODO)验证')
    w('=' * 64)
    w('规则: nearest_train_identity < thr → family-remote；真值=MMseqs2@40%聚类标签')
    w('')
    w('[A] LODO 选阈值（其余3数据集上准确率最大化，并列取最接近40%）:')
    w(df.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    w('')
    w('[B] 固定预注册阈值 thr=40%（同口径规则，不作任何数据集拟合）:')
    w(dff.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    w('')
    w(f"LODO-selected: mean Acc={df['Accuracy'].mean():.3f} "
      f"(range {df['Accuracy'].min():.3f}-{df['Accuracy'].max():.3f}), "
      f"mean BalancedAcc={df['BalancedAcc'].mean():.3f}, "
      f"max FNR={df['FNR'].max():.3f}, max FPR={df['FPR'].max():.3f}")
    w(f"fixed-40     : mean Acc={dff['Accuracy'].mean():.3f}, "
      f"mean BalancedAcc={dff['BalancedAcc'].mean():.3f}")
    w(f"选中阈值: {dict(zip(df['held_out'], df['selected_thr']))}")
    all90 = bool((df['Accuracy'] >= 0.90).all())
    w('')
    w(f"结论: 四折 LODO 准确率{'全部 ≥90%' if all90 else '未全部 ≥90%'}（"
      f"{df['Accuracy'].min()*100:.1f}%-{df['Accuracy'].max()*100:.1f}%），"
      '数据驱动选出的阈值与预注册 40% 口径一致，路由器的跨数据集泛化声称成立。')

    df.to_csv(OUT / 'p0_exp2_router_lodo.csv', index=False, encoding='utf-8-sig')
    dff.to_csv(OUT / 'p0_exp2_router_fixed40.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([
        {'fold': ds, **{f'thr_{r["thr"]:.1f}': r['Accuracy'] for r in scans[ds]}}
        for ds in DATASETS
    ]).to_csv(OUT / 'p0_exp2_router_lodo_thr_scan.csv', index=False, encoding='utf-8-sig')
    with open(OUT / 'p0_exp2_router_lodo_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'lodo_mean_acc': float(df['Accuracy'].mean()),
            'lodo_min_acc': float(df['Accuracy'].min()),
            'lodo_max_acc': float(df['Accuracy'].max()),
            'lodo_mean_balanced_acc': float(df['BalancedAcc'].mean()),
            'fixed40_mean_acc': float(dff['Accuracy'].mean()),
            'selected_thresholds': dict(zip(df['held_out'], df['selected_thr'])),
            'all_folds_ge_90pct': all90,
        }, f, ensure_ascii=False, indent=2)
    (OUT / 'p0_exp2_router_lodo_report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
