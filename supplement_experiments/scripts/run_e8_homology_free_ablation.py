# -*- coding: utf-8 -*-
"""
实验2：40% identity 无同源子集消融
对每个数据集，仅保留与训练集无 ≥40% 同源性的测试靶点（cross-cluster），
在该子集上重算所有可用模型的指标，与完整测试集结果对比。

口径：per_test_target_identity.csv 中 same_cluster_40 == False 的靶点
输出: homology_free_ablation_results.csv + per_dataset/*.json
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE = Path(r'd:\lht')
GDT_DATA = BASE / 'GrapthDTA' / 'data'
GNN_PREDS = BASE / '修改supplement_output' / 'paper_revision修改V1' / 'gnn_preds'
CLUSTER_DIR = BASE / '修改supplement_output' / 'protein_cluster'
OUT_DIR = BASE / '修改supplement_output' / 'revision2_experiments' / 'e8_homology_free_ablation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

GNN_MODELS = ['GCNNet', 'GATNet', 'GAT_GCN', 'GINConvNet']
REPTILE_MODELS = {
    'davis': BASE / 'reptile_ablation_output_davis' / 'morgan_maccs' / 'predictions.npz',
    'kiba': BASE / 'reptile_ablation_output_kiba' / 'morgan_maccs' / 'predictions.npz',
    'bindingdb': BASE / 'reptile_ablation_output_bindingdb' / 'morgan_maccs' / 'predictions.npz',
}


def calc_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(y_true)
    if n < 2:
        return {k: np.nan for k in ['R2', 'RMSE', 'MAE', 'EF@1%', 'EF@5%', 'EF@10%', 'ECE']}
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    active_count = max(5, int(n * 0.2))
    thr = np.sort(y_true)[::-1][min(active_count - 1, n - 1)]
    y_sorted = y_true[np.argsort(y_pred)[::-1]]

    def ef(pct):
        k = int(n * pct / 100)
        return float((np.sum(y_sorted[:k] >= thr) / k) / (active_count / n)) if k else 0.0

    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    rng = hi - lo
    if rng < 1e-12:
        ece = 0.0
    else:
        yt_n, yp_n = (y_true - lo) / rng, (y_pred - lo) / rng
        bins = np.digitize(yp_n, np.linspace(0, 1, 11)[1:-1])
        ece = 0.0
        for b in range(10):
            m = bins == b
            if m.sum():
                ece += (m.sum() / n) * abs(yp_n[m].mean() - yt_n[m].mean())

    return {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae),
            'EF@1%': ef(1), 'EF@5%': ef(5), 'EF@10%': ef(10), 'ECE': float(ece)}


def load_gnn_pred(dataset):
    """加载 GNN 预测，与 GrapthDTA test CSV 行对齐"""
    te = pd.read_csv(GDT_DATA / f'{dataset}_test.csv')
    preds = {}
    for m in GNN_MODELS:
        f = GNN_PREDS / f'{m}_{dataset}.npz'
        if not f.exists():
            continue
        d = np.load(f)
        preds[m] = {'y_true': d['y_true'], 'y_pred': d['y_pred']}
    return te, preds


def build_reptile_target_map(dataset):
    """从预处理目录构建 reptile 测试集的 (target, row_slice) 映射"""
    prep = BASE / f'3_{dataset}_preprocessed' / 'test_set'
    target_dirs = sorted([d for d in prep.iterdir() if d.is_dir()])
    # reptile 遍历: test_set/<category>/<target>/activities.csv
    # 但有些数据集直接是 test_set/<target>，需要递归找 activities.csv
    targets = []
    for cat_dir in target_dirs:
        if (cat_dir / 'activities.csv').exists():
            targets.append((cat_dir.name, cat_dir))
        else:
            for tdir in sorted(cat_dir.iterdir()):
                if tdir.is_dir() and (tdir / 'activities.csv').exists():
                    targets.append((tdir.name, tdir))
    return targets


def read_fasta_seq(fasta_path):
    """读取 fasta 文件的序列"""
    with open(fasta_path) as f:
        lines = f.readlines()
    seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
    return seq


def load_reptile_pred_with_targets(dataset):
    """加载 reptile 预测并标注每个样本的 target_id（test_XXXX）"""
    f = REPTILE_MODELS.get(dataset)
    if f is None or not f.exists():
        return None, None
    d = np.load(f)
    y_true, y_pred = d['y_true'], d['y_pred']

    # 建立 sequence -> test_XXXX 映射（与 supplement_gnn_explain.py 一致）
    te = pd.read_csv(GDT_DATA / f'{dataset}_test.csv')
    test_seqs = sorted(set(te['target_sequence'].dropna()))
    seq2tid = {s: f'test_{i:04d}' for i, s in enumerate(test_seqs)}

    # 构建测试集顺序（遍历 test_set 目录，与 reptile 训练脚本一致）
    prep = BASE / f'3_{dataset}_preprocessed' / 'test_set'
    target_ids = []

    def find_activities_and_seq(tdir):
        """返回 (activities_df, sequence_str)，支持两种目录结构"""
        # 结构1: activities.csv + sequence.fasta (davis)
        if (tdir / 'activities.csv').exists():
            act = pd.read_csv(tdir / 'activities.csv')
            seq = read_fasta_seq(tdir / 'sequence.fasta')
            return act, seq
        # 结构2: <name>_processed_activities.csv + <name>_processed_protein_sequence.txt (kiba/bindingdb)
        for f in tdir.iterdir():
            if f.name.endswith('_processed_activities.csv'):
                act = pd.read_csv(f)
                seq_file = tdir / f.name.replace('_processed_activities.csv', '_processed_protein_sequence.txt')
                seq = seq_file.read_text().strip() if seq_file.exists() else ''
                return act, seq
        return None, ''

    # 第一层可能是 category(kinases/default) 或直接 target
    entries = sorted(prep.iterdir())
    for entry in entries:
        if entry.is_dir():
            act, seq = find_activities_and_seq(entry)
            if act is not None:
                tid = seq2tid.get(seq, entry.name)
                target_ids.extend([tid] * len(act))
            else:
                # category 目录下有多个 target
                for tdir in sorted(entry.iterdir()):
                    if tdir.is_dir():
                        act, seq = find_activities_and_seq(tdir)
                        if act is not None:
                            tid = seq2tid.get(seq, tdir.name)
                            target_ids.extend([tid] * len(act))

    if len(target_ids) != len(y_true):
        print(f'  [WARN] {dataset}: reptile rows {len(y_true)} != built {len(target_ids)}')
        return None, None
    print(f'  [OK] {dataset}: reptile aligned, {len(set(target_ids))} unique targets')
    return {'y_true': y_true, 'y_pred': y_pred}, np.array(target_ids)


def get_cross_cluster_targets(dataset):
    """返回 40% identity 下 cross-cluster 的靶点 ID 集合"""
    pid = pd.read_csv(CLUSTER_DIR / dataset / 'per_test_target_identity.csv')
    cross = pid[~pid['same_cluster_40']]['target_id'].tolist()
    return set(cross)


def main():
    all_results = []

    for dataset in ['davis', 'kiba', 'bindingdb']:
        print(f'\n========== {dataset.upper()} ==========')
        cross_ids = get_cross_cluster_targets(dataset)
        print(f'  cross-cluster targets (40%): {len(cross_ids)}')

        # ---- GNN 模型 ----
        te, gnn_preds = load_gnn_pred(dataset)
        # 建立 target_id 映射（与 per_test_target_identity.csv 一致）
        test_seqs = sorted(set(te['target_sequence'].dropna()))
        seq2tid = {s: f'test_{i:04d}' for i, s in enumerate(test_seqs)}
        te['target_id'] = te['target_sequence'].map(seq2tid)
        cross_mask = te['target_id'].isin(cross_ids).values

        for m, p in gnn_preds.items():
            yt, yp = p['y_true'], p['y_pred']
            full = calc_metrics(yt, yp)
            sub = calc_metrics(yt[cross_mask], yp[cross_mask])
            all_results.append({
                'dataset': dataset, 'model': m, 'family': 'GNN',
                'n_full': len(yt), 'n_cross': int(cross_mask.sum()),
                **{f'full_{k}': v for k, v in full.items()},
                **{f'sub_{k}': v for k, v in sub.items()},
            })
            print(f'  {m:12s} full R2={full["R2"]:.3f} EF1={full["EF@1%"]:.2f} | '
                  f'sub R2={sub["R2"]:.3f} EF1={sub["EF@1%"]:.2f}')

        # ---- Reptile 模型 ----
        rep_pred, rep_targets = load_reptile_pred_with_targets(dataset)
        if rep_pred is not None:
            yt, yp = rep_pred['y_true'], rep_pred['y_pred']
            cross_mask_rep = np.array([t in cross_ids for t in rep_targets])
            if cross_mask_rep.sum() > 0:
                full = calc_metrics(yt, yp)
                sub = calc_metrics(yt[cross_mask_rep], yp[cross_mask_rep])
                all_results.append({
                    'dataset': dataset, 'model': 'Reptile-Transformer', 'family': 'non-GNN',
                    'n_full': len(yt), 'n_cross': int(cross_mask_rep.sum()),
                    **{f'full_{k}': v for k, v in full.items()},
                    **{f'sub_{k}': v for k, v in sub.items()},
                })
                print(f'  {"Reptile":12s} full R2={full["R2"]:.3f} EF1={full["EF@1%"]:.2f} | '
                      f'sub R2={sub["R2"]:.3f} EF1={sub["EF@1%"]:.2f}')
            else:
                print(f'  [WARN] Reptile: no cross-cluster targets matched')
        else:
            print(f'  [WARN] Reptile predictions not available for {dataset}')

    df = pd.DataFrame(all_results)
    df.to_csv(OUT_DIR / 'homology_free_ablation_results.csv', index=False, encoding='utf-8-sig')

    # 汇总：每个数据集上 full vs sub 的 winner 是否变化
    summary = {}
    for ds in df['dataset'].unique():
        sub = df[df['dataset'] == ds]
        full_winner_r2 = sub.loc[sub['full_R2'].idxmax(), 'model']
        sub_winner_r2 = sub.loc[sub['sub_R2'].idxmax(), 'model']
        full_winner_ef1 = sub.loc[sub['full_EF@1%'].idxmax(), 'model']
        sub_winner_ef1 = sub.loc[sub['sub_EF@1%'].idxmax(), 'model']
        summary[ds] = {
            'full_R2_winner': full_winner_r2,
            'sub_R2_winner': sub_winner_r2,
            'R2_winner_changed': full_winner_r2 != sub_winner_r2,
            'full_EF1_winner': full_winner_ef1,
            'sub_EF1_winner': sub_winner_ef1,
            'EF1_winner_changed': full_winner_ef1 != sub_winner_ef1,
        }
        print(f'\n{ds}: R2 winner {full_winner_r2} -> {sub_winner_r2} (changed={full_winner_r2 != sub_winner_r2})')
        print(f'{ds}: EF1 winner {full_winner_ef1} -> {sub_winner_ef1} (changed={full_winner_ef1 != sub_winner_ef1})')

    with open(OUT_DIR / 'homology_free_ablation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()
