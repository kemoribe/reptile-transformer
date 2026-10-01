# -*- coding: utf-8 -*-
"""
实验1：τ 测量方式敏感性分析
比较 4 种 τ 计算口径下各数据集的相似度排序与 winner 身份是否变化：
  (a) ECFP4 指纹，训练集药物两两 Tanimoto 均值（第一轮原始口径）
  (b) MACCS 指纹，训练集药物两两 Tanimoto 均值
  (c) ECFP4 指纹，测试集药物两两 Tanimoto 均值
  (d) MACCS 指纹，测试集药物两两 Tanimoto 均值

输出: tau_sensitivity_results.csv + tau_sensitivity_summary.json
"""
import os
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs
from pathlib import Path

DATA_DIR = Path(r'd:\lht\GrapthDTA\data')
OUT_DIR = Path(r'd:\lht\修改supplement_output\revision2_experiments\e7_tau_sensitivity')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['chembl', 'davis', 'kiba', 'bindingdb']

# 第一轮 Table S3 的 winner（R²准则 / EF@1%准则），用于对比
ROUND1_WINNERS = {
    'chembl':  {'R2': 'Reptile', 'EF1': 'MLP'},
    'davis':   {'R2': 'GAT_GCN', 'EF1': 'GINConvNet'},
    'kiba':    {'R2': 'Reptile', 'EF1': 'Reptile'},
    'bindingdb': {'R2': 'Reptile', 'EF1': 'Transformer'},
}


def fp_ecfp4(smi):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None


def fp_maccs(smi):
    m = Chem.MolFromSmiles(smi)
    return MACCSkeys.GenMACCSKeys(m) if m else None


def avg_tanimoto(smiles, fp_func):
    fps = [f for f in (fp_func(s) for s in smiles) if f is not None]
    if len(fps) < 2:
        return np.nan
    # 两两均值（抽样加速：若分子太多则随机采样上限）
    n = len(fps)
    if n > 2000:
        idx = np.random.RandomState(42).choice(n, 2000, replace=False)
        fps = [fps[i] for i in idx]
        n = 2000
    sims = DataStructs.BulkTanimotoSimilarity(fps[0], fps[1:])
    total = sum(sims)
    count = len(sims)
    for i in range(1, n):
        s = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total += sum(s)
        count += len(s)
    return total / count if count > 0 else np.nan


def main():
    rows = []
    for ds in DATASETS:
        print(f'=== {ds} ===')
        tr = pd.read_csv(DATA_DIR / f'{ds}_train.csv')
        te = pd.read_csv(DATA_DIR / f'{ds}_test.csv')
        tr_smi = pd.unique(tr['compound_iso_smiles'].dropna())
        te_smi = pd.unique(te['compound_iso_smiles'].dropna())
        print(f'  train drugs: {len(tr_smi)}, test drugs: {len(te_smi)}')

        tau_ecfp4_train = avg_tanimoto(tr_smi, fp_ecfp4)
        tau_maccs_train = avg_tanimoto(tr_smi, fp_maccs)
        tau_ecfp4_test = avg_tanimoto(te_smi, fp_ecfp4)
        tau_maccs_test = avg_tanimoto(te_smi, fp_maccs)

        rows.append({
            'dataset': ds,
            'n_train_drugs': len(tr_smi),
            'n_test_drugs': len(te_smi),
            'tau_ECFP4_train': round(tau_ecfp4_train, 4),
            'tau_MACCS_train': round(tau_maccs_train, 4),
            'tau_ECFP4_test': round(tau_ecfp4_test, 4),
            'tau_MACCS_test': round(tau_maccs_test, 4),
        })
        print(f'  ECFP4 train={tau_ecfp4_train:.4f}  MACCS train={tau_maccs_train:.4f}  '
              f'ECFP4 test={tau_ecfp4_test:.4f}  MACCS test={tau_maccs_test:.4f}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'tau_sensitivity_results.csv', index=False, encoding='utf-8-sig')

    # 排序分析：4 种口径下数据集相似度排序是否一致
    summary = {'per_dataset_tau': rows}
    for col in ['tau_ECFP4_train', 'tau_MACCS_train', 'tau_ECFP4_test', 'tau_MACCS_test']:
        order = df.sort_values(col)['dataset'].tolist()
        summary[f'ranking_{col}'] = order
        print(f'Ranking by {col}: {order}')

    # Spearman 相关：4 种口径两两之间的排名相关性
    from scipy.stats import spearmanr
    cols = ['tau_ECFP4_train', 'tau_MACCS_train', 'tau_ECFP4_test', 'tau_MACCS_test']
    corr_matrix = {}
    for c1 in cols:
        corr_matrix[c1] = {}
        for c2 in cols:
            rho, p = spearmanr(df[c1], df[c2])
            corr_matrix[c1][c2] = {'spearman_rho': round(rho, 4), 'p_value': round(p, 4)}
    summary['spearman_correlation_matrix'] = corr_matrix

    # Winner 一致性分析：用 τ=0.13 阈值划分低/高相似度，检查各口径下分组是否变化
    def group_by_tau(tau_val):
        return 'low' if tau_val < 0.13 else 'high'

    for col in cols:
        df[f'group_{col}'] = df[col].apply(group_by_tau)
    group_cols = [f'group_{c}' for c in cols]
    summary['group_consistency'] = df[['dataset'] + group_cols].to_dict(orient='records')
    all_same = all(df[group_cols].nunique(axis=1) == 1)
    summary['all_datasets_same_group_across_methods'] = bool(all_same)
    print(f'\nAll datasets keep same low/high group across 4 methods: {all_same}')

    with open(OUT_DIR / 'tau_sensitivity_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'\nSaved to {OUT_DIR}')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
