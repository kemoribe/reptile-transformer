"""
preprocess_kiba.py - 将 DeepDTA kiba 数据集转换为 chembl targets 格式

默认输入: data/raw/kiba
默认输出: data/processed/kiba

划分方式: 按靶点(protein_id)冷启动划分 70% train / 15% val / 15% test (seed=42)
          与 Davis、BindingDB 保持一致，测试靶点在训练集中从未出现。

格式:
- train_set/default/PROTEIN_ID/
  - activities.csv (包含 molecule_id, smiles, paffinity 等)
  - sequence.fasta (蛋白质序列)
"""

import argparse
import os
import sys
import json
import pickle
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import OrderedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
KIBA_DIR = REPO_ROOT / "data" / "raw" / "kiba"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "kiba"

def main():
    global KIBA_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Preprocess the KIBA dataset")
    parser.add_argument("--input", type=Path, default=KIBA_DIR, help="KIBA raw-data directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Processed-data directory")
    parser.add_argument("--seed", type=int, default=42, help="Target split seed")
    args = parser.parse_args()

    KIBA_DIR = args.input.resolve()
    OUTPUT_DIR = args.output.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("处理 kiba 数据集")
    print("=" * 60)
    print(f"输入目录: {KIBA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    ligands = json.load(open(KIBA_DIR / "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(KIBA_DIR / "proteins.txt"), object_pairs_hook=OrderedDict)
    Y = pickle.load(open(KIBA_DIR / "Y", "rb"), encoding='latin1')

    print(f"\n药物数量: {len(ligands)}")
    print(f"蛋白质数量: {len(proteins)}")
    print(f"亲和力矩阵: {Y.shape}")

    drug_ids = list(ligands.keys())
    protein_ids = list(proteins.keys())

    all_activities = []

    for row_idx, col_idx in zip(*np.where(~np.isnan(Y))):
        drug_id = drug_ids[row_idx]
        protein_id = protein_ids[col_idx]
        affinity = Y[row_idx, col_idx]

        all_activities.append({
            'molecule_id': drug_id,
            'smiles': ligands[drug_id],
            'protein_id': protein_id,
            'affinity': affinity,
        })

    df = pd.DataFrame(all_activities)
    print(f"\n总活性记录: {len(df)}")

    # 按靶点(protein_id)冷启动划分，与 Davis/BindingDB 一致
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    seed = args.seed

    unique_proteins = sorted(df['protein_id'].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_proteins)

    n = len(unique_proteins)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_proteins = set(unique_proteins[:n_train])
    val_proteins = set(unique_proteins[n_train:n_train + n_val])
    test_proteins = set(unique_proteins[n_train + n_val:])

    def assign_split(pid):
        if pid in train_proteins:
            return 'train_set'
        elif pid in val_proteins:
            return 'val_set'
        else:
            return 'test_set'

    df['split'] = df['protein_id'].map(assign_split)

    print(f"\n靶点划分 (冷启动, seed={seed}):")
    print(f"  训练靶点: {len(train_proteins)} ({len(df[df['split']=='train_set'])} 条记录)")
    print(f"  验证靶点: {len(val_proteins)} ({len(df[df['split']=='val_set'])} 条记录)")
    print(f"  测试靶点: {len(test_proteins)} ({len(df[df['split']=='test_set'])} 条记录)")

    # 清理旧的输出目录（避免旧划分残留）
    for split in ['train_set', 'val_set', 'test_set']:
        old_dir = OUTPUT_DIR / split
        if old_dir.exists():
            shutil.rmtree(old_dir)
            print(f"  已清理旧目录: {old_dir}")

    os.makedirs(OUTPUT_DIR / "train_set" / "default", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "val_set" / "default", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "test_set" / "default", exist_ok=True)
    
    protein_counts = df.groupby('protein_id').size()
    valid_proteins = protein_counts[protein_counts >= 3].index
    print(f"\n有效蛋白质(>=3 drugs): {len(valid_proteins)}")
    
    for protein_id in valid_proteins:
        df_protein = df[df['protein_id'] == protein_id]
        seq = proteins.get(protein_id, '')
        
        for split in ['train_set', 'val_set', 'test_set']:
            df_split = df_protein[df_protein['split'] == split]
            if len(df_split) == 0:
                continue
            
            output_dir = OUTPUT_DIR / split / "default" / protein_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            df_out = df_split[['molecule_id', 'smiles', 'affinity']].copy()
            df_out.columns = ['molecule_id', 'smiles', 'paffinity']
            df_out['paffinity_type'] = 'pKd'
            df_out['affinity_type'] = 'Kd'
            df_out['measurement_count'] = 1
            df_out['paffinity_std'] = 0.0
            
            activities_file = output_dir / f"{protein_id}_processed_activities.csv"
            df_out.to_csv(activities_file, index=False)
            
            seq_file = output_dir / f"{protein_id}_processed_protein_sequence.txt"
            with open(seq_file, 'w') as f:
                f.write(seq)
    
    df.to_csv(OUTPUT_DIR / "combined_activities.csv", index=False)
    
    print("\n" + "=" * 60)
    print("预处理完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
