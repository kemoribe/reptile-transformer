"""
将 BindingDB 预处理数据转换为 GraphDTA 格式

生成文件（保存到 GrapthDTA/data/bindingdb/）:
  - ligands_can.txt: JSON OrderedDict {drug_id: canonical_smiles}
  - proteins.txt:    JSON OrderedDict {protein_id: sequence}
  - Y:               pickle, 2D numpy array (n_drugs x n_proteins), NaN 表示缺失
  - target_split.json: {protein_id: 'train_set'|'val_set'|'test_set'}

冷启动划分策略:
  同一蛋白序列可能出现在多个 split 中。
  按 test_set > val_set > train_set 优先级，将每个蛋白分配到唯一 split，
  确保测试靶点的所有数据都进入测试集。
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles


# ==========================
# 工具函数
# ==========================
def canonical_smiles(smiles):
    """规范化 SMILES，失败返回 None"""
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None
    return MolToSmiles(mol, isomericSmiles=True)


def read_sequence(seq_file):
    """读取蛋白质序列文件（兼容纯文本和 FASTA 格式）"""
    with open(seq_file, 'r') as f:
        content = f.read().strip()
    lines = [line.strip() for line in content.split('\n') if not line.startswith('>')]
    sequence = ''.join(lines)
    sequence = ''.join([c for c in sequence if c in 'ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy'])
    return sequence.upper() if sequence else None


def find_activities_file(target_dir, target_name):
    """查找靶点的活性 CSV 文件"""
    patterns = [
        f"{target_name}_activities.csv",
        f"{target_name}_processed_activities.csv",
        "activities.csv",
    ]
    for pattern in patterns:
        f = target_dir / pattern
        if f.exists() and f.stat().st_size > 0:
            return f
    return None


def find_sequence_file(target_dir, target_name):
    """查找靶点的序列文件"""
    patterns = [
        f"{target_name}_sequence.fasta",
        f"{target_name}_processed_protein_sequence.txt",
        f"{target_name}_sequence.txt",
    ]
    for pattern in patterns:
        f = target_dir / pattern
        if f.exists() and f.stat().st_size > 0:
            return f
    for ext in ['.fasta', '.fa', '.faa', '.txt']:
        f = target_dir / f"sequence{ext}"
        if f.exists() and f.stat().st_size > 0:
            return f
    for f in target_dir.iterdir():
        if f.suffix in ['.fasta', '.fa', '.faa', '.txt'] and 'sequence' in f.name.lower():
            if f.stat().st_size > 0:
                return f
    return None


def read_activities(activities_file):
    """读取活性 CSV，返回 [(smiles, affinity), ...]"""
    df = pd.read_csv(activities_file)

    if 'smiles' in df.columns and 'paffinity' in df.columns:
        return list(zip(df['smiles'].tolist(),
                        df['paffinity'].values.astype(float)))

    if 'canonical_smiles' in df.columns and 'standard_value' in df.columns:
        return list(zip(df['canonical_smiles'].tolist(),
                        df['standard_value'].values.astype(float)))

    if 'standardized_smiles' in df.columns and 'paffinity' in df.columns:
        return list(zip(df['standardized_smiles'].tolist(),
                        df['paffinity'].values.astype(float)))

    if 'smiles' in df.columns and 'affinity' in df.columns:
        return list(zip(df['smiles'].tolist(),
                        df['affinity'].values.astype(float)))

    raise ValueError(f"无法识别的 CSV 格式: {activities_file}\n列: {list(df.columns)}")


# ==========================
# 主处理函数
# ==========================
SPLIT_PRIORITY = {'test_set': 3, 'val_set': 2, 'train_set': 1}


def process_dataset(src_dir, dataset_name, output_base):
    """处理单个数据集，保存为 GraphDTA 格式"""
    src_dir = Path(src_dir)
    output_dir = Path(output_base) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"源目录: {src_dir}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")

    all_pairs = []
    n_targets = 0
    n_targets_skipped = 0

    for split in ['train_set', 'val_set', 'test_set']:
        split_dir = src_dir / split
        if not split_dir.exists():
            continue

        for category_dir in sorted(split_dir.iterdir()):
            if not category_dir.is_dir():
                continue

            for target_dir in sorted(category_dir.iterdir()):
                if not target_dir.is_dir():
                    continue

                target_name = target_dir.name
                activities_file = find_activities_file(target_dir, target_name)
                seq_file = find_sequence_file(target_dir, target_name)

                if activities_file is None or seq_file is None:
                    n_targets_skipped += 1
                    continue

                sequence = read_sequence(seq_file)
                if sequence is None or len(sequence) == 0:
                    n_targets_skipped += 1
                    continue

                try:
                    pairs = read_activities(activities_file)
                except Exception as e:
                    print(f"  警告: 读取 {activities_file} 失败: {e}")
                    n_targets_skipped += 1
                    continue

                n_targets += 1
                for smiles, affinity in pairs:
                    canon = canonical_smiles(smiles)
                    if canon is None:
                        continue
                    if not np.isfinite(affinity):
                        continue
                    all_pairs.append((canon, sequence, float(affinity), target_name, split))

    print(f"  扫描靶点: {n_targets} 个有效, {n_targets_skipped} 个跳过")
    print(f"  收集配对: {len(all_pairs)} 条")

    if len(all_pairs) == 0:
        print("  错误: 没有有效数据!")
        return

    drug_set = OrderedDict()
    protein_set = OrderedDict()

    protein_splits = OrderedDict()

    for smiles, sequence, affinity, target_name, split in all_pairs:
        if smiles not in drug_set:
            drug_set[smiles] = f"drug_{len(drug_set)}"
        if sequence not in protein_set:
            pid = f"protein_{len(protein_set)}"
            protein_set[sequence] = pid
            protein_splits[sequence] = set()
        protein_splits[sequence].add(split)

    target_split = {}
    split_protein_counts = {'train_set': 0, 'val_set': 0, 'test_set': 0}
    duplicate_proteins = 0

    for sequence, pid in protein_set.items():
        splits = protein_splits[sequence]
        if len(splits) > 1:
            duplicate_proteins += 1
        best_split = max(splits, key=lambda s: SPLIT_PRIORITY.get(s, 0))
        target_split[pid] = best_split
        split_protein_counts[best_split] = split_protein_counts.get(best_split, 0) + 1

    n_drugs = len(drug_set)
    n_proteins = len(protein_set)
    print(f"  唯一药物: {n_drugs}")
    print(f"  唯一蛋白: {n_proteins} (其中 {duplicate_proteins} 个出现在多个 split 中)")
    print(f"  冷启动划分: {split_protein_counts}")

    drug_to_idx = {s: i for i, s in enumerate(drug_set.keys())}
    prot_to_idx = {s: i for i, s in enumerate(protein_set.keys())}

    print(f"  构建 Y 矩阵: {n_drugs} x {n_proteins} ...")
    Y = np.full((n_drugs, n_proteins), np.nan, dtype=np.float32)

    n_filled = 0
    n_averaged = 0
    for smiles, sequence, affinity, target_name, split in all_pairs:
        i = drug_to_idx[smiles]
        j = prot_to_idx[sequence]
        if np.isnan(Y[i, j]):
            Y[i, j] = affinity
            n_filled += 1
        else:
            Y[i, j] = (Y[i, j] + affinity) / 2.0
            n_averaged += 1

    print(f"  Y 矩阵填充: {n_filled} 个配对 (其中 {n_averaged} 个多次测量取平均)")
    print(f"  Y 矩阵非 NaN 比例: {np.sum(~np.isnan(Y)) / Y.size:.4f}")

    split_pair_counts = {'train_set': 0, 'val_set': 0, 'test_set': 0}
    for sequence, pid in protein_set.items():
        j = prot_to_idx[sequence]
        split = target_split[pid]
        col = Y[:, j]
        n_valid = np.sum(~np.isnan(col))
        split_pair_counts[split] = split_pair_counts.get(split, 0) + n_valid
    print(f"  各 split 配对数: {split_pair_counts}")

    ligands_dict = OrderedDict()
    for smiles, drug_id in drug_set.items():
        ligands_dict[drug_id] = smiles
    with open(output_dir / 'ligands_can.txt', 'w') as f:
        json.dump(ligands_dict, f)
    print(f"  已保存 ligands_can.txt ({len(ligands_dict)} 个药物)")

    proteins_dict = OrderedDict()
    for sequence, protein_id in protein_set.items():
        proteins_dict[protein_id] = sequence
    with open(output_dir / 'proteins.txt', 'w') as f:
        json.dump(proteins_dict, f)
    print(f"  已保存 proteins.txt ({len(proteins_dict)} 个蛋白)")

    with open(output_dir / 'Y', 'wb') as f:
        pickle.dump(Y, f)
    print(f"  已保存 Y ({Y.shape})")

    with open(output_dir / 'target_split.json', 'w') as f:
        json.dump(target_split, f, indent=2)
    print(f"  已保存 target_split.json")

    print(f"\n  [OK] 数据集 {dataset_name} 转换完成!")


if __name__ == '__main__':
    import argparse
    BASE = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert BindingDB to GraphDTA format")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=BASE / "data")
    args = parser.parse_args()
    process_dataset(args.input_dir, "bindingdb", args.output_root)
