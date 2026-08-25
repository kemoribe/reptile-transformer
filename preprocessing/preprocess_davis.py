"""
Davis数据集预处理脚本

将Davis数据集处理成和3_all_data_chembl_targets_preprocessed.py一样的格式：
- 目录结构: train_set/kinases/DAVISXXX/
- 文件: activities.csv, sequence.fasta, target_info.json

运行方式：
    python preprocessing/preprocess_davis.py
    python preprocessing/preprocess_davis.py --input data/raw/davis --output data/processed/davis
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover

# 数据目录配置
REPO_ROOT = Path(__file__).resolve().parents[1]
DAVIS_DIR = REPO_ROOT / "data" / "raw" / "davis"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "davis"

# 分割比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 蛋白质过滤条件
PROTEIN_MIN_LEN = 50
PROTEIN_MAX_LEN = 2000
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')


def standardize_smiles(smiles):
    """标准化SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        AllChem.SanitizeMol(mol)

        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)

        for atom in mol.GetAtoms():
            atom.SetIsotope(0)

        Chem.SanitizeMol(mol, catchErrors=True)

        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

        if canonical_smiles == '' or '.' in canonical_smiles:
            return None

        return canonical_smiles
    except:
        return None


def validate_protein_sequence(sequence):
    """验证蛋白质序列"""
    sequence = sequence.replace('\n', '').replace(' ', '').upper()
    
    if len(sequence) < PROTEIN_MIN_LEN or len(sequence) > PROTEIN_MAX_LEN:
        return None
    
    non_standard = sum(1 for aa in sequence if aa not in STANDARD_AA)
    if non_standard > 0:
        return None
    
    return sequence


def main():
    global DAVIS_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Preprocess the Davis dataset")
    parser.add_argument("--input", type=Path, default=DAVIS_DIR, help="Davis raw-data directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Processed-data directory")
    parser.add_argument("--seed", type=int, default=42, help="Target split seed")
    args = parser.parse_args()

    DAVIS_DIR = args.input.resolve()
    OUTPUT_DIR = args.output.resolve()

    print("=" * 60)
    print("📦 Davis数据集预处理")
    print("=" * 60)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 读取数据
    print("\n📥 读取Davis数据集...")
    
    drugs_df = pd.read_csv(DAVIS_DIR / "drugs.csv")
    proteins_df = pd.read_csv(DAVIS_DIR / "proteins.csv")
    affinity_df = pd.read_csv(DAVIS_DIR / "drug_protein_affinity.csv")
    
    print(f"   药物数量: {len(drugs_df)}")
    print(f"   蛋白质数量: {len(proteins_df)}")
    print(f"   亲和力样本: {len(affinity_df)}")
    
    # 合并数据
    print("\n🔗 合并数据...")
    
    # 将药物索引映射到SMILES
    drug_map = drugs_df.set_index('Drug_Index')['Canonical_SMILES'].to_dict()
    
    # 将蛋白质索引映射到信息
    protein_map = proteins_df.set_index('Protein_Index')[['Accession_Number', 'Gene_Name', 'Sequence']].to_dict('index')
    
    # 添加药物和蛋白质信息到亲和力数据
    affinity_df['smiles'] = affinity_df['Drug_Index'].map(drug_map)
    affinity_df['protein_id'] = affinity_df['Protein_Index'].apply(lambda x: protein_map.get(x, {}).get('Accession_Number', f'protein_{x}'))
    affinity_df['gene_name'] = affinity_df['Protein_Index'].apply(lambda x: protein_map.get(x, {}).get('Gene_Name', f'protein_{x}'))
    affinity_df['sequence'] = affinity_df['Protein_Index'].apply(lambda x: protein_map.get(x, {}).get('Sequence', ''))
    
    # 标准化SMILES
    print("\n🧹 标准化SMILES...")
    affinity_df['standardized_smiles'] = affinity_df['smiles'].apply(standardize_smiles)
    affinity_df = affinity_df.dropna(subset=['standardized_smiles'])
    
    # 验证蛋白质序列
    print("🧹 验证蛋白质序列...")
    affinity_df['valid_sequence'] = affinity_df['sequence'].apply(validate_protein_sequence)
    affinity_df = affinity_df.dropna(subset=['valid_sequence'])
    
    # 重命名亲和力列（Davis数据集的Affinity已经是pKd格式）
    affinity_df['paffinity'] = affinity_df['Affinity']
    affinity_df['affinity_type'] = 'Kd'
    affinity_df['paffinity_type'] = 'pKd'
    
    # 统计每个靶点的药物数量
    print("\n📊 统计靶点信息...")
    drug_counts = affinity_df.groupby('gene_name')['Drug_Index'].nunique()
    print(f"   有效靶点数量: {len(drug_counts)}")
    print(f"   每个靶点平均药物数: {drug_counts.mean():.1f}")
    print(f"   药物数范围: [{drug_counts.min()}, {drug_counts.max()}]")
    
    # 划分训练/验证/测试集（按蛋白质划分）
    print("\n📋 划分数据集...")
    
    unique_proteins = affinity_df['gene_name'].unique()
    np.random.seed(args.seed)
    np.random.shuffle(unique_proteins)
    
    n_train = int(len(unique_proteins) * TRAIN_RATIO)
    n_val = int(len(unique_proteins) * VAL_RATIO)
    
    train_proteins = set(unique_proteins[:n_train])
    val_proteins = set(unique_proteins[n_train:n_train+n_val])
    test_proteins = set(unique_proteins[n_train+n_val:])
    
    affinity_df['set_name'] = affinity_df['gene_name'].apply(
        lambda x: 'train_set' if x in train_proteins else ('val_set' if x in val_proteins else 'test_set')
    )
    
    print(f"   训练靶点: {len(train_proteins)}")
    print(f"   验证靶点: {len(val_proteins)}")
    print(f"   测试靶点: {len(test_proteins)}")
    
    # 按靶点分组处理
    print("\n🔧 按靶点生成文件...")
    
    grouped = affinity_df.groupby('gene_name')
    
    stats = []
    for gene_name, group in grouped:
        # 获取靶点信息
        protein_info = group.iloc[0]
        set_name = protein_info['set_name']
        accession_number = protein_info['protein_id']
        sequence = protein_info['valid_sequence']
        
        # 创建靶点目录（格式：set_name/kinases/DAVIS_<gene_name>）
        target_dir = OUTPUT_DIR / set_name / "kinases" / f"DAVIS_{gene_name}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成activities.csv
        activities_df = group[['standardized_smiles', 'paffinity', 'affinity_type', 'paffinity_type']].copy()
        activities_df.columns = ['canonical_smiles', 'standard_value', 'standard_type', 'paffinity_type']
        activities_df['standard_units'] = 'nM'
        activities_df['standard_relation'] = '='
        activities_df['molecule_chembl_id'] = group['Drug_Index'].astype(str)
        
        activities_output = target_dir / "activities.csv"
        activities_df.to_csv(activities_output, index=False)
        
        # 生成sequence.fasta
        seq_output = target_dir / "sequence.fasta"
        with open(seq_output, 'w') as f:
            f.write(f">{accession_number} | {gene_name}\n")
            f.write(f"{sequence}\n")
        
        # 生成target_info.json
        target_info = {
            'target_name': f'DAVIS_{gene_name}',
            'accession_number': accession_number,
            'gene_name': gene_name,
            'protein_length': len(sequence),
            'n_drugs': len(group),
            'affinity_range': {
                'min': float(group['paffinity'].min()),
                'max': float(group['paffinity'].max()),
                'mean': float(group['paffinity'].mean())
            }
        }
        
        info_output = target_dir / "target_info.json"
        with open(info_output, 'w') as f:
            json.dump(target_info, f, indent=2)
        
        stats.append({
            'target_name': f'DAVIS_{gene_name}',
            'set_name': set_name,
            'gene_name': gene_name,
            'activities': len(group),
            'protein_length': len(sequence)
        })
        
        print(f"   {set_name}/kinases/DAVIS_{gene_name}: {len(group)} activities")
    
    # 生成统计文件
    print("\n📊 生成统计文件...")
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUTPUT_DIR / "target_stats.csv", index=False)
    
    # 生成合并后的activities文件
    combined_df = affinity_df[['standardized_smiles', 'paffinity', 'affinity_type', 'gene_name', 'set_name']].copy()
    combined_df.to_csv(OUTPUT_DIR / "combined_activities.csv", index=False)
    
    # 打印总结
    print("\n" + "="*60)
    print("预处理完成")
    print("="*60)
    
    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"\n数据统计:")
    print(f"  训练集: {stats_df[stats_df['set_name']=='train_set']['activities'].sum()} activities, {len(train_proteins)} targets")
    print(f"  验证集: {stats_df[stats_df['set_name']=='val_set']['activities'].sum()} activities, {len(val_proteins)} targets")
    print(f"  测试集: {stats_df[stats_df['set_name']=='test_set']['activities'].sum()} activities, {len(test_proteins)} targets")
    print(f"  总计: {stats_df['activities'].sum()} activities, {len(stats_df)} targets")
    
    print(f"\n目录结构示例:")
    print(f"  {OUTPUT_DIR}/train_set/kinases/DAVIS_<gene_name>/")
    print(f"    ├── activities.csv")
    print(f"    ├── sequence.fasta")
    print(f"    └── target_info.json")


if __name__ == "__main__":
    main()
