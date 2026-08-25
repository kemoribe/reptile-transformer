"""
3_all_data_chembl_targets_preprocessed.py - 大规模靶点数据预处理

处理 2_all_data_chembl_targets.py 生成的数据：
- 目录结构: train_set/kinases/CHEMBLXXX/
- 文件: activities.csv, sequence.fasta, target_info.json

运行方式：
    python preprocessing/3_all_data_chembl_targets_preprocessed.py \
        --input data/raw/chembl/2_all_data_chembl_targets \
        --output data/processed/chembl
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover, Descriptors

# 数据目录配置 - 兼容本地和 Kaggle Notebook

KAGGLE_INPUT_DIR = Path("/kaggle/input/reptile_need_chembl")
KAGGLE_WORKING_DIR = Path("/kaggle/working")
REPO_ROOT = Path(__file__).resolve().parents[1]

if KAGGLE_INPUT_DIR.exists():
    print(f"[INFO] Running on Kaggle, using input: {KAGGLE_INPUT_DIR}")
    DATA_DIR = KAGGLE_INPUT_DIR / "2_all_data_chembl_targets"
    OUTPUT_DIR = KAGGLE_WORKING_DIR / "3_all_data_chembl_targets_preprocessed"
else:
    DATA_DIR = REPO_ROOT / "data" / "raw" / "chembl" / "2_all_data_chembl_targets"
    OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "chembl"

PROTEIN_MIN_LEN = 50
PROTEIN_MAX_LEN = 2000
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

# 放宽过滤条件，保留更多靶点数据
ALLOWED_STANDARD_TYPES = ['Ki', 'IC50', 'Kd']  # 保留三种类型
EXACT_RELATIONS = {'='}
MIN_DRUGS_PER_PROTEIN = 3  # 降低最低药物数要求

UNIT_TO_NM = {
    'nm': 1.0,
    'um': 1e3,
    'µm': 1e3,
    'μm': 1e3,
    'mm': 1e6,
    'pm': 1e-3,
    'fm': 1e-6,
    'm': 1e9,
}

def standardize_smiles(smiles):
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

def normalize_to_nm(value, units):
    if pd.isna(value) or value <= 0 or pd.isna(units):
        return None
    unit = str(units).strip().lower()
    factor = UNIT_TO_NM.get(unit)
    if factor is None:
        return None
    return value * factor

def convert_to_plog(value_nm):
    if value_nm is None or pd.isna(value_nm) or value_nm <= 0:
        return None
    try:
        return -np.log10(value_nm * 1e-9)
    except:
        return None

def process_activities(target_dir, target_name):
    """处理活性数据 - 读取 activities.csv"""
    activities_file = target_dir / "activities.csv"
    if not activities_file.exists():
        return None, None

    df = pd.read_csv(activities_file)
    
    # 兼容两种列名格式
    if 'canonical_smiles' in df.columns:
        smiles_col = 'canonical_smiles'
    elif 'smiles' in df.columns:
        smiles_col = 'smiles'
    else:
        print(f"  No SMILES column found")
        return None, None
    
    required_cols = {'standard_value', 'standard_type', 'standard_units'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"  Missing required columns: {sorted(missing_cols)}")
        return None, None

    df = df.dropna(subset=['standard_value', smiles_col, 'standard_type', 'standard_units'])
    df = df[df['standard_type'].isin(ALLOWED_STANDARD_TYPES)]

    if 'standard_relation' in df.columns:
        df = df[df['standard_relation'].fillna('=').astype(str).str.strip().isin(EXACT_RELATIONS)]
    else:
        print("  standard_relation column missing, skipping target for safety")
        return None, None

    df_valid = df.copy()
    df_valid['target_name'] = target_name

    df_valid['paffinity_type'] = df_valid['standard_type'].map({
        'IC50': 'pIC50',
        'Kd': 'pKd',
        'Ki': 'pKi'
    })

    df_valid['standard_value_nm'] = df_valid.apply(
        lambda row: normalize_to_nm(row['standard_value'], row['standard_units']), axis=1
    )
    df_valid = df_valid.dropna(subset=['standard_value_nm'])

    df_valid['plog_value'] = df_valid.apply(
        lambda row: convert_to_plog(row['standard_value_nm']), axis=1
    )
    df_valid = df_valid.dropna(subset=['plog_value'])

    df_valid['standardized_smiles'] = df_valid[smiles_col].apply(standardize_smiles)
    df_valid = df_valid.dropna(subset=['standardized_smiles'])

    group_keys = ['standardized_smiles', 'target_name', 'standard_type']
    agg_spec = {
        'plog_value': ['median', 'std', 'count'],
        'paffinity_type': 'first',
        smiles_col: 'first'
    }
    if 'molecule_chembl_id' in df_valid.columns:
        agg_spec['molecule_chembl_id'] = 'first'

    df_grouped = df_valid.groupby(group_keys).agg(agg_spec).reset_index()
    df_grouped.columns = [
        'standardized_smiles',
        'target_name',
        'standard_type',
        'plog_value',
        'plog_std',
        'measurement_count',
        'paffinity_type',
        'original_smiles',
        'molecule_id'
    ] if 'molecule_chembl_id' in df_valid.columns else [
        'standardized_smiles',
        'target_name',
        'standard_type',
        'plog_value',
        'plog_std',
        'measurement_count',
        'paffinity_type',
        'original_smiles'
    ]

    if 'molecule_id' not in df_grouped.columns:
        df_grouped['molecule_id'] = df_grouped['standardized_smiles']

    df_grouped['plog_std'] = df_grouped['plog_std'].fillna(0.0)

    result = df_grouped[
        ['molecule_id', 'standardized_smiles', 'plog_value', 'paffinity_type', 'standard_type', 'measurement_count', 'plog_std']
    ].copy()
    result.columns = ['molecule_id', 'smiles', 'paffinity', 'paffinity_type', 'affinity_type', 'measurement_count', 'paffinity_std']

    return result, df_valid

def process_protein_sequence(target_dir, target_name):
    """处理蛋白序列 - 读取 sequence.fasta"""
    seq_file = target_dir / "sequence.fasta"
    if not seq_file.exists():
        return None

    with open(seq_file, 'r') as f:
        lines = f.readlines()
    
    # FASTA格式：第一行是描述，后面是序列
    sequence = ''
    for line in lines:
        if not line.startswith('>'):
            sequence += line.strip()

    sequence = sequence.replace('\n', '').replace(' ', '').upper()

    if len(sequence) < PROTEIN_MIN_LEN or len(sequence) > PROTEIN_MAX_LEN:
        return None

    non_standard = sum(1 for aa in sequence if aa not in STANDARD_AA)
    if non_standard > 0:
        return None

    return sequence

def filter_proteins_by_drug_count(df_activities, min_drugs=MIN_DRUGS_PER_PROTEIN):
    drug_counts = df_activities.groupby('target_name')['molecule_id'].nunique()
    valid_proteins = drug_counts[drug_counts >= min_drugs].index
    return set(valid_proteins)

def get_all_targets():
    """获取所有靶点（遍历 train_set/val_set/test_set 下的所有类别和靶点）"""
    targets = []
    target_paths = []
    
    for set_dir in DATA_DIR.iterdir():
        if not set_dir.is_dir():
            continue
        
        for category_dir in set_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            for target_dir in category_dir.iterdir():
                if target_dir.is_dir():
                    chembl_id = target_dir.name
                    targets.append(chembl_id)
                    target_paths.append((set_dir.name, category_dir.name, chembl_id))
    
    return sorted(targets), target_paths

def main():
    global DATA_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Preprocess target-grouped ChEMBL data")
    parser.add_argument("--input", type=Path, default=DATA_DIR, help="Input target directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Processed-data directory")
    args = parser.parse_args()

    DATA_DIR = args.input.resolve()
    OUTPUT_DIR = args.output.resolve()
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"ChEMBL input directory not found: {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets, target_paths = get_all_targets()
    print(f"Found {len(targets)} unique targets")

    all_activities = []
    train_activities = []
    stats = {}

    for set_name, category, chembl_id in target_paths:
        target_dir = DATA_DIR / set_name / category / chembl_id
        output_target_dir = OUTPUT_DIR / set_name / category / chembl_id
        output_target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {set_name}/{category}/{chembl_id}...")

        df_activities, df_raw = process_activities(target_dir, chembl_id)

        protein_seq = process_protein_sequence(target_dir, chembl_id)

        if df_activities is not None and len(df_activities) > 0:
            activities_output = output_target_dir / f"{chembl_id}_processed_activities.csv"
            df_activities.to_csv(activities_output, index=False)
            print(f"  Valid activities: {len(df_activities)} records")

            df_activities['target_name'] = chembl_id
            df_activities['set_name'] = set_name
            df_activities['category'] = category
            all_activities.append(df_activities)
            if set_name == 'train_set':
                train_activities.append(df_activities.copy())
        else:
            print(f"  No valid activities found")
            df_activities = pd.DataFrame(
                columns=['molecule_id', 'smiles', 'paffinity', 'paffinity_type', 'affinity_type', 'measurement_count', 'paffinity_std']
            )

        if protein_seq is not None:
            seq_output = output_target_dir / f"{chembl_id}_processed_protein_sequence.txt"
            with open(seq_output, 'w') as f:
                f.write(protein_seq)
            print(f"  Protein sequence: length={len(protein_seq)}")
        else:
            print(f"  Invalid protein sequence")

        ic50_count = len(df_activities[df_activities['affinity_type'] == 'IC50']) if len(df_activities) > 0 else 0
        kd_count = len(df_activities[df_activities['affinity_type'] == 'Kd']) if len(df_activities) > 0 else 0
        ki_count = len(df_activities[df_activities['affinity_type'] == 'Ki']) if len(df_activities) > 0 else 0

        stats[chembl_id] = {
            'set_name': set_name,
            'category': category,
            'activities': len(df_activities),
            'protein_length': len(protein_seq) if protein_seq else 0,
            'ic50_count': ic50_count,
            'kd_count': kd_count,
            'ki_count': ki_count
        }

    if len(all_activities) > 0:
        combined = pd.concat(all_activities, ignore_index=True)
        train_combined = pd.concat(train_activities, ignore_index=True) if len(train_activities) > 0 else pd.DataFrame()
        all_drug_counts = combined.groupby('target_name')['molecule_id'].nunique()
        valid_proteins = all_drug_counts[all_drug_counts >= MIN_DRUGS_PER_PROTEIN].index
        combined = combined[combined['target_name'].isin(valid_proteins)].copy()

        print(f"\n{'='*60}")
        print(f"After filtering using all data (min {MIN_DRUGS_PER_PROTEIN} drugs per protein):")
        print(f"Valid proteins: {len(valid_proteins)}")

        # 按类别统计
        category_stats = combined.groupby(['set_name', 'category'])['target_name'].nunique().unstack(fill_value=0)
        print("\nTarget distribution by set and category:")
        print(category_stats)

        # 删除不符合条件的靶点目录
        for chembl_id, info in stats.items():
            if chembl_id not in valid_proteins:
                target_dir = OUTPUT_DIR / info['set_name'] / info['category'] / chembl_id
                if target_dir.exists():
                    for f in target_dir.glob('*'):
                        f.unlink()
                    target_dir.rmdir()
                    print(f"  Removed {chembl_id} (insufficient drugs)")

        # 保存合并后的所有数据
        combined.to_csv(OUTPUT_DIR / 'combined_activities.csv', index=False)
        print(f"\nCombined data saved to: {OUTPUT_DIR / 'combined_activities.csv'}")

    print("\n" + "="*60)
    print("Preprocessing Summary")
    print("="*60)
    
    # 按类别和数据集统计
    print(f"\n{'Set':<10} {'Category':<15} {'Target':<15} {'Activities':<12} {'Protein Len':<14}")
    print("-"*70)
    for chembl_id, s in stats.items():
        print(f"{s['set_name']:<10} {s['category']:<15} {chembl_id:<15} {s['activities']:<12} {s['protein_length']:<14}")

    # 总体统计
    total_targets = len(stats)
    total_activities = sum(s['activities'] for s in stats.values())
    valid_targets = len([s for s in stats.values() if s['activities'] >= MIN_DRUGS_PER_PROTEIN])
    
    print(f"\n{'='*60}")
    print(f"Total targets: {total_targets}")
    print(f"Valid targets (>= {MIN_DRUGS_PER_PROTEIN} drugs): {valid_targets}")
    print(f"Total activities: {total_activities}")
    print(f"{'='*60}")

    print(f"\nPreprocessed data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
