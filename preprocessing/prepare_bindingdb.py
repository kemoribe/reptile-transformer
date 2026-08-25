"""
从 Harvard Dataverse 直接下载 BindingDB 数据，处理成与 3_all_data_chembl_targets_preprocessed 相同的结构。

三种下载方式（自动选择最快的）:
  1. 本地已有缓存文件 → 直接使用
  2. 手动下载 → 用户自行下载文件放到 data/ 目录
  3. 直接从 Harvard Dataverse 下载（带进度条 + 断点续传）

BindingDB 子数据集:
  - BindingDB_Kd   (~52K pairs, 1.4K proteins)  file_id=4291555
  - BindingDB_IC50  (~991K pairs, 5K proteins)   file_id=4291560
  - BindingDB_Ki    (~375K pairs, 3K proteins)   file_id=4291556

输出结构:
  output_dir/
  ├── train_set/{category}/{target_id}/{target_id}_processed_activities.csv
  ├── train_set/{category}/{target_id}/{target_id}_processed_protein_sequence.txt
  ├── val_set/...
  ├── test_set/...
  └── combined_activities.csv

用法:
  python preprocessing/prepare_bindingdb.py
  python preprocessing/prepare_bindingdb.py --subset BindingDB_Kd
  python preprocessing/prepare_bindingdb.py --subset all --output data/processed/bindingdb
  python preprocessing/prepare_bindingdb.py --manual
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Harvard Dataverse 文件 ID
DATASET_INFO = {
    'BindingDB_Kd':   {'file_id': '4291555', 'ext': 'tab', 'paffinity_type': 'pKd',   'affinity_type': 'Kd'},
    'BindingDB_IC50': {'file_id': '4291560', 'ext': 'csv', 'paffinity_type': 'pIC50', 'affinity_type': 'IC50'},
    'BindingDB_Ki':   {'file_id': '4291556', 'ext': 'csv', 'paffinity_type': 'pKi',   'affinity_type': 'Ki'},
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "bindingdb"


def download_file(url, filepath):
    """带进度条 + 断点续传的下载"""
    import requests
    from tqdm import tqdm

    # 检查断点续传
    resume_pos = 0
    if filepath.exists():
        resume_pos = filepath.stat().st_size
        print(f"  发现已有文件 ({resume_pos / 1024 / 1024:.1f}MB), 尝试断点续传...")

    headers = {}
    if resume_pos > 0:
        headers['Range'] = f'bytes={resume_pos}-'

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)

        if response.status_code == 416:  # 文件已完成
            print(f"  文件已完整下载")
            return True

        if response.status_code not in (200, 206):
            print(f"  下载失败: HTTP {response.status_code}")
            return False

        total_size = int(response.headers.get('content-length', 0))
        if response.status_code == 206:
            total_size += resume_pos  # 续传时总大小 = 已有 + 剩余

        mode = 'ab' if resume_pos > 0 else 'wb'
        chunk_size = 1024 * 256  # 256KB chunks

        with open(filepath, mode) as f:
            with tqdm(total=total_size, initial=resume_pos, unit='B',
                      unit_scale=True, desc='  下载', ncols=80) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        actual_size = filepath.stat().st_size
        print(f"  下载完成: {actual_size / 1024 / 1024:.1f}MB")
        return True

    except Exception as e:
        print(f"  下载出错: {e}")
        return False


def load_bindingdb(subset, manual=False):
    """加载 BindingDB 子数据集"""
    if subset not in DATASET_INFO:
        raise ValueError(f"未知子集: {subset}, 可选: {list(DATASET_INFO.keys())}")

    info = DATASET_INFO[subset]
    filename = f"{subset.lower()}.{info['ext']}"
    filepath = DATA_DIR / filename
    url = f"https://dataverse.harvard.edu/api/access/datafile/{info['file_id']}"

    print(f"\n  加载 {subset} ...")

    # 方式1: 本地已有完整文件
    if filepath.exists() and filepath.stat().st_size > 1000:
        print(f"  使用本地缓存: {filepath} ({filepath.stat().st_size / 1024 / 1024:.1f}MB)")
    # 方式2: 手动下载模式
    elif manual:
        print(f"\n  [手动下载模式]")
        print(f"  请手动下载以下文件:")
        print(f"  URL: {url}")
        print(f"  保存到: {filepath}")
        print(f"  下载完成后重新运行此脚本。")
        sys.exit(0)
    # 方式3: 自动下载
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  从 Harvard Dataverse 下载...")
        print(f"  URL: {url}")

        max_retries = 3
        for attempt in range(max_retries):
            success = download_file(url, filepath)
            if success and filepath.exists() and filepath.stat().st_size > 1000:
                break
            print(f"  [尝试 {attempt + 1}/{max_retries}] 下载不完整, 重试...")
            if filepath.exists() and filepath.stat().st_size < 1000:
                filepath.unlink()
        else:
            print(f"\n  自动下载失败! 请尝试手动下载:")
            print(f"  1. 浏览器打开: {url}")
            print(f"  2. 保存到: {filepath}")
            print(f"  3. 重新运行: python prepare_bindingdb.py --manual")
            sys.exit(1)

    # 读取数据
    sep = '\t' if info['ext'] == 'tab' else ','
    try:
        df = pd.read_csv(filepath, sep=sep)
    except Exception as e:
        print(f"  文件解析失败: {e}")
        print(f"  文件可能不完整, 删除后重试...")
        filepath.unlink()
        return load_bindingdb(subset, manual)

    print(f"  原始数据: {len(df)} 条, 列: {list(df.columns)}")
    return df


def nm_to_paffinity(nm_value):
    """nM → p-affinity: pX = 9 - log10(value_nM)"""
    if pd.isna(nm_value) or nm_value <= 0:
        return np.nan
    return 9.0 - np.log10(float(nm_value))


def classify_target(target_id):
    """按靶点名称简单分类"""
    tid = str(target_id).upper()
    for kw in ['KINASE', 'TYROSINE', 'MAPK', 'AKT', 'CDK', 'EGFR', 'VEGFR', 'SRC', 'ABL', 'JAK']:
        if kw in tid:
            return 'kinases'
    for kw in ['RECEPTOR', 'GPCR', 'ADRENO', 'DOPAMIN', 'SEROTONIN', 'OPIOID', 'HISTAMINE']:
        if kw in tid:
            return 'gpcrs'
    for kw in ['CHANNEL', 'SODIUM', 'POTASSIUM', 'CALCIUM', 'TRP']:
        if kw in tid:
            return 'ion_channels'
    for kw in ['NUCLEAR', 'ANDROGEN', 'ESTROGEN', 'GLUCOCORTICOID', 'PPAR']:
        if kw in tid:
            return 'nuclear_receptors'
    return 'default'


def filter_valid(df):
    """过滤无效数据"""
    # 统一列名 (TDC BindingDB 格式: ID1, X1, ID2, X2, Y)
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ('id1', 'drug_id', 'drugid', 'compound_id'):
            col_map[c] = 'Drug_ID'
        elif cl in ('x1', 'drug', 'smiles', 'compound_smiles', 'canonical_smiles'):
            col_map[c] = 'Drug'
        elif cl in ('id2', 'target_id', 'targetid', 'uniprot_id'):
            col_map[c] = 'Target_ID'
        elif cl in ('x2', 'target', 'target_sequence', 'protein_sequence', 'sequence'):
            col_map[c] = 'Target'
        elif cl in ('y', 'affinity', 'binding_affinity', 'value'):
            col_map[c] = 'Y'
    df = df.rename(columns=col_map)

    required = ['Drug', 'Target', 'Y']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少列 {col}, 当前列: {list(df.columns)}")

    if 'Drug_ID' not in df.columns:
        df['Drug_ID'] = [f"DRUG_{i:08d}" for i in range(len(df))]
    if 'Target_ID' not in df.columns:
        df['Target_ID'] = [f"TARGET_{i:06d}" for i in range(len(df))]

    df = df.dropna(subset=['Drug', 'Target', 'Y'])
    df = df[df['Y'].apply(lambda x: isinstance(x, (int, float)) and np.isfinite(x))]
    df = df[df['Target'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    df = df[df['Drug'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    df = df[df['Y'] > 0]
    print(f"  过滤后: {len(df)} 条")
    return df


def split_by_target(df, train_ratio=0.7, val_ratio=0.15, seed=42):
    """按 Target_ID 冷启动划分"""
    target_ids = sorted(df['Target_ID'].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(target_ids)

    n = len(target_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_t = set(target_ids[:n_train])
    val_t = set(target_ids[n_train:n_train + n_val])

    splits = {}
    for tid in df['Target_ID'].unique():
        splits[tid] = 'train_set' if tid in train_t else ('val_set' if tid in val_t else 'test_set')
    return splits


def write_target_data(target_id, target_seq, target_df, split_dir, paffinity_type, affinity_type):
    """写出单个靶点的文件"""
    target_dir = split_dir / target_id
    target_dir.mkdir(parents=True, exist_ok=True)

    clean_seq = ''.join(c for c in str(target_seq).strip().upper()
                        if c in 'ACDEFGHIKLMNPQRSTVWY')
    (target_dir / f"{target_id}_processed_protein_sequence.txt").write_text(clean_seq, encoding='utf-8')

    paffinity_values = target_df['Y'].apply(nm_to_paffinity).values
    out_df = pd.DataFrame({
        'molecule_id': target_df['Drug_ID'].values,
        'smiles': target_df['Drug'].values,
        'paffinity': paffinity_values,
        'paffinity_type': paffinity_type,
        'affinity_type': affinity_type,
        'measurement_count': 1,
        'paffinity_std': 0.0,
    })
    out_df = out_df.dropna(subset=['paffinity'])
    out_df = out_df[out_df['paffinity'].apply(np.isfinite)]
    out_df.to_csv(target_dir / f"{target_id}_processed_activities.csv", index=False)
    return len(out_df)


def process_subset(subset_name, output_dir, min_samples=5, max_targets=None, manual=False):
    """处理单个子数据集"""
    info = DATASET_INFO[subset_name]
    print(f"\n{'=' * 70}")
    print(f"处理: {subset_name} ({info['affinity_type']} → {info['paffinity_type']})")
    print(f"{'=' * 70}")

    df = load_bindingdb(subset_name, manual=manual)
    df = filter_valid(df)

    target_stats = df.groupby('Target_ID').size().sort_values(ascending=False)
    print(f"  靶点数: {len(target_stats)}, 样本数: min={target_stats.min()}, "
          f"median={target_stats.median():.0f}, max={target_stats.max()}")

    valid_t = target_stats[target_stats >= min_samples].index
    df = df[df['Target_ID'].isin(valid_t)]
    print(f"  样本数 >= {min_samples} 的靶点: {len(valid_t)} 个, 数据: {len(df)} 条")

    if max_targets:
        top_t = target_stats.head(max_targets).index
        df = df[df['Target_ID'].isin(top_t)]
        print(f"  限制 top {max_targets} 靶点, 数据: {len(df)} 条")

    splits = split_by_target(df)
    sc = {'train_set': 0, 'val_set': 0, 'test_set': 0}
    for s in splits.values():
        sc[s] += 1
    print(f"  划分: train={sc['train_set']}, val={sc['val_set']}, test={sc['test_set']}")

    total_pairs = 0
    n_written = 0
    for target_id, group_df in df.groupby('Target_ID'):
        split = splits.get(target_id, 'train_set')
        target_seq = group_df['Target'].iloc[0]
        category = classify_target(target_id)
        split_dir = output_dir / split / category
        n = write_target_data(target_id, target_seq, group_df, split_dir,
                              info['paffinity_type'], info['affinity_type'])
        if n > 0:
            total_pairs += n
            n_written += 1
        if n_written % 500 == 0 and n_written > 0:
            print(f"    已处理 {n_written} 靶点, {total_pairs} 配对...")

    print(f"  完成: {n_written} 靶点, {total_pairs} 配对")

    # combined
    combined = df[['Drug_ID', 'Drug', 'Target_ID', 'Target', 'Y']].copy()
    combined.columns = ['molecule_id', 'smiles', 'target_id', 'target_sequence', 'affinity_nm']
    combined['paffinity'] = combined['affinity_nm'].apply(nm_to_paffinity)
    combined['subset'] = subset_name
    combined['split'] = combined['target_id'].map(lambda t: splits.get(t, 'train_set'))
    return combined, total_pairs, n_written


def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(description='下载并处理 BindingDB 数据')
    parser.add_argument('--subset', type=str, default='all',
                        choices=['all', 'BindingDB_Kd', 'BindingDB_IC50', 'BindingDB_Ki'])
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'data' / 'processed' / 'bindingdb')
    parser.add_argument('--download-dir', type=Path, default=DATA_DIR,
                        help='BindingDB 原始下载文件目录')
    parser.add_argument('--min-samples', type=int, default=5)
    parser.add_argument('--max-targets', type=int, default=None)
    parser.add_argument('--manual', action='store_true',
                        help='手动下载模式（打印下载URL，不自动下载）')
    args = parser.parse_args()

    DATA_DIR = args.download_dir.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    subsets = list(DATASET_INFO.keys()) if args.subset == 'all' else [args.subset]

    print(f"\n{'=' * 70}")
    print(f"BindingDB 数据处理")
    print(f"  子集: {subsets}")
    print(f"  输出: {output_dir}")
    print(f"  模式: {'手动下载' if args.manual else '自动下载'}")
    print(f"{'=' * 70}")

    all_combined = []
    grand_pairs = grand_targets = 0

    for subset_name in subsets:
        combined, n_pairs, n_targets = process_subset(
            subset_name, output_dir, args.min_samples, args.max_targets, args.manual)
        grand_pairs += n_pairs
        grand_targets += n_targets
        all_combined.append(combined)

    if all_combined:
        combined_path = output_dir / "combined_activities.csv"
        all_df = pd.concat(all_combined, ignore_index=True)
        all_df.to_csv(combined_path, index=False)
        print(f"\ncombined_activities.csv 已保存 ({len(all_df)} 行)")

    print(f"\n{'=' * 70}")
    print(f"全部完成!")
    print(f"  总靶点: {grand_targets}, 总配对: {grand_pairs}")
    for split in ['train_set', 'val_set', 'test_set']:
        sd = output_dir / split
        if sd.exists():
            n = sum(1 for c in sd.iterdir() for t in c.iterdir() if t.is_dir())
            print(f"  {split}: {n} 靶点")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
