"""
数据预处理和特征提取模块

功能：
1. 加载CHEMBL靶点数据
2. 提取分子特征（Morgan指纹 + RDKit描述符）
3. 预计算蛋白质特征（ESM2）
4. 数据归一化和划分
"""

import os
import gc
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# PyTorch
import torch

# 配置
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED_DIR = BASE_DIR / "data" / "processed" / "chembl"
PREPROCESSED_DIR = DEFAULT_PREPROCESSED_DIR
OUTPUT_DIR = BASE_DIR / "reptile_output"
CACHE_DIR = OUTPUT_DIR / "cache"


def get_preprocessed_dir():
    """Resolve the dataset path at runtime so --data_dir takes effect."""
    return Path(os.environ.get("PREPROCESSED_DIR", PREPROCESSED_DIR)).resolve()

for d in [OUTPUT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ==========================
# 全局变量
# ==========================
MORGAN_BITS = 2048
MORGAN_RADIUS = 2
DESC_DIM = 10
ESM2_DIM = 480  # facebook/esm2_t12_35M_UR50D hidden size
FEATURE_FORMAT_VERSION = 2

# 缓存
MORGAN_CACHE = {}
DESC_CACHE = {}
PROTEIN_CACHE = {}


# ==========================
# 分子特征提取
# ==========================
class MoleculeFeatureExtractor:
    """分子特征提取器
    
    创新点：多尺度分子特征融合
    - Morgan指纹：局部子结构信息
    - MACCS keys：预定义官能团模式
    - RDKit描述符：全局理化性质
    """
    
    def __init__(self):
        self.morgan_cache = {}
        self.desc_cache = {}
        self.maccs_cache = {}
    
    def get_morgan_fingerprint(self, smiles):
        """获取Morgan指纹"""
        if smiles in self.morgan_cache:
            return self.morgan_cache[smiles]
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fp = np.zeros(MORGAN_BITS, dtype=np.float32)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
            fp = np.array(fp, dtype=np.float32)
        
        self.morgan_cache[smiles] = fp
        return fp
    
    def get_maccs_keys(self, smiles):
        """获取MACCS keys指纹（创新点：补充官能团信息）"""
        if smiles in self.maccs_cache:
            return self.maccs_cache[smiles]
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fp = np.zeros(167, dtype=np.float32)  # MACCS keys固定167位
        else:
            fp = AllChem.GetMACCSKeysFingerprint(mol)
            fp = np.array(fp, dtype=np.float32)
        
        self.maccs_cache[smiles] = fp
        return fp
    
    def get_descriptors(self, smiles):
        """获取RDKit描述符"""
        if smiles in self.desc_cache:
            return self.desc_cache[smiles]
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(DESC_DIM, dtype=np.float32)
        
        desc_names = [
            'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors',
            'NumRotatableBonds', 'RingCount', 'TPSA',
            'FractionCsp3', 'HeavyAtomCount', 'NumAromaticRings'
        ]
        
        desc_values = []
        for name in desc_names:
            try:
                desc_values.append(float(getattr(Descriptors, name)(mol)))
            except:
                desc_values.append(0.0)
        
        result = np.array(desc_values, dtype=np.float32)
        self.desc_cache[smiles] = result
        return result
    
    def extract(self, smiles):
        """提取完整分子特征"""
        return {
            'morgan': self.get_morgan_fingerprint(smiles),
            'maccs': self.get_maccs_keys(smiles),
            'descriptors': self.get_descriptors(smiles)
        }


# ==========================
# 蛋白质特征提取（ESM2）
# ==========================
class ProteinFeatureExtractor:
    """蛋白质特征提取器（使用ESM2）"""
    
    def __init__(self, model_path=None, output_dim=ESM2_DIM, use_gpu=False):
        self.output_dim = output_dim
        self.cache = {}
        self.use_gpu = use_gpu
        self.model_path = str(model_path or os.environ.get(
            "ESM2_MODEL", "facebook/esm2_t12_35M_UR50D"
        ))
        self._load_esm2(self.model_path)
    
    def _load_esm2(self, model_path):
        """使用与 checkpoint 匹配的 Hugging Face 实现加载 ESM-2。"""
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            add_pooling_layer=False,
        )
        hidden_dim = self.model.config.hidden_size
        if self.output_dim != hidden_dim:
            raise ValueError(
                f"ESM-2 output_dim must be {hidden_dim}; got {self.output_dim}. "
                "Do not apply an untrained random projection to cached features."
            )
        print(
            f"   ESM-2 loaded with transformers.AutoModel: "
            f"{model_path} (hidden_size={hidden_dim})"
        )
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 移到设备（默认CPU，避免CUDA碎片化；预计算不需要GPU）
        esm_device = torch.device('cuda:0' if self.use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(esm_device)
        self.model.eval()
        self.esm_device = esm_device
        
        # 如果在GPU上加载，立即清理缓存以避免碎片化
        if esm_device.type == 'cuda':
            torch.cuda.empty_cache()
    
    @torch.no_grad()
    def extract(self, sequence):
        """提取蛋白质特征"""
        if sequence in self.cache:
            return self.cache[sequence]
        
        # 截断过长序列
        if len(sequence) > 1022:
            sequence = sequence[:1022]
        
        # 清理序列
        sequence = ''.join([c for c in sequence if c in 'ACDEFGHIKLMNPQRSTVWY'])
        
        encoded = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = encoded.pop("special_tokens_mask").to(self.esm_device)
        encoded = {key: value.to(self.esm_device) for key, value in encoded.items()}
        token_repr = self.model(**encoded).last_hidden_state

        # 仅对真实氨基酸 token 做平均池化，排除 CLS/EOS/PAD。
        content_mask = encoded["attention_mask"].bool() & ~special_tokens_mask.bool()
        embedding = (
            token_repr * content_mask.unsqueeze(-1)
        ).sum(dim=1) / content_mask.sum(dim=1, keepdim=True).clamp_min(1)
        embedding = embedding.squeeze(0)
        
        result = embedding.cpu().numpy().astype(np.float32)
        self.cache[sequence] = result
        
        return result
    
    def save_cache(self, path):
        """保存缓存"""
        with open(path, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def load_cache(self, path):
        """加载缓存"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.cache = pickle.load(f)


# ==========================
# 数据加载器
# ==========================
def load_target_data(target_info):
    """加载单个靶点的数据（兼容 ChEMBL 和 Davis 两种格式）"""
    target_dir = target_info['path']
    target_name = target_info['name']
    split_name = target_info['split']
    category_name = target_info['category']

    # 文件名兼容：ChEMBL 用 {target}_processed_activities.csv，Davis 用 activities.csv
    activities_file = target_dir / f"{target_name}_processed_activities.csv"
    if not activities_file.exists():
        activities_file = target_dir / "activities.csv"
    if not activities_file.exists():
        return None

    try:
        df = pd.read_csv(activities_file)
    except:
        return None

    # 列名兼容：smiles / canonical_smiles
    smiles_col = 'smiles' if 'smiles' in df.columns else 'canonical_smiles'
    # 活性值列：paffinity（已转换）/ standard_value（原始 Kd nM，需转 pKd）
    if 'paffinity' in df.columns:
        smiles_list = df[smiles_col].tolist()
        affinities = df['paffinity'].values.astype(np.float32)
    elif 'standard_value' in df.columns:
        smiles_list = df[smiles_col].tolist()
        raw_val = df['standard_value'].values.astype(np.float32)
        # standard_value 是 Kd(nM)，转 pKd = 9 - log10(Kd_nM)
        affinities = 9.0 - np.log10(np.maximum(raw_val, 1e-10))
        affinities = affinities.astype(np.float32)
    else:
        return None

    # 获取蛋白质序列
    sequence = None
    seq_file = target_dir / f"{target_name}_processed_protein_sequence.txt"
    if seq_file.exists():
        with open(seq_file, 'r') as f:
            content = f.read().strip()
            sequence = ''.join([line.strip() for line in content.split('\n') if not line.startswith('>')])

    if sequence is None:
        for f in target_dir.iterdir():
            if f.suffix in ['.fasta', '.fa', '.faa', '.txt']:
                with open(f, 'r') as fh:
                    content = fh.read().strip()
                    seq = ''.join([line.strip() for line in content.split('\n') if not line.startswith('>')])
                    if len(seq) > 0:
                        sequence = seq
                        break

    if sequence is None or len(sequence) == 0:
        return None

    # 清理序列
    sequence = ''.join([c for c in sequence if c in 'ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy'])
    if len(sequence) == 0:
        return None

    return {
        'smiles': smiles_list,
        'affinities': affinities,
        'sequence': sequence,
        'target_name': target_name
    }


def get_all_targets(test_mode=False, max_targets=None):
    """获取所有靶点（支持两种目录结构）"""
    all_targets = {'train': [], 'val': [], 'test': []}
    preprocessed_dir = get_preprocessed_dir()
    
    for split in ['train_set', 'val_set', 'test_set']:
        split_dir = preprocessed_dir / split
        if not split_dir.exists():
            continue
        
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if not subdirs:
            continue
        
        first_subdir = subdirs[0]
        has_category_layer = any(str(f).endswith('_activities.csv') for f in first_subdir.iterdir())
        
        if has_category_layer:
            for target_dir in subdirs:
                all_targets[split[:-4]].append({
                    'name': target_dir.name,
                    'path': target_dir,
                    'split': split,
                    'category': 'default'
                })
        else:
            for category_dir in subdirs:
                if not category_dir.is_dir():
                    continue
                
                for target_dir in category_dir.iterdir():
                    if not target_dir.is_dir():
                        continue
                    
                    all_targets[split[:-4]].append({
                        'name': target_dir.name,
                        'path': target_dir,
                        'split': split,
                        'category': category_dir.name
                    })
    
    if max_targets:
        all_targets['train'] = all_targets['train'][:max_targets]
        val_count = min(10, len(all_targets['val'])) if all_targets['val'] else 0
        test_count = min(5, len(all_targets['test'])) if all_targets['test'] else 0
        all_targets['val'] = all_targets['val'][:val_count]
        all_targets['test'] = all_targets['test'][:test_count]
    
    return all_targets


# ==========================
# 数据预处理
# ==========================
class TargetScaler:
    """靶点级别的数据归一化器"""
    
    def __init__(self):
        self.global_mean = None
        self.global_std = None
        self.target_means = {}
        self.target_stds = {}
    
    def fit(self, affinities, target_names=None):
        """拟合归一化器"""
        aff = np.array(affinities, dtype=np.float32)
        
        # 全局统计量
        self.global_mean = float(np.mean(aff))
        self.global_std = float(np.std(aff))
        if self.global_std < 1e-6:
            self.global_std = 1.0
        
        print(f"✅ Global mean: {self.global_mean:.4f}, std: {self.global_std:.4f}")
        
        # 每个靶点的统计量（使用numpy向量化避免MemoryError）
        if target_names is not None:
            target_arr = np.array(target_names)
            unique_targets = np.unique(target_arr)
            for target in unique_targets:
                mask = target_arr == target
                target_aff = aff[mask]
                if len(target_aff) > 0:
                    self.target_means[target] = float(np.mean(target_aff))
                    self.target_stds[target] = float(np.std(target_aff))
                    if self.target_stds[target] < 1e-6:
                        self.target_stds[target] = 1.0
        
        return self
    
    def transform(self, affinities):
        """归一化"""
        return (np.array(affinities, dtype=np.float32) - self.global_mean) / self.global_std
    
    def transform_with_names(self, affinities, target_names):
        """按靶点归一化（带 target_names）"""
        aff = np.array(affinities, dtype=np.float32)
        result = (aff - self.global_mean) / self.global_std
        
        if len(self.target_means) > 0 and target_names is not None:
            target_arr = np.array(target_names)
            for target in np.unique(target_arr):
                mask = target_arr == target
                if target in self.target_means:
                    tmean = self.target_means[target]
                    tstd = self.target_stds.get(target, 1.0)
                    result[mask] = (aff[mask] - tmean) / tstd
        
        return result
    
    def inverse_transform(self, preds, target_name=None):
        """反归一化"""
        result = np.array(preds, dtype=np.float32) * self.global_std + self.global_mean
        
        if target_name is not None and target_name in self.target_means:
            # 使用靶点级别的修正
            result = result * (self.target_stds[target_name] / self.global_std) + (
                self.target_means[target_name] - self.global_mean * (self.target_stds[target_name] / self.global_std)
            )
        
        return result
    
    def save(self, path):
        """保存归一化器"""
        data = {
            'global_mean': self.global_mean,
            'global_std': self.global_std,
            'target_means': self.target_means,
            'target_stds': self.target_stds
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path):
        """加载归一化器"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.global_mean = data['global_mean']
        self.global_std = data['global_std']
        self.target_means = data['target_means']
        self.target_stds = data['target_stds']
        return self


# ==========================
# 预计算所有特征
# ==========================
def clear_precomputed_features(save_path):
    """Remove a feature archive and temporary arrays created beside it."""
    save_path = Path(save_path)
    candidates = [
        save_path,
        save_path.parent / '_morgan.npy',
        save_path.parent / '_maccs.npy',
        save_path.parent / '_desc.npy',
        save_path.parent / '_protein.npy',
        save_path.parent / '_y.npy',
    ]
    removed = []
    for path in candidates:
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def validate_precomputed_features(save_path):
    """Reject caches made before the corrected Hugging Face ESM-2 pipeline."""
    save_path = Path(save_path)
    with np.load(save_path, mmap_mode='r', allow_pickle=False) as data:
        if 'feature_metadata' not in data.files:
            raise RuntimeError(
                f"{save_path} has no feature metadata and may use the old ESM-2 "
                "loader. Run again with --rebuild_features --force."
            )
        metadata = json.loads(str(data['feature_metadata'].item()))

    expected = {
        'format_version': FEATURE_FORMAT_VERSION,
        'esm2_loader': 'transformers.AutoModel',
        'pooling': 'mean_non_special_tokens',
        'esm2_hidden_size': ESM2_DIM,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        details = ', '.join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in mismatches.items()
        )
        raise RuntimeError(
            f"{save_path} is incompatible with the current feature pipeline: "
            f"{details}. Run again with --rebuild_features --force."
        )
    return metadata


def precompute_features(all_targets, protein_extractor, mol_extractor, save_path):
    """预计算所有特征并保存到磁盘（内存友好版本：边提取边写入mmap）"""
    print("\n🔧 预计算所有特征...")
    
    # 第一阶段：先统计总样本数
    n_total = 0
    for split_name in ['train', 'val', 'test']:
        for target_info in all_targets[split_name]:
            target_data = load_target_data(target_info)
            if target_data is None:
                continue
            n_total += len(target_data['smiles'])
    
    print(f"   总样本数: {n_total}")
    
    # 创建mmap文件
    n_samples = n_total
    morgan_path = save_path.parent / '_morgan.npy'
    maccs_path = save_path.parent / '_maccs.npy'
    desc_path = save_path.parent / '_desc.npy'
    protein_path = save_path.parent / '_protein.npy'
    y_path = save_path.parent / '_y.npy'
    
    # 预先创建目标名称和split的python列表（最后转np.array）
    target_names_list = []
    splits_list = []
    
    idx = 0
    for split_name in ['train', 'val', 'test']:
        for target_info in tqdm(all_targets[split_name], desc=f"{split_name} targets"):
            target_data = load_target_data(target_info)
            if target_data is None:
                continue
            
            smiles_list = target_data['smiles']
            affinities = target_data['affinities']
            sequence = target_data['sequence']
            target_name = target_data['target_name']
            
            # 预计算蛋白质特征
            protein_feat = protein_extractor.extract(sequence)
            
            # 提取分子特征，直接写入mmap
            for smiles, affinity in zip(smiles_list, affinities):
                mol_feat = mol_extractor.extract(smiles)
                
                # 直接写入mmap文件（按需创建）
                if idx == 0:
                    morgan_mmap = np.lib.format.open_memmap(str(morgan_path), dtype=np.float32, mode='w+', shape=(n_samples, MORGAN_BITS))
                    maccs_mmap = np.lib.format.open_memmap(str(maccs_path), dtype=np.float32, mode='w+', shape=(n_samples, 167))
                    desc_mmap = np.lib.format.open_memmap(str(desc_path), dtype=np.float32, mode='w+', shape=(n_samples, DESC_DIM))
                    protein_mmap = np.lib.format.open_memmap(str(protein_path), dtype=np.float32, mode='w+', shape=(n_samples, ESM2_DIM))
                    y_mmap = np.lib.format.open_memmap(str(y_path), dtype=np.float32, mode='w+', shape=(n_samples,))
                
                morgan_mmap[idx] = mol_feat['morgan']
                maccs_mmap[idx] = mol_feat['maccs']
                desc_mmap[idx] = mol_feat['descriptors']
                protein_mmap[idx] = protein_feat
                y_mmap[idx] = affinity
                target_names_list.append(target_name)
                splits_list.append(split_name)
                idx += 1
    
    # 刷新mmap
    morgan_mmap.flush()
    maccs_mmap.flush()
    desc_mmap.flush()
    protein_mmap.flush()
    y_mmap.flush()
    
    del morgan_mmap, maccs_mmap, desc_mmap, protein_mmap, y_mmap
    gc.collect()
    
    # 保存为npz（使用mmap模式读取，避免全量加载）
    print(f"\n💾 保存到 {save_path}...")
    morgan_data = np.load(str(morgan_path), mmap_mode='r')
    maccs_data = np.load(str(maccs_path), mmap_mode='r')
    desc_data = np.load(str(desc_path), mmap_mode='r')
    protein_data = np.load(str(protein_path), mmap_mode='r')
    y_data = np.load(str(y_path), mmap_mode='r')
    
    feature_metadata = {
        'format_version': FEATURE_FORMAT_VERSION,
        'esm2_loader': 'transformers.AutoModel',
        'esm2_model': protein_extractor.model_path,
        'esm2_hidden_size': protein_extractor.output_dim,
        'pooling': 'mean_non_special_tokens',
    }
    np.savez(
        save_path,
        morgan=morgan_data,
        maccs=maccs_data,
        descriptors=desc_data,
        protein=protein_data,
        y=y_data,
        target_names=np.array(target_names_list),
        splits=np.array(splits_list),
        feature_metadata=np.array(json.dumps(feature_metadata)),
    )
    
    # 先关闭所有 mmap 句柄，再清理临时文件
    del morgan_data, maccs_data, desc_data, protein_data, y_data
    gc.collect()
    
    # 清理临时文件（Windows 下可能因句柄未释放而失败，加重试）
    import time as _time
    for p in [morgan_path, maccs_path, desc_path, protein_path, y_path]:
        for _ in range(5):
            try:
                if p.exists():
                    os.remove(str(p))
                break
            except PermissionError:
                _time.sleep(0.5)
            except Exception:
                break
    
    print("   ✅ 特征预计算完成!")
    
    # 保存蛋白质缓存
    protein_cache_path = CACHE_DIR / 'protein_cache.pkl'
    protein_extractor.save_cache(protein_cache_path)
    print(f"   ✅ 蛋白质缓存保存到 {protein_cache_path}")
    
    return save_path


# ==========================
# 惰性子集加载器（解决大数据集内存溢出）
# ==========================
class LazySubset:
    """只保存索引，按需从 memmap 数组读取 batch 数据，不占用大量内存"""
    
    def __init__(self, data_mmap, indices, scaler=None):
        self._data = data_mmap  # np.lib.npyio.NpzFile (mmap_mode='r')
        self._indices = np.asarray(indices)
        self._scaler = scaler
        self.size = len(self._indices)
        
        # 只保留索引引用，不复制数据
        # 通过 __getitem__ 按需读取
        
    def __len__(self):
        return self.size
    
    def get_batch(self, batch_indices):
        """获取一个 batch 的数据（batch_indices 是相对于 subset 的索引）"""
        real_indices = self._indices[batch_indices]
        morgan = self._data['morgan'][real_indices].astype(np.float32)
        maccs = self._data['maccs'][real_indices].astype(np.float32)
        descriptors = self._data['descriptors'][real_indices].astype(np.float32)
        protein = self._data['protein'][real_indices].astype(np.float32)
        y_raw = self._data['y'][real_indices].astype(np.float32)
        
        if self._scaler is not None:
            target_names = self._data['target_names'][real_indices]
            y_norm = self._scaler.transform_with_names(y_raw, target_names)
        else:
            y_norm = y_raw
        
        return {
            'morgan': morgan, 'maccs': maccs, 'descriptors': descriptors,
            'protein': protein, 'y': y_raw, 'y_norm': y_norm
        }
    
    def get_all_y(self):
        """获取所有 y 值（用于统计）"""
        return self._data['y'][self._indices].astype(np.float32)
    
    def get_all_target_names(self):
        """获取所有 target 名"""
        return self._data['target_names'][self._indices]


# ==========================
# 数据加载器（用于训练）
# ==========================
class TargetDataLoader:
    """靶点数据加载器"""
    
    def __init__(self, data_path, target_scaler):
        self.data_path = data_path
        self.target_scaler = target_scaler
        
        # 加载数据
        self.data = np.load(data_path, mmap_mode='r', allow_pickle=True)
        
        # 按靶点分组
        self.target_groups = defaultdict(list)
        for i, target_name in enumerate(self.data['target_names']):
            self.target_groups[target_name].append(i)
        
        self.target_names = list(self.target_groups.keys())
        print(f"✅ 加载了 {len(self.target_names)} 个靶点")
    
    def get_target_data(self, target_name):
        """获取单个靶点的数据"""
        indices = self.target_groups[target_name]
        
        morgan = torch.tensor(self.data['morgan'][indices], dtype=torch.float32, device=device)
        maccs = torch.tensor(self.data['maccs'][indices], dtype=torch.float32, device=device)
        descriptors = torch.tensor(self.data['descriptors'][indices], dtype=torch.float32, device=device)
        protein = torch.tensor(self.data['protein'][indices], dtype=torch.float32, device=device)
        y = torch.tensor(self.data['y'][indices], dtype=torch.float32, device=device)
        
        # 归一化y值
        y_norm = torch.tensor(self.target_scaler.transform(y.cpu().numpy()), dtype=torch.float32, device=device)
        
        return morgan, maccs, descriptors, protein, y_norm
    
    def get_random_target(self):
        """随机获取一个靶点"""
        import random
        target_name = random.choice(self.target_names)
        return target_name, self.get_target_data(target_name)
    
    def get_all_targets(self):
        """获取所有靶点名称"""
        return self.target_names
    
    def close(self):
        """关闭数据"""
        del self.data
        gc.collect()


# ==========================
# 评估函数
# ==========================
def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from scipy.stats import pearsonr, spearmanr, kendalltau
    
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    
    # 过滤无效值
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 2:
        return {
            'RMSE': 0.0,
            'MAE': 0.0,
            'R2': 0.0,
            'Pearson': 0.0,
            'Spearman': 0.0,
            'Kendall': 0.0,
            'n_samples': 0
        }
    
    metrics = {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)),
        'Pearson': float(pearsonr(y_true, y_pred)[0]),
        'Spearman': float(spearmanr(y_true, y_pred)[0]),
        'Kendall': float(kendalltau(y_true, y_pred)[0]),
        'n_samples': len(y_true)
    }
    
    # 处理NaN
    for k, v in metrics.items():
        if np.isnan(v):
            metrics[k] = 0.0
    
    return metrics


def evaluate_model(model, data_loader, target_scaler, split='val'):
    """评估模型"""
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for target_name in data_loader.get_all_targets():
            # 检查是否属于指定split
            # 这里简化处理，假设data_loader包含所有数据
            morgan, maccs, descriptors, protein, y_norm = data_loader.get_target_data(target_name)
            
            preds, _, _ = model(morgan, maccs, descriptors, protein)
            
            # 反归一化
            y_true = target_scaler.inverse_transform(y_norm.cpu().numpy(), target_name)
            y_pred = target_scaler.inverse_transform(preds.cpu().numpy(), target_name)
            
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
    
    metrics = compute_metrics(all_y_true, all_y_pred)
    model.train()
    
    return metrics
