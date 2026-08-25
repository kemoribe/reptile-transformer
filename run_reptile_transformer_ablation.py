"""
Reptile-Transformer - 特征消融实验

固定 Reptile-Transformer 元学习网络不变，仅修改输入特征组合，对比四组方案：
1. morgan_only:        仅Morgan+ESM2   （单特征，Morgan代表SMILES衍生特征）
2. morgan_maccs:       Morgan+MACCS+ESM2（双指纹）
3. morgan_descriptors: Morgan+理化+ESM2  （指纹+理化描述符）
4. full:               Morgan+MACCS+理化+ESM2（全套原始方案）

默认数据集：data/processed/chembl（可用 --data_dir 切换）

运行方式：
    # 运行单个变体
    python run_reptile_transformer_ablation.py --variant morgan_only
    # 运行全部四个变体
    python run_reptile_transformer_ablation.py --all
    # 从断点恢复
    python run_reptile_transformer_ablation.py --all --resume
    # 指定数据集
    python run_reptile_transformer_ablation.py --all --data_dir data/processed/davis
"""

import os
import sys
import gc
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_DIR = str(BASE_DIR / "data" / "processed" / "chembl")
os.environ.setdefault('PREPROCESSED_DIR', DEFAULT_DATA_DIR)

sys.path.insert(0, str(BASE_DIR))

import torch
from data_preprocessing import (
    MoleculeFeatureExtractor,
    ProteinFeatureExtractor,
    get_all_targets,
    precompute_features,
    clear_precomputed_features,
    validate_precomputed_features,
    TargetScaler,
)
from reptile_transformer_model import ReptileTransformer, count_params, init_weights
from reptile_training import ReptileTrainer, TrainingConfig

# ==========================
# 变体配置：固定网络不变，仅通过特征开关控制输入
# ==========================
VARIANTS = {
    'morgan_only': {
        'desc': '仅Morgan+ESM2 (单特征, Morgan代表SMILES衍生)',
        'use_morgan': True,
        'use_maccs': False,
        'use_descriptors': False,
    },
    'morgan_maccs': {
        'desc': 'Morgan+MACCS+ESM2 (双指纹)',
        'use_morgan': True,
        'use_maccs': True,
        'use_descriptors': False,
    },
    'morgan_descriptors': {
        'desc': 'Morgan+理化+ESM2 (指纹+理化描述符)',
        'use_morgan': True,
        'use_maccs': False,
        'use_descriptors': True,
    },
    'full': {
        'desc': '全套 Morgan+MACCS+理化+ESM2 (原始方案)',
        'use_morgan': True,
        'use_maccs': True,
        'use_descriptors': True,
    },
}

# 共享预计算特征（根据数据集名称自动区分，避免不同数据集结果冲突）
# 实际路径在 main() 里根据 --data_dir 设置
SHARED_OUTPUT_DIR = BASE_DIR / "reptile_ablation_output"
SHARED_FEATURES = SHARED_OUTPUT_DIR / "precomputed_features.npz"
SHARED_SCALER = SHARED_OUTPUT_DIR / "target_scaler.json"


# ==========================
# 消融 DataLoader：包装原始 LazyDataLoader，对不用的特征置零
# ==========================
class AblationLazyDataLoader:
    """
    消融实验专用 DataLoader：包装原始 LazyDataLoader，
    在 get_target_data 返回时根据变体配置对不用的特征置零。
    ReptileTrainer 完全不用改。
    """

    def __init__(self, base_loader, variant_cfg):
        self.base = base_loader
        self.cfg = variant_cfg
        # 代理 target_groups 等属性
        self.target_groups = base_loader.target_groups

    def get_all_targets(self):
        return self.base.get_all_targets()

    def get_target_data(self, target_name):
        result = self.base.get_target_data(target_name)
        if result is None:
            return None

        morgan, maccs, descriptors, protein, y_norm = result

        # 对不用的特征置零（网络完全不变，只是输入特征被屏蔽）
        if not self.cfg['use_maccs']:
            maccs = torch.zeros_like(maccs)
        if not self.cfg['use_descriptors']:
            descriptors = torch.zeros_like(descriptors)
        # morgan 始终使用（作为 SMILES 衍生特征的基础）

        return morgan, maccs, descriptors, protein, y_norm

    def clear_cache(self):
        if hasattr(self.base, 'clear_cache'):
            self.base.clear_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Reptile-Transformer 特征消融实验")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help="预处理数据目录")
    parser.add_argument('--variant', type=str, default=None,
                        choices=list(VARIANTS.keys()),
                        help="运行单个变体: morgan_only / morgan_maccs / morgan_descriptors / full")
    parser.add_argument('--all', action='store_true',
                        help="运行全部四个变体")
    parser.add_argument('--resume', action='store_true', help="从断点恢复训练")
    parser.add_argument('--force', action='store_true', help="强制重新训练（清除旧断点）")
    parser.add_argument('--gpu', type=str, default='0', help="GPU设备编号")
    parser.add_argument(
        '--esm2_model',
        type=str,
        default=None,
        help='Hugging Face ESM-2 本地目录或模型名',
    )
    parser.add_argument(
        '--rebuild_features',
        action='store_true',
        help='删除共享旧特征并重新运行全部所选实验',
    )
    parser.add_argument('--batch_size', type=int, default=2048, help="批处理大小")
    parser.add_argument('--epochs', type=int, default=220, help="训练轮数")
    parser.add_argument('--inner_steps', type=int, default=3, help="内循环步数")
    parser.add_argument('--gradient_accumulation', type=int, default=1, help="梯度累积步数")
    parser.add_argument('--list', action='store_true', help="列出变体状态")
    parser.add_argument('--max_samples_per_target', type=int, default=None,
                        help="少样本实验: 每个靶点最多使用N个训练样本 (None=全部)")
    return parser.parse_args()


def train_one_variant(variant_name, args, shared_data, splits_info):
    """训练单个变体"""
    cfg = VARIANTS[variant_name]
    output_dir = SHARED_OUTPUT_DIR / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    final_results_path = output_dir / 'final_results.json'
    log_path = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # 日志文件
    log_file = open(log_path, 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log("\n" + "=" * 70)
    log(f"Reptile-Transformer 消融变体: {variant_name}")
    log(f"描述: {cfg['desc']}")
    log(f"特征: morgan={cfg['use_morgan']}, maccs={cfg['use_maccs']}, descriptors={cfg['use_descriptors']}")
    log(f"输出: {output_dir}")
    log("=" * 70)

    # 已完成检查
    if final_results_path.exists() and not args.force and not args.resume:
        log("✅ 已完成，跳过（用 --force 强制重跑或 --resume 断点恢复）")
        log_file.close()
        return True

    # --force 清除旧断点
    if args.force:
        for ckpt_name in ['checkpoint_latest.pt', 'checkpoint_best.pt', 'best_model.pt']:
            ckpt_path = output_dir / ckpt_name
            if ckpt_path.exists():
                ckpt_path.unlink()
                log(f"🗑 已清除旧断点: {ckpt_path}")

    # 设备
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device('cuda:0')
        props = torch.cuda.get_device_properties(0)
        log(f"   GPU: {props.name}, {props.total_memory / 1e9:.2f}GB")
    else:
        device = torch.device('cpu')
    log(f"   Device: {device}")

    # 解包共享数据
    data = shared_data['data']
    train_indices = splits_info['train']
    val_indices = splits_info['val']
    test_indices = splits_info['test']
    scaler = shared_data['scaler']
    target_names_array = shared_data['target_names_array']

    # ==========================
    # 创建基础 LazyDataLoader（带 GPU 缓存，和 run_reptile_transformer.py 一致）
    # ==========================
    from collections import defaultdict

    class LazyDataLoader:
        def __init__(self, data_mmap, indices, scaler, target_names_array=None):
            self._data = data_mmap
            self._scaler = scaler

            self.target_groups = defaultdict(list)
            if target_names_array is not None:
                targets_subset = target_names_array[indices]
                unique_targets, inverse_indices = np.unique(targets_subset, return_inverse=True)
                for idx, target_name in enumerate(unique_targets):
                    self.target_groups[target_name] = np.where(inverse_indices == idx)[0].tolist()
            else:
                target_names = data_mmap['target_names'][:]
                for i, idx in enumerate(indices):
                    self.target_groups[target_names[idx]].append(i)

            self._indices = np.asarray(indices)
            self._gpu_cache = {}
            log(f"   样本数: {len(self._indices)}, 靶点数: {len(self.target_groups)}")

        def get_all_targets(self):
            return list(self.target_groups.keys())

        def get_target_data(self, target_name):
            if target_name in self._gpu_cache:
                return self._gpu_cache[target_name]

            rel_indices = self.target_groups[target_name]
            if len(rel_indices) == 0:
                return None

            real_indices = self._indices[rel_indices]
            morgan = torch.tensor(self._data['morgan'][real_indices].astype(np.float32), dtype=torch.float32, device=device)
            maccs = torch.tensor(self._data['maccs'][real_indices].astype(np.float32), dtype=torch.float32, device=device)
            descriptors = torch.tensor(self._data['descriptors'][real_indices].astype(np.float32), dtype=torch.float32, device=device)
            protein = torch.tensor(self._data['protein'][real_indices].astype(np.float32), dtype=torch.float32, device=device)
            y_raw = self._data['y'][real_indices].astype(np.float32)
            y_norm = self._scaler.transform(y_raw)
            y_norm = torch.tensor(y_norm, dtype=torch.float32, device=device)

            cached = (morgan, maccs, descriptors, protein, y_norm)
            self._gpu_cache[target_name] = cached
            return cached

        def clear_cache(self):
            self._gpu_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 创建基础 loader
    base_train_loader = LazyDataLoader(data, train_indices, scaler, target_names_array)
    base_val_loader = LazyDataLoader(data, val_indices, scaler, target_names_array)
    base_test_loader = LazyDataLoader(data, test_indices, scaler, target_names_array)

    # 用消融包装器包装（对不用的特征置零）
    train_loader = AblationLazyDataLoader(base_train_loader, cfg)
    val_loader = AblationLazyDataLoader(base_val_loader, cfg)
    test_loader = AblationLazyDataLoader(base_test_loader, cfg)

    # ==========================
    # 创建模型（网络完全不变）
    # ==========================
    model = ReptileTransformer()
    model.apply(init_weights)
    model = model.to(device)

    total_params, trainable_params = count_params(model)
    log(f"   Total params: {total_params:,}")
    log(f"   Trainable params: {trainable_params:,}")

    # ==========================
    # 创建训练配置和训练器
    # ==========================
    config = TrainingConfig()
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.INNER_STEPS = args.inner_steps
    config.GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation

    trainer = ReptileTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        target_scaler=scaler,
        output_dir=str(output_dir),
        config=config
    )

    # ==========================
    # 训练
    # ==========================
    log(f"\n🔥 开始训练 [{variant_name}] ...")
    log(f"   Epochs: {args.epochs}, Inner Steps: {args.inner_steps}, Batch Size: {args.batch_size}")
    training_history = trainer.train(resume=args.resume)

    # ==========================
    # 测试评估
    # ==========================
    log("\n" + "=" * 70)
    log("📊 测试集评估")
    log("=" * 70)
    test_metrics = trainer.evaluate_test()

    # ==========================
    # 保存结果
    # ==========================
    results = {
        'training_history': training_history,
        'test_metrics': test_metrics,
        'variant': variant_name,
        'variant_desc': cfg['desc'],
        'features': {
            'morgan': cfg['use_morgan'],
            'maccs': cfg['use_maccs'],
            'descriptors': cfg['use_descriptors'],
            'protein_esm2': True,
        },
        'config': {
            'inner_lr': trainer.config.INNER_LR,
            'inner_steps': trainer.config.INNER_STEPS,
            'meta_lr': trainer.config.META_LR,
            'epochs': trainer.config.EPOCHS,
            'batch_size': trainer.config.BATCH_SIZE,
        },
        'best_val_r2': getattr(trainer, 'best_val_r2', None),
        'best_val_rmse': getattr(trainer, 'best_val_rmse', None),
    }

    with open(final_results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # 打印关键指标
    log(f"\n📊 [{variant_name}] 最终测试结果:")
    log(f"   R²:       {test_metrics.get('R2', 'N/A')}")
    log(f"   RMSE:     {test_metrics.get('RMSE', 'N/A')}")
    log(f"   MAE:      {test_metrics.get('MAE', 'N/A')}")
    log(f"   Pearson:  {test_metrics.get('Pearson', 'N/A')}")
    log(f"   Spearman: {test_metrics.get('Spearman', 'N/A')}")
    log(f"   EF@1%:    {test_metrics.get('EF@1%', 'N/A')}")
    log(f"   EF@5%:    {test_metrics.get('EF@5%', 'N/A')}")
    log(f"   EF@10%:   {test_metrics.get('EF@10%', 'N/A')}")
    log(f"   ECE:      {test_metrics.get('ECE', 'N/A')}")
    log(f"   AUPR:     {test_metrics.get('AUPR', 'N/A')}")
    log(f"   Samples:  {test_metrics.get('samples', 'N/A')}")

    log(f"\n✅ 完成! 输出: {output_dir}")
    log_file.close()

    # 清理显存
    del model, trainer
    del base_train_loader, base_val_loader, base_test_loader
    del train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def main():
    args = parse_args()

    if args.esm2_model:
        os.environ['ESM2_MODEL'] = args.esm2_model
        print(f"🧬 使用 ESM-2 模型: {args.esm2_model}")

    # 更新数据目录
    if args.data_dir:
        os.environ['PREPROCESSED_DIR'] = args.data_dir
        import data_preprocessing
        data_preprocessing.PREPROCESSED_DIR = Path(args.data_dir)

    # 根据数据集名称设置输出目录（避免不同数据集结果冲突）
    dataset_name = Path(args.data_dir).name
    if 'chembl' in dataset_name.lower():
        suffix = 'chembl'
    elif 'davis' in dataset_name.lower():
        suffix = 'davis'
    elif 'kiba' in dataset_name.lower():
        suffix = 'kiba'
    elif 'bindingdb' in dataset_name.lower():
        suffix = 'bindingdb'
    else:
        suffix = dataset_name

    global SHARED_OUTPUT_DIR, SHARED_FEATURES, SHARED_SCALER
    fewshot_tag = f"_fewshot{args.max_samples_per_target}" if args.max_samples_per_target else ""
    SHARED_OUTPUT_DIR = BASE_DIR / f"reptile_ablation_output_{suffix}{fewshot_tag}"
    SHARED_FEATURES = SHARED_OUTPUT_DIR / "precomputed_features.npz"
    SHARED_SCALER = SHARED_OUTPUT_DIR / "target_scaler.json"

    # --list 模式
    if args.list:
        print("\nReptile-Transformer 消融变体状态:")
        for name, cfg in VARIANTS.items():
            out_dir = SHARED_OUTPUT_DIR / name
            done = (out_dir / 'final_results.json').exists()
            ckpt = (out_dir / 'checkpoint_latest.pt').exists() or (out_dir / 'checkpoint_best.pt').exists()
            status = "[完成]" if done else ("[有断点]" if ckpt else "[未开始]")
            print(f"  {name}: {status} - {cfg['desc']}")
        return

    SHARED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild_features:
        args.force = True
        for path in clear_precomputed_features(SHARED_FEATURES):
            print(f"🗑 已清除旧特征: {path}")
        if SHARED_SCALER.exists():
            SHARED_SCALER.unlink()
            print(f"🗑 已清除旧归一化器: {SHARED_SCALER}")
        for variant_name in VARIANTS:
            output_dir = SHARED_OUTPUT_DIR / variant_name
            for name in ('checkpoint_latest.pt', 'checkpoint_best.pt',
                         'best_model.pt', 'predictions.npz',
                         'final_results.json'):
                path = output_dir / name
                if path.exists():
                    path.unlink()
                    print(f"🗑 已清除旧训练产物: {path}")

    # 确定运行哪些变体
    if args.all:
        variants_to_run = list(VARIANTS.keys())
    elif args.variant:
        variants_to_run = [args.variant]
    else:
        print("请指定 --variant <name> 或 --all")
        return

    print("=" * 70)
    print("Reptile-Transformer 特征消融实验")
    print("=" * 70)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {SHARED_OUTPUT_DIR}")
    print(f"变体: {variants_to_run}")
    print(f"参数: epochs={args.epochs}, batch_size={args.batch_size}, inner_steps={args.inner_steps}, gpu={args.gpu}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"force={args.force}, resume={args.resume}")
    print("=" * 70)

    # ==========================
    # 预计算特征（共享，4个变体只算一次 ESM2）
    # ==========================
    all_targets = get_all_targets(test_mode=False, max_targets=None)
    print(f"   Train: {len(all_targets['train'])}, Val: {len(all_targets['val'])}, Test: {len(all_targets['test'])}")

    if not SHARED_FEATURES.exists():
        print("\n预计算特征（共享，四个变体共用）...")
        mol_extractor = MoleculeFeatureExtractor()
        protein_extractor = ProteinFeatureExtractor(
            model_path=args.esm2_model,
            output_dim=480,
        )
        precompute_features(all_targets, protein_extractor, mol_extractor, SHARED_FEATURES)
        del mol_extractor, protein_extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        metadata = validate_precomputed_features(SHARED_FEATURES)
        print(f"\n📦 使用已验证的特征: {metadata['esm2_loader']}")

    # ==========================
    # 加载共享数据
    # ==========================
    print("\n加载数据...")
    data = np.load(SHARED_FEATURES, mmap_mode='r', allow_pickle=True)
    splits = data['splits']
    target_names_all = data['target_names']

    train_indices = np.where(splits == 'train')[0]
    val_indices = np.where(splits == 'val')[0]
    test_indices = np.where(splits == 'test')[0]

    # 少样本实验：限制每个靶点的训练样本数
    if args.max_samples_per_target is not None:
        n_max = args.max_samples_per_target
        print(f"\n🔬 少样本实验: 每靶点最多 {n_max} 个训练样本")
        rng = np.random.RandomState(42)
        new_train = []
        for tname in np.unique(target_names_all[train_indices]):
            t_indices = train_indices[target_names_all[train_indices] == tname]
            if len(t_indices) > n_max:
                selected = rng.choice(t_indices, size=n_max, replace=False)
                new_train.extend(selected)
            else:
                new_train.extend(t_indices)
        train_indices = np.array(sorted(new_train))
        print(f"   训练样本: {len(train_indices)} (原 {np.sum(splits == 'train')})")

    # 拟合归一化器
    scaler = TargetScaler()
    y_train = data['y'][train_indices]
    target_names_train = target_names_all[train_indices]
    scaler.fit(y_train, target_names_train)
    scaler.save(SHARED_SCALER)

    print(f"   Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    splits_info = {'train': train_indices, 'val': val_indices, 'test': test_indices}
    shared_data = {
        'data': data,
        'scaler': scaler,
        'target_names_array': target_names_all,
    }

    # ==========================
    # 依次运行变体
    # ==========================
    results = {}
    total_start = time.time()

    for variant_name in variants_to_run:
        print(f"\n{'#' * 70}")
        print(f"# Reptile 变体: {variant_name}")
        print(f"{'#' * 70}")

        try:
            success = train_one_variant(variant_name, args, shared_data, splits_info)
            results[variant_name] = success
        except Exception as e:
            print(f"\n❌ {variant_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            results[variant_name] = False

    # ==========================
    # 总结
    # ==========================
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("Reptile-Transformer 消融实验总结")
    print("=" * 70)

    for variant_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {variant_name} - {VARIANTS[variant_name]['desc']}")

    # 打印对比表格（所有10个指标）
    print(f"\n{'='*140}")
    print(f"{'Variant':<22} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'Pearson':>8} {'Spearman':>9} "
          f"{'EF@1%':>6} {'EF@5%':>6} {'EF@10%':>7} {'ECE':>8} {'AUPR':>8}")
    print("-" * 140)
    for variant_name in variants_to_run:
        result_file = SHARED_OUTPUT_DIR / variant_name / 'final_results.json'
        if result_file.exists():
            with open(result_file) as f:
                r = json.load(f)
            m = r.get('test_metrics', r)
            def fmt(key, dec=4):
                v = m.get(key)
                if v is None:
                    return f"{'N/A':>8}"
                if dec == 2:
                    return f"{float(v):>6.2f}"
                return f"{float(v):>8.4f}"
            print(f"{variant_name:<22} {fmt('R2')} {fmt('RMSE')} {fmt('MAE')} "
                  f"{fmt('Pearson')} {fmt('Spearman')} {fmt('EF@1%',2)} {fmt('EF@5%',2)} "
                  f"{fmt('EF@10%',2)} {fmt('ECE')} {fmt('AUPR')}")
    print(f"{'='*140}")
    print(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
        print(f"\n⏱️ 总时间: {time.time() - start_time:.2f} 秒")
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
