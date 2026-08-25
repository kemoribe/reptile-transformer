"""
Reptile-Transformer分子活性预测框架 - 主运行脚本

基于元学习的药物-靶点结合活性预测框架：
1. 分子特征：Morgan指纹 + RDKit描述符
2. 蛋白质特征：ESM2预训练模型（预计算）
3. Transformer融合：双向交叉注意力
4. 元学习：Reptile算法（内循环适应，外循环泛化）
5. 对比学习：增强表示学习

运行方式：
    python run_reptile_transformer.py [--test_mode] [--precompute_only]
"""

import os
import sys
import gc
import json
import time
import argparse
import numpy as np
from pathlib import Path

# 设置路径
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 导入模块
from data_preprocessing import (
    MoleculeFeatureExtractor,
    ProteinFeatureExtractor,
    get_all_targets,
    precompute_features,
    clear_precomputed_features,
    validate_precomputed_features,
    TargetScaler,
    TargetDataLoader,
    LazySubset
)
from reptile_transformer_model import ReptileTransformer, count_params, init_weights
from reptile_training import ReptileTrainer, quick_test

# 配置
OUTPUT_DIR = BASE_DIR / "reptile_output"
FEATURES_PATH = OUTPUT_DIR / "precomputed_features.npz"
SCALER_PATH = OUTPUT_DIR / "target_scaler.json"
CONFIG_PATH = OUTPUT_DIR / "training_config.json"

device = "cuda:0" if __import__('torch').cuda.is_available() else "cpu"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Reptile-Transformer分子活性预测")
    parser.add_argument('--test_mode', action='store_true', help="快速测试模式（5个训练靶点）")
    parser.add_argument('--precompute_only', action='store_true', help="仅预计算特征")
    parser.add_argument('--no_precompute', action='store_true', help="不重新预计算特征")
    parser.add_argument('--quick_test', action='store_true', help="快速测试前向传播")
    parser.add_argument('--num_train_targets', type=int, default=None, help="训练靶点数量限制")
    parser.add_argument('--resume', action='store_true', help="从断点恢复训练")
    parser.add_argument('--batch_size', type=int, default=512, help="批处理大小")
    parser.add_argument('--epochs', type=int, default=200, help="训练轮数")
    parser.add_argument('--inner_steps', type=int, default=3, help="内循环步数")
    parser.add_argument('--gradient_accumulation', type=int, default=1, help="梯度累积步数")
    parser.add_argument('--gpu', type=str, default='0', help="GPU设备编号，如'0'或'0,1'")
    parser.add_argument(
        '--esm2_model',
        type=str,
        default=None,
        help='Hugging Face ESM-2 本地目录或模型名',
    )
    parser.add_argument(
        '--rebuild_features',
        action='store_true',
        help='删除旧特征缓存并用当前 Hugging Face ESM-2 重新生成',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help="预处理数据目录，默认使用 data/processed/chembl",
    )
    parser.add_argument('--output_dir', type=str, default=None, help="输出目录")
    parser.add_argument('--force', action='store_true', help='强制重新训练（忽略已完成状态）')
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'morgan_descriptors'],
                        help='特征消融: none(全部特征) / morgan_descriptors(Morgan+理化+ESM2, 屏蔽MACCS)')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置输出目录（如果指定）
    if args.output_dir:
        global OUTPUT_DIR, FEATURES_PATH, SCALER_PATH, CONFIG_PATH
        OUTPUT_DIR = Path(args.output_dir)
        FEATURES_PATH = OUTPUT_DIR / "precomputed_features.npz"
        SCALER_PATH = OUTPUT_DIR / "target_scaler.json"
        CONFIG_PATH = OUTPUT_DIR / "training_config.json"

    if args.data_dir:
        os.environ['PREPROCESSED_DIR'] = args.data_dir
        print(f"📂 使用自定义数据目录: {args.data_dir}")

    if args.esm2_model:
        os.environ['ESM2_MODEL'] = args.esm2_model
        print(f"🧬 使用 ESM-2 模型: {args.esm2_model}")

    if args.rebuild_features:
        args.force = True

    # 跳过已完成
    final_results_path = OUTPUT_DIR / 'final_results.json'
    if final_results_path.exists() and not args.resume and not getattr(args, 'force', False):
        print(f"✅ 已完成，跳过（用 --force 强制重跑或 --resume 断点恢复）")
        return
    
    # --force 时清除旧断点和旧特征文件（避免不同数据集的 mmap 文件冲突）
    if getattr(args, 'force', False):
        for ckpt_name in ['checkpoint_latest.pt', 'checkpoint_best.pt', 'best_model.pt']:
            ckpt_path = OUTPUT_DIR / ckpt_name
            if ckpt_path.exists():
                ckpt_path.unlink()
                print(f"🗑 已清除旧断点: {ckpt_path}")
        for path in clear_precomputed_features(FEATURES_PATH):
            print(f"🗑 已清除旧特征: {path}")
        for name in ('target_scaler.json', 'final_results.json'):
            path = OUTPUT_DIR / name
            if path.exists():
                path.unlink()
                print(f"🗑 已清除旧训练产物: {path}")
    
    # 确定设备（支持多GPU）
    import torch
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device('cuda:0')
        num_gpus = torch.cuda.device_count()
        print("=" * 70)
        print("🔬 Reptile-Transformer 分子活性预测框架")
        print("=" * 70)
        print(f"   设备: {device}")
        print(f"   GPU数量: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name}, {props.total_memory / 1e9:.2f}GB")
    else:
        device = torch.device('cpu')
        num_gpus = 0
        print("=" * 70)
        print("🔬 Reptile-Transformer 分子活性预测框架")
        print("=" * 70)
        print(f"   设备: {device}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   断点恢复: {'✅' if args.resume else '❌'}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Inner Steps: {args.inner_steps}")
    print(f"   Gradient Accumulation: {args.gradient_accumulation}")
    if args.ablation != 'none':
        print(f"   ⚠️ 特征消融: {args.ablation} (MACCS将被屏蔽)")
    print("=" * 70)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 获取所有靶点
    print("\n📂 加载靶点信息...")
    test_mode = args.test_mode or args.quick_test
    
    # 确定训练靶点数量
    if test_mode:
        max_targets = 5
    elif args.num_train_targets is not None:
        max_targets = args.num_train_targets
    else:
        max_targets = None  # 使用所有训练靶点
    
    all_targets = get_all_targets(test_mode=test_mode, max_targets=max_targets)
    
    print(f"   Train: {len(all_targets['train'])} targets")
    print(f"   Val: {len(all_targets['val'])} targets")
    print(f"   Test: {len(all_targets['test'])} targets")
    
    # 2. 预计算特征（如果需要）
    if not FEATURES_PATH.exists():
        if args.no_precompute:
            raise FileNotFoundError(
                f"未找到特征缓存: {FEATURES_PATH}。移除 --no_precompute 后重新运行。"
            )
        print("\n🔧 预计算特征...")
        
        # 初始化特征提取器
        mol_extractor = MoleculeFeatureExtractor()
        protein_extractor = ProteinFeatureExtractor(
            model_path=args.esm2_model,
            output_dim=480,
        )
        
        # 预计算特征
        precompute_features(all_targets, protein_extractor, mol_extractor, FEATURES_PATH)
        
        # 释放内存
        del mol_extractor, protein_extractor
        gc.collect()
        if __import__('torch').cuda.is_available():
            __import__('torch').cuda.empty_cache()
        
        if args.precompute_only:
            print("\n✅ 特征预计算完成!")
            return
    else:
        metadata = validate_precomputed_features(FEATURES_PATH)
        print(f"\n📦 使用已验证的特征: {metadata['esm2_loader']}")
    
    # 3. 拟合归一化器
    print("\n📏 拟合归一化器...")
    scaler = TargetScaler()
    
    # 加载数据获取训练集y值
    data = __import__('numpy').load(FEATURES_PATH, mmap_mode='r', allow_pickle=True)
    splits = data['splits']
    train_mask = splits == 'train'
    
    y_train = data['y'][train_mask]
    target_names_train = data['target_names'][train_mask]
    
    scaler.fit(y_train, target_names_train)
    scaler.save(SCALER_PATH)
    
    del data, y_train, target_names_train
    gc.collect()
    
    # 4. 创建数据加载器
    print("\n📥 创建数据加载器...")
    
    # 创建训练集加载器
    # 先一次性加载到内存中进行索引创建，避免mmap遍历慢
    train_data = __import__('numpy').load(FEATURES_PATH, mmap_mode='r', allow_pickle=True)
    
    # 将splits和target_names加载到内存进行快速索引
    splits_array = train_data['splits'][:]  # 加载到内存
    target_names_array = train_data['target_names'][:]  # 加载到内存
    
    # 使用numpy向量化操作创建索引
    print("   创建索引...")
    train_mask = splits_array == 'train'
    val_mask = splits_array == 'val'
    test_mask = splits_array == 'test'
    
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    # 创建训练集的索引映射（向量化操作）
    import collections
    train_target_groups = collections.defaultdict(list)
    
    # 使用向量化操作：先分组，再收集索引
    unique_targets, inverse_indices = np.unique(target_names_array[train_indices], return_inverse=True)
    for idx, target_name in enumerate(unique_targets):
        train_target_groups[target_name] = train_indices[inverse_indices == idx].tolist()
    
    print(f"   Train samples: {len(train_indices)}")
    print(f"   Val samples: {len(val_indices)}")
    print(f"   Test samples: {len(test_indices)}")
    
    # 5. 创建模型
    print("\n🧠 创建模型...")
    model = ReptileTransformer()

    # 初始化权重
    model.apply(init_weights)

    # 统计参数
    total_params, trainable_params = count_params(model)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    # 将模型移至GPU（关键：缺失会导致模型在CPU上运行，极度缓慢）
    model = model.to(device)
    print(f"   模型已移至: {device}")

    # 多GPU支持
    if num_gpus > 1:
        print(f"   使用 {num_gpus} 个GPU进行训练")
        model = torch.nn.DataParallel(model)
    
    # 6. 快速测试（如果需要）
    if args.quick_test:
        print("\n🔍 快速测试...")
        
        # 创建临时加载器用于测试（使用预加载的target_names数组）
        class TestLoader:
            def __init__(self, data, indices, target_names_array=None):
                self.data = data
                self.target_groups = __import__('collections').defaultdict(list)
                
                if target_names_array is not None:
                    for i in indices:
                        self.target_groups[target_names_array[i]].append(i)
                else:
                    target_names = data['target_names'][:]
                    for i in indices:
                        self.target_groups[target_names[i]].append(i)
            
            def get_all_targets(self):
                return list(self.target_groups.keys())
            
            def get_target_data(self, target_name):
                import torch
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                indices = self.target_groups[target_name]
                
                morgan = torch.tensor(self.data['morgan'][indices].astype(np.float32), dtype=torch.float32, device=device)
                maccs = torch.tensor(self.data['maccs'][indices].astype(np.float32), dtype=torch.float32, device=device)
                descriptors = torch.tensor(self.data['descriptors'][indices].astype(np.float32), dtype=torch.float32, device=device)
                protein = torch.tensor(self.data['protein'][indices].astype(np.float32), dtype=torch.float32, device=device)
                y = torch.tensor(self.data['y'][indices].astype(np.float32), dtype=torch.float32, device=device)
                y_norm = torch.tensor(scaler.transform(y.cpu().numpy()), dtype=torch.float32, device=device)
                
                return morgan, maccs, descriptors, protein, y_norm
        
        test_loader = TestLoader(train_data, train_indices, target_names_array)
        quick_test(model, test_loader, n_tasks=min(3, len(test_loader.get_all_targets())))
        
        del test_loader
        print("\n✅ 快速测试完成!")
        return
    
    # 7. 创建训练器
    print("\n🚀 创建训练器...")
    
    # 创建数据加载器（使用 LazyDataLoader + GPU缓存，避免重复数据传输）
    class LazyDataLoader:
        def __init__(self, data_mmap, indices, scaler, target_names_array=None, ablation='none'):
            from collections import defaultdict

            self._data = data_mmap
            self._scaler = scaler
            self._ablation = ablation

            # 构建靶点分组索引
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
            # GPU数据缓存：避免每次访问靶点都重新从numpy创建tensor
            self._gpu_cache = {}
            print(f"   样本数: {len(self._indices)}, 靶点数: {len(self.target_groups)}")

        def get_all_targets(self):
            return list(self.target_groups.keys())

        def get_target_data(self, target_name):
            import torch
            # 命中缓存则直接返回（数据已在GPU上）
            if target_name in self._gpu_cache:
                return self._gpu_cache[target_name]

            rel_indices = self.target_groups[target_name]
            if len(rel_indices) == 0:
                return None

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            real_indices = self._indices[rel_indices]
            # 一次性创建并移至GPU，后续访问直接复用
            morgan = torch.tensor(self._data['morgan'][real_indices].astype(np.float32), dtype=torch.float32, device=device)
            maccs = torch.tensor(self._data['maccs'][real_indices].astype(np.float32), dtype=torch.float32, device=device)
            descriptors = torch.tensor(self._data['descriptors'][real_indices].astype(np.float32), dtype=torch.float32, device=device)
            protein = torch.tensor(self._data['protein'][real_indices].astype(np.float32), dtype=torch.float32, device=device)
            y_raw = self._data['y'][real_indices].astype(np.float32)
            y_norm = self._scaler.transform(y_raw)
            y_norm = torch.tensor(y_norm, dtype=torch.float32, device=device)

            cached = (morgan, maccs, descriptors, protein, y_norm)
            # 特征消融：屏蔽MACCS
            if self._ablation == 'morgan_descriptors':
                maccs = torch.zeros_like(maccs)
                cached = (morgan, maccs, descriptors, protein, y_norm)
            self._gpu_cache[target_name] = cached
            return cached

        def clear_cache(self):
            """清理GPU缓存（内存不足时调用）"""
            self._gpu_cache.clear()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 创建数据加载器（使用 LazyDataLoader 按需加载）
    train_loader = LazyDataLoader(train_data, train_indices, scaler, target_names_array, args.ablation)
    val_loader = LazyDataLoader(train_data, val_indices, scaler, target_names_array, args.ablation)
    test_loader = LazyDataLoader(train_data, test_indices, scaler, target_names_array, args.ablation)
    
    # 创建自定义训练配置
    config = __import__('reptile_training').TrainingConfig()
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.INNER_STEPS = args.inner_steps
    config.GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation
    
    # 创建训练器
    trainer = ReptileTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        target_scaler=scaler,
        output_dir=str(OUTPUT_DIR),
        config=config
    )
    
    # 8. 训练
    print("\n🔥 开始训练...")
    training_history = trainer.train(resume=args.resume)
    
    # 9. 测试评估
    print("\n" + "=" * 70)
    print("📊 测试集评估")
    print("=" * 70)
    test_metrics = trainer.evaluate_test()
    
    # 10. 保存最终结果
    print("\n💾 保存结果...")
    results = {
        'training_history': training_history,
        'test_metrics': test_metrics,
        'config': {
            'inner_lr': trainer.config.INNER_LR,
            'inner_steps': trainer.config.INNER_STEPS,
            'meta_lr': trainer.config.META_LR,
            'epochs': trainer.config.EPOCHS,
            'batch_size': trainer.config.BATCH_SIZE
        },
        'best_val_r2': trainer.best_val_r2,
        'best_val_rmse': trainer.best_val_rmse
    }
    
    with open(OUTPUT_DIR / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ 全部完成!")
    print(f"\n📊 最终测试结果:")
    print(f"   R²: {test_metrics['R2']:.4f}")
    print(f"   RMSE: {test_metrics['RMSE']:.4f}")
    print(f"   MAE: {test_metrics['MAE']:.4f}")
    print(f"   Pearson: {test_metrics.get('Pearson', 'N/A'):.4f}")
    print(f"   Spearman: {test_metrics.get('Spearman', 'N/A'):.4f}")
    print(f"   AUPR: {test_metrics.get('AUPR', 'N/A'):.4f}")
    print(f"   EF@1%: {test_metrics.get('EF@1%', 'N/A'):.2f}")
    print(f"   EF@5%: {test_metrics.get('EF@5%', 'N/A'):.2f}")
    print(f"   EF@10%: {test_metrics.get('EF@10%', 'N/A'):.2f}")
    print(f"   ECE: {test_metrics.get('ECE', 'N/A'):.4f}")
    print(f"   Samples: {test_metrics.get('samples', 'N/A')}")
    print(f"\n📂 输出文件:")
    print(f"   最佳模型: {OUTPUT_DIR / 'best_model.pt'}")
    print(f"   检查点: {OUTPUT_DIR / 'checkpoint_best.pt'}")
    print(f"   训练日志: {OUTPUT_DIR / 'logs'}")
    print(f"   训练历史: {OUTPUT_DIR / 'training_history.json'}")
    print(f"   每个靶点结果: {OUTPUT_DIR / 'per_target_results.json'}")
    
    return results


if __name__ == "__main__":
    start_time = time.time()
    
    try:
        results = main()
        print(f"\n⏱️ 总时间: {time.time() - start_time:.2f} 秒")
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
