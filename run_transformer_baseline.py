"""
Transformer基线训练脚本（无元学习）

使用ReptileTransformer模型的交叉注意力机制，但去掉Reptile元学习逻辑，
用正常的梯度下降训练。

运行方式：
    python run_transformer_baseline.py
"""

import os
import sys
import gc
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

# 配置
OUTPUT_DIR = BASE_DIR / "transformer_baseline_output"
FEATURES_PATH = OUTPUT_DIR / "precomputed_features.npz"
SCALER_PATH = OUTPUT_DIR / "target_scaler.json"


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    from scipy import stats
    
    # R²
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Pearson相关系数
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    
    # Spearman相关系数
    spearman_r, _ = stats.spearmanr(y_true, y_pred)
    
    # EF富集因子（按靶点计算）
    # 活性阈值：top 20%
    n_total = len(y_true)
    if n_total >= 10:
        active_count = max(5, int(n_total * 0.2))
        sorted_true = np.sort(y_true)[::-1]
        active_threshold = sorted_true[min(active_count - 1, len(sorted_true) - 1)]
        n_active = active_count
    else:
        active_threshold = float('inf')
        n_active = 0
    
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    def compute_ef(top_percent):
        n_top = int(n_total * top_percent / 100)
        if n_top == 0 or n_active == 0:
            return 0.0
        n_active_top = np.sum(y_true_sorted[:n_top] >= active_threshold)
        ef = (n_active_top / n_top) / (n_active / n_total)
        return ef
    
    ef1 = compute_ef(1)
    ef5 = compute_ef(5)
    ef10 = compute_ef(10)
    
    # ECE校准度
    def compute_ece(n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i+1]
            mask = (y_pred >= lower * (y_true.max() - y_true.min()) + y_true.min()) & \
                   (y_pred < upper * (y_true.max() - y_true.min()) + y_true.min())
            if np.sum(mask) > 0:
                avg_pred = np.mean(y_pred[mask])
                avg_true = np.mean(y_true[mask])
                ece += np.abs(avg_pred - avg_true) * np.sum(mask) / len(y_true)
        return ece
    
    ece = compute_ece()
    
    # AUPR (Area Under Precision-Recall curve)
    n_total = len(y_true)
    if n_total >= 5:
        n_pos = max(1, int(n_total * 0.2))
        sorted_true_desc = np.sort(y_true)[::-1]
        threshold = sorted_true_desc[min(n_pos - 1, len(sorted_true_desc) - 1)]
        is_positive = (y_true >= threshold).astype(int)
        n_total_positive = np.sum(is_positive)
        if n_total_positive > 0:
            sorted_indices = np.argsort(y_pred)[::-1]
            sorted_positives = is_positive[sorted_indices]
            tp = np.cumsum(sorted_positives)
            fp = np.cumsum(1 - sorted_positives)
            precisions = tp / (tp + fp + 1e-10)
            recalls = tp / n_total_positive
            recalls_prev = np.zeros_like(recalls)
            recalls_prev[1:] = recalls[:-1]
            aupr_val = float(np.sum((recalls - recalls_prev) * precisions))
        else:
            aupr_val = 0.0
    else:
        aupr_val = 0.0
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Pearson': pearson_r,
        'Spearman': spearman_r,
        'EF@1%': ef1,
        'EF@5%': ef5,
        'EF@10%': ef10,
        'ECE': ece,
        'AUPR': aupr_val
    }


def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description="Transformer基线训练（无元学习）")
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help="预处理数据目录，默认使用 data/processed/chembl",
    )
    parser.add_argument('--output_dir', type=str, default=None, help="输出目录")
    parser.add_argument('--resume', action='store_true', help="从断点恢复训练")
    parser.add_argument('--force', action='store_true', help="强制重新训练（忽略断点）")
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
        help='删除旧特征缓存并用当前 Hugging Face ESM-2 重新生成',
    )
    parser.add_argument('--batch_size', type=int, default=None, help="批处理大小")
    parser.add_argument('--epochs', type=int, default=None, help="训练轮数")
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'morgan_descriptors'],
                        help='特征消融: none(全部特征) / morgan_descriptors(Morgan+理化+ESM2, 屏蔽MACCS)')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.data_dir:
        os.environ['PREPROCESSED_DIR'] = args.data_dir

    if args.esm2_model:
        os.environ['ESM2_MODEL'] = args.esm2_model
        print(f"🧬 使用 ESM-2 模型: {args.esm2_model}")

    if args.output_dir:
        global OUTPUT_DIR, FEATURES_PATH, SCALER_PATH
        OUTPUT_DIR = Path(args.output_dir)
        FEATURES_PATH = OUTPUT_DIR / "precomputed_features.npz"
        SCALER_PATH = OUTPUT_DIR / "target_scaler.json"

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    scaler_amp = torch.amp.GradScaler('cuda') if use_amp else None

    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU: {props.name}, {props.total_memory / 1e9:.2f}GB")
    print(f"   Device: {device}, AMP: {use_amp}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild_features:
        args.force = True
        for path in clear_precomputed_features(FEATURES_PATH):
            print(f"🗑 已清除旧特征: {path}")
        for name in ('target_scaler.json', 'checkpoint.pt', 'best_model.pt',
                     'predictions.npz', 'final_results.json'):
            path = OUTPUT_DIR / name
            if path.exists():
                path.unlink()
                print(f"🗑 已清除旧训练产物: {path}")

    all_targets = get_all_targets(test_mode=False, max_targets=None)
    print(f"   Train: {len(all_targets['train'])}, Val: {len(all_targets['val'])}, Test: {len(all_targets['test'])}")

    if not FEATURES_PATH.exists():
        print("\n预计算特征...")
        mol_extractor = MoleculeFeatureExtractor()
        protein_extractor = ProteinFeatureExtractor(
            model_path=args.esm2_model,
            output_dim=480,
        )
        precompute_features(all_targets, protein_extractor, mol_extractor, FEATURES_PATH)
        del mol_extractor, protein_extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        metadata = validate_precomputed_features(FEATURES_PATH)
        print(f"   ESM-2 特征缓存已验证: {metadata['esm2_loader']}")

    # 加载数据（mmap 模式，不占用大量内存）
    print("\n加载数据...")
    data = np.load(FEATURES_PATH, mmap_mode='r', allow_pickle=True)
    splits = data['splits']
    target_names_all = data['target_names']

    train_mask = splits == 'train'
    val_mask = splits == 'val'
    test_mask = splits == 'test'

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]

    # 拟合归一化器（只需要 y 值，轻量）
    scaler = TargetScaler()
    y_train = data['y'][train_mask]
    target_names_train = target_names_all[train_mask]
    scaler.fit(y_train, target_names_train)
    scaler.save(SCALER_PATH)

    # 使用 LazySubset：只保存索引，按需 batch 读取
    print(f"   Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    print(f"   预加载数据到GPU（一次性传输，避免每batch重复转换）...")

    train_data = LazySubset(data, train_indices, scaler)
    val_data = LazySubset(data, val_indices, scaler)
    test_data = LazySubset(data, test_indices, scaler)

    # 预加载所有数据到GPU（53万样本约2-3GB，4090的24GB足够）
    if torch.cuda.is_available():
        def preload_to_gpu(loader):
            """一次性把全部数据移至GPU，后续batch直接切片"""
            n = len(loader)
            all_idx = np.arange(n)
            batch = loader.get_batch(all_idx)
            loader._gpu_data = {
                'morgan': torch.from_numpy(batch['morgan']).to(device),
                'maccs': torch.from_numpy(batch['maccs']).to(device),
                'descriptors': torch.from_numpy(batch['descriptors']).to(device),
                'protein': torch.from_numpy(batch['protein']).to(device),
                'y_norm': torch.from_numpy(batch['y_norm']).to(device),
                'y': torch.from_numpy(batch['y']).to(device),
            }
            print(f"      已预加载 {n} 样本到 GPU")

        preload_to_gpu(train_data)
        preload_to_gpu(val_data)
        preload_to_gpu(test_data)

        # 覆盖 get_batch，直接从GPU切片
        def gpu_get_batch(self, batch_indices):
            idx = torch.from_numpy(np.asarray(batch_indices)).long()
            maccs = self._gpu_data['maccs'][idx]
            # 特征消融：屏蔽MACCS
            if args.ablation == 'morgan_descriptors':
                maccs = torch.zeros_like(maccs)
            return {
                'morgan': self._gpu_data['morgan'][idx],
                'maccs': maccs,
                'descriptors': self._gpu_data['descriptors'][idx],
                'protein': self._gpu_data['protein'][idx],
                'y_norm': self._gpu_data['y_norm'][idx],
                'y': self._gpu_data['y'][idx],
            }
        import types
        train_data.get_batch = types.MethodType(gpu_get_batch, train_data)
        val_data.get_batch = types.MethodType(gpu_get_batch, val_data)
        test_data.get_batch = types.MethodType(gpu_get_batch, test_data)

    # 不再需要 data 全量引用，gc 可以回收
    del splits, target_names_all, train_mask, val_mask, test_mask
    gc.collect()

    # 创建模型
    model = ReptileTransformer()
    model.apply(init_weights)
    model.to(device)
    total_params, trainable_params = count_params(model)
    print(f"   Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # 训练配置
    batch_size = args.batch_size if args.batch_size is not None else 2048
    epochs = args.epochs if args.epochs is not None else 300
    lr = 0.00015
    patience = 50
    warmup_epochs = 30

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

    contrastive_weight = 0.03
    consistency_weight = 0.05
    ranking_weight = 0.5
    calibration_weight = 0.1

    total_batches = (len(train_data) + batch_size - 1) // batch_size
    print("\n开始训练（无元学习）...")
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    checkpoint_path = OUTPUT_DIR / 'checkpoint.pt'
    final_results_path = OUTPUT_DIR / 'final_results.json'
    
    # 如果已有 final_results.json 且不强制重跑，直接跳过
    if final_results_path.exists() and not args.force and not args.resume:
        print(f"✅ 已完成，跳过（用 --force 强制重跑）")
        return
    
    # --force 时清除旧断点（避免CPU训练的断点不兼容GPU）
    if args.force and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"🗑 已清除旧断点: {checkpoint_path}")
    
    # 断点恢复
    if args.resume and checkpoint_path.exists():
        print(f"💾 加载断点: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        patience_counter = ckpt['patience_counter']
        print(f"   从 epoch {start_epoch} 恢复, best_val_loss={best_val_loss:.4f}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * warmup_factor
        elif epoch == warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        train_perm = np.random.permutation(len(train_data))

        for i in range(0, len(train_data), batch_size):
            bi = train_perm[i:i+batch_size]
            batch = train_data.get_batch(bi)
            # 数据可能已在GPU上（预加载模式），否则从numpy转换
            morgan = batch['morgan'] if torch.is_tensor(batch['morgan']) else torch.from_numpy(batch['morgan']).to(device)
            maccs = batch['maccs'] if torch.is_tensor(batch['maccs']) else torch.from_numpy(batch['maccs']).to(device)
            descriptors = batch['descriptors'] if torch.is_tensor(batch['descriptors']) else torch.from_numpy(batch['descriptors']).to(device)
            protein = batch['protein'] if torch.is_tensor(batch['protein']) else torch.from_numpy(batch['protein']).to(device)
            y_norm = batch['y_norm'] if torch.is_tensor(batch['y_norm']) else torch.from_numpy(batch['y_norm']).to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    preds, consistency_loss, contrastive_loss = model(morgan, maccs, descriptors, protein)
                    mse_loss = F.mse_loss(preds, y_norm)
                    total_loss_val = mse_loss

                    if epoch >= warmup_epochs:
                        if contrastive_weight > 0:
                            total_loss_val += contrastive_weight * contrastive_loss
                        if consistency_weight > 0:
                            total_loss_val += consistency_weight * consistency_loss
                        if ranking_weight > 0 and len(bi) >= 2:
                            pred_probs = F.softmax(preds, dim=0)
                            target_probs = F.softmax(y_norm, dim=0)
                            pred_probs = torch.clamp(pred_probs, min=1e-10)
                            target_probs = torch.clamp(target_probs, min=1e-10)
                            ranking_loss = F.kl_div(torch.log(pred_probs), target_probs, reduction='batchmean')
                            total_loss_val += ranking_weight * ranking_loss
                        if calibration_weight > 0:
                            preds_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-10)
                            y_norm_norm = (y_norm - y_norm.min()) / (y_norm.max() - y_norm.min() + 1e-10)
                            calibration_loss = F.mse_loss(preds_norm, y_norm_norm)
                            total_loss_val += calibration_weight * calibration_loss

                scaler_amp.scale(total_loss_val).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                preds, consistency_loss, contrastive_loss = model(morgan, maccs, descriptors, protein)
                mse_loss = F.mse_loss(preds, y_norm)
                total_loss_val = mse_loss

                if epoch >= warmup_epochs:
                    if contrastive_weight > 0:
                        total_loss_val += contrastive_weight * contrastive_loss
                    if consistency_weight > 0:
                        total_loss_val += consistency_weight * consistency_loss
                    if ranking_weight > 0 and len(bi) >= 2:
                        pred_probs = F.softmax(preds, dim=0)
                        target_probs = F.softmax(y_norm, dim=0)
                        pred_probs = torch.clamp(pred_probs, min=1e-10)
                        target_probs = torch.clamp(target_probs, min=1e-10)
                        ranking_loss = F.kl_div(torch.log(pred_probs), target_probs, reduction='batchmean')
                        total_loss_val += ranking_weight * ranking_loss
                    if calibration_weight > 0:
                        preds_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-10)
                        y_norm_norm = (y_norm - y_norm.min()) / (y_norm.max() - y_norm.min() + 1e-10)
                        calibration_loss = F.mse_loss(preds_norm, y_norm_norm)
                        total_loss_val += calibration_weight * calibration_loss

                total_loss_val.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += total_loss_val.item() * len(bi)

        avg_train_loss = total_loss / len(train_data)
        if epoch >= warmup_epochs:
            scheduler.step()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                bi = np.arange(i, min(i+batch_size, len(val_data)))
                batch = val_data.get_batch(bi)
                morgan = batch['morgan'] if torch.is_tensor(batch['morgan']) else torch.from_numpy(batch['morgan']).to(device)
                maccs = batch['maccs'] if torch.is_tensor(batch['maccs']) else torch.from_numpy(batch['maccs']).to(device)
                descriptors = batch['descriptors'] if torch.is_tensor(batch['descriptors']) else torch.from_numpy(batch['descriptors']).to(device)
                protein = batch['protein'] if torch.is_tensor(batch['protein']) else torch.from_numpy(batch['protein']).to(device)
                y_norm = batch['y_norm'] if torch.is_tensor(batch['y_norm']) else torch.from_numpy(batch['y_norm']).to(device)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        preds, consistency_loss, contrastive_loss = model(morgan, maccs, descriptors, protein)
                        vloss = F.mse_loss(preds, y_norm)
                        if contrastive_weight > 0:
                            vloss += contrastive_weight * contrastive_loss
                        if consistency_weight > 0:
                            vloss += consistency_weight * consistency_loss
                        val_loss += vloss.item() * len(bi)
                else:
                    preds, consistency_loss, contrastive_loss = model(morgan, maccs, descriptors, protein)
                    vloss = F.mse_loss(preds, y_norm)
                    if contrastive_weight > 0:
                        vloss += contrastive_weight * contrastive_loss
                    if consistency_weight > 0:
                        vloss += consistency_weight * consistency_loss
                    val_loss += vloss.item() * len(bi)

        avg_val_loss = val_loss / len(val_data)

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pt')
        else:
            patience_counter += 1
        
        # 保存断点
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
        }, checkpoint_path)
        
        if patience_counter >= patience:
            print(f"早停: {patience}个epoch无改善")
            break

    # 测试评估
    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)

    model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model.pt', map_location=device))
    model.eval()

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            bi = np.arange(i, min(i+batch_size, len(test_data)))
            batch = test_data.get_batch(bi)
            morgan = batch['morgan'] if torch.is_tensor(batch['morgan']) else torch.from_numpy(batch['morgan']).to(device)
            maccs = batch['maccs'] if torch.is_tensor(batch['maccs']) else torch.from_numpy(batch['maccs']).to(device)
            descriptors = batch['descriptors'] if torch.is_tensor(batch['descriptors']) else torch.from_numpy(batch['descriptors']).to(device)
            protein = batch['protein'] if torch.is_tensor(batch['protein']) else torch.from_numpy(batch['protein']).to(device)
            y_raw = batch['y'] if torch.is_tensor(batch['y']) else torch.from_numpy(batch['y']).to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    preds, _, _ = model(morgan, maccs, descriptors, protein)
            else:
                preds, _, _ = model(morgan, maccs, descriptors, protein)

            y_pred_raw = scaler.inverse_transform(preds.cpu().numpy())
            all_y_true.extend(y_raw.cpu().numpy())
            all_y_pred.extend(y_pred_raw)

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    metrics = compute_metrics(y_true, y_pred)
    metrics['samples'] = len(y_true)

    print(f"   R2: {metrics['R2']:.4f}")
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   MAE: {metrics['MAE']:.4f}")
    print(f"   Pearson: {metrics['Pearson']:.4f}")
    print(f"   Spearman: {metrics['Spearman']:.4f}")
    print(f"   EF@1%: {metrics['EF@1%']:.2f}")
    print(f"   EF@5%: {metrics['EF@5%']:.2f}")
    print(f"   EF@10%: {metrics['EF@10%']:.2f}")
    print(f"   ECE: {metrics['ECE']:.4f}")
    print(f"   AUPR: {metrics['AUPR']:.4f}")
    print(f"   Samples: {metrics['samples']}")

    metrics_serializable = {k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}
    with open(OUTPUT_DIR / 'final_results.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    print(f"\n完成! 输出: {OUTPUT_DIR}")


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
