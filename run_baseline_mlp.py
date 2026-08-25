"""
非元学习基线训练脚本 - 用于对比实验

不使用Reptile元学习，直接训练一个简单的MLP模型：
分子特征 + 蛋白质特征 → MLP → 预测

GPU优化：pin_memory + AMP混合精度 + 非阻塞传输
"""

import os
import sys
import gc
import json
import time
import argparse
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

OUTPUT_DIR = BASE_DIR / "baseline_output"
FEATURES_PATH = OUTPUT_DIR / "precomputed_features.npz"
SCALER_PATH = OUTPUT_DIR / "target_scaler.json"

MORGAN_BITS = 2048
MACCS_DIM = 167
DESC_DIM = 10
ESM2_DIM = 480


class BaselineMLP(nn.Module):
    def __init__(self, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.morgan_proj = nn.Sequential(
            nn.Linear(MORGAN_BITS, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.maccs_proj = nn.Sequential(
            nn.Linear(MACCS_DIM, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.desc_proj = nn.Sequential(
            nn.Linear(DESC_DIM, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.protein_proj = nn.Sequential(
            nn.Linear(ESM2_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim // 4 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))

    def forward(self, morgan_fp, maccs_fp, descriptors, protein):
        morgan_feat = self.morgan_proj(morgan_fp)
        maccs_feat = self.maccs_proj(maccs_fp)
        desc_feat = self.desc_proj(descriptors)
        protein_feat = self.protein_proj(protein)
        combined = torch.cat([morgan_feat, maccs_feat, desc_feat, protein_feat], dim=-1)
        fused = self.fusion(combined)
        preds = self.predictor(fused).squeeze()
        preds = preds * self.output_scale + self.output_bias
        return preds, 0.0, 0.0


def compute_metrics(y_true, y_pred):
    from scipy import stats
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    spearman_r, _ = stats.spearmanr(y_true, y_pred)

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
        return (n_active_top / n_top) / (n_active / n_total)

    ef1, ef5, ef10 = compute_ef(1), compute_ef(5), compute_ef(10)

    def compute_ece(n_bins=10):
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        range_val = max_val - min_val
        if range_val == 0:
            return 0.0
        y_true_norm = (y_true - min_val) / range_val
        y_pred_norm = (y_pred - min_val) / range_val
        bin_indices = np.digitize(y_pred_norm, np.linspace(0, 1, n_bins + 1)[1:-1])
        ece = 0.0
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) == 0:
                continue
            ece += np.abs(np.mean(y_pred_norm[mask]) - np.mean(y_true_norm[mask])) * np.sum(mask) / len(y_true)
        return ece

    ece = compute_ece()

    if n_total >= 5:
        n_pos = max(1, int(n_total * 0.2))
        threshold = np.sort(y_true)[::-1][min(n_pos - 1, n_total - 1)]
        is_positive = (y_true >= threshold).astype(int)
        n_total_positive = np.sum(is_positive)
        if n_total_positive > 0:
            sorted_positives = is_positive[np.argsort(y_pred)[::-1]]
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
        'R2': r2, 'RMSE': rmse, 'MAE': mae,
        'Pearson': pearson_r, 'Spearman': spearman_r,
        'EF@1%': ef1, 'EF@5%': ef5, 'EF@10%': ef10,
        'ECE': ece, 'AUPR': aupr_val
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline MLP")
    parser.add_argument('--data_dir', type=str, default=None, help='预处理数据目录')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--resume', action='store_true', help='从断点恢复训练')
    parser.add_argument('--force', action='store_true', help='强制重新训练（忽略断点）')
    parser.add_argument('--gpu', type=str, default='0', help='GPU设备编号')
    parser.add_argument('--no_precompute', action='store_true', help='不重新预计算特征')
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
    parser.add_argument('--batch_size', type=int, default=None, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'morgan_descriptors'],
                        help='特征消融: none(全部特征) / morgan_descriptors(Morgan+理化+ESM2, 屏蔽MACCS)')
    parser.add_argument('--max_samples_per_target', type=int, default=None,
                        help='少样本实验: 每个靶点最多使用N个训练样本 (None=全部)')
    return parser.parse_args()


def main():
    global OUTPUT_DIR, FEATURES_PATH, SCALER_PATH
    args = parse_args()

    if args.data_dir:
        os.environ['PREPROCESSED_DIR'] = args.data_dir
        print(f"📂 使用数据目录: {args.data_dir}")

    if args.esm2_model:
        os.environ['ESM2_MODEL'] = args.esm2_model
        print(f"🧬 使用 ESM-2 模型: {args.esm2_model}")

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        FEATURES_PATH = OUTPUT_DIR / "precomputed_features.npz"
        SCALER_PATH = OUTPUT_DIR / "target_scaler.json"
    elif args.max_samples_per_target is not None:
        dataset_name = Path(args.data_dir).name if args.data_dir else "default"
        OUTPUT_DIR = BASE_DIR / f"baseline_output_{dataset_name}_fewshot{args.max_samples_per_target}"
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
    print(f"   Train: {len(all_targets['train'])} targets, Val: {len(all_targets['val'])} targets, Test: {len(all_targets['test'])} targets")

    if not FEATURES_PATH.exists():
        if args.no_precompute:
            raise FileNotFoundError(
                f"未找到特征缓存: {FEATURES_PATH}。移除 --no_precompute 后重新运行。"
            )
        print("\n预计算特征...")
        mol_extractor = MoleculeFeatureExtractor()
        protein_extractor = ProteinFeatureExtractor(
            model_path=args.esm2_model,
            output_dim=ESM2_DIM,
        )
        precompute_features(all_targets, protein_extractor, mol_extractor, FEATURES_PATH)
        del mol_extractor, protein_extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        metadata = validate_precomputed_features(FEATURES_PATH)
        print(f"   ESM-2 特征缓存已验证: {metadata['esm2_loader']}")

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
        print(f"   训练样本: {len(train_indices)} (原 {np.sum(train_mask)})")

    # 拟合归一化器
    scaler = TargetScaler()
    y_train = data['y'][train_indices]
    target_names_train = target_names_all[train_indices]
    scaler.fit(y_train, target_names_train)
    scaler.save(SCALER_PATH)

    # 使用 LazySubset + 按需 batch 加载（避免内存溢出）
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

    del splits, target_names_all, train_mask, val_mask, test_mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 创建模型
    model = BaselineMLP(hidden_dim=512)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数: {total_params:,}")

    batch_size = args.batch_size if args.batch_size is not None else 2048
    epochs = args.epochs if args.epochs is not None else 220
    lr = 0.001
    patience = 30

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n开始训练...")
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
        train_perm = np.random.permutation(len(train_data))

        for i in range(0, len(train_data), batch_size):
            bi = train_perm[i:i+batch_size]
            batch = train_data.get_batch(bi)
            morgan = batch['morgan'] if torch.is_tensor(batch['morgan']) else torch.from_numpy(batch['morgan']).to(device)
            maccs = batch['maccs'] if torch.is_tensor(batch['maccs']) else torch.from_numpy(batch['maccs']).to(device)
            descriptors = batch['descriptors'] if torch.is_tensor(batch['descriptors']) else torch.from_numpy(batch['descriptors']).to(device)
            protein = batch['protein'] if torch.is_tensor(batch['protein']) else torch.from_numpy(batch['protein']).to(device)
            y_norm = batch['y_norm'] if torch.is_tensor(batch['y_norm']) else torch.from_numpy(batch['y_norm']).to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    preds, _, _ = model(morgan, maccs, descriptors, protein)
                    loss = F.mse_loss(preds, y_norm)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                preds, _, _ = model(morgan, maccs, descriptors, protein)
                loss = F.mse_loss(preds, y_norm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(bi)

        avg_train_loss = total_loss / len(train_data)
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
                        preds, _, _ = model(morgan, maccs, descriptors, protein)
                        val_loss += F.mse_loss(preds, y_norm).item() * len(bi)
                else:
                    preds, _, _ = model(morgan, maccs, descriptors, protein)
                    val_loss += F.mse_loss(preds, y_norm).item() * len(bi)

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

    # 保存预测值（用于画散点图）
    np.savez(OUTPUT_DIR / 'predictions.npz', y_true=y_true, y_pred=y_pred)
    print(f"   预测值已保存至 {OUTPUT_DIR / 'predictions.npz'}")

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
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
