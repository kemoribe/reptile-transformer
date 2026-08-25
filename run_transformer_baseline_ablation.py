"""
Transformer基线 - 特征消融实验

固定纯Transformer网络不变，仅修改输入特征组合，对比四组方案：
1. morgan_only:        仅Morgan+ESM2   （单特征，Morgan代表SMILES衍生特征）
2. morgan_maccs:       Morgan+MACCS+ESM2（双指纹）
3. morgan_descriptors: Morgan+理化+ESM2  （指纹+理化描述符）
4. full:               Morgan+MACCS+理化+ESM2（全套原始方案）

默认数据集：data/processed/chembl（可用 --data_dir 切换）

运行方式：
    # 运行单个变体
    python run_transformer_baseline_ablation.py --variant morgan_only
    # 运行全部四个变体
    python run_transformer_baseline_ablation.py --all
    # 从断点恢复
    python run_transformer_baseline_ablation.py --all --resume
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
from datetime import datetime

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

BASE_DIR = Path(__file__).resolve().parent

# 在导入 data_preprocessing 前设置默认数据目录
DEFAULT_DATA_DIR = str(BASE_DIR / "data" / "processed" / "chembl")
os.environ.setdefault('PREPROCESSED_DIR', DEFAULT_DATA_DIR)

sys.path.insert(0, str(BASE_DIR))

from data_preprocessing import (
    MoleculeFeatureExtractor,
    ProteinFeatureExtractor,
    get_all_targets,
    precompute_features,
    clear_precomputed_features,
    validate_precomputed_features,
    TargetScaler,
    LazySubset
)
from reptile_transformer_model import ReptileTransformer, count_params, init_weights

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
SHARED_OUTPUT_DIR = BASE_DIR / "transformer_ablation_output"
SHARED_FEATURES = SHARED_OUTPUT_DIR / "precomputed_features.npz"
SHARED_SCALER = SHARED_OUTPUT_DIR / "target_scaler.json"


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    from scipy import stats

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

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

    def compute_ece(n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (y_pred >= lower * (y_true.max() - y_true.min()) + y_true.min()) & \
                   (y_pred < upper * (y_true.max() - y_true.min()) + y_true.min())
            if np.sum(mask) > 0:
                ece += np.abs(np.mean(y_pred[mask]) - np.mean(y_true[mask])) * np.sum(mask) / len(y_true)
        return ece

    if n_total >= 5:
        n_pos = max(1, int(n_total * 0.2))
        sorted_true_desc = np.sort(y_true)[::-1]
        threshold = sorted_true_desc[min(n_pos - 1, len(sorted_true_desc) - 1)]
        is_positive = (y_true >= threshold).astype(int)
        n_total_positive = np.sum(is_positive)
        if n_total_positive > 0:
            si = np.argsort(y_pred)[::-1]
            sp = is_positive[si]
            tp = np.cumsum(sp)
            fp = np.cumsum(1 - sp)
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
        'EF@1%': compute_ef(1), 'EF@5%': compute_ef(5), 'EF@10%': compute_ef(10),
        'ECE': compute_ece(), 'AUPR': aupr_val
    }


def get_variant_features(batch, cfg, device):
    """根据变体配置返回特征，不用的特征置零（网络完全不变）"""
    morgan = batch['morgan']
    maccs = batch['maccs']
    descriptors = batch['descriptors']
    protein = batch['protein']

    if not torch.is_tensor(morgan):
        morgan = torch.from_numpy(morgan).to(device)
        maccs = torch.from_numpy(maccs).to(device)
        descriptors = torch.from_numpy(descriptors).to(device)
        protein = torch.from_numpy(protein).to(device)
    else:
        morgan = morgan.to(device)
        maccs = maccs.to(device)
        descriptors = descriptors.to(device)
        protein = protein.to(device)

    # 不用的特征置零
    if not cfg['use_maccs']:
        maccs = torch.zeros_like(maccs)
    if not cfg['use_descriptors']:
        descriptors = torch.zeros_like(descriptors)

    return morgan, maccs, descriptors, protein


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Transformer基线 - 特征消融实验")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help="预处理数据目录")
    parser.add_argument('--variant', type=str, default=None,
                        choices=list(VARIANTS.keys()),
                        help="运行单个变体: morgan_only / morgan_maccs / morgan_descriptors / full")
    parser.add_argument('--all', action='store_true',
                        help="运行全部四个变体")
    parser.add_argument('--resume', action='store_true', help="从断点恢复训练")
    parser.add_argument('--force', action='store_true', help="强制重新训练")
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
    parser.add_argument('--epochs', type=int, default=300, help="训练轮数")
    parser.add_argument('--list', action='store_true', help="列出变体状态")
    parser.add_argument('--max_samples_per_target', type=int, default=None,
                        help="少样本实验: 每个靶点最多使用N个训练样本 (None=全部)")
    return parser.parse_args()


def train_one_variant(variant_name, args, shared_data, splits_info):
    """训练单个变体"""
    cfg = VARIANTS[variant_name]
    output_dir = SHARED_OUTPUT_DIR / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / 'checkpoint.pt'
    final_results_path = output_dir / 'final_results.json'
    log_path = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # 日志文件
    log_file = open(log_path, 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log("\n" + "=" * 70)
    log(f"变体: {variant_name} - {cfg['desc']}")
    log(f"特征: morgan={cfg['use_morgan']}, maccs={cfg['use_maccs']}, descriptors={cfg['use_descriptors']}")
    log(f"输出: {output_dir}")
    log("=" * 70)

    # 已完成检查
    if final_results_path.exists() and not args.force and not args.resume:
        log("✅ 已完成，跳过（用 --force 强制重跑）")
        log_file.close()
        return True

    # --force 清除断点
    if args.force and checkpoint_path.exists():
        checkpoint_path.unlink()
        log(f"🗑 已清除旧断点: {checkpoint_path}")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    scaler_amp = torch.amp.GradScaler('cuda') if use_amp else None

    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        log(f"   GPU: {props.name}, {props.total_memory / 1e9:.2f}GB")
    log(f"   Device: {device}, AMP: {use_amp}")

    # 解包共享数据
    data = shared_data['data']
    train_indices = splits_info['train']
    val_indices = splits_info['val']
    test_indices = splits_info['test']
    scaler = shared_data['scaler']

    # 预加载到GPU
    log("   预加载数据到GPU...")
    train_data = LazySubset(data, train_indices, scaler)
    val_data = LazySubset(data, val_indices, scaler)
    test_data = LazySubset(data, test_indices, scaler)

    if torch.cuda.is_available():
        def preload_to_gpu(loader):
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

        def gpu_get_batch(self, batch_indices):
            idx = torch.from_numpy(np.asarray(batch_indices)).long()
            return {
                'morgan': self._gpu_data['morgan'][idx],
                'maccs': self._gpu_data['maccs'][idx],
                'descriptors': self._gpu_data['descriptors'][idx],
                'protein': self._gpu_data['protein'][idx],
                'y_norm': self._gpu_data['y_norm'][idx],
                'y': self._gpu_data['y'][idx],
            }

        import types
        for loader in [train_data, val_data, test_data]:
            preload_to_gpu(loader)
            loader.get_batch = types.MethodType(gpu_get_batch, loader)

    gc.collect()

    # 创建模型（网络完全不变）
    model = ReptileTransformer()
    model.apply(init_weights)
    model.to(device)
    total_params, trainable_params = count_params(model)
    log(f"   Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # 训练配置
    batch_size = args.batch_size
    epochs = args.epochs
    lr = 0.00015
    patience = 50
    warmup_epochs = 30

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

    contrastive_weight = 0.03
    consistency_weight = 0.05
    ranking_weight = 0.5
    calibration_weight = 0.1

    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0

    # 断点恢复
    if args.resume and checkpoint_path.exists():
        log(f"💾 加载断点: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        patience_counter = ckpt['patience_counter']
        log(f"   从 epoch {start_epoch} 恢复, best_val_loss={best_val_loss:.4f}")

    log(f"\n开始训练 [{variant_name}] ...")
    total_batches = (len(train_data) + batch_size - 1) // batch_size

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr * warmup_factor
        elif epoch == warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        train_perm = np.random.permutation(len(train_data))

        for i in range(0, len(train_data), batch_size):
            bi = train_perm[i:i + batch_size]
            batch = train_data.get_batch(bi)
            morgan, maccs, descriptors, protein = get_variant_features(batch, cfg, device)
            y_norm = batch['y_norm']
            if not torch.is_tensor(y_norm):
                y_norm = torch.from_numpy(y_norm).to(device)
            else:
                y_norm = y_norm.to(device)

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
                bi = np.arange(i, min(i + batch_size, len(val_data)))
                batch = val_data.get_batch(bi)
                morgan, maccs, descriptors, protein = get_variant_features(batch, cfg, device)
                y_norm = batch['y_norm']
                if not torch.is_tensor(y_norm):
                    y_norm = torch.from_numpy(y_norm).to(device)
                else:
                    y_norm = y_norm.to(device)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        preds, cl, ctl = model(morgan, maccs, descriptors, protein)
                        vloss = F.mse_loss(preds, y_norm)
                        if contrastive_weight > 0:
                            vloss += contrastive_weight * ctl
                        if consistency_weight > 0:
                            vloss += consistency_weight * cl
                        val_loss += vloss.item() * len(bi)
                else:
                    preds, cl, ctl = model(morgan, maccs, descriptors, protein)
                    vloss = F.mse_loss(preds, y_norm)
                    if contrastive_weight > 0:
                        vloss += contrastive_weight * ctl
                    if consistency_weight > 0:
                        vloss += consistency_weight * cl
                    val_loss += vloss.item() * len(bi)

        avg_val_loss = val_loss / len(val_data)

        if (epoch + 1) % 2 == 0:
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, GPU: {gpu_mem:.2f}GB")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
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
            log(f"早停: {patience}个epoch无改善")
            break

    # 测试评估
    log("\n" + "=" * 70)
    log("测试集评估")
    log("=" * 70)

    model.load_state_dict(torch.load(output_dir / 'best_model.pt', map_location=device))
    model.eval()

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            bi = np.arange(i, min(i + batch_size, len(test_data)))
            batch = test_data.get_batch(bi)
            morgan, maccs, descriptors, protein = get_variant_features(batch, cfg, device)
            y_raw = batch['y']
            if not torch.is_tensor(y_raw):
                y_raw = torch.from_numpy(y_raw).to(device)
            else:
                y_raw = y_raw.to(device)

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
    np.savez(output_dir / 'predictions.npz', y_true=y_true, y_pred=y_pred)
    log(f"   预测值已保存至 {output_dir / 'predictions.npz'}")

    metrics = compute_metrics(y_true, y_pred)
    metrics['variant'] = variant_name
    metrics['variant_desc'] = cfg['desc']
    metrics['features'] = {
        'morgan': cfg['use_morgan'],
        'maccs': cfg['use_maccs'],
        'descriptors': cfg['use_descriptors'],
        'protein_esm2': True,
    }
    metrics['samples'] = len(y_true)

    log(f"   R2: {metrics['R2']:.4f}")
    log(f"   RMSE: {metrics['RMSE']:.4f}")
    log(f"   MAE: {metrics['MAE']:.4f}")
    log(f"   Pearson: {metrics['Pearson']:.4f}")
    log(f"   Spearman: {metrics['Spearman']:.4f}")
    log(f"   EF@1%: {metrics['EF@1%']:.2f}")
    log(f"   EF@5%: {metrics['EF@5%']:.2f}")
    log(f"   EF@10%: {metrics['EF@10%']:.2f}")
    log(f"   ECE: {metrics['ECE']:.4f}")
    log(f"   AUPR: {metrics['AUPR']:.4f}")
    log(f"   Samples: {metrics['samples']}")

    metrics_serializable = {k: (float(v) if isinstance(v, np.floating) else v)
                           for k, v in metrics.items()}
    with open(final_results_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)

    log(f"\n✅ 完成! 输出: {output_dir}")
    log_file.close()

    # 清理显存
    del model
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
    SHARED_OUTPUT_DIR = BASE_DIR / f"transformer_ablation_output_{suffix}{fewshot_tag}"
    SHARED_FEATURES = SHARED_OUTPUT_DIR / "precomputed_features.npz"
    SHARED_SCALER = SHARED_OUTPUT_DIR / "target_scaler.json"

    # --list 模式
    if args.list:
        print("\n变体状态:")
        for name, cfg in VARIANTS.items():
            out_dir = SHARED_OUTPUT_DIR / name
            done = (out_dir / 'final_results.json').exists()
            ckpt = (out_dir / 'checkpoint.pt').exists()
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
            for name in ('checkpoint.pt', 'best_model.pt', 'predictions.npz',
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
    print("Transformer基线 - 特征消融实验")
    print("=" * 70)
    print(f"数据目录: {args.data_dir}")
    print(f"变体: {variants_to_run}")
    print(f"参数: epochs={args.epochs}, batch_size={args.batch_size}, gpu={args.gpu}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 预计算特征（共享，只算一次）
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
        print(f"\n使用已验证的特征: {metadata['esm2_loader']}")

    # 加载共享数据
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

    scaler = TargetScaler()
    y_train = data['y'][train_indices]
    target_names_train = target_names_all[train_indices]
    scaler.fit(y_train, target_names_train)
    scaler.save(SHARED_SCALER)

    splits_info = {'train': train_indices, 'val': val_indices, 'test': test_indices}
    shared_data = {'data': data, 'scaler': scaler}

    print(f"   Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # 依次运行变体
    results = {}
    total_start = time.time()

    for variant_name in variants_to_run:
        print(f"\n{'#' * 70}")
        print(f"# 变体: {variant_name}")
        print(f"{'#' * 70}")

        try:
            success = train_one_variant(variant_name, args, shared_data, splits_info)
            results[variant_name] = success
        except Exception as e:
            print(f"\n❌ {variant_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            results[variant_name] = False

    # 总结
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("消融实验总结")
    print("=" * 70)

    for variant_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {variant_name} - {VARIANTS[variant_name]['desc']}")

    # 打印对比表格（补齐所有10个指标）
    print(f"\n{'='*140}")
    print(f"{'Variant':<16} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'Pearson':>8} {'Spearman':>9} "
          f"{'EF@1%':>6} {'EF@5%':>6} {'EF@10%':>7} {'ECE':>8} {'AUPR':>8}")
    print("-" * 140)
    for variant_name in variants_to_run:
        result_file = SHARED_OUTPUT_DIR / variant_name / 'final_results.json'
        if result_file.exists():
            with open(result_file) as f:
                m = json.load(f)
            print(f"{variant_name:<16} {m.get('R2',0):>8.4f} {m.get('RMSE',0):>8.4f} "
                  f"{m.get('MAE',0):>8.4f} {m.get('Pearson',0):>8.4f} {m.get('Spearman',0):>9.4f} "
                  f"{m.get('EF@1%',0):>6.2f} {m.get('EF@5%',0):>6.2f} {m.get('EF@10%',0):>7.2f} "
                  f"{m.get('ECE',0):>8.4f} {m.get('AUPR',0):>8.4f}")
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
