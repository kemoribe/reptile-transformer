import numpy as np
import pandas as pd
import sys, os, gc, time, json
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *

# GPU 优化配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # 允许 TF32 加速 (Ampere架构)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train(model, device, train_loader, optimizer, epoch, scaler, use_amp=True):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast():
                output = model(data)
                loss = loss_fn(output, data.y.view(-1, 1).float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float())
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        if batch_idx % LOG_INTERVAL == 0:
            avg_loss = total_loss / n_batches
            print('  [Train] Epoch {} | Batch {}/{} ({:.0f}%) | Avg Loss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), avg_loss))


def predicting_gpu(model, device, loader, desc=""):
    """GPU 上累积预测结果"""
    model.eval()
    preds_list = []
    labels_list = []
    n_samples = len(loader.dataset)
    print(f'  [Pred] {desc} {n_samples} samples...')
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device, non_blocking=True)
            with autocast(enabled=USE_AMP):
                output = model(data)
            preds_list.append(output.view(-1))
            labels_list.append(data.y.view(-1))
    preds_gpu = torch.cat(preds_list, dim=0)
    labels_gpu = torch.cat(labels_list, dim=0)
    elapsed = time.time() - t0
    print(f'  [Pred] Done in {elapsed:.1f}s ({n_samples/elapsed:.0f} samples/s)')
    return labels_gpu, preds_gpu


def evaluate_model(model, device, loader, desc=""):
    """完整评估，返回所有指标 (GPU加速)"""
    labels_gpu, preds_gpu = predicting_gpu(model, device, loader, desc)
    t0 = time.time()
    metrics = compute_all_metrics_gpu(labels_gpu, preds_gpu)
    t1 = time.time()
    metrics['samples'] = len(labels_gpu)
    metrics['compute_time'] = t1 - t0
    return metrics


def print_metrics(metrics, prefix=""):
    """格式化打印指标"""
    items = [
        ('RMSE', '.4f'), ('MSE', '.4f'), ('MAE', '.4f'), ('R2', '.4f'),
        ('Pearson', '.4f'), ('Spearman', '.4f'), ('CI', '.4f'),
        ('EF@1%', '.2f'), ('EF@5%', '.2f'), ('EF@10%', '.2f'),
        ('ECE', '.4f'), ('AUPR', '.4f'),
    ]
    lines = []
    for name, fmt in items:
        if name in metrics:
            lines.append(f"{name}={metrics[name]:{fmt}}")
    line = ', '.join(lines)
    time_info = f", compute_time={metrics.get('compute_time', 0):.3f}s" if 'compute_time' in metrics else ""
    print(f'  [{prefix}] {line}{time_info}')


# ==========================
# 命令行参数
# ==========================
DATASET_NAMES = ['chembl', 'davis', 'kiba', 'bindingdb']
MODEL_NAMES = [GINConvNet, GATNet, GAT_GCN, GCNNet]

dataset_idx = int(sys.argv[1])
model_idx = int(sys.argv[2])
dataset_name = DATASET_NAMES[dataset_idx]
modeling = MODEL_NAMES[model_idx]
model_st = modeling.__name__

gpu_id = 0
if len(sys.argv) > 3:
    gpu_id = int(sys.argv[3])

# 早停和续训参数
resume_from = None
has_resume_flag = '--resume' in sys.argv
if has_resume_flag:
    resume_idx = sys.argv.index('--resume')
    if resume_idx + 1 < len(sys.argv) and sys.argv[resume_idx + 1].isdigit():
        resume_from = int(sys.argv[resume_idx + 1])

cuda_name = "cuda:" + str(gpu_id)
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024
LR = 0.0005
LOG_INTERVAL = 50
NUM_EPOCHS = 1000
NUM_WORKERS = 0
EARLY_STOP_PATIENCE = 50
USE_COMPILE = False

USE_AMP = torch.cuda.is_available()

print('=' * 70)
print('Configuration:')
print(f'  Dataset: {dataset_name}')
print(f'  Model: {model_st}')
print(f'  GPU: {cuda_name}')
print(f'  Batch size: {TRAIN_BATCH_SIZE}')
print(f'  Learning rate: {LR}')
print(f'  Max epochs: {NUM_EPOCHS}')
print(f'  Early stop patience: {EARLY_STOP_PATIENCE}')
print(f'  AMP: {USE_AMP}')
print(f'  Torch compile: {USE_COMPILE}')
print('=' * 70)

processed_data_file_train = 'data/processed/' + dataset_name + '_train.pt'
processed_data_file_val = 'data/processed/' + dataset_name + '_val.pt'
processed_data_file_test = 'data/processed/' + dataset_name + '_test.pt'

if not os.path.isfile(processed_data_file_train):
    print('please run create_data.py to prepare data in pytorch format!')
    sys.exit(1)

print('\nLoading data...')
train_data = TestbedDataset(root='data', dataset=dataset_name + '_train')
val_data = TestbedDataset(root='data', dataset=dataset_name + '_val')
test_data = TestbedDataset(root='data', dataset=dataset_name + '_test')
print(f'  Train: {len(train_data)} samples')
print(f'  Val:   {len(val_data)} samples')
print(f'  Test:  {len(test_data)} samples')

use_cuda = torch.cuda.is_available()
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=use_cuda)
val_loader = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=use_cuda)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=use_cuda)

device = torch.device(cuda_name if use_cuda else "cpu")
model = modeling().to(device)

if USE_COMPILE and hasattr(torch, 'compile'):
    print('Using torch.compile for model optimization...')
    model = torch.compile(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
scaler = GradScaler(enabled=USE_AMP)

if use_cuda:
    print(f'GPU Memory: allocated={torch.cuda.memory_allocated(device)/1e9:.2f}GB, reserved={torch.cuda.memory_reserved(device)/1e9:.2f}GB')

best_val_mse = float('inf')
best_epoch = 0
patience_counter = 0
start_epoch = 0

model_file_name = 'model_' + model_st + '_' + dataset_name + '.model'
checkpoint_file = 'checkpoint_' + model_st + '_' + dataset_name + '.pt'
result_file_name = 'result_' + model_st + '_' + dataset_name + '.json'

# 断点续训
if (has_resume_flag or resume_from is not None) and os.path.exists(checkpoint_file):
    print(f'\n[Resume] Loading checkpoint...')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_mse = checkpoint['best_val_mse']
    best_epoch = checkpoint['best_epoch']
    patience_counter = checkpoint['patience_counter']
    if resume_from is not None and resume_from < start_epoch:
        start_epoch = resume_from
    print(f'  Resuming from epoch {start_epoch}, best_val_mse={best_val_mse:.6f}')

print('\n' + '=' * 70)
print(f'Starting training: {dataset_name} + {model_st}')
print('=' * 70)

train_start_time = time.time()
history = []

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_start = time.time()
    
    train(model, device, train_loader, optimizer, epoch + 1, scaler, use_amp=USE_AMP)
    epoch_time = time.time() - epoch_start
    
    # 验证
    val_metrics = evaluate_model(model, device, val_loader, f"Val Epoch {epoch+1}")
    val_mse = val_metrics['MSE']
    val_metrics['epoch_time'] = epoch_time
    history.append(val_metrics)
    
    print_metrics(val_metrics, f"Val Epoch {epoch+1} ({epoch_time:.1f}s)")
    
    # 学习率调度
    scheduler.step(val_mse)
    current_lr = optimizer.param_groups[0]['lr']
    
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_epoch = epoch + 1
        patience_counter = 0
        
        # 保存最佳模型
        torch.save(model.state_dict(), model_file_name)
        
        # 评估测试集
        test_metrics = evaluate_model(model, device, test_loader, f"Test (best epoch {best_epoch})")
        test_metrics['best_epoch'] = best_epoch
        test_metrics['best_val_mse'] = best_val_mse
        print_metrics(test_metrics, f"Test (epoch {best_epoch})")
        
        # 保存完整结果
        with open(result_file_name, 'w') as f:
            json.dump({
                'dataset': dataset_name,
                'model': model_st,
                'best_epoch': best_epoch,
                'best_val_mse': best_val_mse,
                'test_metrics': test_metrics,
                'history': history[-10:]  # 最近10轮历史
            }, f, indent=2)
        print(f'  [Save] Best model saved to {model_file_name}')
    else:
        patience_counter += 1
        print(f'  [No improve] Patience: {patience_counter}/{EARLY_STOP_PATIENCE} (best_epoch={best_epoch}, best_val_mse={best_val_mse:.6f})')
    
    # 保存断点
    if (epoch + 1) % 10 == 0 or val_mse < best_val_mse:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_mse': best_val_mse,
            'best_epoch': best_epoch,
            'patience_counter': patience_counter,
        }, checkpoint_file)
    
    # 早停检查
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f'\n[Early Stop] No improvement for {EARLY_STOP_PATIENCE} epochs. Stopping at epoch {epoch+1}.')
        break
    
    # 清理 GPU 缓存
    if use_cuda and (epoch + 1) % 5 == 0:
        torch.cuda.empty_cache()

total_time = time.time() - train_start_time

# 最终评估
print('\n' + '=' * 70)
print('Final Evaluation (best model)')
print('=' * 70)

if os.path.exists(model_file_name):
    model.load_state_dict(torch.load(model_file_name, map_location=device))

final_test_metrics = evaluate_model(model, device, test_loader, "Final Test")
print_metrics(final_test_metrics, "Final Test")

# 保存预测值（用于校准曲线）
final_labels, final_preds = predicting_gpu(model, device, test_loader, "Save predictions")
np.savez(f'predictions_{model_st}_{dataset_name}.npz',
         y_true=final_labels.cpu().numpy(), y_pred=final_preds.cpu().numpy())
print(f'  [Save] Predictions saved to predictions_{model_st}_{dataset_name}.npz')

print('\n' + '=' * 70)
print('Training Complete!')
print(f'  Dataset: {dataset_name}')
print(f'  Model: {model_st}')
print(f'  Best epoch: {best_epoch}')
print(f'  Best val MSE: {best_val_mse:.6f}')
print(f'  Total time: {total_time:.1f}s ({total_time/60:.1f}min)')
print(f'  Model: {model_file_name}')
print(f'  Results: {result_file_name}')
print(f'  Checkpoint: {checkpoint_file}')
print('=' * 70)