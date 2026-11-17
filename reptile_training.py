"""
Reptile元学习训练循环 - 增强版

核心算法：
1. 内循环（Inner Loop）：在单个靶点上进行快速适应
2. 外循环（Outer Loop）：跨靶点泛化，更新元参数

增强特性：
- 完整日志系统（logging模块）
- 断点保存与恢复
- Pearson/Spearman相关系数指标
- GPU最大化利用：混合精度、梯度累积、批量处理
- 余弦退火学习率调度器
- 梯度裁剪防止梯度爆炸
- 对比学习增强表示学习
- 一致性损失保证双向注意力一致性
- 早停机制防止过拟合
"""

import os
import gc
import json
import time
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

# CUDA优化设置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ==========================
# 日志配置
# ==========================
def setup_logging(output_dir):
    """配置日志系统"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('reptile_trainer')
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 文件handler（详细日志）
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台handler（简洁日志）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ==========================
# 训练配置
# ==========================
class TrainingConfig:
    """训练配置 - 优化GPU利用率和模型性能"""
    
    def __init__(self):
        # 超参数 - 最佳配置（R²=0.2229）
        self.INNER_LR = 0.01  # 稳定的内循环学习率
        self.INNER_STEPS = 3  # 最优内循环步数
        self.META_LR = 0.001  # 最佳元学习率
        self.EPOCHS = 300  # 增加训练轮数，给模型更多学习时间
        self.META_WARMUP_EPOCHS = 5  # warmup长度
        self.PATIENCE = 40  # 增加耐心值
        
        # 损失权重（平衡排序和回归）
        # 支持环境变量覆盖（供 λ 敏感性分析 supplement_s9 使用，默认值即论文设定）
        self.CONTRASTIVE_WEIGHT = float(os.environ.get('REPTILE_LAMBDA1', 0.05))  # 对比学习权重
        self.CONSISTENCY_WEIGHT = float(os.environ.get('REPTILE_LAMBDA2', 0.1))  # 一致性损失权重
        self.RANKING_WEIGHT = 0.3  # 排序损失权重（平衡排序和回归）
        
        # 批处理 - 最大化GPU利用率
        self.BATCH_SIZE = 512
        self.GRADIENT_ACCUMULATION_STEPS = 1
        
        # 早停
        self.EARLY_STOP_PATIENCE = 25
        self.EARLY_STOP_MIN_DELTA = 0.001
        
        # 日志
        self.LOG_INTERVAL = 1
        self.VAL_INTERVAL = 10
        
        # GPU优化
        self.USE_COMPILE = False
        self.COMPILE_MODE = 'reduce-overhead'
        self.MIXED_PRECISION = True
        # 外循环 delta 在线累加：峰值显存 O(参数量) 而非 O(任务数×参数量)
        # 与离线 stack→mean 数学等价；靶点数多（>300）时必须开启
        self.ONLINE_DELTA_ACCUM = False
        # 在线累加的分块大小：每 N 个靶点做一次向量化 stack/sum，避免逐靶点 add_ 开销
        self.ONLINE_ACCUM_CHUNK = 32
        
        # 断点保存
        self.CHECKPOINT_INTERVAL = 10
        self.SAVE_BEST_ONLY = True
        
        # 梯度裁剪
        self.GRADIENT_CLIP = 5.0
        
        # 学习率调度
        self.LR_SCHEDULER = 'cosine'
        
        # 权重衰减
        self.WEIGHT_DECAY = 1e-5
    
    def save(self, path):
        """保存配置"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path):
        """加载配置"""
        config = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        config.__dict__.update(data)
        return config


# ==========================
# 自定义学习率调度器（带warmup）
# ==========================
class WarmupCosineScheduler:
    """带warmup的余弦退火调度器"""
    
    def __init__(self, optimizer, max_epochs, warmup_epochs=5, eta_min=1e-6):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch):
        """更新学习率"""
        if epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = base_lr * warmup_factor
        else:
            # Cosine退火阶段
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = self.eta_min + (base_lr - self.eta_min) * cosine_decay
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


# ==========================
# Reptile训练器
# ==========================
class ReptileTrainer:
    """Reptile元学习训练器 - 增强版"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, target_scaler,
                 output_dir='./reptile_output', config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.target_scaler = target_scaler
        self.output_dir = output_dir
        self.config = config or TrainingConfig()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 配置日志
        self.logger = setup_logging(output_dir)

        # 确保模型在GPU上（关键修复：避免模型在CPU而数据在GPU）
        if torch.cuda.is_available() and not next(self.model.parameters()).is_cuda:
            self.model = self.model.to(device)
            self.logger.info(f"✅ 模型已移至GPU: {device}")

        # GPU优化：编译模型
        if self.config.USE_COMPILE and torch.cuda.is_available():
            self.logger.info("🔧 编译模型以加速GPU计算...")
            self.model = torch.compile(self.model, mode=self.config.COMPILE_MODE)
        
        # 元优化器（使用权重衰减）
        self.meta_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.META_LR,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # 学习率调度器（带warmup）
        self.scheduler = WarmupCosineScheduler(
            self.meta_optimizer,
            max_epochs=self.config.EPOCHS,
            warmup_epochs=self.config.META_WARMUP_EPOCHS,
            eta_min=1e-6
        )
        
        # 记录
        self.best_val_r2 = -float('inf')
        self.best_val_rmse = float('inf')
        self.best_val_pearson = -float('inf')
        self.training_history = []
        self.patience_counter = 0
        self.current_epoch = 0
        self.epochs_run = 0
        self.best_epoch = None
        # 训练预算记录（用于 6.1 预算匹配对照）
        self.total_inner_updates = 0
        self.total_meta_updates = 0
        self.total_samples_seen = 0
        
        # 混合精度训练
        if torch.cuda.is_available() and self.config.MIXED_PRECISION:
            self.scaler = torch.amp.GradScaler('cuda')
            self.autocast = torch.amp.autocast('cuda')
            self.logger.info(f"✅ 混合精度训练已启用 (GPU: {torch.cuda.get_device_name(0)})")
            self.logger.info(f"   GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.scaler = None
            self.autocast = torch.no_grad()
            self.logger.warning("⚠️ 未检测到GPU或禁用混合精度，使用CPU训练")
        
        # 预创建内循环优化器（避免重复创建）
        self.inner_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.INNER_LR
        )

        # ===== 梯度冲突诊断（E3，默认关闭；REPTILE_GRAD_DIAG=1 开启）=====
        # 对 MSE / ListNet排序 / 对比 / 一致性 四个损失分别求梯度，记录两两余弦
        self.grad_diag_on = os.environ.get('REPTILE_GRAD_DIAG', '0') == '1'
        self.grad_diag_epochs = {
            int(x) for x in os.environ.get('REPTILE_GRAD_DIAG_EPOCHS', '1,2,3,10,20').split(',') if x.strip()
        }
        self.grad_diag_max_tasks = int(os.environ.get('REPTILE_GRAD_DIAG_TASKS', '8'))
        self.grad_diag_records = []
        self._diag_epoch = 0
        self._diag_task_idx = 0

    def _record_grad_conflict(self, l_mse, l_rank, l_con, l_cons):
        """E3：四个损失分别对可训练参数求梯度，计算两两余弦相似度（不影响正常反传）"""
        import itertools
        params = [p for p in self.model.parameters() if p.requires_grad]
        named_losses = [('MSE', l_mse), ('Rank', l_rank), ('Contrast', l_con), ('Consist', l_cons)]
        vecs = {}
        for name, loss in named_losses:
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            # 未被该损失使用的参数补零，保证四个向量等长（标准做法：在全参数空间比较方向）
            flat = torch.cat([
                (g.detach() if g is not None else torch.zeros_like(p)).reshape(-1).float()
                for g, p in zip(grads, params)
            ])
            vecs[name] = flat
        rec = {'epoch': self._diag_epoch + 1, 'task_idx': self._diag_task_idx}
        for name, loss in named_losses:
            rec[f'loss_{name}'] = float(loss.detach().item())
        for (na, _), (nb, _) in itertools.combinations(named_losses, 2):
            va, vb = vecs[na], vecs[nb]
            if va.numel() == 0 or vb.numel() == 0:
                cos = float('nan')
            else:
                cos = float(F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item())
            rec[f'cos_{na}_{nb}'] = cos
        self.grad_diag_records.append(rec)

    def _dump_grad_conflict(self):
        """训练结束写出梯度冲突诊断（JSON）"""
        if not (self.grad_diag_on and self.grad_diag_records):
            return
        import csv as _csv
        out_json = os.path.join(self.output_dir, 'grad_conflict.json')
        with open(out_json, 'w') as f:
            json.dump(self.grad_diag_records, f, indent=2)
        keys = list(self.grad_diag_records[0].keys())
        out_csv = os.path.join(self.output_dir, 'grad_conflict.csv')
        with open(out_csv, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.grad_diag_records)
        self.logger.info(f"📈 梯度冲突诊断已保存: {out_json} / {out_csv} ({len(self.grad_diag_records)} 条)")

    def _save_checkpoint(self, epoch, val_metrics, is_best=False):
        """保存断点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': {k: v for k, v in self.model.state_dict().items() 
                                if 'protein_encoder' not in k},
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'scheduler_state_dict': {
                'base_lrs': self.scheduler.base_lrs,
                'current_epoch': epoch
            },
            'best_val_r2': self.best_val_r2,
            'best_val_rmse': self.best_val_rmse,
            'best_val_pearson': self.best_val_pearson,
            'patience_counter': self.patience_counter,
            'training_history': self.training_history,
            'val_metrics': val_metrics,
            'config': self.config.__dict__,
            'timestamp': time.time()
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.output_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"📦 保存最佳检查点: epoch={epoch}, R2={val_metrics['R2']:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % self.config.CHECKPOINT_INTERVAL == 0:
            periodic_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, periodic_path)
            self.logger.info(f"📦 保存定期检查点: {periodic_path}")
    
    def _load_checkpoint(self, checkpoint_path=None):
        """加载断点"""
        if checkpoint_path is None:
            # 优先加载最新检查点
            checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pt')
            if not os.path.exists(checkpoint_path):
                # 尝试加载最佳检查点
                checkpoint_path = os.path.join(self.output_dir, 'checkpoint_best.pt')
        
        if not os.path.exists(checkpoint_path):
            self.logger.warning("⚠️ 未找到检查点文件，从头开始训练")
            return False
        
        self.logger.info(f"🔄 加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 加载优化器状态
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复调度器状态
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.base_lrs = checkpoint['scheduler_state_dict']['base_lrs']
        
        # 恢复训练状态
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
        self.best_val_rmse = checkpoint.get('best_val_rmse', float('inf'))
        self.best_val_pearson = checkpoint.get('best_val_pearson', -float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.training_history = checkpoint.get('training_history', [])
        
        self.logger.info(f"✅ 检查点加载成功，从epoch {self.current_epoch} 继续训练")
        self.logger.info(f"   最佳验证R2: {self.best_val_r2:.4f}")
        return True
    
    def _compute_metrics(self, y_true, y_pred):
        """计算评估指标（包括EF富集因子和校准度）"""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R²
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # 总平方和
        ss_res = np.sum((y_true - y_pred) ** 2)  # 残差平方和
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # Pearson相关系数
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        
        # Spearman相关系数
        spearman_r, _ = stats.spearmanr(y_true, y_pred)
        
        # EF富集因子（Enrichment Factor）- 虚拟筛选关键指标
        # 使用相对活性阈值：每个靶点的top 20%作为活性化合物（虚拟筛选标准做法）
        n_total = len(y_true)
        
        # 计算活性阈值（取top 20%）
        if n_total >= 10:  # 至少需要10个样本才有意义
            active_count = max(5, int(n_total * 0.2))  # 至少5个活性化合物，最多20%
            sorted_true = np.sort(y_true)[::-1]  # 降序排列真实值
            active_threshold = sorted_true[min(active_count - 1, len(sorted_true) - 1)]
            n_active = active_count
        else:
            active_threshold = float('inf')
            n_active = 0
        
        # 按预测值排序，取top百分比
        sorted_indices = np.argsort(y_pred)[::-1]  # 降序排列（预测值越高越好）
        y_true_sorted = y_true[sorted_indices]
        
        def compute_ef(top_percent):
            n_top = int(n_total * top_percent / 100)
            if n_top == 0 or n_active == 0:
                return 0.0
            n_active_top = np.sum(y_true_sorted[:n_top] >= active_threshold)
            # EF = (活性化合物在top%中的比例) / (活性化合物在总体中的比例)
            ef = (n_active_top / n_top) / (n_active / n_total)
            return ef
        
        ef1 = compute_ef(1)
        ef5 = compute_ef(5)
        ef10 = compute_ef(10)
        
        # 校准度指标（Expected Calibration Error）
        # 将预测值和真实值分为多个bin，计算每个bin的校准误差
        def compute_ece(n_bins=10):
            # 归一化预测值和真实值到[0,1]区间
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            range_val = max_val - min_val
            if range_val == 0:
                return 0.0
            
            y_true_norm = (y_true - min_val) / range_val
            y_pred_norm = (y_pred - min_val) / range_val
            
            # 按预测值分bin
            bin_indices = np.digitize(y_pred_norm, np.linspace(0, 1, n_bins + 1)[1:-1])
            
            ece = 0.0
            n_samples = len(y_true_norm)
            
            for bin_idx in range(n_bins):
                mask = bin_indices == bin_idx
                if np.sum(mask) == 0:
                    continue
                
                bin_pred = np.mean(y_pred_norm[mask])
                bin_true = np.mean(y_true_norm[mask])
                bin_weight = np.sum(mask) / n_samples
                
                ece += bin_weight * np.abs(bin_pred - bin_true)
            
            return ece
        
        ece = compute_ece()
        
        # AUPR (Area Under Precision-Recall curve)
        # 定义top 20%真实值为正样本，按预测值排序计算PR曲线下面积
        def compute_aupr():
            n = len(y_true)
            if n < 5:
                return 0.0
            n_pos = max(1, int(n * 0.2))
            sorted_true_desc = np.sort(y_true)[::-1]
            threshold = sorted_true_desc[min(n_pos - 1, len(sorted_true_desc) - 1)]
            is_positive = (y_true >= threshold).astype(int)
            n_total_positive = np.sum(is_positive)
            if n_total_positive == 0:
                return 0.0
            sorted_indices = np.argsort(y_pred)[::-1]
            sorted_positives = is_positive[sorted_indices]
            tp = np.cumsum(sorted_positives)
            fp = np.cumsum(1 - sorted_positives)
            precisions = tp / (tp + fp + 1e-10)
            recalls = tp / n_total_positive
            recalls_prev = np.zeros_like(recalls)
            recalls_prev[1:] = recalls[:-1]
            aupr = np.sum((recalls - recalls_prev) * precisions)
            return aupr
        
        aupr = compute_aupr()
        
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
            'AUPR': aupr
        }
    
    def _inner_loop(self, morgan, maccs, descriptors, protein, y_norm, restore=True):
        """
        内循环：在单个靶点上进行快速适应（GPU优化）

        Args:
            ...
            restore: True=Reptile（内循环后恢复初始参数）；False=direct_supervised
                     （保留适应后的参数，直接作为监督更新）
        """
        # 保存初始参数（仅保存一次；direct 模式不需要恢复）
        if restore:
            initial_params = {name: param.data.clone() for name, param in self.model.named_parameters()
                              if param.requires_grad}
        
        # 任务自适应内循环学习率（创新点）
        # 数据量多的靶点：小学习率（已有足够信息）
        # 数据量少的靶点：大学习率（需要快速适应）
        n_samples = morgan.shape[0]
        base_lr = self.config.INNER_LR
        # 根据样本数量调整学习率（log缩放）
        if n_samples < 100:
            adaptive_lr = base_lr * 2.0  # 少量样本用大学习率
        elif n_samples < 500:
            adaptive_lr = base_lr * 1.5  # 中等样本用中等学习率
        elif n_samples > 2000:
            adaptive_lr = base_lr * 0.5  # 大量样本用小学习率
        else:
            adaptive_lr = base_lr  # 默认学习率
        
        # 重置内循环优化器（设置自适应学习率）
        for param_group in self.inner_optimizer.param_groups:
            param_group['lr'] = adaptive_lr
        
        # 内循环步数
        losses = []
        
        for step in range(self.config.INNER_STEPS):
            # 随机采样批次（大batch最大化GPU利用率）
            if n_samples > self.config.BATCH_SIZE:
                indices = torch.randperm(n_samples, device=morgan.device)[:self.config.BATCH_SIZE]
                batch_morgan = morgan[indices]
                batch_maccs = maccs[indices]
                batch_descriptors = descriptors[indices]
                batch_protein = protein[indices]
                batch_y = y_norm[indices]
            else:
                batch_morgan = morgan
                batch_maccs = maccs
                batch_descriptors = descriptors
                batch_protein = protein
                batch_y = y_norm
            
            # 前向传播（混合精度）
            with self.autocast:
                preds, consistency_loss, contrastive_loss = self.model(
                    batch_morgan, batch_maccs, batch_descriptors, batch_protein
                )
                
                # 计算损失（标准MSE损失）
                mse_loss = F.mse_loss(preds, batch_y)
                total_loss = mse_loss
                
                # ListNet排序损失（提升EF和Spearman，比成对损失更强）
                if self.config.RANKING_WEIGHT > 0 and batch_y.shape[0] >= 2:
                    n = batch_y.shape[0]
                    
                    # ListNet损失：对预测值和真实值分别计算softmax，然后用交叉熵
                    # 预测分布：softmax(preds)
                    # 目标分布：softmax(y)（真实值越高，概率越大）
                    
                    pred_probs = F.softmax(preds, dim=0)
                    target_probs = F.softmax(batch_y, dim=0)
                    
                    # 避免零概率导致log(0)
                    pred_probs = torch.clamp(pred_probs, min=1e-10)
                    target_probs = torch.clamp(target_probs, min=1e-10)
                    
                    # KL散度损失（交叉熵的一种形式）
                    ranking_loss = F.kl_div(torch.log(pred_probs), target_probs, reduction='batchmean')
                    total_loss += self.config.RANKING_WEIGHT * ranking_loss
                
                if self.config.CONTRASTIVE_WEIGHT > 0:
                    total_loss += self.config.CONTRASTIVE_WEIGHT * contrastive_loss
                
                if self.config.CONSISTENCY_WEIGHT > 0:
                    total_loss += self.config.CONSISTENCY_WEIGHT * consistency_loss

                # E3 梯度冲突诊断（指定 epoch 的前 N 个任务、每个任务第 0 个内步采样一次）
                if (self.grad_diag_on and step == 0
                        and (self._diag_epoch + 1) in self.grad_diag_epochs
                        and self._diag_task_idx < self.grad_diag_max_tasks
                        and batch_y.shape[0] >= 2 and 'ranking_loss' in locals()):
                    self._record_grad_conflict(mse_loss, ranking_loss,
                                               contrastive_loss, consistency_loss)

            # 反向传播（混合精度）
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.inner_optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.inner_optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.inner_optimizer.step()
            
            self.inner_optimizer.zero_grad()
            self.total_inner_updates += 1
            self.total_samples_seen += int(batch_morgan.shape[0])
            # 保留 GPU 张量，避免每步 .item() 强制同步（每 epoch 约 400 次）
            losses.append(total_loss.detach())

        # direct_supervised：保留适应后的参数，不做 Reptile 恢复/插值
        if not restore:
            return None, torch.stack(losses).mean().item()

        # 计算参数变化（在GPU上完成，避免CPU拷贝）
        adapted_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                adapted_params[name] = param.data.clone()
        
        # 恢复初始参数（原地操作，避免额外内存分配）
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(initial_params[name])
        
        return adapted_params, torch.stack(losses).mean().item()
    
    def _meta_update(self, task_deltas):
        """
        外循环：跨靶点泛化更新（GPU优化）
        
        Args:
            task_deltas: list of {name: delta_tensor}
        """
        # 清空梯度
        self.meta_optimizer.zero_grad()
        self.total_meta_updates += 1
        
        # 计算平均梯度
        n_tasks = len(task_deltas)
        if n_tasks == 0:
            return
        
        # 梯度累积步数
        accumulation_steps = max(1, n_tasks // self.config.GRADIENT_ACCUMULATION_STEPS)
        
        # 对每个参数计算平均delta（在GPU上完成）
        avg_deltas = {}
        for name in task_deltas[0].keys():
            # 使用torch.stack和mean替代循环sum，更高效
            deltas_tensor = torch.stack([delta[name] for delta in task_deltas])
            avg_deltas[name] = deltas_tensor.mean(dim=0) / accumulation_steps
        
        # 应用元梯度（原地操作）
        # Reptile算法：θ = θ + meta_lr * (θ' - θ) = θ - meta_lr * (θ - θ')
        # 所以梯度应该是 -avg_deltas（avg_deltas = θ' - θ）
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in avg_deltas:
                param.grad = -avg_deltas[name]
        
        # 更新元参数
        self.meta_optimizer.step()

    def _accumulate_delta_chunk(self, sum_delta, chunk):
        """把一批任务的 delta 向量化求和后累加到 sum_delta（O(参数量) 显存）。

        与「把所有任务 delta stack 后 mean」数学等价，但避免逐靶点对每个参数
        张量单独 add_ 的 Python/launch 开销。
        """
        if not chunk:
            return sum_delta
        names = list(chunk[0].keys())
        if sum_delta is None:
            sum_delta = {}
            for name in names:
                sum_delta[name] = torch.stack([d[name] for d in chunk], dim=0).sum(dim=0).detach()
        else:
            with torch.no_grad():
                for name in names:
                    chunk_sum = torch.stack([d[name] for d in chunk], dim=0).sum(dim=0)
                    sum_delta[name].add_(chunk_sum)
                    del chunk_sum
        return sum_delta

    def _meta_update_online(self, sum_delta, n_tasks):
        """E4：在线累加版外循环更新。与 _meta_update 数学完全一致：
        stock: grad = -(stack(deltas).mean) / accumulation_steps，
               accumulation_steps = max(1, n_tasks // GRADIENT_ACCUMULATION_STEPS)。
        峰值显存由 O(任务数 × 参数量) 降为 O(参数量)。"""
        self.meta_optimizer.zero_grad()
        self.total_meta_updates += 1
        if n_tasks == 0:
            return
        accumulation_steps = max(1, n_tasks // self.config.GRADIENT_ACCUMULATION_STEPS)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in sum_delta:
                    param.grad = -(sum_delta[name] / n_tasks) / accumulation_steps
        self.meta_optimizer.step()
    
    def train(self, resume=False):
        """完整训练流程（GPU优化）"""
        # 尝试恢复断点
        if resume:
            self._load_checkpoint()
        
        self.logger.info("\n🚀 开始Reptile元学习训练...")
        self.logger.info(f"   Epochs: {self.config.EPOCHS}")
        self.logger.info(f"   Inner LR: {self.config.INNER_LR}, Inner Steps: {self.config.INNER_STEPS}")
        self.logger.info(f"   Meta LR: {self.config.META_LR}")
        self.logger.info(f"   Batch Size: {self.config.BATCH_SIZE}")
        self.logger.info(f"   Gradient Accumulation: {self.config.GRADIENT_ACCUMULATION_STEPS}")
        self.logger.info(f"   Mixed Precision: {self.config.MIXED_PRECISION}")
        
        # 记录开始时间
        total_start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.EPOCHS):
            epoch_start_time = time.time()
            self.epochs_run = epoch + 1
            self._diag_epoch = epoch

            # 训练模式
            self.model.train()

            # 收集所有任务的参数变化
            task_deltas = []
            # E4 在线模式：不保存所有任务的整模型 delta（任务数×参数量会爆显存），
            # 改为分块向量化累加；数学与「stack→mean」完全等价
            online_accum = getattr(self.config, 'ONLINE_DELTA_ACCUM', False)
            accum_chunk = max(1, int(getattr(self.config, 'ONLINE_ACCUM_CHUNK', 32)))
            control_mode = getattr(self.config, 'CONTROL_MODE', 'reptile')
            direct = control_mode == 'direct_supervised'
            chunk_deltas = []
            sum_delta = None
            total_inner_loss = 0.0
            n_tasks = 0
            self._diag_task_idx = 0
            
            # 打乱训练靶点顺序
            target_names = self.train_loader.get_all_targets()
            np.random.shuffle(target_names)
            
            # 遍历每个靶点
            for target_name in target_names:
                try:
                    # 获取靶点数据（已在GPU上）
                    morgan, maccs, descriptors, protein, y_norm = self.train_loader.get_target_data(target_name)
                    
                    # 跳过数据量过少的靶点
                    if morgan.shape[0] < 5:
                        continue
                    
                    # 内循环适应；direct_supervised 保留更新后的参数，不做 Reptile 插值
                    adapted_params, inner_loss = self._inner_loop(
                        morgan, maccs, descriptors, protein, y_norm, restore=not direct)

                    if direct:
                        total_inner_loss += inner_loss
                        n_tasks += 1
                        self._diag_task_idx += 1
                        continue

                    # 计算参数变化（delta = adapted - initial）
                    delta = {}
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            delta[name] = adapted_params[name] - param.data

                    if online_accum:
                        # 分块累加：每 accum_chunk 个靶点做一次向量化 stack/sum
                        chunk_deltas.append(delta)
                        if len(chunk_deltas) >= accum_chunk:
                            sum_delta = self._accumulate_delta_chunk(sum_delta, chunk_deltas)
                            chunk_deltas.clear()
                        del adapted_params
                    else:
                        task_deltas.append(delta)
                    total_inner_loss += inner_loss
                    n_tasks += 1
                    self._diag_task_idx += 1
                    
                except Exception as e:
                    self.logger.error(f"⚠️ Error processing {target_name}: {e}")
                    continue

            # 处理最后一个不足 accum_chunk 的分块
            if online_accum and chunk_deltas:
                sum_delta = self._accumulate_delta_chunk(sum_delta, chunk_deltas)
                chunk_deltas.clear()
            
            # 外循环更新（direct_supervised 不做 meta update）
            if n_tasks > 0:
                if not direct:
                    if online_accum:
                        self._meta_update_online(sum_delta, n_tasks)
                    else:
                        self._meta_update(task_deltas)
                    self.scheduler.step(epoch)
                
                avg_inner_loss = total_inner_loss / n_tasks
                current_lr = (self.inner_optimizer.param_groups[0]['lr'] if direct
                              else self.scheduler.get_last_lr()[0])
                
                epoch_time = time.time() - epoch_start_time
                
                # 打印GPU状态
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1e9
                    gpu_mem_cached = torch.cuda.memory_reserved() / 1e9
                    
                    # 尝试获取GPU利用率（需要pynvml）
                    try:
                        gpu_util = torch.cuda.utilization()
                    except (ModuleNotFoundError, AttributeError):
                        gpu_util = 'N/A'
                    
                    log_msg = (
                        f"\n📊 [Epoch {epoch+1}/{self.config.EPOCHS}] "
                        f"| Loss: {avg_inner_loss:.4f} "
                        f"| Tasks: {n_tasks} "
                        f"| LR: {current_lr:.6f} "
                        f"| GPU: {gpu_mem:.2f}GB/{gpu_mem_cached:.2f}GB "
                        f"| Time: {epoch_time:.2f}s"
                    )
                    self.logger.info(log_msg)
                else:
                    log_msg = (
                        f"\n📊 [Epoch {epoch+1}/{self.config.EPOCHS}] "
                        f"| Loss: {avg_inner_loss:.4f} "
                        f"| Tasks: {n_tasks} "
                        f"| LR: {current_lr:.6f} "
                        f"| Time: {epoch_time:.2f}s"
                    )
                    self.logger.info(log_msg)
            
            # 验证
            if (epoch + 1) % self.config.VAL_INTERVAL == 0 or epoch == self.config.EPOCHS - 1:
                val_metrics = self._validate()
                self._log_epoch(epoch, avg_inner_loss if n_tasks > 0 else 0, val_metrics)
                
                # 保存检查点
                is_best = val_metrics['R2'] > self.best_val_r2
                if is_best:
                    self.best_epoch = epoch + 1
                self._save_checkpoint(epoch, val_metrics, is_best)
                
                # 检查早停
                if self._check_early_stop(val_metrics):
                    self.logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                    break
            
            # 清理内存（减少GPU内存碎片）
            # 注：empty_cache 会强制设备同步，每 epoch 调用造成 GPU 周期性停顿；
            # 24GB 显存充足，降为每 10 epoch 清理一次（不影响数值结果）
            del task_deltas
            gc.collect()
            if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        total_time = time.time() - total_start_time
        self.logger.info(f"\n✅ 训练完成! 总时间: {total_time:.2f}s")
        
        # 保存训练历史
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)

        # 训练预算记录（6.1 预算匹配对照）
        epochs_run = int(getattr(self, 'epochs_run', self.current_epoch + 1))
        best_epoch = self.best_epoch
        if best_epoch is None and self.training_history:
            best = max(self.training_history, key=lambda r: float(r.get('val_r2', -1e9)))
            best_epoch = int(best.get('epoch', 0)) + 1
        budget = {
            'control_mode': getattr(self.config, 'CONTROL_MODE', 'reptile'),
            'epochs_run': epochs_run,
            'n_train_targets': len(self.train_loader.get_all_targets()),
            'inner_steps': int(self.config.INNER_STEPS),
            'inner_updates': int(self.total_inner_updates),
            'meta_updates': int(self.total_meta_updates),
            'samples_seen': int(self.total_samples_seen),
            'inner_updates_per_epoch': (self.total_inner_updates / max(1, epochs_run)),
            'best_epoch': best_epoch,
            'best_val_r2': float(self.best_val_r2),
            'walltime_s': float(total_time),
            'batch_size': int(self.config.BATCH_SIZE),
            'inner_lr': float(self.config.INNER_LR),
            'meta_lr': float(self.config.META_LR),
        }
        with open(os.path.join(self.output_dir, 'training_budget.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(budget, f, indent=2, ensure_ascii=False)
        self.logger.info(f"📦 训练预算: {json.dumps(budget, ensure_ascii=False)}")

        # E3：写出梯度冲突诊断
        self._dump_grad_conflict()
        
        return self.training_history
    
    def _validate(self):
        """验证模型（GPU优化）"""
        self.logger.info("   📝 Validating...")
        self.model.eval()
        
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for target_name in self.val_loader.get_all_targets():
                morgan, maccs, descriptors, protein, y_norm = self.val_loader.get_target_data(target_name)
                
                # 前向传播（混合精度）
                with self.autocast:
                    preds, _, _ = self.model(morgan, maccs, descriptors, protein)
                
                # 反归一化（仅在需要时拷贝到CPU）
                y_true = self.target_scaler.inverse_transform(y_norm.cpu().numpy(), target_name)
                y_pred = self.target_scaler.inverse_transform(preds.cpu().numpy(), target_name)
                
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
        
        # 计算指标
        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        
        metrics = self._compute_metrics(y_true, y_pred)
        
        self.logger.info(
            f"   Val R2: {metrics['R2']:.4f} | "
            f"RMSE: {metrics['RMSE']:.4f} | "
            f"MAE: {metrics['MAE']:.4f} | "
            f"Pearson: {metrics['Pearson']:.4f} | "
            f"Spearman: {metrics['Spearman']:.4f} | "
            f"EF@1%: {metrics['EF@1%']:.2f} | "
            f"EF@5%: {metrics['EF@5%']:.2f} | "
            f"EF@10%: {metrics['EF@10%']:.2f} | "
            f"ECE: {metrics['ECE']:.4f} | "
            f"AUPR: {metrics['AUPR']:.4f}"
        )
        
        return metrics
    
    def _log_epoch(self, epoch, train_loss, val_metrics):
        """记录训练日志"""
        self.training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_r2': val_metrics['R2'],
            'val_rmse': val_metrics['RMSE'],
            'val_mae': val_metrics['MAE'],
            'val_pearson': val_metrics['Pearson'],
            'val_spearman': val_metrics['Spearman'],
            'lr': self.scheduler.get_last_lr()[0],
            'time': time.time()
        })
        
        # 更新最佳指标
        if val_metrics['R2'] > self.best_val_r2:
            self.best_val_r2 = val_metrics['R2']
            self.best_val_rmse = val_metrics['RMSE']
            self.best_val_pearson = val_metrics['Pearson']
            self.model.save(os.path.join(self.output_dir, 'best_model.pt'))
            self.logger.info(f"🎉 新最佳模型保存! R2={val_metrics['R2']:.4f}, Pearson={val_metrics['Pearson']:.4f}")
    
    def _check_early_stop(self, val_metrics):
        """检查早停条件"""
        if val_metrics['R2'] > self.best_val_r2 + self.config.EARLY_STOP_MIN_DELTA:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.EARLY_STOP_PATIENCE
    
    def evaluate_test(self, load_best=True):
        """评估测试集（新靶点）"""
        self.logger.info("\n📊 测试集评估...")
        
        # 加载最佳模型
        if load_best:
            best_model_path = os.path.join(self.output_dir, 'best_model.pt')
            checkpoint_path = os.path.join(self.output_dir, 'checkpoint_best.pt')
            
            if os.path.exists(best_model_path):
                self.model.load(best_model_path)
                self.logger.info("   ✅ 加载最佳模型")
            elif os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.logger.info("   ✅ 从检查点加载最佳模型")
            else:
                self.logger.warning("   ⚠️ 未找到最佳模型，使用当前模型")
        
        self.model.eval()
        
        all_y_true = []
        all_y_pred = []
        per_target_metrics = {}
        
        with torch.no_grad():
            for target_name in self.test_loader.get_all_targets():
                morgan, maccs, descriptors, protein, y_norm = self.test_loader.get_target_data(target_name)
                
                # 前向传播（混合精度）
                with self.autocast:
                    preds, _, _ = self.model(morgan, maccs, descriptors, protein)
                
                # 反归一化
                y_true = self.target_scaler.inverse_transform(y_norm.cpu().numpy(), target_name)
                y_pred = self.target_scaler.inverse_transform(preds.cpu().numpy(), target_name)
                
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                
                # 计算单个靶点指标（跳过 <2 样本的靶点，避免 pearsonr 报错）
                if len(y_true) >= 2:
                    target_metrics = self._compute_metrics(np.array(y_true), np.array(y_pred))
                    target_metrics['samples'] = len(y_true)
                    per_target_metrics[target_name] = target_metrics
        
        # 计算整体指标（R²、RMSE等全局计算）
        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        
        metrics = self._compute_metrics(y_true, y_pred)
        metrics['samples'] = len(y_true)
        
        # 按靶点计算EF并取平均值（虚拟筛选标准做法）
        # 不同EF百分比需要不同的最小样本数：
        # - EF@1%: 需要>=100个样本（选1个化合物）
        # - EF@5%: 需要>=20个样本（选1个化合物）
        # - EF@10%: 需要>=10个样本（选1个化合物）
        ef1_list = []
        ef5_list = []
        ef10_list = []
        
        for target_name, target_metrics in per_target_metrics.items():
            n_samples = target_metrics.get('samples', 0)
            if n_samples >= 100:  # EF@1%需要至少100个样本
                ef1_list.append(target_metrics['EF@1%'])
            if n_samples >= 20:  # EF@5%需要至少20个样本
                ef5_list.append(target_metrics['EF@5%'])
            if n_samples >= 10:  # EF@10%需要至少10个样本
                ef10_list.append(target_metrics['EF@10%'])
        
        metrics['EF@1%'] = np.mean(ef1_list) if len(ef1_list) > 0 else 0.0
        metrics['EF@5%'] = np.mean(ef5_list) if len(ef5_list) > 0 else 0.0
        metrics['EF@10%'] = np.mean(ef10_list) if len(ef10_list) > 0 else 0.0
        metrics['EF1_targets'] = len(ef1_list)
        metrics['EF5_targets'] = len(ef5_list)
        metrics['EF10_targets'] = len(ef10_list)
        
        # 按靶点计算AUPR并取平均
        aupr_list = []
        for target_name, target_metrics in per_target_metrics.items():
            n_samples = target_metrics.get('samples', 0)
            if n_samples >= 5:
                aupr_list.append(target_metrics.get('AUPR', 0.0))
        metrics['AUPR'] = np.mean(aupr_list) if len(aupr_list) > 0 else 0.0
        metrics['AUPR_targets'] = len(aupr_list)
        
        metrics['per_target'] = per_target_metrics
        
        # 打印结果
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🎯 测试集结果 (新靶点)")
        self.logger.info("=" * 60)
        self.logger.info(f"   R²: {metrics['R2']:.4f}")
        self.logger.info(f"   RMSE: {metrics['RMSE']:.4f}")
        self.logger.info(f"   MAE: {metrics['MAE']:.4f}")
        self.logger.info(f"   Pearson: {metrics['Pearson']:.4f}")
        self.logger.info(f"   Spearman: {metrics['Spearman']:.4f}")
        self.logger.info(f"   EF@1%: {metrics['EF@1%']:.2f} (targets: {metrics['EF1_targets']})")
        self.logger.info(f"   EF@5%: {metrics['EF@5%']:.2f} (targets: {metrics['EF5_targets']})")
        self.logger.info(f"   EF@10%: {metrics['EF@10%']:.2f} (targets: {metrics['EF10_targets']})")
        self.logger.info(f"   ECE (校准度): {metrics['ECE']:.4f}")
        self.logger.info(f"   AUPR: {metrics['AUPR']:.4f} (targets: {metrics['AUPR_targets']})")
        self.logger.info(f"   Samples: {len(y_true)}")
        
        # 保存详细结果
        with open(os.path.join(self.output_dir, 'per_target_results.json'), 'w') as f:
            json.dump(per_target_metrics, f, indent=2, default=str)

        # 保存预测值（用于画散点图）
        np.savez(os.path.join(self.output_dir, 'predictions.npz'),
                 y_true=y_true, y_pred=y_pred)
        self.logger.info(f"   预测值已保存至 {self.output_dir}/predictions.npz")

        return metrics


# ==========================
# 快速测试工具
# ==========================
def quick_test(model, train_loader, n_tasks=3):
    """快速测试训练流程"""
    print("\n🔍 快速测试训练流程...")
    
    target_names = train_loader.get_all_targets()
    selected_targets = target_names[:n_tasks]
    
    for target_name in selected_targets:
        morgan, maccs, descriptors, protein, y_norm = train_loader.get_target_data(target_name)
        
        print(f"   Target: {target_name}, Samples: {morgan.shape[0]}")
        
        # 前向传播测试
        preds, consistency_loss, contrastive_loss = model(morgan, maccs, descriptors, protein)
        print(f"   Predictions shape: {preds.shape}")
        print(f"   Consistency loss: {consistency_loss.item():.4f}")
        print(f"   Contrastive loss: {contrastive_loss.item():.4f}")
        
        # 反向传播测试
        loss = F.mse_loss(preds, y_norm)
        loss.backward()
        
        # 检查梯度
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = np.sqrt(total_grad_norm)
        print(f"   Gradient norm: {total_grad_norm:.4f}")
        
        # 清空梯度
        model.zero_grad()
    
    print("✅ 快速测试通过!")


if __name__ == "__main__":
    print("Reptile Training Module")
