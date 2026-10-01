# -*- coding: utf-8 -*-
"""
E6 补充实验：Reptile vs MAML 计算成本对比（KIBA）
=====================================================
在主实验基础上把 Reptile 元学习换成 MAML（Finn et al., 2017），其余一切保持不变，
测量并对比计算成本（逐 epoch 耗时 / 总时长 / 峰值显存）与测试集指标。

与 stock Reptile 的对齐点（保证公平对比）：
  - 同 KIBA 论文配方：3_kiba_preprocessed, morgan_descriptors, batch=2048, epochs=220, inner_steps=3
  - 同模型 ReptileTransformer + init_weights；同 canonical 特征缓存（supplement_exp_common）
  - 同损失组合：MSE + 0.3*ListNet + λ1*对比 + λ2*一致性（λ1=0.05, λ2=0.10）
  - 同自适应内循环 lr 规则（n<100:×2, n<500:×1.5, n>2000:×0.5）+ 内循环梯度裁剪 5.0
  - 同外循环 Adam(meta_lr=0.001, wd=1e-5) + WarmupCosine 调度 + 相同的 grad 归一化公式
  - 同验证间隔/早停/检查点/测试评估代码路径（继承自 ReptileTrainer）

MAML 特有设定（论文需注明）：
  - 每任务按 support_frac=0.8 划分支持集/查询集（MAML 标准做法；Reptile 内循环用全量任务数据）
  - 默认二阶 MAML：torch.func.functional_call + create_graph=True，元梯度经内循环流回原参数
    --first_order 切换 FOMAML（一阶近似，内循环 detach）
  - bf16 autocast：二阶图下 fp16+GradScaler 不可行；4090 原生 bf16 与 stock 的 fp16 速度同量级

运行：
  python supplement_e6_maml.py --epochs 220            # 正式（默认 KIBA 配方）
  python supplement_e6_maml.py --epochs 2              # 烟雾测试
  python supplement_e6_maml.py --first_order ...       # FOMAML 变体
"""
import os
import sys

# GBK 控制台防崩（logger 的 📊 等 emoji）；须在创建任何 handler 之前
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# CUDA_VISIBLE_DEVICES 需在 torch 导入前生效
_gpu = '0'
if '--gpu' in sys.argv:
    _gpu = sys.argv[sys.argv.index('--gpu') + 1]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', _gpu)

import gc
import csv
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import data_preprocessing as _dp
from data_preprocessing import TargetScaler, get_all_targets
from reptile_transformer_model import ReptileTransformer, init_weights, count_params
from reptile_training import ReptileTrainer, TrainingConfig
import supplement_exp_common as common

OUTROOT = common.OUTROOT / 'e6_maml_vs_reptile'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    p = argparse.ArgumentParser(description='E6: Reptile vs MAML 计算成本对比（KIBA 配方）')
    p.add_argument('--dataset', type=str, default='kiba', choices=list(common.RECIPES.keys()))
    p.add_argument('--epochs', type=int, default=None, help='默认取该数据集论文配方（kiba=220）')
    p.add_argument('--inner_steps', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=None, help='默认取该数据集论文配方（kiba=2048）')
    p.add_argument('--first_order', action='store_true', help='FOMAML（一阶近似）而非二阶 MAML')
    p.add_argument('--support_frac', type=float, default=0.8, help='每任务支持集比例（其余为查询集）')
    p.add_argument('--meta_lr', type=float, default=None, help='覆盖 META_LR（补救调参用，如同配方失败后试 1e-4）')
    p.add_argument('--inner_lr', type=float, default=None, help='覆盖 INNER_LR（补救调参用，如 0.005）')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--test_mode', action='store_true', help='仅 5 个训练靶点（调试用）')
    p.add_argument('--num_train_targets', type=int, default=None)
    p.add_argument('--output_dir', type=str, default=None)
    p.add_argument('--force', action='store_true', help='忽略已完成状态强制重跑')
    return p.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class LazyDataLoader:
    """与 run_reptile_transformer.py main() 内的实现逐字节一致（GPU 缓存 + 消融屏蔽 MACCS）"""

    def __init__(self, data_mmap, indices, scaler, target_names_array=None, ablation='none'):
        from collections import defaultdict
        self._data = data_mmap
        self._scaler = scaler
        self._ablation = ablation
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
        print(f'   样本数: {len(self._indices)}, 靶点数: {len(self.target_groups)}')

    def get_all_targets(self):
        return list(self.target_groups.keys())

    def get_target_data(self, target_name):
        if target_name in self._gpu_cache:
            return self._gpu_cache[target_name]
        rel_indices = self.target_groups[target_name]
        if len(rel_indices) == 0:
            return None
        real_indices = self._indices[rel_indices]
        morgan = torch.tensor(self._data['morgan'][real_indices].astype(np.float32),
                              dtype=torch.float32, device=DEVICE)
        maccs = torch.tensor(self._data['maccs'][real_indices].astype(np.float32),
                             dtype=torch.float32, device=DEVICE)
        descriptors = torch.tensor(self._data['descriptors'][real_indices].astype(np.float32),
                                   dtype=torch.float32, device=DEVICE)
        protein = torch.tensor(self._data['protein'][real_indices].astype(np.float32),
                               dtype=torch.float32, device=DEVICE)
        y_raw = self._data['y'][real_indices].astype(np.float32)
        y_norm = self._scaler.transform(y_raw)
        y_norm = torch.tensor(y_norm, dtype=torch.float32, device=DEVICE)
        if self._ablation == 'morgan_descriptors':
            maccs = torch.zeros_like(maccs)
        cached = (morgan, maccs, descriptors, protein, y_norm)
        self._gpu_cache[target_name] = cached
        return cached


class MAMLTrainer(ReptileTrainer):
    """继承 ReptileTrainer：复用 _validate / evaluate_test / _save_checkpoint / _log_epoch /
    _check_early_stop / _compute_metrics；重写内循环为 MAML、外循环为查询集元梯度。"""

    def __init__(self, *args, first_order=False, support_frac=0.8, **kw):
        super().__init__(*args, **kw)
        self.first_order = first_order
        self.support_frac = support_frac
        # 训练用 bf16 autocast（二阶图下 fp16+GradScaler 不可行；4090 原生 bf16，速度同量级）；
        # 验证/评估沿用 stock 的 fp16 autocast（self.autocast 不动，行为逐字节一致）
        self.train_autocast = torch.amp.autocast('cuda', dtype=torch.bfloat16) \
            if torch.cuda.is_available() else None
        self.epoch_times = []
        self.n_tasks_last_epoch = 0

    # ---- 损失组合：与 stock _inner_loop 内的公式逐项一致 ----
    def _combined_loss(self, preds, batch_y, consistency_loss, contrastive_loss):
        total = F.mse_loss(preds, batch_y)
        if self.config.RANKING_WEIGHT > 0 and batch_y.shape[0] >= 2:
            pred_probs = F.softmax(preds, dim=0)
            target_probs = F.softmax(batch_y, dim=0)
            pred_probs = torch.clamp(pred_probs, min=1e-10)
            target_probs = torch.clamp(target_probs, min=1e-10)
            ranking_loss = F.kl_div(torch.log(pred_probs), target_probs, reduction='batchmean')
            total = total + self.config.RANKING_WEIGHT * ranking_loss
        if self.config.CONTRASTIVE_WEIGHT > 0:
            total = total + self.config.CONTRASTIVE_WEIGHT * contrastive_loss
        if self.config.CONSISTENCY_WEIGHT > 0:
            total = total + self.config.CONSISTENCY_WEIGHT * consistency_loss
        return total

    def _task_loss(self, params, buffers, morgan, maccs, descriptors, protein, y_norm):
        with self.train_autocast:
            preds, consistency_loss, contrastive_loss = functional_call(
                self.model, {**params, **buffers},
                (morgan, maccs, descriptors, protein))
            return self._combined_loss(preds, y_norm, consistency_loss, contrastive_loss)

    def _adaptive_inner_lr(self, n_samples):
        """与 stock _inner_loop 完全相同的任务自适应学习率规则"""
        base_lr = self.config.INNER_LR
        if n_samples < 100:
            return base_lr * 2.0
        elif n_samples < 500:
            return base_lr * 1.5
        elif n_samples > 2000:
            return base_lr * 0.5
        return base_lr

    @staticmethod
    def _clip_grads(grads, max_norm=5.0):
        """全局范数裁剪（对齐 stock 内循环 clip_grad_norm_(5.0)；系数 detach，不参与二阶图）"""
        gnorm = torch.sqrt(torch.stack([
            (g.detach() ** 2).sum() for g in grads if g.numel() > 0
        ]).sum())
        coef = (max_norm / (gnorm + 1e-12)).clamp(max=1.0).detach()
        return [g * coef for g in grads]

    def _maml_task(self, morgan, maccs, descriptors, protein, y_norm):
        """单个任务：支持集 inner SGD（可微/或 detach）→ 查询集元梯度。
        返回 (meta_grads{name: tensor}, inner_loss, meta_loss)。"""
        model = self.model
        n = morgan.shape[0]
        perm = torch.randperm(n, device=morgan.device)
        n_support = min(max(int(n * self.support_frac), 2), n - 2)
        s_idx, q_idx = perm[:n_support], perm[n_support:]
        lr = self._adaptive_inner_lr(n)

        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if self.first_order:
            # FOMAML：初始参数切断图连接，元梯度 = 查询损失对适应后参数的梯度
            params = {name: p.detach().clone().requires_grad_(True) for name, p in trainable}
        else:
            # 二阶 MAML：clone 保留对原参数的图连接，元梯度经内循环流回原参数
            params = {name: p.clone() for name, p in trainable}
        buffers = dict(model.named_buffers())

        inner_losses = []
        for _step in range(self.config.INNER_STEPS):
            if n_support > self.config.BATCH_SIZE:
                sub = torch.randperm(n_support, device=morgan.device)[:self.config.BATCH_SIZE]
                bm = morgan[s_idx][sub]
                bc = maccs[s_idx][sub]
                bd = descriptors[s_idx][sub]
                bp = protein[s_idx][sub]
                by = y_norm[s_idx][sub]
            else:
                bm, bc, bd, bp, by = morgan[s_idx], maccs[s_idx], descriptors[s_idx], protein[s_idx], y_norm[s_idx]

            loss = self._task_loss(params, buffers, bm, bc, bd, bp, by)
            grads = torch.autograd.grad(loss, list(params.values()),
                                        create_graph=not self.first_order, allow_unused=True)
            grads = [torch.zeros_like(v) if g is None else g
                     for g, v in zip(grads, params.values())]
            grads = self._clip_grads(grads)

            if self.first_order:
                with torch.no_grad():
                    params = {name: (v - lr * g).detach().requires_grad_(True)
                              for (name, _), v, g in zip(trainable, params.values(), grads)}
            else:
                params = {name: v - lr * g
                          for (name, _), v, g in zip(trainable, params.values(), grads)}
            inner_losses.append(loss.detach())

        qm, qc, qd, qp, qy = (morgan[q_idx], maccs[q_idx], descriptors[q_idx],
                              protein[q_idx], y_norm[q_idx])
        meta_loss = self._task_loss(params, buffers, qm, qc, qd, qp, qy)

        if self.first_order:
            glist = torch.autograd.grad(meta_loss, list(params.values()), allow_unused=True)
            meta_grads = {name: (torch.zeros_like(v) if g is None else g.detach())
                          for (name, _), g, v in zip(trainable, glist, params.values())}
        else:
            glist = torch.autograd.grad(meta_loss, [p for _, p in trainable], allow_unused=True)
            meta_grads = {name: (torch.zeros_like(p) if g is None else g.detach())
                          for (name, p), g in zip(trainable, glist)}
        return meta_grads, torch.stack(inner_losses).mean().item(), meta_loss.detach().item()

    def train(self, resume=False):
        """与 stock train() 相同骨架；差异：MAML 内循环 + 查询集元梯度在线累加（O(参数量) 显存）"""
        algo = 'FOMAML' if self.first_order else 'MAML(2nd)'
        self.logger.info(f'\n🚀 开始 {algo} 元学习训练...')
        self.logger.info(f'   Epochs: {self.config.EPOCHS} | Inner LR: {self.config.INNER_LR} '
                         f'| Inner Steps: {self.config.INNER_STEPS} | Meta LR: {self.config.META_LR}')
        self.logger.info(f'   Batch Size: {self.config.BATCH_SIZE} | support/query: '
                         f'{self.support_frac}/{1 - self.support_frac:.1f} | Mixed Precision: bf16')

        total_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(self.current_epoch, self.config.EPOCHS):
            epoch_start_time = time.time()
            self._diag_epoch = epoch
            self.model.train()

            sum_grads = None
            total_inner_loss = 0.0
            total_meta_loss = 0.0
            n_tasks = 0
            self._diag_task_idx = 0

            target_names = self.train_loader.get_all_targets()
            np.random.shuffle(target_names)

            for target_name in target_names:
                try:
                    morgan, maccs, descriptors, protein, y_norm = \
                        self.train_loader.get_target_data(target_name)
                    if morgan.shape[0] < 5:
                        continue
                    meta_grads, inner_loss, meta_loss = \
                        self._maml_task(morgan, maccs, descriptors, protein, y_norm)
                    if sum_grads is None:
                        sum_grads = {k: v.clone() for k, v in meta_grads.items()}
                    else:
                        for k, v in meta_grads.items():
                            sum_grads[k] += v
                    total_inner_loss += inner_loss
                    total_meta_loss += meta_loss
                    n_tasks += 1
                except Exception as e:
                    self.logger.error(f'⚠️ Error processing {target_name}: {e}')
                    continue

            avg_inner_loss = total_inner_loss / n_tasks if n_tasks else 0.0
            avg_meta_loss = total_meta_loss / n_tasks if n_tasks else 0.0

            if n_tasks > 0:
                self.n_tasks_last_epoch = n_tasks
                # 与 stock 相同的归一化公式（GAS=1 时 accumulation_steps = n_tasks）
                self.meta_optimizer.zero_grad()
                accumulation_steps = max(1, n_tasks // self.config.GRADIENT_ACCUMULATION_STEPS)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in sum_grads:
                        # MAML：grad = +meta_grad（optimizer.step 做 -= lr*grad）
                        param.grad = (sum_grads[name] / n_tasks) / accumulation_steps
                self.meta_optimizer.step()
                self.scheduler.step(epoch)

                epoch_time = time.time() - epoch_start_time
                self.epoch_times.append(epoch_time)

                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1e9
                    gpu_mem_cached = torch.cuda.memory_reserved() / 1e9
                    self.logger.info(
                        f'\n📊 [Epoch {epoch + 1}/{self.config.EPOCHS}] '
                        f'| MetaLoss: {avg_meta_loss:.4f} | InnerLoss: {avg_inner_loss:.4f} '
                        f'| Tasks: {n_tasks} | LR: {self.scheduler.get_last_lr()[0]:.6f} '
                        f'| GPU: {gpu_mem:.2f}GB/{gpu_mem_cached:.2f}GB '
                        f'| Time: {epoch_time:.2f}s')
                else:
                    self.logger.info(
                        f'\n📊 [Epoch {epoch + 1}/{self.config.EPOCHS}] '
                        f'| MetaLoss: {avg_meta_loss:.4f} | InnerLoss: {avg_inner_loss:.4f} '
                        f'| Tasks: {n_tasks} | Time: {epoch_time:.2f}s')

            if (epoch + 1) % self.config.VAL_INTERVAL == 0 or epoch == self.config.EPOCHS - 1:
                val_metrics = self._validate()
                self._log_epoch(epoch, avg_meta_loss, val_metrics)
                is_best = val_metrics['R2'] > self.best_val_r2
                self._save_checkpoint(epoch, val_metrics, is_best)
                if self._check_early_stop(val_metrics):
                    self.logger.info(f'⏹️ Early stopping at epoch {epoch + 1}')
                    break

            del sum_grads
            gc.collect()
            if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
                torch.cuda.empty_cache()

        total_time = time.time() - total_start_time
        self.logger.info(f'\n✅ 训练完成! 总时间: {total_time:.2f}s')

        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        return self.training_history

    def cost_summary(self, total_walltime_sec):
        ep = self.epoch_times
        cost = {
            'algorithm': 'FOMAML' if self.first_order else 'MAML-2nd',
            'total_train_sec': round(sum(ep), 2),
            'total_walltime_sec': round(total_walltime_sec, 2),
            'mean_epoch_sec': round(float(np.mean(ep)), 3) if ep else None,
            'median_epoch_sec': round(float(np.median(ep)), 3) if ep else None,
            'n_epochs_trained': len(ep),
            'n_tasks_per_epoch': self.n_tasks_last_epoch,
            'peak_mem_alloc_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                                 if torch.cuda.is_available() else None,
            'peak_mem_reserved_gb': round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)
                                    if torch.cuda.is_available() else None,
            'n_trainable_params': int(sum(p.numel() for p in self.model.parameters()
                                          if p.requires_grad)),
            'support_frac': self.support_frac,
            'inner_steps': self.config.INNER_STEPS,
            'batch_size': self.config.BATCH_SIZE,
        }
        return cost


def main():
    args = parse_args()
    recipe = common.RECIPES[args.dataset]

    variant = 'fomaml' if args.first_order else 'maml'
    output_dir = Path(args.output_dir) if args.output_dir else \
        OUTROOT / args.dataset / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / 'final_results.json'
    if final_path.exists() and not args.force:
        print(f'✅ 已完成，跳过: {final_path}')
        return

    # TF32（与 stock 一致，保证成本对比公平）
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if args.seed is not None:
        set_seed(args.seed)
        print(f'🎲 随机种子已固定: {args.seed}')

    data_dir = common.data_path(args.dataset)
    _dp.PREPROCESSED_DIR = Path(data_dir)    # 显式重绑模块全局（陷阱：须为 Path，模块内做 PREPROCESSED_DIR / split）
    os.environ['PREPROCESSED_DIR'] = data_dir

    print('=' * 70)
    print(f'🔬 E6: Reptile vs MAML 计算成本对比 | dataset={args.dataset} | '
          f'algorithm={"FOMAML" if args.first_order else "MAML(2nd)"}')
    print(f'   设备: {DEVICE} | 输出: {output_dir}')
    print('=' * 70)

    script_start = time.time()

    # 1. 靶点（与 stock 相同的划分）
    all_targets = get_all_targets(test_mode=args.test_mode,
                                  max_targets=5 if args.test_mode else args.num_train_targets)
    print(f'   Train: {len(all_targets["train"])} targets | '
          f'Val: {len(all_targets["val"])} | Test: {len(all_targets["test"])}')

    # 2. canonical 特征缓存（与 E1-E5 逐字节一致；KIBA 已缓存，秒级完成）
    feat_npz = common.ensure_features(args.dataset, gpu=args.gpu)
    print(f'   特征: {feat_npz}')

    # 3. 归一化器（与 stock 完全相同的拟合方式）
    scaler = TargetScaler()
    data = np.load(feat_npz, mmap_mode='r', allow_pickle=True)
    splits = data['splits'][:]
    train_mask = splits == 'train'
    scaler.fit(data['y'][train_mask], data['target_names'][train_mask])
    scaler.save(output_dir / 'target_scaler.json')

    target_names_array = data['target_names'][:]
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(splits == 'val')[0]
    test_indices = np.where(splits == 'test')[0]
    print(f'   Samples: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}')

    # 4. 模型（与 stock 完全相同的构建与初始化）
    model = ReptileTransformer()
    model.apply(init_weights)
    model = model.to(DEVICE)
    total_params, trainable_params = count_params(model)
    print(f'   Total params: {total_params:,} | Trainable: {trainable_params:,}')

    # 5. 数据加载器
    train_loader = LazyDataLoader(data, train_indices, scaler, target_names_array,
                                  recipe['ablation'])
    val_loader = LazyDataLoader(data, val_indices, scaler, target_names_array,
                                recipe['ablation'])
    test_loader = LazyDataLoader(data, test_indices, scaler, target_names_array,
                                 recipe['ablation'])

    # 6. 配置：论文配方（kiba: batch=2048, epochs=220）
    config = TrainingConfig()
    config.BATCH_SIZE = args.batch_size if args.batch_size is not None else recipe['batch']
    config.EPOCHS = args.epochs if args.epochs is not None else recipe['epochs']
    config.INNER_STEPS = args.inner_steps
    config.GRADIENT_ACCUMULATION_STEPS = 1
    if args.meta_lr is not None:
        config.META_LR = args.meta_lr
    if args.inner_lr is not None:
        config.INNER_LR = args.inner_lr

    # 7. 训练
    trainer = MAMLTrainer(model=model, train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader, target_scaler=scaler,
                          output_dir=str(output_dir), config=config,
                          first_order=args.first_order, support_frac=args.support_frac)
    training_history = trainer.train()

    # 8. 测试评估（复用 stock 代码路径）
    test_metrics = trainer.evaluate_test()

    # 9. 成本与结果落盘（final_results.json 与 stock 同 schema → read_metrics 兼容）
    total_walltime = time.time() - script_start
    cost = trainer.cost_summary(total_walltime)

    # 逐 epoch 耗时 CSV
    with open(output_dir / 'epoch_times.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'time_sec'])
        for i, t in enumerate(trainer.epoch_times, 1):
            w.writerow([i, f'{t:.3f}'])

    results = {
        'training_history': training_history,
        'test_metrics': test_metrics,
        'cost': cost,
        'config': {
            'algorithm': cost['algorithm'],
            'inner_lr': trainer.config.INNER_LR,
            'inner_steps': trainer.config.INNER_STEPS,
            'meta_lr': trainer.config.META_LR,
            'epochs': trainer.config.EPOCHS,
            'batch_size': trainer.config.BATCH_SIZE,
            'ablation': recipe['ablation'],
            'support_frac': args.support_frac,
            'seed': args.seed,
        },
        'best_val_r2': trainer.best_val_r2,
        'best_val_rmse': trainer.best_val_rmse,
    }
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    (output_dir / 'walltime_sec.txt').write_text(str(total_walltime), encoding='utf-8')

    # 10. 与 stock Reptile 基线（E1 steps_3）对比打印
    print('\n' + '=' * 70)
    print(f'⏱️ 总时间: {total_walltime:.1f} s（训练 {cost["total_train_sec"]:.1f} s，'
          f'均值 {cost["mean_epoch_sec"]} s/epoch）')
    print(f'💾 峰值显存: {cost["peak_mem_alloc_gb"]} GB (allocated) / '
          f'{cost["peak_mem_reserved_gb"]} GB (reserved)')
    print(f'📊 测试集 R²: {test_metrics["R2"]:.4f} | EF@1%: {test_metrics["EF@1%"]:.2f} '
          f'| ECE: {test_metrics["ECE"]:.4f}')
    print('=' * 70)

    baseline_dir = common.OUTROOT / 'e1_inner_steps' / args.dataset / 'steps_3'
    base = common.read_metrics(baseline_dir)
    if base is not None:
        print('\n📋 Reptile（E1 baseline steps_3）vs 本 run 对比：')
        print(f'   {"":<10}{"R2":>10}{"EF@1%":>10}{"ECE":>10}{"walltime":>12}')
        bw = base.get('_walltime_min', float('nan'))
        print(f'   {"Reptile":<10}{base["R2"]:>10.4f}{base["EF@1%"]:>10.2f}'
              f'{base["ECE"]:>10.4f}{bw:>10.1f}m*')
        print(f'   {cost["algorithm"]:<10}{test_metrics["R2"]:>10.4f}'
              f'{test_metrics["EF@1%"]:>10.2f}{test_metrics["ECE"]:>10.4f}'
              f'{total_walltime / 60:>10.1f}m')
        print('   * E1 基线当时与填充作业并行跑，walltime 偏高；干净对比请另跑 '
              'reptile_solo（见 E6 说明）')
    print(f'\n✅ E6 完成! 结果: {final_path}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n⚠️ 用户中断')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ 错误: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
