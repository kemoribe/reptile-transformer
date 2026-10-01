# -*- coding: utf-8 -*-
"""
E4 — ESM2 冻结策略消融（KIBA）：Frozen / Partial(last-2) / Full
===============================================================
Frozen（论文现状、本实验基准）：ESM2 128 维嵌入离线预计算、完全冻结，
   编码器不在训练图中 —— 直接用 run_reptile_transformer.py 的 KIBA 论文配方
   (morgan_descriptors + batch2048 + 220ep)，由本脚本 --mode frozen 自动调用。

Partial / Full（本脚本在线训练）：
   - 同一批训练样本、同一 KIBA 配方（batch2048/220ep/inner_steps3/λ1=0.05/λ2=0.1）；
   - ESM2(t12, 35M, 480d) 进入训练图，每靶点只需对【单条】序列前向（批内蛋白相同，
     128 维表征再 expand），因此开销远小于逐化合物过 ESM2；
   - Partial：仅最后 2 个 transformer block + 最终 LayerNorm + 480→128 投影可训练；
   - Full：ESM2 全部参数 + 投影可训练；
   - 优化：头模型沿用 Reptile（内 SGD lr=0.01 / 外 Adam lr=1e-3）；
     ESM2 参数分组学习率 = 头模型 × 0.1（Partial）/ × 0.01（Full，全量微调惯例），
     weight_decay=0；其余（AMP/TF32/warmup5/余弦/早停/逐靶点 EF/ECE）与主模型完全一致。

用法:
  python supplement_e4_esm2_freeze.py --mode frozen  --gpu 0
  python supplement_e4_esm2_freeze.py --mode partial --gpu 0
  python supplement_e4_esm2_freeze.py --mode full    --gpu 0
  python supplement_e4_esm2_freeze.py --mode partial --epochs 60 --smoke   # 快速验证
输出: supplement_output/revision2_experiments/e4_esm2_freeze/<dataset>/<mode>/
"""
import argparse
import gc
import json
import os
import sys
import collections
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from data_preprocessing import get_all_targets, load_target_data, TargetScaler  # noqa
from reptile_transformer_model import (ReptileTransformer, init_weights, count_params,  # noqa
                                       HIDDEN_DIM, ESM2_DIM, DROPOUT)
from reptile_training import ReptileTrainer, TrainingConfig, device  # noqa
from supplement_exp_common import RECIPES, stage_dir, ensure_features, run_stock, read_metrics  # noqa

ESM2_MODEL_PATH = str(BASE / 'esm2_model')
AA = set('ACDEFGHIKLMNPQRSTVWY')


# ============================ 在线 ESM2 编码器 ============================
class OnlineESM2Encoder(torch.nn.Module):
    """ESM2 + mean-pool + Linear(480→128)；mode: partial / full"""

    def __init__(self, mode, model_path=ESM2_MODEL_PATH, output_dim=ESM2_DIM, seed=42):
        super().__init__()
        import esm
        assert mode in ('partial', 'full')
        self.mode = mode

        # 与 data_preprocessing.ProteinFeatureExtractor 完全一致的构建方式
        with open(os.path.join(model_path, 'config.json')) as f:
            cfg = json.load(f)
        hidden_dim = cfg.get('hidden_size', 480)
        num_layers = cfg.get('num_hidden_layers', 12)
        num_heads = cfg.get('num_attention_heads', 20)
        self.num_layers = num_layers

        self.alphabet = esm.Alphabet.from_architecture('ESM-1b')
        self.esm = esm.ESM2(num_layers=num_layers, embed_dim=hidden_dim,
                            attention_heads=num_heads, alphabet=self.alphabet)
        wp = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(wp):
            sd = torch.load(wp, map_location='cpu', weights_only=True)
        else:
            from safetensors.torch import load_file
            sd = load_file(os.path.join(model_path, 'model.safetensors'))
        self.esm.load_state_dict(sd, strict=False)
        self.batch_converter = self.alphabet.get_batch_converter()

        # 480→128 投影：与离线管线同构（离线为冻结随机投影；解冻档该投影参与训练）
        torch.manual_seed(seed)
        self.proj = torch.nn.Linear(hidden_dim, output_dim)

        # 冻结策略
        for p in self.esm.parameters():
            p.requires_grad = False
        if mode == 'full':
            for p in self.esm.parameters():
                p.requires_grad = True
        elif mode == 'partial':
            for p in self.esm.layers[-1].parameters():
                p.requires_grad = True
            for p in self.esm.layers[-2].parameters():
                p.requires_grad = True
            if hasattr(self.esm, 'emb_layer_norm_after'):
                for p in self.esm.emb_layer_norm_after.parameters():
                    p.requires_grad = True
        for p in self.proj.parameters():
            p.requires_grad = True

    def tokenize(self, sequence):
        sequence = ''.join(c for c in sequence if c in AA)[:1022]
        _, _, toks = self.batch_converter([('protein', sequence)])
        return toks  # (1, L+2)

    def forward(self, tokens):
        # tokens: (1, L) —— 一个靶点只过一次 ESM2
        out = self.esm(tokens, repr_layers=[self.num_layers])
        h = out['representations'][self.num_layers].mean(dim=1)   # (1,480)
        return self.proj(h)                                        # (1,128)


# ============================ 在线模型 ============================
class OnlineReptileTransformer(ReptileTransformer):
    """第 4 个入参由预计算 128d 向量改为 token ids；ESM2 在图内"""

    def __init__(self, mode, seed=42):
        super().__init__()                       # 头模型（15.45M），protein_encoder=None
        self.esm_core = OnlineESM2Encoder(mode, seed=seed)
        self.esm_mode = mode
        self.to(device)

    def forward(self, morgan_fp, maccs_fp, descriptors, tokens):
        n = morgan_fp.shape[0]
        p128 = self.esm_core(tokens)             # (1,128)
        protein_feat = p128.expand(n, -1)
        # 以下与父类 forward 完全一致
        mol_feat = self.molecule_encoder(morgan_fp, maccs_fp, descriptors)
        protein_feat_proj = self.protein_proj(protein_feat)
        fused_feat, consistency_loss = self.cross_attention(mol_feat, protein_feat_proj)
        preds = self.predictor(fused_feat)
        preds = preds * self.output_scale + self.output_bias
        contrastive_loss = self.contrastive_loss(mol_feat, protein_feat_proj)
        return preds.squeeze(-1), consistency_loss, contrastive_loss

    def save(self, path):
        torch.save(self.state_dict(), path)      # 含 esm_core（解冻权重必须落盘）

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=False),
                             strict=False)
        return self


# ============================ token 数据器 ============================
class OnlineTokenLoader:
    """与 stock LazyDataLoader 同构，第 4 项换成单条 token (1,L)（按靶点缓存）"""

    def __init__(self, data_mmap, indices, scaler, seq_map, tokenize_fn,
                 target_names_array=None, zero_maccs=False):
        self._data = data_mmap
        self._scaler = scaler
        self._seq_map = seq_map
        self._tokenize = tokenize_fn
        self._zero_maccs = zero_maccs
        self.target_groups = collections.defaultdict(list)
        targets_subset = target_names_array[indices]
        uniq, inv = np.unique(targets_subset, return_inverse=True)
        for k, name in enumerate(uniq):
            self.target_groups[name] = np.where(inv == k)[0].tolist()
        self._indices = np.asarray(indices)
        self._gpu_cache = {}
        print(f"   [online] 样本 {len(self._indices)}, 靶点 {len(self.target_groups)}")

    def get_all_targets(self):
        return list(self.target_groups.keys())

    def get_target_data(self, target_name):
        if target_name in self._gpu_cache:
            return self._gpu_cache[target_name]
        rel = self.target_groups[target_name]
        if len(rel) == 0:
            return None
        real = self._indices[rel]
        morgan = torch.tensor(self._data['morgan'][real].astype(np.float32), device=device)
        maccs = self._data['maccs'][real].astype(np.float32)
        if self._zero_maccs:
            maccs = np.zeros_like(maccs)
        maccs = torch.tensor(maccs, device=device)
        desc = torch.tensor(self._data['descriptors'][real].astype(np.float32), device=device)
        y_raw = self._data['y'][real].astype(np.float32)
        y = torch.tensor(self._scaler.transform(y_raw), device=device)
        seq = self._seq_map[target_name]
        tokens = self._tokenize(seq).to(device)
        cached = (morgan, maccs, desc, tokens, y)
        self._gpu_cache[target_name] = cached
        return cached

    def clear_cache(self):
        self._gpu_cache.clear()
        torch.cuda.empty_cache()


# ============================ 分组学习率训练器 ============================
class ESM2FinetuneTrainer(ReptileTrainer):
    """ESM2 参数使用更小的内/外学习率；其余与 ReptileTrainer 完全一致"""

    def __init__(self, *args, esm_lr_scale=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.esm_lr_scale = esm_lr_scale
        m = self.model
        head_p, esm_p = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            (esm_p if n.startswith('esm_core.') else head_p).append(p)
        self.meta_optimizer = torch.optim.Adam([
            {'params': head_p, 'lr': self.config.META_LR, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': esm_p, 'lr': self.config.META_LR * esm_lr_scale, 'weight_decay': 0.0},
        ])
        self.inner_optimizer = torch.optim.SGD([
            {'params': head_p, 'lr': self.config.INNER_LR},
            {'params': esm_p, 'lr': self.config.INNER_LR * esm_lr_scale},
        ])
        # 重建调度器（读取新 optimizer 的 base_lrs）
        from reptile_training import WarmupCosineScheduler
        self.scheduler = WarmupCosineScheduler(self.meta_optimizer,
                                               max_epochs=self.config.EPOCHS,
                                               warmup_epochs=self.config.META_WARMUP_EPOCHS,
                                               eta_min=1e-6)

    def _inner_loop(self, morgan, maccs, descriptors, tokens, y_norm):
        """stock _inner_loop 的精确复制，仅内循环 lr 按参数组缩放"""
        import torch.nn.functional as F
        initial_params = {n: p.data.clone() for n, p in self.model.named_parameters()
                          if p.requires_grad}
        n_samples = morgan.shape[0]
        base_lr = self.config.INNER_LR
        if n_samples < 100:
            adaptive_lr = base_lr * 2.0
        elif n_samples < 500:
            adaptive_lr = base_lr * 1.5
        elif n_samples > 2000:
            adaptive_lr = base_lr * 0.5
        else:
            adaptive_lr = base_lr
        for pg, sc in zip(self.inner_optimizer.param_groups, (1.0, self.esm_lr_scale)):
            pg['lr'] = adaptive_lr * sc

        losses = []
        for step in range(self.config.INNER_STEPS):
            if n_samples > self.config.BATCH_SIZE:
                idx = torch.randperm(n_samples, device=morgan.device)[:self.config.BATCH_SIZE]
                bm, bma, bd, by = morgan[idx], maccs[idx], descriptors[idx], y_norm[idx]
            else:
                bm, bma, bd, by = morgan, maccs, descriptors, y_norm
            with self.autocast:
                preds, consistency_loss, contrastive_loss = self.model(bm, bma, bd, tokens)
                mse_loss = F.mse_loss(preds, by)
                total_loss = mse_loss
                if self.config.RANKING_WEIGHT > 0 and by.shape[0] >= 2:
                    pp = torch.clamp(F.softmax(preds, dim=0), min=1e-10)
                    tp = torch.clamp(F.softmax(by, dim=0), min=1e-10)
                    ranking_loss = F.kl_div(torch.log(pp), tp, reduction='batchmean')
                    total_loss += self.config.RANKING_WEIGHT * ranking_loss
                if self.config.CONTRASTIVE_WEIGHT > 0:
                    total_loss += self.config.CONTRASTIVE_WEIGHT * contrastive_loss
                if self.config.CONSISTENCY_WEIGHT > 0:
                    total_loss += self.config.CONSISTENCY_WEIGHT * consistency_loss
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
            losses.append(total_loss.detach())

        adapted = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(initial_params[n])
        return adapted, torch.stack(losses).mean().item()


# ============================ main ============================
def build_seq_map(dataset):
    from pathlib import Path as _P
    data_dir = _P(BASE) / RECIPES[dataset]['data_dir']
    os.environ['PREPROCESSED_DIR'] = str(data_dir)
    import data_preprocessing as _dp
    # PREPROCESSED_DIR 在 import 时已绑定（默认 ChEMBL），必须显式重绑
    _dp.PREPROCESSED_DIR = data_dir
    targets = _dp.get_all_targets()
    seq_map = {}
    for split in ('train', 'val', 'test'):
        for info in targets[split]:
            td = _dp.load_target_data(info)
            if td is not None and td.get('sequence'):
                seq_map[td['target_name']] = td['sequence']
    print(f'[online] 序列字典: {len(seq_map)} 个靶点')
    return seq_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['frozen', 'partial', 'full'], required=True)
    ap.add_argument('--dataset', default='kiba')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--inner_steps', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--smoke', action='store_true', help='2 epochs 烟雾测试')
    args = ap.parse_args()

    recipe = RECIPES[args.dataset]
    epochs = 2 if args.smoke else (args.epochs or recipe['epochs'])
    batch = args.batch_size or recipe['batch']
    outroot = stage_dir('e4_esm2_freeze') / args.dataset
    outroot.mkdir(parents=True, exist_ok=True)

    # ---- Frozen：完全等价的离线冻结管线（stock runner）----
    if args.mode == 'frozen':
        run_dir = outroot / 'frozen'
        if (run_dir / 'final_results.json').exists():
            print(f'[skip] E4 frozen 已完成: {read_metrics(run_dir)}')
            return
        rc = run_stock(args.dataset, run_dir, epochs=epochs, inner_steps=args.inner_steps,
                       seed=args.seed, gpu=args.gpu, log_suffix='E4 frozen(offline)')
        print(f'frozen done rc={rc}: {read_metrics(run_dir)}')
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = outroot / args.mode
    run_dir.mkdir(parents=True,exist_ok=True)
    if (run_dir / 'final_results.json').exists():
        print(f'[skip] E4 {args.mode} 已完成: {read_metrics(run_dir)}')
        return

    # ---- 数据（复用 canonical 全特征 npz；MACCS 按配方在 loader 内置零）----
    feat_npz = ensure_features(args.dataset, gpu=args.gpu)
    data = np.load(feat_npz, mmap_mode='r', allow_pickle=True)
    splits = data['splits'][:]
    tnames = data['target_names'][:]
    train_idx = np.where(splits == 'train')[0]
    val_idx = np.where(splits == 'val')[0]
    test_idx = np.where(splits == 'test')[0]

    scaler = TargetScaler()
    scaler.fit(data['y'][train_idx], tnames[train_idx])
    scaler.save(run_dir / 'target_scaler.json')

    seq_map = build_seq_map(args.dataset)
    zero_maccs = (recipe['ablation'] == 'morgan_descriptors')

    model = OnlineReptileTransformer(args.mode, seed=args.seed)
    # 只初始化头模块；绝不能 apply 到 esm_core（会用 Xavier 冲掉 ESM2 预训练的 q/k/v 线性层）
    for name, module in list(model.named_children()):
        if name != 'esm_core':
            module.apply(init_weights)
    model = model.to(device)
    total_p, train_p = count_params(model)
    print(f'[online] mode={args.mode} total={total_p:,} trainable={train_p:,}')

    tok = model.esm_core.tokenize
    train_loader = OnlineTokenLoader(data, train_idx, scaler, seq_map, tok, tnames, zero_maccs)
    val_loader = OnlineTokenLoader(data, val_idx, scaler, seq_map, tok, tnames, zero_maccs)
    test_loader = OnlineTokenLoader(data, test_idx, scaler, seq_map, tok, tnames, zero_maccs)

    cfg = TrainingConfig()
    cfg.BATCH_SIZE = batch
    cfg.EPOCHS = epochs
    cfg.INNER_STEPS = args.inner_steps
    cfg.USE_COMPILE = False
    # 在线累加 task delta：峰值显存 O(参数量) 而非 O(任务数×参数量)，与 stock 数学等价
    cfg.ONLINE_DELTA_ACCUM = True
    scale = 0.1 if args.mode == 'partial' else 0.01
    trainer = ESM2FinetuneTrainer(model, train_loader, val_loader, test_loader, scaler,
                                  output_dir=str(run_dir), config=cfg, esm_lr_scale=scale)
    history = trainer.train()
    test_metrics = trainer.evaluate_test()

    results = {
        'training_history': history,
        'test_metrics': test_metrics,
        'config': {'esm2_mode': args.mode, 'esm_lr_scale': scale, 'dataset': args.dataset,
                   'epochs': epochs, 'batch_size': batch, 'inner_steps': args.inner_steps,
                   'ablation': recipe['ablation'], 'seed': args.seed,
                   'total_params': total_p, 'trainable_params': train_p},
        'best_val_r2': trainer.best_val_r2,
    }
    with open(run_dir / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n[E4 {args.mode}] R2={test_metrics["R2"]:.4f} EF@1%={test_metrics["EF@1%"]:.3f} '
          f'ECE={test_metrics["ECE"]:.4f} trainable={train_p:,}')
    del data
    gc.collect()


if __name__ == '__main__':
    main()
