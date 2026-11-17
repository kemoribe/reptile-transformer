# -*- coding: utf-8 -*-
"""
模块删除消融：Reptile / ranking / contrastive / consistency / 双向融合
=====================================================================
数据：KIBA 18,478 测试对，morgan_descriptors（屏蔽 MACCS）
训练：同 ReptileTransformer 骨干、同 global target normalization、同 composite loss 权重
      ranking=0.3, contrastive=0.05, consistency=0.10
对照：去掉 Reptile（标准监督训练），以及逐个去掉 ranking/contrastive/consistency/双向融合
Seeds：11/22/33，mean ± SD

用法（PyCharm 终端，工作目录 D:\\lht）：
    python run_module_ablation.py
    python run_module_ablation.py --force
    python run_module_ablation.py --variants no_reptile,no_ranking,no_contrastive,no_consistency,no_bidir_fusion,mse_only
输出：
    D:\\lht\\module_ablation\\
      per_seed.csv
      summary.csv
      REPORT.md
      {variant}\\seed_{seed}\\predictions.npz
"""
import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(r"D:\lht")
sys.path.insert(0, str(ROOT))
from data_preprocessing import TargetScaler  # noqa
from reptile_transformer_model import ReptileTransformer, init_weights, count_params  # noqa
import run_table2_kiba_baselines as common  # noqa

DATA = ROOT / "3_kiba_preprocessed"
FEATURE = ROOT / "transformer_ablation_output_kiba" / "precomputed_features.npz"
OUT = ROOT / "module_ablation"
SEEDS = [11, 22, 33]
EXPECTED_N = 18478

VARIANTS = {
    "full_reptile":     {"kind": "reptile",   "rank": 0.3, "con": 0.05, "cons": 0.10, "fusion": "bidirectional"},
    "no_reptile":       {"kind": "supervised","rank": 0.3, "con": 0.05, "cons": 0.10, "fusion": "bidirectional"},
    "no_ranking":       {"kind": "supervised","rank": 0.0, "con": 0.05, "cons": 0.10, "fusion": "bidirectional"},
    "no_contrastive":   {"kind": "supervised","rank": 0.3, "con": 0.0,  "cons": 0.10, "fusion": "bidirectional"},
    "no_consistency":   {"kind": "supervised","rank": 0.3, "con": 0.05, "cons": 0.0,  "fusion": "bidirectional"},
    "no_bidir_fusion":  {"kind": "supervised","rank": 0.3, "con": 0.05, "cons": 0.0,  "fusion": "concat"},
    "mse_only":         {"kind": "supervised","rank": 0.0, "con": 0.0,  "cons": 0.0,  "fusion": "bidirectional"},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class GlobalLoader:
    """与 Reptile 训练一致的 global target normalization；支持一次性预加载到 GPU。"""
    def __init__(self, data, indices, scaler):
        self.data = data
        self.indices = np.asarray(indices)
        self.scaler = scaler
        self._gpu_data = None

    def __len__(self):
        return len(self.indices)

    def get_batch(self, batch_indices):
        if self._gpu_data is not None:
            idx = torch.from_numpy(np.asarray(batch_indices)).long().to(self._gpu_data["y"].device)
            return {k: v[idx] for k, v in self._gpu_data.items()}
        real = self.indices[np.asarray(batch_indices)]
        y_raw = self.data["y"][real].astype(np.float32)
        y_norm = (y_raw - self.scaler.global_mean) / self.scaler.global_std
        return {
            "morgan": np.asarray(self.data["morgan"][real], dtype=np.float32),
            "maccs": np.asarray(self.data["maccs"][real], dtype=np.float32),
            "descriptors": np.asarray(self.data["descriptors"][real], dtype=np.float32),
            "protein": np.asarray(self.data["protein"][real], dtype=np.float32),
            "y": y_raw,
            "y_norm": y_norm.astype(np.float32),
        }

    def preload_gpu(self, device):
        n = len(self)
        b = self.get_batch(np.arange(n))
        self._gpu_data = {
            "morgan": torch.from_numpy(b["morgan"]).to(device),
            "maccs": torch.zeros_like(torch.from_numpy(b["maccs"])).to(device),  # morgan_descriptors: 屏蔽 MACCS
            "descriptors": torch.from_numpy(b["descriptors"]).to(device),
            "protein": torch.from_numpy(b["protein"]).to(device),
            "y": torch.from_numpy(b["y"]).to(device),
            "y_norm": torch.from_numpy(b["y_norm"]).to(device),
        }
        print(f"   preload {n} samples -> GPU")


def supervised_train(variant, cfg, seed, train_data, val_data, test_data, test_targets,
                     epochs=300, batch_size=2048, lr=1.5e-4, patience=50, warmup=30):
    out = OUT / variant / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.npz"
    if pred_path.exists() and not FORCE:
        yt, yp = common.load_pred(pred_path)
        return evaluate(yt, yp, test_targets, variant, seed, out, walltime=None)

    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ReptileTransformer(fusion_mode=cfg["fusion"])
    model.apply(init_weights)
    model.to(device)
    total, trainable = count_params(model)
    print(f"[{variant} seed {seed}] params={total:,} fusion={cfg['fusion']} "
          f"rank={cfg['rank']} con={cfg['con']} cons={cfg['cons']}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup, 1), eta_min=1e-6)
    amp_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best_val = float("inf")
    bad = 0
    best_path = out / "best_model.pt"
    t0 = __import__("time").time()

    for epoch in range(epochs):
        model.train()
        if epoch < warmup:
            for pg in optimizer.param_groups:
                pg["lr"] = lr * (epoch + 1) / warmup
        elif epoch == warmup:
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        perm = np.random.permutation(len(train_data))
        for i in range(0, len(train_data), batch_size):
            bi = perm[i:i + batch_size]
            b = train_data.get_batch(bi)
            morgan, maccs = b["morgan"], b["maccs"]
            desc, protein, y = b["descriptors"], b["protein"], b["y_norm"]
            with torch.amp.autocast("cuda") if device.type == "cuda" else torch.no_grad():
                preds, cons_loss, con_loss = model(morgan, maccs, desc, protein)
                loss = F.mse_loss(preds, y)
                if cfg["rank"] > 0 and len(bi) >= 2:
                    p = torch.clamp(F.softmax(preds, dim=0), min=1e-10)
                    t = torch.clamp(F.softmax(y, dim=0), min=1e-10)
                    loss = loss + cfg["rank"] * F.kl_div(torch.log(p), t, reduction="batchmean")
                if cfg["con"] > 0:
                    loss = loss + cfg["con"] * con_loss
                if cfg["cons"] > 0:
                    loss = loss + cfg["cons"] * cons_loss
            if amp_scaler is not None:
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # validation: MSE only
        model.eval()
        val_sum, n = 0.0, 0
        with torch.no_grad():
            for j in range(0, len(val_data), batch_size):
                b = val_data.get_batch(np.arange(j, min(j + batch_size, len(val_data))))
                with torch.amp.autocast("cuda") if device.type == "cuda" else torch.no_grad():
                    preds, _, _ = model(b["morgan"], b["maccs"], b["descriptors"], b["protein"])
                    l = F.mse_loss(preds, b["y_norm"])
                val_sum += float(l) * len(preds)
                n += len(preds)
        val_loss = val_sum / max(n, 1)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"   early stop at epoch {epoch+1}, best_val={best_val:.4f}")
                break
        if epoch >= warmup:
            scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"   epoch {epoch+1}/{epochs} val_mse={val_loss:.4f} best={best_val:.4f}")

    walltime = __import__("time").time() - t0
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for j in range(0, len(test_data), batch_size):
            b = test_data.get_batch(np.arange(j, min(j + batch_size, len(test_data))))
            with torch.amp.autocast("cuda") if device.type == "cuda" else torch.no_grad():
                preds, _, _ = model(b["morgan"], b["maccs"], b["descriptors"], b["protein"])
            ys.append(b["y"].float().cpu().numpy())
            # 使用 global inverse（与 Reptile 训练一致）
            ps.append(preds.float().cpu().numpy() * SCALER.global_std + SCALER.global_mean)
    yt = np.concatenate(ys).astype(np.float64)
    yp = np.concatenate(ps).astype(np.float64)
    np.savez(pred_path, y_true=yt, y_pred=yp)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return evaluate(yt, yp, test_targets, variant, seed, out, walltime=walltime)


def evaluate(yt, yp, targets, variant, seed, out, walltime=None):
    g = common.global_metrics(yt, yp)
    mm = common.macro_metrics(yt, yp, targets)
    row = {
        "variant": variant, "seed": seed,
        "R2": g["R2"], "RMSE": g["RMSE"], "ECE": g["ECE"],
        "AUPR": mm["AUPR"], "EF@1%": mm["EF@1%"],
        "N": len(yt), "walltime_s": walltime if walltime is not None else np.nan,
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    print(f"[{variant} seed {seed}] R2={row['R2']:.4f} RMSE={row['RMSE']:.4f} "
          f"AUPR={row['AUPR']:.4f} EF@1%={row['EF@1%']:.3f} ECE={row['ECE']:.4f}")
    return row


def full_reptile_rows(targets, seeds):
    rows = []
    for seed in seeds:
        d = ROOT / "修改supplement_output" / "revision2_experiments" / "e5_repeats" / "kiba" / f"seed_{seed}"
        yt, yp = common.load_pred(d / "predictions.npz")
        g = common.global_metrics(yt, yp)
        mm = common.reptile_macro_from_json(d / "per_target_results.json")
        rows.append({"variant": "full_reptile", "seed": seed,
                     "R2": g["R2"], "RMSE": g["RMSE"], "ECE": g["ECE"],
                     "AUPR": mm["AUPR"], "EF@1%": mm["EF@1%"],
                     "N": len(yt), "walltime_s": np.nan})
        print(f"[full_reptile seed {seed}] R2={g['R2']:.4f} RMSE={g['RMSE']:.4f}")
    return rows


def main():
    global FORCE, SCALER
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS.keys()))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    FORCE = args.force
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    data = np.load(FEATURE, mmap_mode="r", allow_pickle=True)
    splits = data["splits"][:]
    target_names = data["target_names"][:]
    train_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]
    if len(test_idx) != EXPECTED_N:
        raise RuntimeError(f"test N={len(test_idx)} != {EXPECTED_N}")
    SCALER = TargetScaler()
    SCALER.fit(data["y"][train_idx], target_names[train_idx])
    test_targets = target_names[test_idx]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data = GlobalLoader(data, train_idx, SCALER)
    val_data = GlobalLoader(data, val_idx, SCALER)
    test_data = GlobalLoader(data, test_idx, SCALER)
    print("preloading features to GPU (once, reused across all variants)...")
    train_data.preload_gpu(device)
    val_data.preload_gpu(device)
    test_data.preload_gpu(device)

    rows = []
    for variant in variants:
        cfg = VARIANTS[variant]
        if cfg["kind"] == "reptile":
            rows.extend(full_reptile_rows(test_targets, seeds))
            continue
        for seed in seeds:
            rows.append(supervised_train(
                variant, cfg, seed, train_data, val_data, test_data, test_targets,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                patience=args.patience, warmup=args.warmup))

    import pandas as pd
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "per_seed.csv", index=False, encoding="utf-8-sig")

    summary = []
    for variant in per_seed["variant"].unique():
        s = per_seed[per_seed["variant"] == variant]
        row = {"variant": variant, "n_seeds": len(s)}
        for k in ["R2", "RMSE", "AUPR", "EF@1%", "ECE"]:
            row[f"{k}_mean"] = float(s[k].mean())
            row[f"{k}_sd"] = float(s[k].std(ddof=1)) if len(s) > 1 else np.nan
        row["walltime_mean_s"] = float(s["walltime_s"].mean()) if s["walltime_s"].notna().any() else np.nan
        summary.append(row)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")

    lines = ["# 模块删除消融（KIBA 18,478, morgan_descriptors）", ""]
    lines.append("同骨干、同 global normalization、同 loss 权重（rank=0.3, contrastive=0.05, consistency=0.10）。")
    lines.append("")
    for _, r in summ.iterrows():
        lines.append(f"## {r['variant']} (n={int(r['n_seeds'])})")
        lines.append(f"- R2: {r['R2_mean']:.4f} ± {r['R2_sd']:.4f}")
        lines.append(f"- RMSE: {r['RMSE_mean']:.4f} ± {r['RMSE_sd']:.4f}")
        lines.append(f"- AUPR: {r['AUPR_mean']:.4f} ± {r['AUPR_sd']:.4f}")
        lines.append(f"- EF@1%: {r['EF@1%_mean']:.3f} ± {r['EF@1%_sd']:.3f}")
        lines.append(f"- ECE: {r['ECE_mean']:.4f} ± {r['ECE_sd']:.4f}")
        lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + summ.to_string(index=False))
    print("[saved]", OUT / "summary.csv")


if __name__ == "__main__":
    main()
