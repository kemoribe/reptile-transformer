# -*- coding: utf-8 -*-
"""
补 ChEMBL 5-seed Reptile-Transformer（morgan_descriptors）
========================================================
配置与论文主表一致：
  data_dir = 3_all_data_chembl_targets_preprocessed
  ablation = morgan_descriptors（屏蔽 MACCS）
  batch_size = 2048, epochs = 220, inner_steps = 3
  seeds = 11, 22, 33, 44, 55
  （与已有 reptile_ablation_output_chembl/morgan_descriptors 单模型配方一致）

每个 seed 完成后：
  1. 用 best_model.pt 重新生成 raw-scale predictions_raw.npz；
  2. 用统一口径计算 R2/RMSE/ECE（全局）与 EF@1%/AUPR（per-target macro）；
  3. 输出汇总 mean ± SD。

用法（PyCharm 终端，工作目录 D:\\lht）：
    python run_chembl_reptile_morgan_5seeds.py
    python run_chembl_reptile_morgan_5seeds.py --force   # 忽略已完成 seed
输出：
    D:\\lht\\chembl_morgan_5seeds_b2048\\seed_{11,22,33,44,55}\\
    D:\\lht\\chembl_morgan_5seeds_b2048\\summary.csv
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "3_all_data_chembl_targets_preprocessed"
FEATURE_SRC = (ROOT / "修改supplement_output" / "revision2_experiments"
               / "e5_repeats" / "chembl" / "seed_11" / "precomputed_features.npz")
OUT_ROOT = ROOT / "chembl_morgan_5seeds_b2048"
SEEDS = [11, 22, 33, 44, 55]
BATCH_SIZE = 2048
EPOCHS = 220

sys.path.insert(0, str(ROOT))
import recompute_si_tables as rc  # noqa


def env_for_run():
    env = dict(os.environ)
    env["PREPROCESSED_DIR"] = str(DATA)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def link_features(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "precomputed_features.npz"
    if dst.exists():
        return
    if not FEATURE_SRC.exists():
        raise FileNotFoundError(FEATURE_SRC)
    try:
        os.link(FEATURE_SRC, dst)
        print(f"[link] {dst}")
    except OSError:
        print(f"[copy] {dst} (5GB, 可能较慢)")
        shutil.copyfile(FEATURE_SRC, dst)


def run_one(seed: int, force=False):
    out = OUT_ROOT / f"seed_{seed}"
    link_features(out)
    final = out / "final_results.json"
    if force:
        # 强制重跑时清掉旧断点，避免 --resume 接着旧 batch/epoch 配方
        for f in ["checkpoint_latest.pt", "best_model.pt", "final_results.json", "predictions.npz"]:
            p = out / f
            if p.exists():
                p.unlink()
    if final.exists() and not force:
        print(f"[skip] seed {seed} 已完成: {out}")
    else:
        cmd = [sys.executable, ROOT / "run_reptile_transformer.py",
               "--data_dir", DATA,
               "--output_dir", out,
               "--no_precompute",
               "--batch_size", str(BATCH_SIZE),
               "--epochs", str(EPOCHS),
               "--inner_steps", "3",
               "--ablation", "morgan_descriptors",
               "--gpu", "0",
               "--seed", str(seed)]
        if (out / "checkpoint_latest.pt").exists() and not force:
            cmd.append("--resume")
        print("\n" + "=" * 80)
        print("$", " ".join(str(c) for c in cmd))
        print("=" * 80, flush=True)
        r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env_for_run())
        if r.returncode != 0:
            raise SystemExit(f"seed {seed} 训练失败 returncode={r.returncode}")

    # 生成 raw-scale 预测并统一评估（morgan_descriptors 必须屏蔽 MACCS）
    rc.make_raw_chembl_predictions(out, mask_maccs=True, force=force)
    row = rc.row_from_run(out, "chembl")
    row["seed"] = seed
    print(f"[seed {seed}] R2={row['R2']:.4f} RMSE={row['RMSE']:.4f} "
          f"EF@1%={row['EF']:.3f} AUPR={row['AUPR']:.4f} ECE={row['ECE']:.4f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="忽略已完成 seed，强制重跑")
    args = ap.parse_args()

    if not DATA.exists():
        raise FileNotFoundError(DATA)
    if not FEATURE_SRC.exists():
        raise FileNotFoundError(FEATURE_SRC)

    rows = [run_one(s, force=args.force) for s in SEEDS]
    keys = ["R2", "RMSE", "EF", "AUPR", "ECE"]
    print("\n" + "=" * 80)
    print("ChEMBL morgan_descriptors, 5 seeds (raw-scale)")
    for k in keys:
        vals = [r[k] for r in rows]
        print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")
    print("=" * 80)

    csv = OUT_ROOT / "summary.csv"
    with open(csv, "w", encoding="utf-8-sig") as f:
        f.write("seed,R2,RMSE,EF@1%,AUPR,ECE\n")
        for r in rows:
            f.write(f"{r['seed']},{r['R2']:.6f},{r['RMSE']:.6f},"
                    f"{r['EF']:.6f},{r['AUPR']:.6f},{r['ECE']:.6f}\n")
    print("[saved]", csv)


if __name__ == "__main__":
    main()
