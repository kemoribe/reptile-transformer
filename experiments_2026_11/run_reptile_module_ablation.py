# -*- coding: utf-8 -*-
"""
完整 Reptile 模型的模块删除消融
================================
与 run_module_ablation.py 的区别：
- run_module_ablation.py 的 no_ranking/no_contrastive/... 是 **supervised** 族（从 no_reptile 删除）；
- 本脚本从 **full Reptile** 删除模块，用于回答“Reptile 模型内部各模块贡献”。

数据：KIBA 18,478 测试对，morgan_descriptors（屏蔽 MACCS）
配方：batch 2048, epochs 220, inner_steps 3, 与 e5/full Reptile 一致
Seeds：11/22/33
变体：
  reptile_no_ranking       ranking_weight=0
  reptile_no_contrastive   contrastive_weight=0
  reptile_no_consistency   consistency_weight=0
  reptile_no_bidir_fusion  fusion_mode=concat, consistency_weight=0
  full_reptile             可选（--include-full），用于补齐 walltime

用法（PyCharm 终端，工作目录 D:\\lht）：
    python run_reptile_module_ablation.py --pilot          # 只跑 seed 11，约 1.2h
    python run_reptile_module_ablation.py                  # 4 变体 x 3 seeds，约 3.6h
    python run_reptile_module_ablation.py --include-full    # 额外 3 个 full seeds，约 4.5h
    python run_reptile_module_ablation.py --force
输出：
    D:\\lht\\reptile_module_ablation\\
      summary.csv
      per_seed.csv
      REPORT.md
      {variant}\\seed_{seed}\\
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\lht")
DATA = ROOT / "3_kiba_preprocessed"
FEATURE_SRC = ROOT / "transformer_ablation_output_kiba" / "precomputed_features.npz"
OUT_ROOT = ROOT / "reptile_module_ablation"
SEEDS = [11, 22, 33]

VARIANTS = {
    "reptile_no_ranking":     dict(rank=0.0, con=0.05, cons=0.10, fusion="bidirectional"),
    "reptile_no_contrastive": dict(rank=0.3, con=0.0,  cons=0.10, fusion="bidirectional"),
    "reptile_no_consistency": dict(rank=0.3, con=0.05, cons=0.0,  fusion="bidirectional"),
    "reptile_no_bidir_fusion": dict(rank=0.3, con=0.05, cons=0.0, fusion="concat"),
    "full_reptile":           dict(rank=0.3, con=0.05, cons=0.10, fusion="bidirectional"),
}

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
    try:
        os.link(FEATURE_SRC, dst)
        print(f"[link] {dst}")
    except OSError:
        print(f"[copy] {dst} (5GB, 可能较慢)")
        shutil.copyfile(FEATURE_SRC, dst)


def run_one(variant: str, cfg: dict, seed: int, force: bool):
    out = OUT_ROOT / variant / f"seed_{seed}"
    link_features(out)
    final = out / "final_results.json"
    if force:
        for f in ["checkpoint_latest.pt", "best_model.pt", "final_results.json", "predictions.npz"]:
            p = out / f
            if p.exists():
                p.unlink()
    if final.exists() and not force:
        print(f"[skip] {variant} seed {seed}: {out}")
    else:
        cmd = [sys.executable, ROOT / "run_reptile_transformer.py",
               "--data_dir", DATA,
               "--output_dir", out,
               "--no_precompute",
               "--batch_size", "2048",
               "--epochs", "220",
               "--inner_steps", "3",
               "--ablation", "morgan_descriptors",
               "--fusion_mode", cfg["fusion"],
               "--ranking_weight", str(cfg["rank"]),
               "--contrastive_weight", str(cfg["con"]),
               "--consistency_weight", str(cfg["cons"]),
               "--gpu", "0",
               "--seed", str(seed)]
        if (out / "checkpoint_latest.pt").exists() and not force:
            cmd.append("--resume")
        print("\n" + "=" * 80)
        print("$", " ".join(str(c) for c in cmd))
        print("=" * 80, flush=True)
        t0 = time.time()
        r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env_for_run())
        wall = time.time() - t0
        if r.returncode != 0:
            raise SystemExit(f"{variant} seed {seed} 训练失败 returncode={r.returncode}")
        (out / "walltime_sec.txt").write_text(f"{wall:.1f}\n", encoding="utf-8")

    wall = None
    wp = out / "walltime_sec.txt"
    if wp.exists():
        wall = float(wp.read_text(encoding="utf-8").strip())
    row = rc.row_from_run(out, "kiba")
    row.update({"variant": variant, "seed": seed, "walltime_s": wall})
    print(f"[{variant} seed {seed}] R2={row['R2']:.4f} RMSE={row['RMSE']:.4f} "
          f"EF@1%={row['EF']:.3f} AUPR={row['AUPR']:.4f} ECE={row['ECE']:.4f} "
          f"wall={wall if wall is not None else 'NA'}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join([k for k in VARIANTS if k != "full_reptile"]))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--pilot", action="store_true", help="只跑 seed 11")
    ap.add_argument("--include-full", action="store_true", help="额外训练 full_reptile 3 seeds")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.include_full and "full_reptile" not in variants:
        variants.append("full_reptile")
    seeds = [11] if args.pilot else [int(s) for s in args.seeds.split(",") if s.strip()]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        for seed in seeds:
            rows.append(run_one(variant, VARIANTS[variant], seed, args.force))

    import pandas as pd
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT_ROOT / "per_seed.csv", index=False, encoding="utf-8-sig")

    summary = []
    for variant in per_seed["variant"].unique():
        s = per_seed[per_seed["variant"] == variant]
        row = {"variant": variant, "n_seeds": len(s)}
        for k in ["R2", "RMSE", "EF", "AUPR", "ECE"]:
            row[f"{k}_mean"] = float(s[k].mean())
            row[f"{k}_sd"] = float(s[k].std(ddof=1)) if len(s) > 1 else np.nan
        wt = s["walltime_s"].dropna()
        row["walltime_mean_s"] = float(wt.mean()) if len(wt) else np.nan
        summary.append(row)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT_ROOT / "summary.csv", index=False, encoding="utf-8-sig")

    lines = ["# 完整 Reptile 模块删除消融（KIBA 18,478）", ""]
    lines.append("从 full Reptile 删除模块；batch 2048, epochs 220, inner_steps 3, morgan_descriptors。")
    lines.append("")
    for _, r in summ.iterrows():
        lines.append(f"## {r['variant']} (n={int(r['n_seeds'])})")
        lines.append(f"- R2: {r['R2_mean']:.4f} ± {r['R2_sd']:.4f}")
        lines.append(f"- RMSE: {r['RMSE_mean']:.4f} ± {r['RMSE_sd']:.4f}")
        lines.append(f"- AUPR: {r['AUPR_mean']:.4f} ± {r['AUPR_sd']:.4f}")
        lines.append(f"- EF@1%: {r['EF_mean']:.3f} ± {r['EF_sd']:.3f}")
        lines.append(f"- ECE: {r['ECE_mean']:.4f} ± {r['ECE_sd']:.4f}")
        lines.append(f"- walltime: {r['walltime_mean_s']:.1f}s")
        lines.append("")
    (OUT_ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + summ.to_string(index=False))
    print("[saved]", OUT_ROOT / "summary.csv")


if __name__ == "__main__":
    main()
