# -*- coding: utf-8 -*-
"""
6.1 预算匹配监督对照：direct_supervised
========================================
目的：隔离“Reptile 参数插值/外循环”本身，而不是训练轮数、样本访问量或模型选择差异。

做法：
- 数据、特征、标签、骨干、融合、损失权重、按靶点采样、batch、inner_steps、
  epochs、验证频率、最佳检查点规则全部与 full Reptile 一致；
- 唯一改变：每个靶点内循环结束后**保留更新后的参数**，不做 Reptile 的
  “恢复初始参数 → 计算 delta → 外循环插值”；
- 因此内循环梯度更新次数、样本访问量与 full Reptile 相同，meta 更新为 0。

用法（PyCharm 终端，工作目录 D:\\lht）：
    python run_budget_matched_control.py --pilot           # 只跑 seed 11
    python run_budget_matched_control.py                   # 5 seeds 串行
    python run_budget_matched_control.py --parallel 2      # 同卡并行 2 seeds
    python run_budget_matched_control.py --force

输出：
    D:\\lht\\control_budget_matched\\seed_{11..55}\\
      final_results.json / predictions.npz / training_budget.json
    D:\\lht\\control_budget_matched\\summary.csv
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\lht")
DATA = ROOT / "3_kiba_preprocessed"
FEATURE_SRC = ROOT / "transformer_ablation_output_kiba" / "precomputed_features.npz"
OUT_ROOT = ROOT / "control_budget_matched"
SEEDS = [11, 22, 33, 44, 55]

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
        print(f"[copy] {dst} (1.1GB)")
        shutil.copyfile(FEATURE_SRC, dst)


def run_stream(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("$", " ".join(str(c) for c in cmd))
    print("=" * 80, flush=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(ROOT), env=env_for_run(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end="")
            log.write(line)
        p.wait()
    return p.returncode


def build_cmd(seed, out, epochs, force):
    cmd = [sys.executable, ROOT / "run_reptile_transformer.py",
           "--data_dir", DATA, "--output_dir", out, "--no_precompute",
           "--batch_size", "2048", "--epochs", str(epochs), "--inner_steps", "3",
           "--ablation", "morgan_descriptors", "--gpu", "0", "--seed", str(seed),
           "--control_mode", "direct_supervised"]
    if (out / "checkpoint_latest.pt").exists() and not force:
        cmd.append("--resume")
    return cmd


def prepare(seed, force):
    out = OUT_ROOT / f"seed_{seed}"
    link_features(out)
    if force:
        for f in ["checkpoint_latest.pt", "checkpoint_best.pt", "best_model.pt",
                  "final_results.json", "predictions.npz", "per_target_results.json",
                  "training_budget.json"]:
            p = out / f
            if p.exists():
                p.unlink()
    return out


def launch(seed, out, epochs, force):
    log = OUT_ROOT / "logs" / f"control_seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "w", encoding="utf-8", errors="replace")
    print(f"\n[start] seed {seed}  log={log}", flush=True)
    p = subprocess.Popen(
        [str(c) for c in build_cmd(seed, out, epochs, force)],
        cwd=str(ROOT), env=env_for_run(),
        stdout=f, stderr=subprocess.STDOUT, text=True)
    return p, f, log


def row_from_out(seed, out):
    row = rc.row_from_run(out, "kiba")
    row["seed"] = seed
    budget_path = out / "training_budget.json"
    if budget_path.exists():
        b = json.loads(budget_path.read_text(encoding="utf-8"))
        for k in ["epochs_run", "inner_updates", "meta_updates", "samples_seen",
                  "best_epoch", "walltime_s"]:
            row[k] = b.get(k)
    print(f"[seed {seed}] R2={row['R2']:.4f} RMSE={row['RMSE']:.4f} "
          f"EF@1%={row['EF']:.3f} AUPR={row['AUPR']:.4f} ECE={row['ECE']:.4f}")
    return row


def main():
    global OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="只跑 seed 11")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--parallel", type=int, default=1,
                    help="同卡并行 seed 数；每个约 1.3GB，建议 2")
    ap.add_argument("--output_root", default=None, help="仅用于短测/重定向")
    args = ap.parse_args()

    if not FEATURE_SRC.exists():
        raise FileNotFoundError(FEATURE_SRC)
    if args.output_root:
        OUT_ROOT = Path(args.output_root)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    seeds = [11] if args.pilot else [int(s) for s in args.seeds.split(",") if s.strip()]
    parallel = max(1, int(args.parallel))
    rows, pending = [], []

    for seed in seeds:
        out = prepare(seed, args.force)
        if (out / "final_results.json").exists() and not args.force:
            print(f"[skip] seed {seed} 已完成")
            rows.append(row_from_out(seed, out))
        else:
            pending.append((seed, out))

    for i in range(0, len(pending), parallel):
        batch = pending[i:i + parallel]
        if len(batch) == 1:
            seed, out = batch[0]
            rc_ = run_stream(build_cmd(seed, out, args.epochs, args.force),
                             OUT_ROOT / "logs" / f"control_seed_{seed}.log")
            if rc_ != 0:
                raise SystemExit(f"control seed {seed} 失败 returncode={rc_}")
        else:
            procs = []
            for seed, out in batch:
                procs.append((seed, out) + launch(seed, out, args.epochs, args.force))
            for seed, out, p, f, log in procs:
                rc_ = p.wait()
                f.close()
                print(f"[done] seed {seed} returncode={rc_} log={log}", flush=True)
                if rc_ != 0:
                    raise SystemExit(f"control seed {seed} 失败 returncode={rc_}")
        for seed, out in batch:
            rows.append(row_from_out(seed, out))

    rows = sorted(rows, key=lambda r: r["seed"])
    keys = ["R2", "RMSE", "EF", "AUPR", "ECE"]
    print("\n" + "=" * 80)
    print("direct_supervised（预算匹配对照）")
    for k in keys:
        vals = [r[k] for r in rows]
        print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")
    print("=" * 80)

    csv = OUT_ROOT / "summary.csv"
    cols = ["seed", "R2", "RMSE", "EF", "AUPR", "ECE",
            "epochs_run", "inner_updates", "meta_updates", "samples_seen",
            "best_epoch", "walltime_s"]
    pd.DataFrame(rows).reindex(columns=cols).to_csv(csv, index=False, encoding="utf-8-sig")
    print("[saved]", csv)


if __name__ == "__main__":
    main()
