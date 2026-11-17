# -*- coding: utf-8 -*-
"""
面板统一口径训练：BindingDB / Davis（morgan_descriptors）
========================================================
特征：Morgan 2048 + 10 理化描述符 + ESM2-128，屏蔽 MACCS。
模型：
  MLP         : run_baseline_mlp.py，5 次独立运行（batch 2048 / epochs 220）
  Transformer : run_module_ablation.supervised_train 的 no_reptile 配置，
                5 seeds（11/22/33/44/55），AdamW 1.5e-4 / batch 2048 / epochs 300
  Reptile     : run_reptile_transformer.py，5 seeds，
                batch 2048 / epochs 220 / inner_steps 3

数据集：
  bindingdb : test 7,087 个唯一配对 / 112 靶点（bindingdb_panel_preprocessed）
  davis     : test 4,290 对 / 65 靶点（davis_panel_preprocessed）

用法（PyCharm 终端，工作目录 D:\\lht）：
    python prepare_bindingdb_panel.py
    python prepare_davis_panel.py
    python run_panel_morgan.py --dataset bindingdb --stage all
    python run_panel_morgan.py --dataset davis --stage all
    # 也可分阶段：
    python run_panel_morgan.py --dataset davis --stage precompute
    python run_panel_morgan.py --dataset davis --stage mlp
    python run_panel_morgan.py --dataset davis --stage transformer
    python run_panel_morgan.py --dataset davis --stage reptile
    python run_panel_morgan.py --dataset davis --stage reptile --parallel 2
    python run_panel_morgan.py --dataset davis --stage summary

输出：
    D:\\lht\\{dataset}_panel_morgan\\
      _features\\precomputed_features.npz
      mlp\\mlp_run{1..5}\\predictions.npz
      transformer\\no_reptile\\seed_{11..55}\\predictions.npz
      reptile\\seed_{11..55}\\predictions_raw.npz
      summary.csv
"""
import argparse
import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"D:\lht")
sys.path.insert(0, str(ROOT))

SEEDS = [11, 22, 33, 44, 55]

DATASETS = {
    "bindingdb": dict(
        data=ROOT / "bindingdb_panel_preprocessed",
        out=ROOT / "bindingdb_panel_morgan",
        expected_n=7087,
        expected_targets=112,
        ef1="macro",
        ef5="macro",
    ),
    "davis": dict(
        data=ROOT / "davis_panel_preprocessed",
        out=ROOT / "davis_panel_morgan",
        expected_n=4290,
        expected_targets=65,
        ef1="pooled",
        ef5="macro",
    ),
}


def feat_path(cfg):
    return cfg["out"] / "_features" / "precomputed_features.npz"


# ---------------------------------------------------------------- 通用
def env_for_run(cfg):
    env = dict(os.environ)
    env["PREPROCESSED_DIR"] = str(cfg["data"])
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["ESM2_USE_GPU"] = "1"
    return env


def run_stream(cmd, log_path, cfg):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("$", " ".join(str(c) for c in cmd))
    print("=" * 80, flush=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(ROOT), env=env_for_run(cfg),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end="")
            log.write(line)
        p.wait()
    return p.returncode


def link_features(cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "precomputed_features.npz"
    if dst.exists():
        return dst
    src = feat_path(cfg)
    if not src.exists():
        raise FileNotFoundError(f"缺少共享特征文件 {src}，先跑 --stage precompute")
    try:
        os.link(src, dst)
        print(f"[link] {dst}")
    except OSError:
        print(f"[copy] {dst}（可能较慢）")
        shutil.copyfile(src, dst)
    return dst


def load_feature_index(cfg):
    fp = feat_path(cfg)
    d = np.load(fp, mmap_mode="r", allow_pickle=True)
    splits = d["splits"][:]
    names = d["target_names"][:]
    te = np.where(splits == "test")[0]
    if len(te) != cfg["expected_n"]:
        raise RuntimeError(f"test N={len(te)} != {cfg['expected_n']}")
    n_targets = len(np.unique(names[te]))
    if n_targets != cfg["expected_targets"]:
        raise RuntimeError(f"test targets={n_targets} != {cfg['expected_targets']}")
    return (d, splits, names,
            np.where(splits == "train")[0],
            np.where(splits == "val")[0],
            te)


# ---------------------------------------------------------------- precompute
def stage_precompute(cfg, force=False):
    fp = feat_path(cfg)
    if fp.exists() and fp.stat().st_size > 1_000_000 and not force:
        print(f"[skip] 共享特征已存在: {fp} ({fp.stat().st_size / 1e6:.0f} MB)")
    else:
        cmd = [sys.executable, ROOT / "run_reptile_transformer.py",
               "--data_dir", cfg["data"], "--output_dir", fp.parent,
               "--precompute_only", "--ablation", "morgan_descriptors", "--gpu", "0"]
        rc = run_stream(cmd, cfg["out"] / "logs" / "precompute.log", cfg)
        if rc != 0:
            raise SystemExit(f"特征预计算失败 returncode={rc}")

    d, splits, names, tr, va, te = load_feature_index(cfg)
    print(f"[ok] precomputed features: train={len(tr)} val={len(va)} test={len(te)} "
          f"targets={len(np.unique(names[te]))}")
    del d
    gc.collect()


# ---------------------------------------------------------------- MLP
def stage_mlp(cfg, force=False):
    for i in range(1, 6):
        out = cfg["out"] / "mlp" / f"mlp_run{i}"
        link_features(cfg, out)
        done = out / "final_results.json"
        if done.exists() and not force:
            print(f"[skip] {out} 已完成")
            continue
        cmd = [sys.executable, ROOT / "run_baseline_mlp.py",
               "--data_dir", cfg["data"], "--output_dir", out,
               "--no_precompute", "--ablation", "morgan_descriptors", "--gpu", "0"]
        if force:
            cmd.append("--force")
        rc = run_stream(cmd, cfg["out"] / "logs" / f"mlp_run{i}.log", cfg)
        if rc != 0:
            raise SystemExit(f"MLP run{i} 失败 returncode={rc}")


# ---------------------------------------------------------------- Transformer (supervised composite)
def stage_transformer(cfg, force=False):
    import run_module_ablation as rma

    rma.OUT = cfg["out"] / "transformer"
    rma.FORCE = force

    d, splits, names, tr, va, te = load_feature_index(cfg)
    scaler = rma.TargetScaler()
    scaler.fit(d["y"][tr], names[tr])
    rma.SCALER = scaler

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data = rma.GlobalLoader(d, tr, scaler)
    val_data = rma.GlobalLoader(d, va, scaler)
    test_data = rma.GlobalLoader(d, te, scaler)
    print("[transformer] preloading features to GPU ...")
    train_data.preload_gpu(device)
    val_data.preload_gpu(device)
    test_data.preload_gpu(device)

    for seed in SEEDS:
        rma.supervised_train(
            "no_reptile", rma.VARIANTS["no_reptile"], seed,
            train_data, val_data, test_data, names[te],
            epochs=300, batch_size=2048, lr=1.5e-4, patience=50, warmup=30)

    del train_data, val_data, test_data, d
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- Reptile
def _launch_reptile(cfg, seed, out, force):
    cmd = [sys.executable, ROOT / "run_reptile_transformer.py",
           "--data_dir", cfg["data"], "--output_dir", out, "--no_precompute",
           "--batch_size", "2048", "--epochs", "220", "--inner_steps", "3",
           "--ablation", "morgan_descriptors", "--gpu", "0", "--seed", str(seed),
           "--online_delta_accum"]
    if (out / "checkpoint_latest.pt").exists() and not force:
        cmd.append("--resume")
    log = cfg["out"] / "logs" / f"reptile_seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "w", encoding="utf-8", errors="replace")
    print("\n" + "=" * 80)
    print(f"[start] seed {seed}  log={log}")
    print("$", " ".join(str(c) for c in cmd))
    print("=" * 80, flush=True)
    p = subprocess.Popen([str(c) for c in cmd], cwd=str(ROOT), env=env_for_run(cfg),
                         stdout=f, stderr=subprocess.STDOUT, text=True)
    return p, f, log


def _post_reptile(cfg, seed, out, force, rc):
    # morgan_descriptors：推理时必须屏蔽 MACCS
    rc.make_raw_chembl_predictions(out, mask_maccs=True, force=force)
    p = out / "predictions_raw.npz"
    if not p.exists():
        raise RuntimeError(f"seed {seed} 缺少 predictions_raw.npz")
    if len(np.load(p)["y_true"]) != cfg["expected_n"]:
        raise RuntimeError(f"seed {seed} N != {cfg['expected_n']}")


def stage_reptile(cfg, force=False, seeds=None, parallel=1):
    import recompute_si_tables as rc

    seeds = list(seeds) if seeds else list(SEEDS)
    parallel = max(1, int(parallel))
    pending = []

    for seed in seeds:
        out = cfg["out"] / "reptile" / f"seed_{seed}"
        link_features(cfg, out)
        if force:
            for fn in ["checkpoint_latest.pt", "best_model.pt", "final_results.json",
                       "predictions.npz", "predictions_raw.npz"]:
                p = out / fn
                if p.exists():
                    p.unlink()
        done = (out / "final_results.json").exists()
        if done and not force:
            print(f"[skip] seed {seed} 已完成")
            if not (out / "predictions_raw.npz").exists() or force:
                _post_reptile(cfg, seed, out, force, rc)
        else:
            pending.append((seed, out))

    if not pending:
        return

    # 同卡并行：每个 seed 约 3–4GB，2 个 seed 峰值约 10GB，GPU 利用率可到 90%
    for i in range(0, len(pending), parallel):
        batch = pending[i:i + parallel]
        if len(batch) == 1:
            seed, out = batch[0]
            cmd = [sys.executable, ROOT / "run_reptile_transformer.py",
                   "--data_dir", cfg["data"], "--output_dir", out, "--no_precompute",
                   "--batch_size", "2048", "--epochs", "220", "--inner_steps", "3",
                   "--ablation", "morgan_descriptors", "--gpu", "0", "--seed", str(seed),
                   "--online_delta_accum"]
            if (out / "checkpoint_latest.pt").exists() and not force:
                cmd.append("--resume")
            rc_ = run_stream(cmd, cfg["out"] / "logs" / f"reptile_seed_{seed}.log", cfg)
            if rc_ != 0:
                raise SystemExit(f"Reptile seed {seed} 失败 returncode={rc_}")
        else:
            procs = []
            for seed, out in batch:
                procs.append((seed, out) + _launch_reptile(cfg, seed, out, force))
            for seed, out, p, f, log in procs:
                rc_ = p.wait()
                f.close()
                print(f"[done] seed {seed} returncode={rc_} log={log}", flush=True)
                if rc_ != 0:
                    raise SystemExit(f"Reptile seed {seed} 失败 returncode={rc_}")
        for seed, out in batch:
            _post_reptile(cfg, seed, out, force, rc)
            print(f"[post] seed {seed} predictions_raw ok", flush=True)


# ---------------------------------------------------------------- summary
def evaluate_pred(cfg, yt, yp, targets, feat_y):
    if len(yt) != cfg["expected_n"]:
        raise RuntimeError(f"N={len(yt)} != {cfg['expected_n']}")
    max_diff = float(np.max(np.abs(yt.astype(np.float64) - feat_y.astype(np.float64))))
    if max_diff > 1e-3:
        raise RuntimeError(f"y_true 与特征 test 顺序不一致 max_diff={max_diff}")
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    ss_res = float(np.sum((yt - yp) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    from recompute_s18_s19 import ef_at, macro_ef
    if cfg["ef1"] == "pooled":
        ef1, used1, total = ef_at(yt, yp, 1), 1, 1
    else:
        ef1, used1, total = macro_ef(yt, yp, targets, 1, 100)
    ef5, used5, _ = macro_ef(yt, yp, targets, 5, 20)
    return dict(R2=r2, RMSE=rmse, EF1=ef1, EF5=ef5,
                n_used1=used1, n_used5=used5, n_total=total,
                max_label_diff=max_diff)


def stage_summary(cfg):
    d, splits, names, tr, va, te = load_feature_index(cfg)
    targets = names[te]
    feat_y = np.asarray(d["y"][te], dtype=np.float64)

    jobs = []
    for i in range(1, 6):
        jobs.append(("MLP", f"run{i}",
                     cfg["out"] / "mlp" / f"mlp_run{i}" / "predictions.npz"))
    for seed in SEEDS:
        jobs.append(("Transformer", f"seed_{seed}",
                     cfg["out"] / "transformer" / "no_reptile" / f"seed_{seed}" / "predictions.npz"))
    for seed in SEEDS:
        jobs.append(("Reptile-Transformer", f"seed_{seed}",
                     cfg["out"] / "reptile" / f"seed_{seed}" / "predictions_raw.npz"))

    rows = []
    for family, run_name, path in jobs:
        if not path.exists():
            print(f"[missing] {family} {run_name}: {path}")
            continue
        p = np.load(path)
        m = evaluate_pred(cfg, p["y_true"].astype(np.float64),
                          p["y_pred"].astype(np.float64), targets, feat_y)
        rows.append(dict(family=family, run=run_name, **m))
        print(f"[{family} {run_name}] R2={m['R2']:.4f} RMSE={m['RMSE']:.4f} "
              f"EF@1%={m['EF1']:.3f} EF@5%={m['EF5']:.3f}")

    if not rows:
        raise RuntimeError("没有任何预测文件可汇总")
    df = pd.DataFrame(rows)
    csv = cfg["out"] / "summary.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    print("\n[saved]", csv)

    ef1_kind = "pooled" if cfg["ef1"] == "pooled" else "macro (n>=100)"
    lines = [f"# {cfg['out'].name}（morgan_descriptors）", "",
             f"test N={cfg['expected_n']}，targets={cfg['expected_targets']}。", "",
             f"EF@1% 聚合：{ef1_kind}；EF@5% 聚合：macro (n>=20)。", ""]
    for fam, g in df.groupby("family"):
        lines.append(f"## {fam} (n={len(g)})")
        for k in ["R2", "RMSE", "EF1", "EF5"]:
            val = g[k].astype(float)
            sd = val.std(ddof=1) if len(val) > 1 else np.nan
            lines.append(f"- {k}: {val.mean():.4f} ± {sd:.4f}")
        lines.append(f"- EF@1% coverage: {int(g['n_used1'].iloc[0])}/{int(g['n_total'].iloc[0])}")
        lines.append("")
    (cfg["out"] / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[saved]", cfg["out"] / "REPORT.md")
    print(df.groupby("family")[["R2", "RMSE", "EF1", "EF5"]].agg(["mean", "std"]).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bindingdb",
                    choices=list(DATASETS.keys()) + ["all"])
    ap.add_argument("--stage", default="all",
                    choices=["precompute", "mlp", "transformer", "reptile", "summary", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                    help="Reptile / Transformer seeds，逗号分隔")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Reptile 同卡并行 seed 数；每个约 3–4GB，建议 2")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in names:
        cfg = DATASETS[name]
        if not cfg["data"].exists():
            raise FileNotFoundError(
                f"{cfg['data']} 不存在，请先运行对应的 prepare_*_panel.py")
        cfg["out"].mkdir(parents=True, exist_ok=True)
        print("\n" + "#" * 80)
        print(f"# dataset = {name}  out = {cfg['out']}")
        print("#" * 80, flush=True)

        if args.stage in ("precompute", "all"):
            stage_precompute(cfg, force=args.force)
        if args.stage in ("mlp", "all"):
            stage_mlp(cfg, force=args.force)
        if args.stage in ("transformer", "all"):
            stage_transformer(cfg, force=args.force)
        if args.stage in ("reptile", "all"):
            stage_reptile(cfg, force=args.force, seeds=seeds, parallel=args.parallel)
        if args.stage in ("summary", "all"):
            stage_summary(cfg)


if __name__ == "__main__":
    main()
