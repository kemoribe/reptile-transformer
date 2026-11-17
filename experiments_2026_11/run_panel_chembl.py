# -*- coding: utf-8 -*-
"""
ChEMBL 展示面板统一口径比较（morgan_descriptors）
=================================================
目的：在 ChEMBL 靶点冷启动划分上，用与论文一致的 morgan_descriptors 特征配置，
比较 Descriptor MLP / MSE-only / composite / Reptile，判断是否适合作为新的主展示 panel。

默认使用已有的 Reptile 单模型（reptile_ablation_output_chembl/morgan_descriptors）；
如需 5 seeds，先手动用 run_reptile_transformer.py 训练，或后续扩展。

用法（PyCharm 终端，工作目录 D:\\lht）：
    python run_panel_chembl.py
    python run_panel_chembl.py --with-mse        # 额外跑一个 MSE-only Transformer
    python run_panel_chembl.py --skip-train      # 只重算指标

输出：
    D:\\lht\\panel_screen\\chembl_morgan_descriptors\\
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "3_all_data_chembl_targets_preprocessed"
FEATURE = ROOT / "transformer_ablation_output_chembl" / "precomputed_features.npz"
OUT = ROOT / "panel_screen" / "chembl_morgan_descriptors"
COMPOSITE_DIR = ROOT / "transformer_ablation_output_chembl" / "morgan_descriptors"
REPTILE_DIR = ROOT / "reptile_ablation_output_chembl" / "morgan_descriptors"
SCALER = ROOT / "修改supplement_output" / "revision2_experiments" / "e5_repeats" / "chembl" / "seed_11" / "target_scaler.json"
EXPECTED_N = 31132


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
        return dst
    try:
        os.link(FEATURE, dst)
    except OSError:
        shutil.copyfile(FEATURE, dst)
    return dst


def run(cmd):
    print("\n" + "=" * 80)
    print("$", " ".join(str(c) for c in cmd))
    print("=" * 80, flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env_for_run())
    if r.returncode != 0:
        raise SystemExit(f"命令失败 returncode={r.returncode}")


def infer_model(model_path, out_path, mask_maccs=True):
    os.environ["PREPROCESSED_DIR"] = str(DATA)
    import torch
    from data_preprocessing import LazySubset, TargetScaler
    from reptile_transformer_model import ReptileTransformer

    data = np.load(FEATURE, mmap_mode="r", allow_pickle=True)
    splits = data["splits"][:]
    target_names = data["target_names"][:]
    test_idx = np.where(splits == "test")[0]
    if len(test_idx) != EXPECTED_N:
        raise RuntimeError(f"测试集 {len(test_idx)} != {EXPECTED_N}")
    scaler = TargetScaler()
    scaler.load(SCALER)
    test_data = LazySubset(data, test_idx, scaler)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ReptileTransformer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    ys, ps = [], []
    with torch.no_grad():
        for i in range(0, len(test_data), 1024):
            bi = np.arange(i, min(i + 1024, len(test_data)))
            b = test_data.get_batch(bi)
            morgan = torch.from_numpy(b["morgan"]).to(device)
            maccs = torch.from_numpy(b["maccs"]).to(device)
            if mask_maccs:
                maccs = torch.zeros_like(maccs)
            desc = torch.from_numpy(b["descriptors"]).to(device)
            protein = torch.from_numpy(b["protein"]).to(device)
            with torch.amp.autocast("cuda"):
                preds, _, _ = model(morgan, maccs, desc, protein)
            ys.append(np.asarray(b["y"], dtype=np.float64))
            ps.append(scaler.inverse_transform(preds.float().cpu().numpy()))
    yt = np.concatenate(ys)
    yp = np.concatenate(ps)
    np.savez(out_path, y_true=yt, y_pred=yp)
    print(f"[saved] {out_path} N={len(yt)} var={yt.var():.6f}")
    return yt, yp, target_names[test_idx]


def load_pred(path: Path):
    d = np.load(path)
    return d["y_true"].astype(np.float64), d["y_pred"].astype(np.float64)


def load_targets():
    data = np.load(FEATURE, mmap_mode="r", allow_pickle=True)
    splits = data["splits"][:]
    target_names = data["target_names"][:]
    return target_names[np.where(splits == "test")[0]]


def ece_global(yt, yp):
    n = len(yt)
    a = min(float(yt.min()), float(yp.min()))
    c = max(float(yt.max()), float(yp.max()))
    if c == a:
        return 0.0
    ytn = (yt - a) / (c - a)
    ypn = (yp - a) / (c - a)
    bins = np.digitize(ypn, np.linspace(0, 1, 11)[1:-1])
    e = 0.0
    for b in range(10):
        m = bins == b
        if m.sum() == 0:
            continue
        e += abs(float(ypn[m].mean()) - float(ytn[m].mean())) * int(m.sum()) / n
    return float(e)


def global_metrics(yt, yp):
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - np.sum((yt - yp) ** 2) / ss_tot
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return {"R2": float(r2), "RMSE": rmse, "ECE": ece_global(yt, yp)}


def ef_at(yt, yp, pct=1):
    n = len(yt)
    if n < 10:
        return np.nan
    active_count = max(5, int(n * 0.2))
    thr = np.sort(yt)[::-1][min(active_count - 1, n - 1)]
    n_top = int(n * pct / 100)
    if n_top == 0:
        return np.nan
    return float((np.sum(yt[np.argsort(yp)[::-1][:n_top]] >= thr) / n_top) / (active_count / n))


def aupr_at(yt, yp):
    n = len(yt)
    if n < 5:
        return np.nan
    n_pos = max(1, int(n * 0.2))
    thr = np.sort(yt)[::-1][min(n_pos - 1, n - 1)]
    pos = (yt >= thr).astype(int)
    n_positive = int(pos.sum())
    if n_positive == 0:
        return 0.0
    pos_sorted = pos[np.argsort(yp)[::-1]]
    tp = np.cumsum(pos_sorted)
    fp = np.cumsum(1 - pos_sorted)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / n_positive
    recall_prev = np.zeros_like(recall)
    recall_prev[1:] = recall[:-1]
    return float(np.sum((recall - recall_prev) * precision))


def macro_metrics(yt, yp, targets):
    ef, aupr = [], []
    for t in np.unique(targets):
        m = targets == t
        n = int(m.sum())
        if n >= 100:
            v = ef_at(yt[m], yp[m])
            if np.isfinite(v):
                ef.append(v)
        if n >= 5:
            v = aupr_at(yt[m], yp[m])
            if np.isfinite(v):
                aupr.append(v)
    return {"EF@1%": float(np.mean(ef)) if ef else np.nan,
            "AUPR": float(np.mean(aupr)) if aupr else np.nan,
            "EF_targets": len(ef), "AUPR_targets": len(aupr)}


def fmt(v, nd=4):
    return f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-mse", action="store_true", help="额外训练一个 MSE-only Transformer")
    ap.add_argument("--skip-train", action="store_true", help="只重算指标")
    ap.add_argument("--force", action="store_true", help="强制重训 MLP/MSE")
    args = ap.parse_args()

    if not FEATURE.exists():
        raise FileNotFoundError(FEATURE)
    OUT.mkdir(parents=True, exist_ok=True)

    targets = load_targets()

    # MLP 5 runs
    mlp_rows = []
    for i in range(1, 6):
        out = OUT / f"mlp_run{i}"
        link_features(out)
        if not args.skip_train:
            cmd = [sys.executable, ROOT / "run_baseline_mlp.py",
                   "--data_dir", DATA, "--output_dir", out,
                   "--ablation", "morgan_descriptors", "--batch_size", "2048",
                   "--epochs", "300", "--gpu", "0"]
            if args.force or not (out / "final_results.json").exists():
                cmd.append("--force")
            run(cmd)
        yt, yp = load_pred(out / "predictions.npz")
        mlp_rows.append({**global_metrics(yt, yp), **macro_metrics(yt, yp, targets)})

    # composite
    comp_yt, comp_yp, targets = infer_model(COMPOSITE_DIR / "best_model.pt", OUT / "composite_raw.npz")
    comp = {**global_metrics(comp_yt, comp_yp), **macro_metrics(comp_yt, comp_yp, targets)}

    # Reptile single model
    rep_yt, rep_yp, _ = infer_model(REPTILE_DIR / "best_model.pt", OUT / "reptile_raw.npz")
    rep = {**global_metrics(rep_yt, rep_yp), **macro_metrics(rep_yt, rep_yp, targets)}

    # MSE optional
    mse = None
    if args.with_mse:
        out = OUT / "transformer_mse"
        link_features(out)
        if not args.skip_train:
            cmd = [sys.executable, ROOT / "run_transformer_morgan_mse.py",
                   "--data_dir", DATA, "--output_dir", out,
                   "--batch_size", "2048", "--epochs", "300", "--gpu", "0"]
            if args.force or not (out / "final_results.json").exists():
                cmd.append("--force")
            run(cmd)
        yt, yp = load_pred(out / "predictions.npz")
        mse = {**global_metrics(yt, yp), **macro_metrics(yt, yp, targets)}

    print("\n" + "=" * 80)
    print("ChEMBL morgan_descriptors, unified metrics")
    print("Model | R2 | RMSE | AUPR | EF@1% | ECE")
    print(f"Descriptor MLP (5 runs) | {fmt(np.mean([r['R2'] for r in mlp_rows]))} ± {fmt(np.std([r['R2'] for r in mlp_rows], ddof=1))} | "
          f"{fmt(np.mean([r['RMSE'] for r in mlp_rows]))} ± {fmt(np.std([r['RMSE'] for r in mlp_rows], ddof=1))} | "
          f"{fmt(np.mean([r['AUPR'] for r in mlp_rows]))} | {fmt(np.mean([r['EF@1%'] for r in mlp_rows]),3)} | "
          f"{fmt(np.mean([r['ECE'] for r in mlp_rows]))}")
    if mse:
        print(f"Transformer, MSE only | {fmt(mse['R2'])} | {fmt(mse['RMSE'])} | {fmt(mse['AUPR'])} | {fmt(mse['EF@1%'],3)} | {fmt(mse['ECE'])}")
    print(f"Transformer, composite | {fmt(comp['R2'])} | {fmt(comp['RMSE'])} | {fmt(comp['AUPR'])} | {fmt(comp['EF@1%'],3)} | {fmt(comp['ECE'])}")
    print(f"Reptile-Transformer (single) | {fmt(rep['R2'])} | {fmt(rep['RMSE'])} | {fmt(rep['AUPR'])} | {fmt(rep['EF@1%'],3)} | {fmt(rep['ECE'])}")
    print("Reptile vs MLP mean:")
    for k in ["R2", "RMSE", "AUPR", "EF@1%", "ECE"]:
        m = float(np.mean([r[k] for r in mlp_rows]))
        print(f"  {k}: {(rep[k]/m-1)*100:+.1f}%")
    print(f"(EF targets={rep['EF_targets']}, AUPR targets={rep['AUPR_targets']})")


if __name__ == "__main__":
    main()
