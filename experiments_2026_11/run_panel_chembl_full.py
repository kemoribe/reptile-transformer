# -*- coding: utf-8 -*-
"""ChEMBL 全特征（含 MACCS）统一口径筛选：Reptile 5 seeds vs MLP 5 runs vs composite。"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\lht")
sys.path.insert(0, str(ROOT))
os.environ["PREPROCESSED_DIR"] = str(ROOT / "3_all_data_chembl_targets_preprocessed")

from data_preprocessing import LazySubset, TargetScaler
from reptile_transformer_model import ReptileTransformer
import run_table2_kiba_baselines as common

PANEL = "chembl"
DATA = ROOT / "3_all_data_chembl_targets_preprocessed"
E5 = ROOT / "修改supplement_output" / "revision2_experiments" / "e5_repeats" / PANEL
COMPOSITE = ROOT / "transformer_ablation_output_chembl" / "full"
FEATURE = ROOT / "transformer_ablation_output_chembl" / "precomputed_features.npz"
OUT = ROOT / "panel_screen" / PANEL


def infer_model(model_path: Path, feature_path: Path, scaler_path: Path, out_path: Path, mask_maccs=False):
    data = np.load(feature_path, mmap_mode="r", allow_pickle=True)
    splits = data["splits"][:]
    target_names = data["target_names"][:]
    test_idx = np.where(splits == "test")[0]
    scaler = TargetScaler()
    scaler.load(scaler_path)
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
    return yt, yp, target_names[test_idx]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep_rows = []
    for d in sorted([x for x in E5.glob("seed_*") if x.is_dir()]):
        out = d / "predictions_raw.npz"
        yt, yp, targets = infer_model(d / "best_model.pt", d / "precomputed_features.npz",
                                     d / "target_scaler.json", out, mask_maccs=False)
        g = common.global_metrics(yt, yp)
        mm = common.macro_metrics(yt, yp, targets)
        rep_rows.append({**g, **mm})
        print(d.name, "global", g, "macro", mm)
    rep_mean = {k: float(np.mean([r[k] for r in rep_rows])) for k in rep_rows[0]}
    rep_sd = {k: float(np.std([r[k] for r in rep_rows], ddof=1)) for k in rep_rows[0]}

    comp_yt, comp_yp, comp_targets = infer_model(
        COMPOSITE / "best_model.pt", FEATURE, E5 / "seed_11" / "target_scaler.json",
        OUT / "composite_raw.npz", mask_maccs=False)
    comp_g = common.global_metrics(comp_yt, comp_yp)
    comp_m = common.macro_metrics(comp_yt, comp_yp, comp_targets)

    mlp_rows = []
    env = dict(os.environ)
    env["PREPROCESSED_DIR"] = str(DATA)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(1, 6):
        out = OUT / f"mlp_run{i}"
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "precomputed_features.npz"
        if not dst.exists():
            os.link(FEATURE, dst)
        cmd = [sys.executable, str(ROOT / "run_baseline_mlp.py"),
               "--data_dir", str(DATA), "--output_dir", str(out),
               "--ablation", "none", "--batch_size", "2048",
               "--epochs", "300", "--gpu", "0", "--force"]
        print("RUN", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if r.returncode != 0:
            raise SystemExit("MLP failed")
        yt, yp = common.load_pred(out / "predictions.npz")
        g = common.global_metrics(yt, yp)
        mm = common.macro_metrics(yt, yp, comp_targets)
        mlp_rows.append({**g, **mm})
        print(out.name, "global", g, "macro", mm)
    mlp_mean = {k: float(np.mean([r[k] for r in mlp_rows])) for k in mlp_rows[0]}
    mlp_sd = {k: float(np.std([r[k] for r in mlp_rows], ddof=1)) for k in mlp_rows[0]}

    print("\n" + "=" * 70)
    print("ChEMBL full-feature screen")
    print("Reptile", rep_mean, rep_sd)
    print("MLP    ", mlp_mean, mlp_sd)
    print("Composite", comp_g, comp_m)
    print("Reptile vs MLP mean %:")
    for k in ["R2", "RMSE", "AUPR", "EF@1%", "ECE"]:
        print(f"  {k}: {(rep_mean[k]/mlp_mean[k]-1)*100:+.1f}%")


if __name__ == "__main__":
    main()
