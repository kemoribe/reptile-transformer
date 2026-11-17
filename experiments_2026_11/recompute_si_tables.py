# -*- coding: utf-8 -*-
"""
重算 6 张“预测已存在”的补充表：
  S23 ChEMBL lambda
  S24 Davis/KIBA lambda
  S27 ESM2 freeze ablation
  S28 protein-encoder capacity
  S29 inner steps
  S30 MAML vs Reptile

统一口径：
  R2 / RMSE / MAE / ECE：全局 pooled；
  EF@1% / AUPR：逐靶点 macro（KIBA、ChEMBL）；
  Davis EF@5%：pooled（per-target EF@1% 不可估）。

S8/S9 需要逐靶点 subset 归属，本脚本不重算，只在 summary 中标注。

用法：
    python recompute_si_tables.py
输出：
    D:\\lht\\table_audit\\recomputed\\S23_chembl_lambda.csv
    ... S24/S27/S28/S29/S30 ...
    D:\\lht\\table_audit\\recomputed\\RECOMPUTED_SUMMARY.md
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\lht")
sys.path.insert(0, str(ROOT))
import run_table2_kiba_baselines as common  # noqa

OUT = ROOT / "table_audit" / "recomputed"
OUT.mkdir(parents=True, exist_ok=True)


def mae(yt, yp):
    return float(np.mean(np.abs(yt - yp)))


def global_all(yt, yp):
    g = common.global_metrics(yt, yp)
    g["MAE"] = mae(yt, yp)
    return g


def macro_ef5_from_json(pt_path: Path):
    """Davis 用逐靶点 EF@5% 的 macro 均值（n>=20），与 S24/S28 原定义一致。"""
    d = json.loads(pt_path.read_text(encoding="utf-8"))
    vals = []
    for v in d.values():
        n = int(float(v.get("samples", 0)))
        if n >= 20:
            x = float(v.get("EF@5%", float("nan")))
            if np.isfinite(x):
                vals.append(x)
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


def row_from_run(run_dir: Path, dataset: str, use_saved=True):
    pred = run_dir / "predictions.npz"
    raw = run_dir / "predictions_raw.npz"
    if dataset == "chembl" and not raw.exists():
        raw = make_raw_chembl_predictions(run_dir)
    p = raw if raw.exists() else pred
    if not p.exists():
        raise FileNotFoundError(f"{run_dir} 缺少 predictions")
    yt, yp = common.load_pred(p)
    g = global_all(yt, yp)
    pt = run_dir / "per_target_results.json"
    if dataset == "davis":
        mm = {"EF@1%": float("nan"), "AUPR": float("nan"), "EF_targets": 0, "AUPR_targets": 0}
        if pt.exists():
            mm = common.reptile_macro_from_json(pt)
            g["EF"], n_ef5 = macro_ef5_from_json(pt)
            g["EF_targets5"] = n_ef5
        else:
            g["EF"] = float("nan")
        g["EF_kind"] = "EF@5% macro"
    else:
        if not pt.exists():
            raise FileNotFoundError(f"{run_dir} 缺少 per_target_results.json")
        mm = common.reptile_macro_from_json(pt)
        g["EF"] = mm["EF@1%"]
        g["EF_kind"] = "EF@1% macro"
    g["AUPR"] = mm["AUPR"]
    g["N"] = len(yt)
    g["var"] = float(yt.var())
    return g


def make_raw_chembl_predictions(run_dir: Path, mask_maccs: bool = False, force: bool = False):
    """ChEMBL 的旧 predictions.npz 是 target-wise 扭曲尺度；用 best_model 重生成 raw。

    mask_maccs=True 用于 morgan_descriptors 模型（训练时 MACCS 被清零）；
    False 用于 full 特征模型（ChEMBL λ 敏感性）。
    """
    import torch
    from data_preprocessing import LazySubset, TargetScaler
    from reptile_transformer_model import ReptileTransformer

    out = run_dir / "predictions_raw.npz"
    if out.exists() and not force:
        return out
    feat = run_dir / "precomputed_features.npz"
    best = run_dir / "best_model.pt"
    scaler_path = run_dir / "target_scaler.json"
    if not (feat.exists() and best.exists() and scaler_path.exists()):
        return None
    data = np.load(feat, mmap_mode="r", allow_pickle=True)
    splits = data["splits"][:]
    test_idx = np.where(splits == "test")[0]
    scaler = TargetScaler()
    scaler.load(scaler_path)
    test_data = LazySubset(data, test_idx, scaler)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ReptileTransformer()
    model.load_state_dict(torch.load(best, map_location=device))
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
    np.savez(out, y_true=yt, y_pred=yp)
    print(f"[raw] {run_dir.name} N={len(yt)} var={yt.var():.6f}")
    return out


def write_csv(name, rows, columns):
    p = OUT / name
    with open(p, "w", encoding="utf-8-sig") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in columns) + "\n")
    print("[saved]", p)


def main():
    # S23 ChEMBL lambda
    s23 = []
    lam = {
        "base": (0.05, 0.10),
        "l1_low": (0.025, 0.10),
        "l1_high": (0.100, 0.10),
        "l2_low": (0.05, 0.05),
        "l2_high": (0.05, 0.20),
    }
    for cfg, (l1, l2) in lam.items():
        d = ROOT / "修改supplement_output" / "lambda_sensitivity" / f"run_{cfg}_l1_{l1:.3f}_l2_{l2:.2f}" / "output"
        if not d.exists():
            # 目录名格式不统一，按前缀匹配
            cand = list((ROOT / "修改supplement_output" / "lambda_sensitivity").glob(f"run_{cfg}*"))
            if not cand:
                raise FileNotFoundError(f"S23 缺少 {cfg}")
            d = cand[0] / "output"
        r = row_from_run(d, "chembl")
        s23.append({"config": cfg, "lambda1": l1, "lambda2": l2, "dataset": "ChEMBL", **r})
    write_csv("S23_chembl_lambda.csv", s23,
              ["dataset", "config", "lambda1", "lambda2", "R2", "RMSE", "MAE", "EF", "EF_kind", "ECE", "AUPR", "N", "var"])

    # S24 Davis / KIBA lambda
    s24 = []
    for ds in ["kiba", "davis"]:
        for cfg in lam:
            cand = list((ROOT / "修改supplement_output" / "revision2_experiments" / "e2_lambda_ofat" / ds).glob(f"run_{cfg}*"))
            if not cand:
                raise FileNotFoundError(f"S24 缺少 {ds}/{cfg}")
            r = row_from_run(cand[0], ds)
            s24.append({"dataset": ds, "config": cfg, "lambda1": lam[cfg][0], "lambda2": lam[cfg][1], **r})
    write_csv("S24_davis_kiba_lambda.csv", s24,
              ["dataset", "config", "lambda1", "lambda2", "R2", "RMSE", "MAE", "EF", "EF_kind", "ECE", "AUPR", "N", "var"])

    # S27 ESM2 freeze
    s27 = []
    for cfg, label in [("frozen", "Frozen (protein tower at 0.1x LR)"),
                       ("partial", "Partial fine-tune (top protein layers unfrozen)"),
                       ("full", "Full fine-tune (entire protein encoder unfrozen)")]:
        d = ROOT / "修改supplement_output" / "revision2_experiments" / "e4_esm2_freeze" / "kiba" / cfg
        r = row_from_run(d, "kiba")
        s27.append({"configuration": label, **r})
    write_csv("S27_esm2_freeze.csv", s27,
              ["configuration", "R2", "RMSE", "MAE", "EF", "EF_kind", "ECE", "AUPR", "N", "var"])

    # S28 encoder capacity: default ESM2 from e5 seeds 11/22/33; ESM1b/ProtT5 from exp12
    s28 = []
    groups = {
        "ESM-2 t12 35M (default)": {
            "kiba": [ROOT / "修改supplement_output" / "revision2_experiments" / "e5_repeats" / "kiba" / f"seed_{s}" for s in (11, 22, 33)],
            "davis": [ROOT / "修改supplement_output" / "revision2_experiments" / "e5_repeats" / "davis" / f"seed_{s}" for s in (11, 22, 33)],
        },
        "ESM-1b 650M": {
            "kiba": [ROOT / "修改supplement_output" / "remaining_experiments" / "exp12_lm_ablation" / f"kiba_esm1b_seed{s}" for s in (11, 22, 33)],
            "davis": [ROOT / "修改supplement_output" / "remaining_experiments" / "exp12_lm_ablation" / f"davis_esm1b_seed{s}" for s in (11, 22, 33)],
        },
        "ProtT5-XL 3B": {
            "kiba": [ROOT / "修改supplement_output" / "remaining_experiments" / "exp12_lm_ablation" / f"kiba_prott5_seed{s}" for s in (11, 22, 33)],
            "davis": [ROOT / "修改supplement_output" / "remaining_experiments" / "exp12_lm_ablation" / f"davis_prott5_seed{s}" for s in (11, 22, 33)],
        },
    }
    for encoder, ds_dirs in groups.items():
        kiba_rows = [row_from_run(d, "kiba") for d in ds_dirs["kiba"]]
        davis_rows = [row_from_run(d, "davis") for d in ds_dirs["davis"]]
        def ms(rows, key):
            vals = [r[key] for r in rows]
            return float(np.mean(vals)), float(np.std(vals, ddof=1))
        k_r2, k_r2s = ms(kiba_rows, "R2")
        k_ef, k_efs = ms(kiba_rows, "EF")
        k_ece, k_eces = ms(kiba_rows, "ECE")
        k_aupr, k_auprs = ms(kiba_rows, "AUPR")
        d_r2, d_r2s = ms(davis_rows, "R2")
        d_ef, d_efs = ms(davis_rows, "EF")
        d_ece, d_eces = ms(davis_rows, "ECE")
        s28.append({
            "protein_encoder": encoder,
            "KIBA_R2": f"{k_r2:.4f} ± {k_r2s:.4f}",
            "KIBA_EF1": f"{k_ef:.3f} ± {k_efs:.3f}",
            "KIBA_ECE": f"{k_ece:.4f} ± {k_eces:.4f}",
            "KIBA_AUPR": f"{k_aupr:.4f} ± {k_auprs:.4f}",
            "Davis_R2": f"{d_r2:.4f} ± {d_r2s:.4f}",
            "Davis_EF5": f"{d_ef:.3f} ± {d_efs:.3f}",
            "Davis_ECE": f"{d_ece:.4f} ± {d_eces:.4f}",
        })
    write_csv("S28_encoder_capacity.csv", s28,
              ["protein_encoder", "KIBA_R2", "KIBA_EF1", "KIBA_ECE", "KIBA_AUPR", "Davis_R2", "Davis_EF5", "Davis_ECE"])

    # S29 inner steps
    s29 = []
    for steps in [1, 3, 5, 10]:
        d = ROOT / "修改supplement_output" / "revision2_experiments" / "e1_inner_steps" / "kiba" / f"steps_{steps}"
        r = row_from_run(d, "kiba")
        wt = d / "walltime_sec.txt"
        r["walltime_s"] = float(wt.read_text().strip()) if wt.exists() else ""
        r["inner_steps"] = steps
        s29.append(r)
    write_csv("S29_inner_steps.csv", s29,
              ["inner_steps", "R2", "RMSE", "MAE", "EF", "EF_kind", "ECE", "AUPR", "walltime_s", "N", "var"])

    # S30 MAML vs Reptile
    s30 = []
    for cfg, label in [("maml", "MAML (default inner LR)"),
                       ("maml_tuned", "MAML (tuned inner LR)"),
                       ("reptile_solo", "Reptile (first-order)")]:
        d = ROOT / "修改supplement_output" / "revision2_experiments" / "e6_maml_vs_reptile" / "kiba" / cfg
        r = row_from_run(d, "kiba")
        wt = d / "walltime_sec.txt"
        r["walltime_s"] = float(wt.read_text().strip()) if wt.exists() else ""
        r["optimizer"] = label
        s30.append(r)
    write_csv("S30_maml_vs_reptile.csv", s30,
              ["optimizer", "R2", "RMSE", "MAE", "EF", "EF_kind", "ECE", "AUPR", "walltime_s", "N", "var"])

    # summary
    lines = ["# 重算结果（6 张表）", ""]
    lines.append("统一口径：R2/RMSE/MAE/ECE 全局；KIBA/ChEMBL EF@1% 与 AUPR 为逐靶点 macro；Davis EF@5% 为 pooled。")
    lines.append("")
    for name, path in [
        ("S23 ChEMBL lambda", OUT / "S23_chembl_lambda.csv"),
        ("S24 Davis/KIBA lambda", OUT / "S24_davis_kiba_lambda.csv"),
        ("S27 ESM2 freeze", OUT / "S27_esm2_freeze.csv"),
        ("S28 encoder capacity", OUT / "S28_encoder_capacity.csv"),
        ("S29 inner steps", OUT / "S29_inner_steps.csv"),
        ("S30 MAML vs Reptile", OUT / "S30_maml_vs_reptile.csv"),
    ]:
        lines.append(f"## {name}")
        lines.append(f"文件：`{path}`")
        lines.append("")
    lines.append("## 未包含")
    lines.append("- S8/S9：需要逐靶点 subset 归属，本脚本未重算；源文件为 `e8_homology_free_ablation_results.csv`。")
    (OUT / "RECOMPUTED_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("[saved]", OUT / "RECOMPUTED_SUMMARY.md")


if __name__ == "__main__":
    main()
