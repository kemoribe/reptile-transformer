# -*- coding: utf-8 -*-
"""
S18/S19 重算：EF@1% 与 EF@5% 跨数据集矩阵
==========================================
统一口径：
- 活性定义：每个靶点真实值 top 20%（至少 5 个）；
- EF@k%：top floor(n*k/100) 个预测中的活性比例 / 全靶点活性比例；
- KIBA/ChEMBL/BindingDB：逐靶点 macro；EF@1% 要求 n>=100，EF@5% 要求 n>=20；
- Davis：66 化合物/靶点，per-target EF@1% 未定义，用 pooled EF@1%；EF@5% 用逐靶点 macro。

输出：
  D:\\lht\\s18_s19_recomputed\\
    S18_long.csv
    S19_long.csv
    S18_matrix.csv
    S19_matrix.csv
    REPORT.md

说明：不训练模型，只从现有 predictions 重算。缺失的预测标为 NA，不伪造。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\lht")
GD = ROOT / "GrapthDTA" / "data"
GNN = ROOT / "修改supplement_output" / "paper_revision修改V1" / "gnn_preds"
OUT = ROOT / "s18_s19_recomputed"
OUT.mkdir(parents=True, exist_ok=True)

KIBA_FEAT = ROOT / "transformer_ablation_output_kiba" / "precomputed_features.npz"
CHEMBL_FEAT = ROOT / "transformer_ablation_output_chembl" / "precomputed_features.npz"

MODELS = ["MLP", "Transformer", "Reptile-Transformer",
          "GATNet", "GAT_GCN", "GCNNet", "GINConvNet"]
GNN_MODELS = ["GATNet", "GAT_GCN", "GCNNet", "GINConvNet"]


def test_targets(feat_path):
    d = np.load(feat_path, mmap_mode="r", allow_pickle=True)
    sp = d["splits"][:]
    tn = d["target_names"][:]
    return tn[sp == "test"]


def load_pred(path):
    d = np.load(path)
    return np.asarray(d["y_true"], float), np.asarray(d["y_pred"], float)


def ef_at(yt, yp, pct):
    n = len(yt)
    if n < 10:
        return np.nan
    active = max(5, int(n * 0.2))
    thr = np.sort(yt)[::-1][min(active - 1, n - 1)]
    n_top = int(n * pct / 100)
    if n_top == 0:
        return np.nan
    top = np.argsort(yp)[::-1][:n_top]
    hits = np.sum(yt[top] >= thr)
    return float((hits / n_top) / (active / n))


def macro_ef(yt, yp, targets, pct, min_n):
    vals = []
    used = 0
    for t in np.unique(targets):
        m = targets == t
        n = int(m.sum())
        if n < min_n:
            continue
        v = ef_at(yt[m], yp[m], pct)
        if np.isfinite(v):
            vals.append(v)
            used += 1
    if not vals:
        return np.nan, 0, len(np.unique(targets))
    return float(np.mean(vals)), used, len(np.unique(targets))


def model_metric(preds, targets, pct, agg):
    vals = []
    used = 0
    total = len(np.unique(targets))
    for yt, yp in preds:
        if agg == "pooled":
            v = ef_at(yt, yp, pct)
            u = 1
            t = 1
        else:
            min_n = 100 if pct == 1 else 20
            v, u, t = macro_ef(yt, yp, targets, pct, min_n)
        vals.append(v)
        used = u
        total = t
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return np.nan, np.nan, 0, total
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan, used, total


def specs():
    kiba_t = test_targets(KIBA_FEAT)
    chembl_t = test_targets(CHEMBL_FEAT)
    # GNN targets from test CSV (order verified by y_true correlation)
    def gnn_targets(ds):
        return pd.read_csv(GD / f"{ds}_test.csv")["target_sequence"].values

    s = {}
    # KIBA
    kiba_mlp_paths = [ROOT / "table2_kiba_rerun" / "mlp" / "predictions.npz"] + [
        ROOT / "table2_kiba_rerun" / f"mlp_run{i}" / "predictions.npz" for i in range(2, 6)
    ]
    s[("KIBA", "MLP")] = ([load_pred(p) for p in kiba_mlp_paths], kiba_t, {1: "macro", 5: "macro"}, "5 runs")
    s[("KIBA", "Transformer")] = ([load_pred(ROOT / "transformer_ablation_output_kiba" / "morgan_descriptors" / "predictions.npz")], kiba_t, {1: "macro", 5: "macro"}, "single")
    s[("KIBA", "Reptile-Transformer")] = ([load_pred(ROOT / "修改supplement_output" / "revision2_experiments" / "e5_repeats" / "kiba" / f"seed_{x}" / "predictions.npz") for x in [11, 22, 33, 44, 55]], kiba_t, {1: "macro", 5: "macro"}, "5 seeds")
    for m in GNN_MODELS:
        s[("KIBA", m)] = ([load_pred(GNN / f"{m}_kiba.npz")], gnn_targets("kiba"), {1: "macro", 5: "macro"}, "single")
    # ChEMBL
    s[("ChEMBL", "MLP")] = ([load_pred(ROOT / "panel_screen" / "chembl_morgan_descriptors" / f"mlp_run{i}" / "predictions.npz") for i in range(1, 6)], chembl_t, {1: "macro", 5: "macro"}, "5 runs")
    s[("ChEMBL", "Transformer")] = ([load_pred(ROOT / "panel_screen" / "chembl_morgan_descriptors" / "composite_raw.npz")], chembl_t, {1: "macro", 5: "macro"}, "single")
    s[("ChEMBL", "Reptile-Transformer")] = ([load_pred(ROOT / "chembl_morgan_5seeds_b2048" / f"seed_{x}" / "predictions_raw.npz") for x in [11, 22, 33, 44, 55]], chembl_t, {1: "macro", 5: "macro"}, "5 seeds")
    # ChEMBL GNN（重训导出；GraphDTA split: 30,180 对 / 26 靶点）
    chembl_gnn_t = gnn_targets("chembl")
    for m in GNN_MODELS:
        p = GNN / f"{m}_chembl.npz"
        if p.exists():
            s[("ChEMBL", m)] = ([load_pred(p)], chembl_gnn_t, {1: "macro", 5: "macro"}, "single")
    # Davis 非 GNN（morgan_descriptors 面板，4,290 对 / 65 靶点）
    dv_feat = ROOT / "davis_panel_morgan" / "_features" / "precomputed_features.npz"
    dv_panel = ROOT / "davis_panel_morgan"
    if dv_feat.exists():
        d = np.load(dv_feat, mmap_mode="r", allow_pickle=True)
        dv_t = d["target_names"][d["splits"][:] == "test"]
        mlp_paths = [dv_panel / "mlp" / f"mlp_run{i}" / "predictions.npz" for i in range(1, 6)]
        if all(p.exists() for p in mlp_paths):
            s[("Davis", "MLP")] = ([load_pred(p) for p in mlp_paths], dv_t, {1: "pooled", 5: "macro"}, "5 runs")
        tr_paths = [dv_panel / "transformer" / "no_reptile" / f"seed_{x}" / "predictions.npz" for x in [11, 22, 33, 44, 55]]
        if all(p.exists() for p in tr_paths):
            s[("Davis", "Transformer")] = ([load_pred(p) for p in tr_paths], dv_t, {1: "pooled", 5: "macro"}, "5 seeds")
        rp_paths = [dv_panel / "reptile" / f"seed_{x}" / "predictions_raw.npz" for x in [11, 22, 33, 44, 55]]
        if all(p.exists() for p in rp_paths):
            s[("Davis", "Reptile-Transformer")] = ([load_pred(p) for p in rp_paths], dv_t, {1: "pooled", 5: "macro"}, "5 seeds")
        del d
    for m in GNN_MODELS:
        s[("Davis", m)] = ([load_pred(GNN / f"{m}_davis.npz")], gnn_targets("davis"), {1: "pooled", 5: "macro"}, "single")
    # BindingDB
    bdb_gnn_t = gnn_targets("bindingdb")
    for m in GNN_MODELS:
        p = GNN / f"{m}_bindingdb.npz"
        if p.exists():
            s[("BindingDB", m)] = ([load_pred(p)], bdb_gnn_t, {1: "macro", 5: "macro"}, "single")

    # BindingDB 非 GNN（morgan_descriptors 面板，7,087 对 / 112 靶点）
    bdb_feat = ROOT / "bindingdb_panel_morgan" / "_features" / "precomputed_features.npz"
    bdb_panel = ROOT / "bindingdb_panel_morgan"
    if bdb_feat.exists():
        d = np.load(bdb_feat, mmap_mode="r", allow_pickle=True)
        bdb_t = d["target_names"][d["splits"][:] == "test"]
        mlp_paths = [bdb_panel / "mlp" / f"mlp_run{i}" / "predictions.npz" for i in range(1, 6)]
        if all(p.exists() for p in mlp_paths):
            s[("BindingDB", "MLP")] = ([load_pred(p) for p in mlp_paths], bdb_t, {1: "macro", 5: "macro"}, "5 runs")
        tr_paths = [bdb_panel / "transformer" / "no_reptile" / f"seed_{x}" / "predictions.npz" for x in [11, 22, 33, 44, 55]]
        if all(p.exists() for p in tr_paths):
            s[("BindingDB", "Transformer")] = ([load_pred(p) for p in tr_paths], bdb_t, {1: "macro", 5: "macro"}, "5 seeds")
        rp_paths = [bdb_panel / "reptile" / f"seed_{x}" / "predictions_raw.npz" for x in [11, 22, 33, 44, 55]]
        if all(p.exists() for p in rp_paths):
            s[("BindingDB", "Reptile-Transformer")] = ([load_pred(p) for p in rp_paths], bdb_t, {1: "macro", 5: "macro"}, "5 seeds")
        del d
    return s


def main():
    specs_dict = specs()
    rows18, rows19 = [], []
    for (ds, model), (preds, targets, aggs, runs) in specs_dict.items():
        for pct, out_rows in [(1, rows18), (5, rows19)]:
            mean, sd, used, total = model_metric(preds, targets, pct, aggs[pct])
            out_rows.append({
                "Dataset": ds, "Model": model,
                "EF": mean, "EF_sd": sd,
                "Aggregation": aggs[pct],
                "n_targets_used": used, "n_targets_total": total,
                "coverage": used / total if total else np.nan,
                "n_pairs": len(preds[0][0]),
                "runs": runs,
                "status": "recomputed",
            })
    # Missing cells
    expected = [(ds, m) for ds in ["KIBA", "ChEMBL", "Davis", "BindingDB"] for m in MODELS]
    missing = [k for k in expected if k not in specs_dict]
    for ds, m in missing:
        if ds == "ChEMBL" and m in GNN_MODELS:
            status = "missing ChEMBL GNN predictions"
        elif ds in ("BindingDB", "Davis") and m in ("MLP", "Transformer", "Reptile-Transformer"):
            status = f"missing {ds} morgan predictions"
        else:
            status = "missing predictions"
        for pct, out_rows in [(1, rows18), (5, rows19)]:
            out_rows.append({"Dataset": ds, "Model": m, "EF": np.nan, "EF_sd": np.nan,
                             "Aggregation": "NA", "n_targets_used": 0, "n_targets_total": 0,
                             "coverage": np.nan, "n_pairs": 0, "runs": "NA",
                             "status": status})
    df18 = pd.DataFrame(rows18)
    df19 = pd.DataFrame(rows19)
    df18.to_csv(OUT / "S18_long.csv", index=False, encoding="utf-8-sig")
    df19.to_csv(OUT / "S19_long.csv", index=False, encoding="utf-8-sig")

    def pivot(df):
        return df.pivot_table(index="Model", columns="Dataset", values="EF", aggfunc="first").reindex(MODELS)
    p18 = pivot(df18)
    p19 = pivot(df19)
    p18.to_csv(OUT / "S18_matrix.csv", encoding="utf-8-sig")
    p19.to_csv(OUT / "S19_matrix.csv", encoding="utf-8-sig")

    lines = ["# S18/S19 重算报告", ""]
    lines.append("EF 定义：每个靶点真实值 top 20%（至少 5 个）为活性；EF@k% = top floor(n*k/100) 个预测中的活性比例 / 全靶点活性比例。")
    lines.append("KIBA/ChEMBL/BindingDB 用逐靶点 macro；EF@1% 要求 n>=100，EF@5% 要求 n>=20。")
    lines.append("Davis 每靶点 66 个化合物，per-target EF@1% 未定义，EF@1% 用 pooled；EF@5% 用逐靶点 macro。")
    lines.append("")
    lines.append("## S18 EF@1%")
    lines.append(p18.round(3).to_string())
    lines.append("")
    lines.append("## S19 EF@5%")
    lines.append(p19.round(3).to_string())
    lines.append("")
    lines.append("## 缺失")
    if missing:
        for ds, m in missing:
            if ds == "ChEMBL" and m in GNN_MODELS:
                st = "missing ChEMBL GNN predictions"
            elif ds in ("BindingDB", "Davis") and m in ("MLP", "Transformer", "Reptile-Transformer"):
                st = f"missing {ds} morgan predictions"
            else:
                st = "missing predictions"
            lines.append(f"- {ds}/{m}: {st}")
    else:
        lines.append("- 无缺失单元格。")
    lines.append("")
    lines.append("## 口径与 caveat")
    lines.append("- ChEMBL GNN 使用 GraphDTA split：30,180 对 / 26 靶点；ChEMBL 非 GNN 为 31,132 对 / 26 靶点，二者不是同一测试集。")
    lines.append("- KIBA GNN 使用 20,502 对 GraphDTA split；KIBA 非 GNN 使用 18,478 对 target-cold split，二者不是同一测试集。")
    lines.append("- BindingDB 非 GNN 使用 morgan_descriptors 面板：7,087 个唯一配对 / 112 靶点，paffinity 已对齐为 GraphDTA affinity。")
    lines.append("- Davis 非 GNN 使用 morgan_descriptors 面板：4,290 对 / 65 靶点；EF@1% 为 pooled，"
                 "不能与 KIBA/ChEMBL/BindingDB 的 macro EF@1% 直接比较。")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(lines[0])
    print(p18.round(3).to_string())
    print("\nS19")
    print(p19.round(3).to_string())
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
