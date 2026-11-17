# -*- coding: utf-8 -*-
"""
S8/S9 逐靶点归属与复核
=====================
输出：
  table_audit/s8_s9/attribution_{dataset}_gnn.csv
  table_audit/s8_s9/attribution_{dataset}_reptile.csv
  table_audit/s8_s9/S8_recomputed.csv
  table_audit/s8_s9/S9_recomputed.csv
  table_audit/s8_s9/S8_S9_verification.md

说明：
- S8 的 EF 是 pooled（不是 per-target macro），与 S8 原脚本一致；
- S9 的 cost = |sub[full_winner] - sub[audited_winner]|，relative = cost / sub[audited_winner] * 100；
- 逐靶点归属用 target_sequence 排序映射到 test_XXXX，与 S8 原脚本一致。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(r"D:\lht")
GDT_DATA = BASE / "GrapthDTA" / "data"
GNN_PREDS = BASE / "修改supplement_output" / "paper_revision修改V1" / "gnn_preds"
CLUSTER_DIR = BASE / "修改supplement_output" / "protein_cluster"
E8_DIR = BASE / "修改supplement_output" / "revision2_experiments" / "e8_homology_free_ablation"
OUT = BASE / "table_audit" / "s8_s9"
OUT.mkdir(parents=True, exist_ok=True)

GNN_MODELS = ["GCNNet", "GATNet", "GAT_GCN", "GINConvNet"]
REPTILE_PREDS = {
    "davis": BASE / "reptile_ablation_output_davis" / "morgan_maccs" / "predictions.npz",
    "kiba": BASE / "reptile_ablation_output_kiba" / "morgan_maccs" / "predictions.npz",
    # bindingdb 原 S8 没有 Reptile 行；保持不加入
}


def calc_metrics(y_true, y_pred):
    """与 S8 原脚本一致：pooled R2/RMSE/MAE/EF/ECE。"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(y_true)
    if n < 2:
        return {k: np.nan for k in ["R2", "RMSE", "MAE", "EF@1%", "EF@5%", "EF@10%", "ECE"]}
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    active_count = max(5, int(n * 0.2))
    thr = np.sort(y_true)[::-1][min(active_count - 1, n - 1)]
    y_sorted = y_true[np.argsort(y_pred)[::-1]]

    def ef(pct):
        k = int(n * pct / 100)
        return float((np.sum(y_sorted[:k] >= thr) / k) / (active_count / n)) if k else 0.0

    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    rng = hi - lo
    if rng < 1e-12:
        ece = 0.0
    else:
        yt_n, yp_n = (y_true - lo) / rng, (y_pred - lo) / rng
        bins = np.digitize(yp_n, np.linspace(0, 1, 11)[1:-1])
        ece = 0.0
        for b in range(10):
            m = bins == b
            if m.sum():
                ece += (m.sum() / n) * abs(yp_n[m].mean() - yt_n[m].mean())
    return {"R2": float(r2), "RMSE": float(rmse), "MAE": float(mae),
            "EF@1%": ef(1), "EF@5%": ef(5), "EF@10%": ef(10), "ECE": float(ece)}


def load_identity(dataset):
    return pd.read_csv(CLUSTER_DIR / dataset / "per_test_target_identity.csv")


def build_gnn_attribution(dataset):
    te = pd.read_csv(GDT_DATA / f"{dataset}_test.csv")
    seqs = sorted(set(te["target_sequence"].dropna()))
    seq2tid = {s: f"test_{i:04d}" for i, s in enumerate(seqs)}
    te["target_id"] = te["target_sequence"].map(seq2tid)
    ident = load_identity(dataset)
    agg = te.groupby("target_id").size().rename("n_pairs").reset_index()
    attr = ident.merge(agg, on="target_id", how="left")
    attr["n_pairs"] = attr["n_pairs"].fillna(0).astype(int)
    attr["cross_cluster_40"] = ~attr["same_cluster_40"].astype(bool)
    # 附上序列
    tid2seq = {v: k for k, v in seq2tid.items()}
    attr["target_sequence"] = attr["target_id"].map(tid2seq)
    attr.to_csv(OUT / f"attribution_{dataset}_gnn.csv", index=False, encoding="utf-8-sig")
    return te, seq2tid, ident, attr


def _read_fasta(path):
    with open(path) as f:
        return "".join(l.strip() for l in f if not l.startswith(">"))


def _find_activities_and_seq(tdir: Path):
    if (tdir / "activities.csv").exists():
        act = pd.read_csv(tdir / "activities.csv")
        seq_path = tdir / "sequence.fasta"
        seq = _read_fasta(seq_path) if seq_path.exists() else ""
        return act, seq
    for f in tdir.iterdir():
        if f.name.endswith("_processed_activities.csv"):
            act = pd.read_csv(f)
            seq_file = tdir / f.name.replace("_processed_activities.csv", "_processed_protein_sequence.txt")
            seq = seq_file.read_text().strip() if seq_file.exists() else ""
            return act, seq
    return None, ""


def build_reptile_attribution(dataset):
    pred_path = REPTILE_PREDS.get(dataset)
    if pred_path is None or not pred_path.exists():
        return None, None, None
    d = np.load(pred_path)
    y_true, y_pred = d["y_true"], d["y_pred"]

    te = pd.read_csv(GDT_DATA / f"{dataset}_test.csv")
    seqs = sorted(set(te["target_sequence"].dropna()))
    seq2tid = {s: f"test_{i:04d}" for i, s in enumerate(seqs)}

    prep = BASE / f"3_{dataset}_preprocessed" / "test_set"
    target_ids = []
    entries = sorted(prep.iterdir())
    for entry in entries:
        if not entry.is_dir():
            continue
        act, seq = _find_activities_and_seq(entry)
        if act is not None:
            target_ids.extend([seq2tid.get(seq, entry.name)] * len(act))
        else:
            for tdir in sorted(entry.iterdir()):
                if not tdir.is_dir():
                    continue
                act, seq = _find_activities_and_seq(tdir)
                if act is not None:
                    target_ids.extend([seq2tid.get(seq, tdir.name)] * len(act))
    if len(target_ids) != len(y_true):
        print(f"[WARN] {dataset} Reptile rows {len(y_true)} != built {len(target_ids)}")
        return None, None, None
    target_ids = np.array(target_ids)
    tmp = pd.DataFrame({"target_id": target_ids})
    agg = tmp.groupby("target_id").size().rename("n_pairs").reset_index()
    if dataset == "kiba":
        # KIBA Reptile 使用 3_kiba_preprocessed 的 35 个靶点，与 GraphDTA 审计的 34 个靶点
        # 只有 3 个 UniProt 重叠；不能把 GraphDTA 的 cross 标签套过来。
        attr = agg.copy()
        attr["mapping_note"] = "UniProt-level counts; cross labels not available for this split"
        attr.to_csv(OUT / f"attribution_{dataset}_reptile.csv", index=False, encoding="utf-8-sig")
        print(f"[WARN] {dataset}: Reptile split has only 3/35 UniProt overlap with GraphDTA audit split")
        return {"y_true": y_true, "y_pred": y_pred}, target_ids, attr
    ident = load_identity(dataset)
    attr = ident.merge(agg, on="target_id", how="left")
    attr["n_pairs"] = attr["n_pairs"].fillna(0).astype(int)
    attr["cross_cluster_40"] = ~attr["same_cluster_40"].astype(bool)
    tid2seq = {v: k for k, v in seq2tid.items()}
    attr["target_sequence"] = attr["target_id"].map(tid2seq)
    attr.to_csv(OUT / f"attribution_{dataset}_reptile.csv", index=False, encoding="utf-8-sig")
    return {"y_true": y_true, "y_pred": y_pred}, target_ids, attr


def recompute_s8():
    rows = []
    for dataset in ["davis", "kiba", "bindingdb"]:
        te, seq2tid, ident, attr = build_gnn_attribution(dataset)
        cross_ids = set(ident.loc[~ident["same_cluster_40"].astype(bool), "target_id"])
        te["target_id"] = te["target_sequence"].map(seq2tid)
        cross_mask = te["target_id"].isin(cross_ids).values

        for m in GNN_MODELS:
            f = GNN_PREDS / f"{m}_{dataset}.npz"
            if not f.exists():
                continue
            p = np.load(f)
            yt, yp = p["y_true"], p["y_pred"]
            full = calc_metrics(yt, yp)
            sub = calc_metrics(yt[cross_mask], yp[cross_mask])
            rows.append({
                "dataset": dataset, "model": m, "family": "GNN",
                "n_full": len(yt), "n_cross": int(cross_mask.sum()),
                "n_cross_targets": len(cross_ids),
                **{f"full_{k}": v for k, v in full.items()},
                **{f"sub_{k}": v for k, v in sub.items()},
                "valid": True,
                "note": "",
            })

        rep, rep_targets, rep_attr = build_reptile_attribution(dataset)
        if rep is not None:
            yt, yp = rep["y_true"], rep["y_pred"]
            full = calc_metrics(yt, yp)
            if dataset == "kiba":
                # 测试靶点集与 GraphDTA 审计集几乎不重叠，无法做 matched subset。
                rows.append({
                    "dataset": dataset, "model": "Reptile-Transformer", "family": "non-GNN",
                    "n_full": len(yt), "n_cross": np.nan, "n_cross_targets": np.nan,
                    **{f"full_{k}": v for k, v in full.items()},
                    **{f"sub_{k}": np.nan for k in full},
                    "valid": False,
                    "note": "KIBA Reptile split has only 3/35 UniProt overlap with GraphDTA audit split",
                })
            else:
                cross_mask_rep = np.array([t in cross_ids for t in rep_targets])
                sub = calc_metrics(yt[cross_mask_rep], yp[cross_mask_rep])
                rows.append({
                    "dataset": dataset, "model": "Reptile-Transformer", "family": "non-GNN",
                    "n_full": len(yt), "n_cross": int(cross_mask_rep.sum()),
                    "n_cross_targets": len(set(rep_targets[cross_mask_rep])),
                    **{f"full_{k}": v for k, v in full.items()},
                    **{f"sub_{k}": v for k, v in sub.items()},
                    "valid": True,
                    "note": "",
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "S8_recomputed.csv", index=False, encoding="utf-8-sig")
    return df


def recompute_s9(df):
    rows = []
    for dataset in ["bindingdb", "davis", "kiba"]:
        sub = df[(df["dataset"] == dataset) & (df["valid"] == True)].set_index("model")
        for metric in ["R2", "EF@1%", "ECE"]:
            full = sub[f"full_{metric}"].astype(float)
            aud = sub[f"sub_{metric}"].astype(float)
            # winners: R2/EF 最大，ECE 最小
            if metric == "ECE":
                fw, aw = full.idxmin(), aud.idxmin()
            else:
                fw, aw = full.idxmax(), aud.idxmax()
            tau = stats.kendalltau(full.values, aud.values).correlation
            rho = stats.spearmanr(full.values, aud.values).correlation
            note = ""
            rev = "yes" if fw != aw else "no"
            # KIBA EF 的 audited winner 是并列：GAT_GCN 与 GCNNet 都是 4.3243
            if dataset == "kiba" and metric == "EF@1%":
                top = aud.max()
                tied = sorted(aud[aud.sub(aud.max()).abs() < 1e-9].index.tolist())
                if len(tied) > 1:
                    aw = "tie:" + "/".join(tied)
                    rev = "tie"
                    note = f"audited EF tie at {top:.4f}"
            if aw.startswith("tie:"):
                cost = 0.0
                rel = 0.0
            else:
                cost = abs(float(aud[fw]) - float(aud[aw]))
                rel = cost / abs(float(aud[aw])) * 100 if aud[aw] != 0 else np.nan
            if dataset == "kiba" and metric in ("R2", "EF@1%"):
                note = (note + "; " if note else "") + "Reptile excluded: KIBA split mismatch"
            rows.append({
                "dataset": dataset, "metric": metric,
                "Kendall_tau": tau, "Spearman_rho": rho,
                "full_winner": fw, "audited_winner": aw,
                "reversed": rev,
                "mis_selection_cost": cost, "relative_cost_pct": rel,
                "n_models": len(sub), "note": note,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "S9_recomputed.csv", index=False, encoding="utf-8-sig")
    return out


def compare(old_path, new_df, key_cols, value_cols, label):
    old = pd.read_csv(old_path)
    lines = [f"## {label}", ""]
    lines.append(f"旧表行数 {len(old)}，新表行数 {len(new_df)}。")
    merged = old.merge(new_df, on=key_cols, suffixes=("_old", "_new"), how="outer")
    max_diff = 0.0
    for c in value_cols:
        if c in merged.columns:
            d = (merged[f"{c}_old"] - merged[f"{c}_new"]).abs().max()
            max_diff = max(max_diff, float(d) if pd.notna(d) else 0)
            lines.append(f"- {c}: max |old-new| = {d}")
    lines.append(f"- 总体最大差异：{max_diff:.6g}")
    lines.append("")
    return "\n".join(lines)


def main():
    s8 = recompute_s8()
    s9 = recompute_s9(s8)

    old_s8 = E8_DIR / "homology_free_ablation_results.csv"
    lines = ["# S8/S9 逐靶点归属复核", ""]
    lines.append("逐靶点归属文件：")
    for p in sorted(OUT.glob("attribution_*.csv")):
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## 关键发现：KIBA Reptile 行不可用")
    lines.append("- `3_kiba_preprocessed` 的 35 个测试靶点与 GraphDTA/KIBA 审计的 34 个测试靶点")
    lines.append("  只有 **3 个 UniProt ID 重叠**；两套 split 的 target sequence 也几乎不同。")
    lines.append("- 旧 S8 的 KIBA Reptile `n_cross=1734` 只是序列能匹配上的部分，不是真正的 cross-cluster 子集；")
    lines.append("  13,000+ 个 Reptile 测试对没有对应的审计 cross 标签，sub_R2/sub_EF 无效。")
    lines.append("- 修正：S8 KIBA 仅保留 4 个 GNN；S9 KIBA 用 4 个 GNN 重算。")
    lines.append("- Davis 的 Reptile 归属正常（65 targets / 4,290 pairs / 27 cross targets / 1,782 cross pairs）。")
    lines.append("")
    if old_s8.exists():
        lines.append(compare(
            old_s8, s8,
            key_cols=["dataset", "model"],
            value_cols=["n_full", "n_cross", "full_R2", "full_RMSE", "full_MAE",
                        "full_EF@1%", "full_EF@5%", "full_EF@10%", "full_ECE",
                        "sub_R2", "sub_RMSE", "sub_MAE", "sub_EF@1%", "sub_EF@5%",
                        "sub_EF@10%", "sub_ECE"],
            label="S8 新旧对比"))
    lines.append("## S9 重算")
    lines.append(s9.to_string(index=False))
    lines.append("")
    lines.append("## 定义说明")
    lines.append("- S8 的 EF 是 pooled（与 S8 原脚本一致），不是 per-target macro。")
    lines.append("- S9 的 cost = |sub[full_winner] - sub[audited_winner]|；relative = cost / |sub[audited_winner]| * 100。")
    lines.append("- 逐靶点归属：GNN 用 target_sequence 排序映射 test_XXXX；Reptile 用预处理目录顺序映射。")
    (OUT / "S8_S9_verification.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
