# -*- coding: utf-8 -*-
"""Audited vs unaudited model evaluation.

Given per-pair model predictions and the audit labels, recompute every model's
metrics on

* **standard panel** — the full nominal target-cold-start test set (the
  convention used in the cold-start literature), and
* **audited panel** — the homology-free subset (test targets cross-cluster at
  the chosen identity threshold, 40% by default),

then compare model rankings (Kendall tau-b, Spearman rho), identify winner
reversals, and quantify the mis-selection cost (audited-panel performance lost
by deploying the standard-panel winner).

Predictions file (CSV, long format)::

    dataset,model,target_sequence,y_true,y_pred
    davis,GCNNet,MKTLLLTL... ,5.0,4.82
    ...

``target_id`` (matching ``per_target_audit.csv``) may replace
``target_sequence`` for joining.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .audit import PER_TARGET_FILE
from .metrics import METRIC_DIRECTION, pooled_metrics, ranks

REQUIRED_COLS = {"dataset", "model", "y_true", "y_pred"}


def _attach_labels(pred: pd.DataFrame, labels: pd.DataFrame,
                   identity_tag: int) -> pd.DataFrame:
    label_col = f"same_cluster_{identity_tag}"
    if label_col not in labels.columns:
        raise ValueError(f"labels missing {label_col}; audit at "
                         f"{identity_tag}% required")
    if "target_id" in pred.columns:
        key = "target_id"
    elif "target_sequence" in pred.columns:
        key = "target_sequence"
    else:
        raise ValueError("predictions must contain target_id or target_sequence")
    if key not in labels.columns:
        raise ValueError(f"labels file lacks join column {key}")
    keep = labels[[key, label_col]].rename(columns={label_col: "_same_cluster"})
    merged = pred.merge(keep, on=key, how="left", validate="many_to_one")
    missing = merged["_same_cluster"].isna().sum()
    if missing:
        raise ValueError(f"{missing} prediction rows could not be matched to an "
                         f"audit label via {key}")
    merged["_same_cluster"] = merged["_same_cluster"].astype(bool)
    return merged


def run_evaluation(predictions_csv: Path, labels_root: Path,
                   datasets: Sequence[str], out_dir: Path,
                   identity_tag: int = 40,
                   metrics: Sequence[str] = ("R2", "EF@1%", "ECE")) -> dict:
    """Compute audited vs standard metrics, ranks, and mis-selection costs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(predictions_csv)
    missing_cols = REQUIRED_COLS - set(pred.columns)
    if missing_cols:
        raise ValueError(f"predictions missing columns: {sorted(missing_cols)}")

    detail_rows: List[dict] = []
    rank_rows: List[dict] = []

    for ds in datasets:
        labels_path = Path(labels_root) / ds / PER_TARGET_FILE
        if not labels_path.exists():
            raise FileNotFoundError(labels_path)
        labels = pd.read_csv(labels_path)
        sub_pred = pred[pred["dataset"] == ds].copy()
        if sub_pred.empty:
            print(f"[warn] no predictions for dataset {ds!r}, skipping")
            continue
        sub_pred = _attach_labels(sub_pred, labels, identity_tag)

        models = sorted(sub_pred["model"].unique())
        per_model: Dict[str, dict] = {}
        for model in models:
            mp = sub_pred[sub_pred["model"] == model]
            full = pooled_metrics(mp["y_true"].to_numpy(float),
                                  mp["y_pred"].to_numpy(float))
            remote = mp[~mp["_same_cluster"]]
            sub = pooled_metrics(remote["y_true"].to_numpy(float),
                                 remote["y_pred"].to_numpy(float))
            per_model[model] = {"full": full, "sub": sub,
                                "n_full": len(mp), "n_sub": len(remote)}

        for metric in metrics:
            direction = METRIC_DIRECTION.get(metric)
            if direction is None:
                raise ValueError(f"unknown metric {metric}; choose from "
                                 f"{sorted(METRIC_DIRECTION)}")
            v_full = np.array([per_model[m]["full"][metric] for m in models])
            v_sub = np.array([per_model[m]["sub"][metric] for m in models])
            r_full = ranks(np.nan_to_num(v_full, nan=-np.inf if direction == "higher"
                                         else np.inf), direction)
            r_sub = ranks(np.nan_to_num(v_sub, nan=-np.inf if direction == "higher"
                                        else np.inf), direction)
            valid = ~np.isnan(v_sub)
            if valid.sum() > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kt = float(kendalltau(r_full[valid], r_sub[valid]).statistic)
                    sr = float(spearmanr(r_full[valid], r_sub[valid]).statistic)
                if np.isnan(kt):
                    kt = np.nan  # constant ranks -> rank correlation undefined
            else:
                kt = sr = np.nan

            i_std = int(np.argmin(r_full))
            i_aud = int(np.argmin(np.where(valid, r_sub, np.iinfo(int).max)))
            if direction == "higher":
                cost = float(v_sub[i_aud] - v_sub[i_std])
            else:
                cost = float(v_sub[i_std] - v_sub[i_aud])
            denom = abs(v_sub[i_aud])
            rel = cost / denom if denom not in (0.0, np.nan) and not np.isnan(denom) else np.nan

            rank_rows.append({
                "dataset": ds, "metric": metric, "n_models": len(models),
                "kendall_tau": round(float(kt), 3) if not np.isnan(kt) else np.nan,
                "spearman_rho": round(float(sr), 3) if not np.isnan(sr) else np.nan,
                "standard_winner": models[i_std],
                "audited_winner": models[i_aud],
                "winner_reversed": models[i_std] != models[i_aud],
                "misselection_cost": round(cost, 4),
                "relative_cost_pct": round(100 * rel, 1) if not np.isnan(rel) else np.nan})

            for j, model in enumerate(models):
                detail_rows.append({
                    "dataset": ds, "metric": metric, "model": model,
                    "n_full": per_model[model]["n_full"],
                    "n_audited": per_model[model]["n_sub"],
                    "standard_value": round(float(v_full[j]), 4),
                    "audited_value": round(float(v_sub[j]), 4),
                    "standard_rank": int(r_full[j]), "audited_rank": int(r_sub[j])})

    detail = pd.DataFrame(detail_rows)
    ranking = pd.DataFrame(rank_rows)
    detail.to_csv(out_dir / "audited_evaluation_per_model.csv", index=False,
                  encoding="utf-8-sig")
    ranking.to_csv(out_dir / "rank_comparison.csv", index=False, encoding="utf-8-sig")

    lines = ["Audited vs standard evaluation", "=" * 70,
             f"audited panel = targets cross-cluster at {identity_tag}% identity",
             "", ranking.to_string(index=False), "",
             "Per-model values (standard -> audited) and ranks:"]
    for ds, d0 in detail.groupby("dataset"):
        lines.append(f"-- {ds} --")
        for metric, d1 in d0.groupby("metric"):
            seq = " > ".join(
                f"{r.model}(audited#{r.audited_rank}/std#{r.standard_rank})"
                for r in d1.sort_values("audited_rank").itertuples())
            lines.append(f"  {metric:7s} {seq}")
    reversed_n = int(ranking["winner_reversed"].sum())
    lines += ["", f"Winner reversals: {reversed_n}/{len(ranking)} dataset-metric "
                  f"panels; non-zero mis-selection cost: "
                  f"{int((ranking['misselection_cost'] > 0).sum())}"]
    (out_dir / "audited_evaluation_report.txt").write_text(
        "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return {"per_model": detail, "ranking": ranking}
