# -*- coding: utf-8 -*-
"""Prediction metrics and ranking helpers.

Metric definitions match the paper's homology-free ablation protocol:

* R² / RMSE / MAE on pooled (target, compound) pairs of the evaluated panel.
* EF@k% — enrichment of the top 20% actives (at least 5 actives) among the
  top k% predictions, pooled within the panel.  Panels with fewer than the
  number of top-k% positions are skipped for that metric (returns NaN).
* ECE — 10 equal-width bins after min-max scaling jointly on y_true/y_pred of
  the evaluated panel.

These pooled definitions deliberately match the cross-cluster subset
evaluation in the manuscript; they are *not* the per-target-averaged EF used
during model training (which is a separate convention).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

METRIC_DIRECTION: Dict[str, str] = {
    "R2": "higher", "RMSE": "lower", "MAE": "lower",
    "EF@1%": "higher", "EF@5%": "higher", "EF@10%": "higher",
    "ECE": "lower",
}


def pooled_metrics(y_true, y_pred, active_quantile: float = 0.2,
                   min_actives: int = 5, bins: int = 10) -> Dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    if n < 2:
        return {k: np.nan for k in METRIC_DIRECTION}

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    n_active = max(min_actives, int(round(n * active_quantile)))
    active_thr = np.sort(y_true)[::-1][min(n_active - 1, n - 1)]
    # y values of true actives among the top-predicted k%
    top_order = np.argsort(y_pred)[::-1]
    y_top = y_true[top_order]

    def ef(pct: float) -> float:
        k = int(n * pct / 100.0)
        if k == 0:
            return np.nan  # panel too small for this enrichment level
        hit_rate = float(np.sum(y_top[:k] >= active_thr) / k)
        return (hit_rate / (n_active / n))

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    span = hi - lo
    if span < 1e-12:
        ece = 0.0
    else:
        yt = (y_true - lo) / span
        yp = (y_pred - lo) / span
        bin_ids = np.digitize(yp, np.linspace(0, 1, bins + 1)[1:-1])
        ece = 0.0
        for b in range(bins):
            mask = bin_ids == b
            if mask.sum():
                ece += (mask.sum() / n) * abs(yp[mask].mean() - yt[mask].mean())

    return {"R2": float(r2), "RMSE": rmse, "MAE": mae,
            "EF@1%": ef(1), "EF@5%": ef(5), "EF@10%": ef(10),
            "ECE": float(ece)}


def ranks(values: Sequence[float], direction: str) -> np.ndarray:
    """Competition ranks (1 = best) according to metric direction."""
    s = np.asarray(values, float)
    order = np.argsort(-s, kind="min") if direction == "higher" else np.argsort(s, kind="min")
    ranks_out = np.empty(len(s), dtype=int)
    # "min" rank semantics: ties share the better (smaller) rank number
    sorted_vals = s[order]
    cur_rank = 0
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        for k in range(i, j + 1):
            ranks_out[order[k]] = i + 1
        cur_rank += 1
        i = j + 1
    return ranks_out
