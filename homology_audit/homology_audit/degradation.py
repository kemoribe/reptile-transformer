# -*- coding: utf-8 -*-
"""Degradation detection: how much does the homology audit cost a benchmark?

Takes the outputs of :mod:`homology_audit.evaluate` (``rank_comparison.csv``
and ``audited_evaluation_per_model.csv``) — or recomputes them from raw
predictions — and produces a verdict per dataset × metric panel:

* **rank flips** — number of model pairs whose relative order changes
  between the standard and the audited panel (Kendall-distance style count,
  computed from the two rank vectors);
* **degraded models** — models whose audited value is worse than their
  standard value under the metric direction;
* **winner reversal + mis-selection cost** — from ``run_evaluation``;
* **severity** —
    ``FAIL``  winner reversed **and** relative cost ≥ ``fail_cost_pct`` (%)
    ``WARN``  winner reversed **or** relative cost ≥ ``warn_cost_pct`` (%)
    ``OK``    otherwise.

A panel-level overall verdict (worst severity across panels) is appended to
the text report.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .audit import PER_TARGET_FILE
from .evaluate import run_evaluation
from .metrics import METRIC_DIRECTION

REPORT_FILE = "degradation_report.txt"
DETAIL_FILE = "degradation_report.csv"


def _count_rank_flips(r_std: Sequence[int], r_aud: Sequence[int]) -> int:
    """Number of model pairs whose relative order differs between ranks."""
    n = len(r_std)
    flips = 0
    for i, j in combinations(range(n), 2):
        a_std, b_std = r_std[i] < r_std[j], r_std[i] > r_std[j]
        a_aud, b_aud = r_aud[i] < r_aud[j], r_aud[i] > r_aud[j]
        if (a_std and b_aud) or (b_std and a_aud):
            flips += 1
    return flips


def detect_degradation(per_model: pd.DataFrame, ranking: pd.DataFrame,
                       warn_cost_pct: float = 5.0,
                       fail_cost_pct: float = 10.0) -> pd.DataFrame:
    """Attach severity/flip columns; return the annotated ranking table."""
    direction_map = METRIC_DIRECTION
    flip_col, deg_col, sev_col = [], [], []
    for row in ranking.itertuples():
        d = per_model[(per_model["dataset"] == row.dataset)
                      & (per_model["metric"] == row.metric)]
        flips = 0
        n_degraded = 0
        if len(d) >= 2:
            r_std = d["standard_rank"].to_numpy(int)
            r_aud = d["audited_rank"].to_numpy(int)
            flips = _count_rank_flips(r_std, r_aud)
            direction = direction_map.get(row.metric, "higher")
            std_v = d["standard_value"].to_numpy(float)
            aud_v = d["audited_value"].to_numpy(float)
            worse = (aud_v < std_v) if direction == "higher" else (aud_v > std_v)
            n_degraded = int(np.sum(worse & np.isfinite(aud_v) & np.isfinite(std_v)))
        rel = row.relative_cost_pct
        rel = np.nan if rel is None or (isinstance(rel, float) and np.isnan(rel)) \
            else float(rel)
        if bool(row.winner_reversed) and (not np.isnan(rel)) and rel >= fail_cost_pct:
            sev = "FAIL"
        elif bool(row.winner_reversed) or (not np.isnan(rel) and rel >= warn_cost_pct):
            sev = "WARN"
        else:
            sev = "OK"
        flip_col.append(flips)
        deg_col.append(n_degraded)
        sev_col.append(sev)
    out = ranking.copy()
    out["rank_flips"] = flip_col
    out["n_degraded_models"] = deg_col
    out["severity"] = sev_col
    return out


def run_degradation(out_dir: Path,
                    ranking_csv: Optional[Path] = None,
                    per_model_csv: Optional[Path] = None,
                    predictions_csv: Optional[Path] = None,
                    labels_root: Optional[Path] = None,
                    datasets: Optional[Sequence[str]] = None,
                    identity_tag: int = 40,
                    metrics: Optional[Sequence[str]] = None,
                    warn_cost_pct: float = 5.0,
                    fail_cost_pct: float = 10.0,
                    verbose: bool = True) -> dict:
    """Detect and report audit-induced degradation.

    Either pass the two ``evaluate`` output CSVs, or raw ``--predictions``
    plus ``--labels-root``/``--datasets`` (then ``evaluate`` is re-run
    internally into ``<out_dir>/evaluation/``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ranking_csv is not None and per_model_csv is not None:
        ranking = pd.read_csv(ranking_csv)
        per_model = pd.read_csv(per_model_csv)
    elif predictions_csv is not None and labels_root is not None:
        if not datasets:
            raise ValueError("--datasets is required with --predictions")
        res = run_evaluation(predictions_csv=predictions_csv,
                             labels_root=labels_root, datasets=datasets,
                             out_dir=out_dir / "evaluation",
                             identity_tag=identity_tag,
                             metrics=metrics or ("R2", "EF@1%", "ECE"),
                             verbose=False)
        ranking, per_model = res["ranking"], res["per_model"]
    else:
        raise ValueError("provide rank_comparison.csv + per-model CSV, or "
                         "--predictions with --labels-root/--datasets")

    report = detect_degradation(per_model, ranking, warn_cost_pct, fail_cost_pct)
    report.to_csv(out_dir / DETAIL_FILE, index=False, encoding="utf-8-sig")

    order = {"FAIL": 0, "WARN": 1, "OK": 2}
    worst = min(report["severity"], key=lambda s: order[s]) if len(report) else "OK"
    n_fail = int((report["severity"] == "FAIL").sum())
    n_warn = int((report["severity"] == "WARN").sum())

    lines = ["Degradation detection report", "=" * 60,
             f"severity rules: FAIL = winner reversed & relative cost >="
             f" {fail_cost_pct:g}%; WARN = winner reversed or cost >="
             f" {warn_cost_pct:g}%", "",
             report.to_string(index=False), "",
             f"Panels: {len(report)}  FAIL={n_fail}  WARN={n_warn}  "
             f"OK={len(report) - n_fail - n_warn}",
             f"Overall verdict: {worst}",
             ("Benchmark conclusions drawn on the standard panel are NOT "
              "homology-robust." if worst != "OK"
              else "No material audit-induced degradation detected.")]
    (out_dir / REPORT_FILE).write_text("\n".join(lines), encoding="utf-8")
    if verbose:
        print("\n".join(lines))
    return {"report": report, "overall": worst,
            "n_fail": n_fail, "n_warn": n_warn, "out_dir": str(out_dir)}
