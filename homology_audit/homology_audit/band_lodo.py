# -*- coding: utf-8 -*-
"""Leave-one-dataset-out validation of a 1-D applicability band/threshold.

Useful when a "GNN-favourable region" was drawn from only a handful of
benchmark panels (x = panel descriptor, y = 1 if the model family wins).  The
fitted rule is a midpoint-bounded band around the positive support; this
module reports its LODO accuracy against 1-nearest-neighbour and the majority
class baseline.  Accuracy at or below the majority baseline means the band is
a descriptive summary, not a deployable threshold.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Point = Tuple[str, float, float, int]  # name, x, effect_size, y

# Paper Table S3 / exp6_conceptual_framework authoritative values (for
# reproducing the manuscript analysis directly; supply your own CSV otherwise).
PAPER_POINTS: List[Point] = [
    ("ChEMBL", 0.122, -0.2041, 0),
    ("Davis", 0.137, 0.1412, 1),
    ("KIBA", 0.250, -0.0989, 0),
    ("BindingDB", 0.360, -0.1340, 0),
]


def fit_band(points: Sequence[Tuple[float, int]]):
    """Midpoint band around the positive support; 'undef' if no positive."""
    pos = [x for x, y in points if y == 1]
    neg = [x for x, y in points if y == 0]
    if not pos:
        return "undef", "undef"
    p = max(pos)
    below = [x for x in neg if x < p]
    above = [x for x in neg if x > p]
    b_lo = (p + max(below)) / 2 if below else None
    b_hi = (p + min(above)) / 2 if above else None
    return b_lo, b_hi


def in_band(x: float, band) -> int:
    b_lo, b_hi = band
    if b_lo == "undef":
        return 0
    ok = True
    if b_lo is not None:
        ok &= x >= b_lo
    if b_hi is not None:
        ok &= x <= b_hi
    return int(ok)


def run_band_lodo(out_dir: Path, points: Optional[List[Point]] = None,
                  points_csv: Optional[Path] = None) -> dict:
    """LODO for midpoint-band and 1-NN rules; writes folds/boundaries/report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if points_csv:
        df0 = pd.read_csv(points_csv)
        cols = {c.lower(): c for c in df0.columns}
        need = ("name", "x", "y")
        if not all(k in cols for k in need):
            raise ValueError(f"points CSV needs columns name,x,y (got {list(df0.columns)})")
        effect_col = cols.get("effect")
        points = [(str(r[cols["name"]]), float(r[cols["x"]]),
                   float(r[effect_col]) if effect_col else np.nan,
                   int(r[cols["y"]])) for r in df0.to_dict("records")]
    if points is None:
        points = list(PAPER_POINTS)

    rows, bounds = [], []
    for i, (name, x, effect, y) in enumerate(points):
        train = [(points[j][1], points[j][3]) for j in range(len(points)) if j != i]
        band = fit_band(train)
        pred = in_band(x, band)
        nn_j = min((j for j in range(len(points)) if j != i),
                   key=lambda j: abs(points[j][1] - x))
        nn_pred = points[nn_j][3]
        rows.append({"held_out": name, "x": x, "effect": effect, "true_label": y,
                     "b_lo": band[0] if band[0] != "undef" else "unidentifiable",
                     "b_hi": band[1] if band[1] != "undef" else "unidentifiable",
                     "band_pred": pred, "band_correct": int(pred == y),
                     "nn_pred": nn_pred, "nn_correct": int(nn_pred == y)})
        for tag, b in (("b_lo", band[0]), ("b_hi", band[1])):
            if b not in (None, "undef"):
                bounds.append({"held_out": name, "boundary": tag, "value": b})

    folds = pd.DataFrame(rows)
    bd = pd.DataFrame(bounds)
    folds.to_csv(out_dir / "band_lodo_folds.csv", index=False, encoding="utf-8-sig")
    bd.to_csv(out_dir / "band_lodo_boundaries.csv", index=False, encoding="utf-8-sig")

    acc_band = float(folds["band_correct"].mean())
    acc_nn = float(folds["nn_correct"].mean())
    majority = float(1 - np.mean([p[3] for p in points]))
    full_band = fit_band([(p[1], p[3]) for p in points])
    summary = {"accuracy_band_rule": acc_band, "accuracy_1nn": acc_nn,
               "majority_baseline": majority,
               "band_supports_deployment": bool(acc_band > majority),
               "full_data_band": [None if b == "undef" else b for b in full_band]}
    with open(out_dir / "band_lodo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = ["Applicability-band LODO validation", "=" * 60,
             folds.to_string(index=False), "",
             f"LODO accuracy (midpoint band) : {acc_band:.3f} "
             f"({int(folds['band_correct'].sum())}/{len(folds)})",
             f"LODO accuracy (1-NN)          : {acc_nn:.3f}",
             f"Majority-class baseline       : {majority:.3f}",
             f"Full-data band (display only) : {summary['full_data_band']}", "",
             "Verdict: band is a DEPLOYABLE threshold only if its LODO accuracy "
             "exceeds the majority baseline; otherwise report it as a "
             "descriptive region."]
    (out_dir / "band_lodo_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return summary
