# -*- coding: utf-8 -*-
"""Family-remote deployment router and its leave-one-dataset-out (LODO) check.

Router rule
-----------
Feature  x = nearest sequence identity of a test target to the training panel.
Label    y = family-remote  ⇔  not in the same 40%-identity MMseqs2 cluster.
Rule     x < threshold  →  route to the family-remote (non-homologous) branch.

Validation protocol
-------------------
For each held-out dataset, select the threshold on the pooled other three
datasets by maximizing accuracy over a grid (ties: higher balanced accuracy,
then closeness to the pre-registered 40% rule), and evaluate on the held-out
dataset.  The fixed pre-registered threshold is reported alongside as the
no-fitting control.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .audit import PER_TARGET_FILE


def load_label_panel(path: Path, identity_tag: int = 40
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Read a per-target audit CSV into (x, y) with y=1 for family-remote."""
    df = pd.read_csv(path)
    col = f"same_cluster_{identity_tag}"
    if col not in df.columns:
        raise ValueError(f"{path} lacks column {col}; re-run audit with "
                         f"threshold {identity_tag}%")
    x = df["nearest_train_identity_pct"].to_numpy(float)
    y = (~df[col].astype(bool)).to_numpy(int)
    return x, y


def confusion(x: np.ndarray, y: np.ndarray, thr: float) -> dict:
    pred = (x < thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    return {"thr": thr, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Accuracy": (tp + tn) / len(y),
            "FPR": 1 - tnr if not np.isnan(tnr) else np.nan,
            "FNR": 1 - tpr if not np.isnan(tpr) else np.nan,
            "BalancedAcc": 0.5 * (np.nan_to_num(tpr) + np.nan_to_num(tnr))
                           if not (np.isnan(tpr) and np.isnan(tnr)) else np.nan}


def pick_threshold(x: np.ndarray, y: np.ndarray,
                   grid: Sequence[float], prereg: float) -> Tuple[float, list]:
    """Accuracy-maximizing threshold; ties → balanced acc → closeness to prereg."""
    results = [confusion(x, y, t) for t in grid]
    best_acc = max(r["Accuracy"] for r in results)
    candidates = [r for r in results if r["Accuracy"] == best_acc]
    best_bacc = max(r["BalancedAcc"] for r in candidates)
    candidates = [r for r in candidates if r["BalancedAcc"] == best_bacc]
    candidates.sort(key=lambda r: abs(r["thr"] - prereg))
    return candidates[0]["thr"], results


def run_router_validation(label_paths: Dict[str, Path], out_dir: Path,
                          identity_tag: int = 40,
                          grid: Sequence[float] = (),
                          prereg: float = 40.0) -> dict:
    """Run LODO + fixed-threshold validation; write CSV/JSON/report.

    Parameters
    ----------
    label_paths:
        ``{dataset_name: path to per_target_audit.csv}`` (at least 2 datasets).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if grid is None or len(grid) == 0:
        grid = np.round(np.arange(25.0, 60.0 + 1e-9, 0.5), 1)
    panels = {ds: load_label_panel(p, identity_tag) for ds, p in label_paths.items()}
    datasets = list(label_paths)

    rows, fixed_rows, scans = [], [], {}
    for ds in datasets:
        others = [d for d in datasets if d != ds]
        xtr = np.concatenate([panels[d][0] for d in others])
        ytr = np.concatenate([panels[d][1] for d in others])
        xte, yte = panels[ds]
        thr, scan = pick_threshold(xtr, ytr, grid, prereg)
        scans[ds] = scan
        ev = confusion(xte, yte, thr)
        ev.update({"held_out": ds, "n_test": len(yte), "n_remote": int(yte.sum()),
                   "selected_thr": thr, "rule": "LODO-selected"})
        rows.append(ev)
        fx = confusion(xte, yte, prereg)
        fx.update({"held_out": ds, "n_test": len(yte), "n_remote": int(yte.sum()),
                   "selected_thr": prereg, "rule": "fixed-preregistered"})
        fixed_rows.append(fx)

    cols = ["held_out", "n_test", "n_remote", "selected_thr", "TP", "FP",
            "TN", "FN", "Accuracy", "BalancedAcc", "FPR", "FNR"]
    df_lodo = pd.DataFrame(rows)[cols]
    df_fixed = pd.DataFrame(fixed_rows)[cols]
    df_lodo.to_csv(out_dir / "router_lodo.csv", index=False, encoding="utf-8-sig")
    df_fixed.to_csv(out_dir / "router_fixed_threshold.csv", index=False,
                    encoding="utf-8-sig")
    pd.DataFrame([{"fold": ds,
                   **{f"thr_{r['thr']:.1f}": r["Accuracy"] for r in scans[ds]}}
                  for ds in datasets]).to_csv(
        out_dir / "router_threshold_scan.csv", index=False, encoding="utf-8-sig")

    summary = {
        "identity_label_pct": identity_tag,
        "lodo_mean_acc": float(df_lodo["Accuracy"].mean()),
        "lodo_min_acc": float(df_lodo["Accuracy"].min()),
        "lodo_max_acc": float(df_lodo["Accuracy"].max()),
        "lodo_mean_balanced_acc": float(df_lodo["BalancedAcc"].mean()),
        "fixed_mean_acc": float(df_fixed["Accuracy"].mean()),
        "fixed_mean_balanced_acc": float(df_fixed["BalancedAcc"].mean()),
        "selected_thresholds": dict(zip(df_lodo["held_out"],
                                        df_lodo["selected_thr"])),
        "max_fnr": float(df_lodo["FNR"].max()),
        "max_fpr": float(df_lodo["FPR"].max()),
    }
    with open(out_dir / "router_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "Family-remote router validation (LODO)", "=" * 60,
        f"Rule: nearest_train_identity_pct < thr -> remote; "
        f"ground truth = MMseqs2 {identity_tag}% clusters", "",
        "[A] LODO-selected thresholds:",
        df_lodo.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "",
        f"[B] Fixed pre-registered threshold ({prereg:g}%):",
        df_fixed.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "",
        f"LODO mean Acc={summary['lodo_mean_acc']:.3f} "
        f"({summary['lodo_min_acc']:.3f}-{summary['lodo_max_acc']:.3f}), "
        f"mean balanced Acc={summary['lodo_mean_balanced_acc']:.3f}, "
        f"max FNR={summary['max_fnr']:.3f}",
        f"Fixed rule mean Acc={summary['fixed_mean_acc']:.3f}; "
        f"selected thresholds={summary['selected_thresholds']}",
    ]
    (out_dir / "router_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return summary


def discover_label_paths(labels_root: Path, datasets: Sequence[str]) -> Dict[str, Path]:
    """Build ``{ds: labels_root/ds/per_target_audit.csv}`` for existing dirs."""
    paths = {}
    for ds in datasets:
        p = Path(labels_root) / ds / PER_TARGET_FILE
        if p.exists():
            paths[ds] = p
    if not paths:
        raise FileNotFoundError(f"no {PER_TARGET_FILE} found under {labels_root}")
    return paths
