# -*- coding: utf-8 -*-
"""Binding-spectrum mirror audit (distributional redundancy check).

Sequence-level audits (MMseqs2 clustering) certify that test and training
targets are *distant in sequence space*.  They cannot see *functional* mirror
redundancy: a test complex may sit in a different sequence cluster yet repeat
the mutational ΔΔG profile of some training complex almost exactly.

Two complementary redundancy scores per (test complex, train complex) pair
are computed:

1. **Shared-mutation Spearman ρ** — over mutations measured in *both*
   complexes, correlate the ΔΔG values.  Requires ``--min-shared-mutations``
   shared mutations (default 3) and non-constant vectors.
2. **KS distribution similarity** — ``similarity = 1 - KS statistic`` of the
   two complexes' ΔΔG distributions.  Requires at least ``--min-sites-ks``
   entries per side (default 3).

For every test complex the *maximum* score over all training complexes is
reported.  A complex is flagged as mirror-redundant when
``max_score >= threshold`` (default 0.7; 0.5 is reported as a moderate band).

Input (long CSV, one row per measured mutation)::

    complex_id,split,mutation,ddG
    1EAW_A_B,test,MA14A,1.23
    ...

Alternatively provide affinity columns instead of ``ddG`` — then
``ddG = R * T * ln(aff_wt / aff_mut)`` is derived per row (``T`` defaults to
298.15 K when a temperature column is absent or unparsable).  Affinity units
cancel inside the ratio, so nM / µM / M all work; the affinities must be
*Kd-like* (larger = weaker binding).

Optional cross-cluster intersection: pass the audit output
``per_target_audit.csv`` (from ``homology-audit audit``) plus the column in
the input that carries the audit ``target_id``.  A complex counts as
cross-cluster only when *all* of its targets are cross-cluster.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

GAS_CONSTANT = 1.987e-3  # kcal/(mol*K)

PER_COMPLEX_FILE = "per_complex_mirror.csv"
SUMMARY_FILE = "mirror_summary.csv"
REPORT_FILE = "mirror_report.txt"


def compute_ddg(df: pd.DataFrame, ddg_col: Optional[str],
                aff_mut_col: Optional[str], aff_wt_col: Optional[str],
                temp_col: Optional[str], default_temp: float) -> pd.Series:
    """Return the ΔΔG column, deriving it from affinities when needed."""
    if ddg_col:
        if ddg_col not in df.columns:
            raise ValueError(f"ddG column {ddg_col!r} not found in input")
        return pd.to_numeric(df[ddg_col], errors="coerce")
    missing = [c for c in (aff_mut_col, aff_wt_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"missing affinity column(s) {missing}; provide either --ddg-col "
            f"or --aff-mut-col/--aff-wt-col")
    aff_mut = pd.to_numeric(df[aff_mut_col], errors="coerce")
    aff_wt = pd.to_numeric(df[aff_wt_col], errors="coerce")
    if temp_col and temp_col in df.columns:
        temp = pd.to_numeric(df[temp_col], errors="coerce").fillna(default_temp)
    else:
        temp = pd.Series(default_temp, index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        ddg = GAS_CONSTANT * temp * np.log(aff_wt / aff_mut)
    return ddg.replace([np.inf, -np.inf], np.nan)


def _build_mutation_dict(sub: pd.DataFrame, complex_col: str, mutation_col: str,
                         ddg_col: str) -> Dict[str, Dict[str, float]]:
    """``{complex: {mutation: mean ΔΔG over repeated measurements}}``."""
    grouped = sub.groupby([complex_col, mutation_col])[ddg_col].mean()
    out: Dict[str, Dict[str, float]] = {}
    for (cx, mut), val in grouped.items():
        out.setdefault(str(cx), {})[str(mut)] = float(val)
    return out


def _build_ddg_vectors(sub: pd.DataFrame, complex_col: str,
                       ddg_col: str) -> Dict[str, np.ndarray]:
    return {str(cx): g[ddg_col].to_numpy(float)
            for cx, g in sub.groupby(complex_col)}


def _best_spearman(test_muts: Dict[str, float],
                   train_muts: Dict[str, Dict[str, float]],
                   min_shared: int):
    best_rho, best_train, n_shared, n_comparable = -2.0, None, 0, 0
    for train_cx, tmuts in train_muts.items():
        shared = set(test_muts) & set(tmuts)
        if len(shared) < min_shared:
            continue
        v_test = np.array([test_muts[m] for m in shared])
        v_train = np.array([tmuts[m] for m in shared])
        if np.std(v_test) == 0 or np.std(v_train) == 0:
            continue
        n_comparable += 1
        rho = spearmanr(v_test, v_train).statistic
        if rho is not None and np.isfinite(rho) and rho > best_rho:
            best_rho, best_train, n_shared = float(rho), train_cx, len(shared)
    return ((best_rho if best_rho > -2 else np.nan), best_train, n_shared,
            n_comparable)


def _best_ks(test_vec: np.ndarray, train_vecs: Dict[str, np.ndarray],
             min_sites: int):
    best_sim, best_train = 0.0, None
    if len(test_vec) < min_sites:
        return np.nan, None
    for train_cx, train_v in train_vecs.items():
        if len(train_v) < min_sites:
            continue
        sim = 1.0 - float(ks_2samp(test_vec, train_v).statistic)
        if sim > best_sim:
            best_sim, best_train = sim, train_cx
    return best_sim, best_train


def run_mirror(data_csv: Path, out_dir: Path, dataset_name: str = "dataset",
               complex_col: str = "complex_id", split_col: str = "split",
               mutation_col: str = "mutation", ddg_col: Optional[str] = "ddG",
               aff_mut_col: Optional[str] = None,
               aff_wt_col: Optional[str] = None,
               temp_col: Optional[str] = None, default_temp: float = 298.15,
               min_shared_mutations: int = 3, min_sites_ks: int = 3,
               thresholds: Sequence[float] = (0.5, 0.7),
               per_target_audit: Optional[Path] = None,
               identity_tag: int = 40, target_id_col: Optional[str] = None,
               verbose: bool = True) -> dict:
    """Run the binding-spectrum mirror audit for one dataset."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted(set(float(t) for t in thresholds))

    df = pd.read_csv(data_csv)
    for col, label in ((complex_col, "complex"), (mutation_col, "mutation"),
                       (split_col, "split")):
        if col not in df.columns:
            raise ValueError(f"{label} column {col!r} not found in {data_csv}")
    df = df.copy()
    df["_ddG"] = compute_ddg(df, ddg_col, aff_mut_col, aff_wt_col,
                             temp_col, default_temp)
    df = df.dropna(subset=["_ddG", mutation_col])
    df = df[(df["_ddG"] != 0)]
    if df.empty:
        raise ValueError("no usable mutation rows after dropping NaN ΔΔG")
    df[complex_col] = df[complex_col].astype(str)

    train_mask = df[split_col].astype(str).str.lower().eq("train")
    test_mask = df[split_col].astype(str).str.lower().eq("test")
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"split column {split_col!r} must contain 'train' and 'test' "
            f"values; found: {sorted(df[split_col].astype(str).unique())}")
    train_df, test_df = df[train_mask], df[test_mask]

    train_muts = _build_mutation_dict(train_df, complex_col, mutation_col, "_ddG")
    test_muts = _build_mutation_dict(test_df, complex_col, mutation_col, "_ddG")
    train_vecs = _build_ddg_vectors(train_df, complex_col, "_ddG")
    test_vecs = _build_ddg_vectors(test_df, complex_col, "_ddG")

    # optional cross-cluster annotation from the sequence audit
    cross_map: Dict[str, bool] = {}
    if per_target_audit is not None:
        audit = pd.read_csv(per_target_audit)
        label_col = f"same_cluster_{identity_tag}"
        if label_col not in audit.columns or target_id_col is None \
                or target_id_col not in df.columns:
            print(f"[warn] mirror: cross-cluster annotation skipped "
                  f"(need {label_col} in audit and {target_id_col!r} in data)")
        else:
            audit_map = dict(zip(audit[target_id_col].astype(str),
                                 audit[label_col].astype(bool)))
            per_complex_targets = (df[[complex_col, target_id_col]]
                                   .dropna().astype(str).groupby(complex_col)
                                   [target_id_col].apply(set).to_dict())
            for cx, ids in per_complex_targets.items():
                known = [audit_map[i] for i in ids if i in audit_map]
                if known:
                    cross_map[cx] = not any(known)

    rows: List[dict] = []
    for test_cx in sorted(test_muts):
        rho, best_rho_train, n_shared, n_comparable = _best_spearman(
            test_muts[test_cx], train_muts, min_shared_mutations)
        sim, best_ks_train = _best_ks(
            test_vecs.get(test_cx, np.array([])), train_vecs, min_sites_ks)
        tvec = test_vecs.get(test_cx, np.array([]))
        rows.append({
            "dataset": dataset_name, "test_complex": test_cx,
            "n_mutations": len(test_muts[test_cx]),
            "n_comparable_train_spearman": n_comparable,
            "max_spearman_rho": rho, "best_train_spearman": best_rho_train,
            "n_shared_best": n_shared,
            "max_ks_similarity": sim, "best_train_ks": best_ks_train,
            "ddG_mean": float(np.mean(tvec)) if len(tvec) else np.nan,
            "ddG_std": float(np.std(tvec)) if len(tvec) else np.nan,
            "cross_cluster": cross_map.get(test_cx, None)})
    res = pd.DataFrame(rows).sort_values("test_complex")

    # redundancy flags per threshold (KS similarity drives the default flag;
    # Spearman needs the stricter shared-mutation support so it is reported
    # separately rather than merged)
    for t in thresholds:
        res[f"mirror_redundant_ks_{int(round(t*100))}"] = (
            res["max_ks_similarity"] >= t)
    res.to_csv(out_dir / PER_COMPLEX_FILE, index=False, encoding="utf-8-sig")

    n_test = len(res)
    summary_rows = []
    for t in thresholds:
        tag = int(round(t * 100))
        n_ks = int((res["max_ks_similarity"] >= t).sum())
        rho_valid = res["max_spearman_rho"].dropna()
        n_rho = int((rho_valid >= t).sum())
        sub = res[res["cross_cluster"] == True]  # noqa: E712  (boolean column)
        n_cross = len(sub)
        n_ks_cross = int((sub["max_ks_similarity"] >= t).sum()) if n_cross else 0
        summary_rows.append({
            "dataset": dataset_name, "threshold": t,
            "n_test_complexes": n_test,
            "n_ks_redundant": n_ks,
            "pct_ks_redundant": round(100 * n_ks / n_test, 1) if n_test else np.nan,
            "n_spearman_comparable": int(len(rho_valid)),
            "n_spearman_redundant": n_rho,
            "n_cross_cluster": n_cross,
            "n_cross_cluster_ks_redundant": n_ks_cross,
            "pct_cross_cluster_ks_redundant":
                round(100 * n_ks_cross / n_cross, 1) if n_cross else np.nan})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / SUMMARY_FILE, index=False, encoding="utf-8-sig")

    lines = ["Binding-spectrum mirror audit", "=" * 60,
             f"dataset={dataset_name}  input={data_csv}",
             f"train complexes={len(train_muts)}  test complexes={n_test}",
             f"shared-mutation Spearman needs >={min_shared_mutations} shared "
             f"mutations; KS needs >={min_sites_ks} entries per side", ""]
    for r in summary_rows:
        cross_note = (f"; cross-cluster: {r['n_cross_cluster_ks_redundant']}/"
                      f"{r['n_cross_cluster']} = "
                      f"{r['pct_cross_cluster_ks_redundant']}%"
                      ) if r["n_cross_cluster"] else ""
        lines.append(
            f"  threshold {r['threshold']:.2f}: KS-similar test complexes "
            f"{r['n_ks_redundant']}/{r['n_test_complexes']} = "
            f"{r['pct_ks_redundant']}%{cross_note}")
    top_ks = res.nlargest(min(10, n_test), "max_ks_similarity")
    lines += ["", "Top complexes by max KS similarity:"]
    for r in top_ks.itertuples():
        lines.append(f"  {r.test_complex}  sim={r.max_ks_similarity:.3f}  "
                     f"n_mut={r.n_mutations}  best_train={r.best_train_ks}")
    rho_valid = res["max_spearman_rho"].dropna()
    if len(rho_valid):
        top_sp = res.dropna(subset=["max_spearman_rho"]).nlargest(
            min(10, len(rho_valid)), "max_spearman_rho")
        lines += ["", "Top complexes by shared-mutation Spearman ρ:"]
        for r in top_sp.itertuples():
            lines.append(f"  {r.test_complex}  rho={r.max_spearman_rho:.3f}  "
                         f"n_shared={r.n_shared_best}  "
                         f"best_train={r.best_train_spearman}")
    (out_dir / REPORT_FILE).write_text("\n".join(lines), encoding="utf-8")
    if verbose:
        print("\n".join(lines))
    return {"per_complex": res, "summary": summary,
            "out_dir": str(out_dir)}
