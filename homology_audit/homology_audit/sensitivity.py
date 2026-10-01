# -*- coding: utf-8 -*-
"""Audit-parameter sensitivity and clustering-algorithm cross-checks.

* :func:`run_grid` re-clusters the merged train+test FASTA over a
  coverage x identity grid and reports cross-cluster ratios.  When a reference
  audit summary exists (the original audit), the reference cells (e.g.
  c=0.8 / 40,60,80%) are checked for exact reproduction.
* :func:`run_algorithm_check` compares MMseqs2 default set-cover / connected
  components (``--cluster-mode 0``) with CD-HIT-style greedy incremental
  (``--cluster-mode 2``): cluster counts, cross-cluster ratios, per-target
  label agreement and Cohen's kappa.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .audit import PER_TARGET_FILE
from .labels import read_clusters, shared_test_targets
from .mmseqs import easy_cluster, resolve_mmseqs


def _shared_and_count(tsv: Path, test_ids: Sequence[str]
                      ) -> Tuple[int, int, set]:
    clusters = read_clusters(tsv)
    shared = shared_test_targets(clusters) & set(test_ids)
    return len(clusters), len(shared), shared


def run_grid(dataset_dir: Path, out_dir: Path,
             coverages: Sequence[float] = (0.7, 0.8, 0.9),
             identities: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
             cov_mode: int = 0, cluster_mode: int = 0, threads: int = 8,
             mmseqs_path: Optional[Path] = None,
             reference_summary: Optional[Path] = None,
             dataset_name: Optional[str] = None) -> pd.DataFrame:
    """Parameter-grid clustering for one audited dataset directory."""
    dataset_dir, out_dir = Path(dataset_dir).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mmseqs = resolve_mmseqs(mmseqs_path)
    fasta = dataset_dir / "sequences_all.fasta"
    test_ids = pd.read_csv(dataset_dir / PER_TARGET_FILE)["target_id"].tolist()
    name = dataset_name or dataset_dir.name

    rows = []
    for cov in coverages:
        for ident in identities:
            tag = f"clu_cov{int(round(cov*100))}_id{int(round(ident*100))}"
            tsv = easy_cluster(mmseqs, fasta, out_dir / tag, out_dir / f"tmp_{tag}",
                               min_seq_id=ident, coverage=cov, cov_mode=cov_mode,
                               cluster_mode=cluster_mode, threads=threads)
            n_clusters, n_same, _ = _shared_and_count(tsv, test_ids)
            n_test = len(test_ids)
            rows.append({"dataset": name, "coverage": cov,
                         "identity_pct": int(round(ident * 100)),
                         "n_clusters": n_clusters, "n_test": n_test,
                         "n_test_same_cluster": n_same,
                         "n_test_cross_cluster": n_test - n_same,
                         "cross_cluster_ratio_pct":
                             round(100 * (n_test - n_same) / n_test, 2)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "param_grid.csv", index=False, encoding="utf-8-sig")

    if reference_summary and Path(reference_summary).exists():
        ref = pd.read_csv(reference_summary)
        checks = []
        ok_all = True
        for t in identities:
            tagp = int(round(t * 100))
            if tagp not in {40, 60, 80}:
                continue
            a = df[(df.coverage == 0.8) & (df.identity_pct == tagp)]
            b = ref[(ref.dataset == name) & (ref.identity_threshold_pct == tagp)]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            match = int(a.n_test_cross_cluster) == int(b.n_test_targets_cross_cluster)
            ok_all &= match
            checks.append({"dataset": name, "identity_pct": tagp,
                           "recalc_cross": int(a.n_test_cross_cluster),
                           "original_cross": int(b.n_test_targets_cross_cluster),
                           "recalc_ratio": a.cross_cluster_ratio_pct,
                           "original_ratio": b.cross_cluster_ratio_pct,
                           "match": bool(match)})
        chk = pd.DataFrame(checks)
        chk.to_csv(out_dir / "reproduction_check.csv", index=False,
                   encoding="utf-8-sig")
        print(f"reproduction check (c=0.8 vs original audit): "
              f"{'ALL MATCH' if ok_all else 'MISMATCH FOUND'}")
    return df


def cohen_kappa(labels_a: Dict[str, bool], labels_b: Dict[str, bool]
                ) -> Tuple[float, int, int]:
    """Cohen's kappa + agreement count over the union of keys."""
    keys = sorted(set(labels_a) | set(labels_b))
    y0 = np.array([int(labels_a[k]) for k in keys])
    y1 = np.array([int(labels_b[k]) for k in keys])
    n = len(keys)
    p_o = float((y0 == y1).mean())
    p_e = float(y0.mean() * y1.mean() + (1 - y0.mean()) * (1 - y1.mean()))
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-12 else 1.0
    return float(kappa), int((y0 == y1).sum()), n


def run_algorithm_check(dataset_dir: Path, out_dir: Path,
                        identities: Sequence[float] = (0.4, 0.6, 0.8),
                        coverage: float = 0.8, threads: int = 8,
                        mmseqs_path: Optional[Path] = None,
                        dataset_name: Optional[str] = None) -> pd.DataFrame:
    """Compare cluster-mode 0 (set-cover) vs mode 2 (CD-HIT-style greedy)."""
    dataset_dir, out_dir = Path(dataset_dir).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mmseqs = resolve_mmseqs(mmseqs_path)
    fasta = dataset_dir / "sequences_all.fasta"
    meta = pd.read_csv(dataset_dir / PER_TARGET_FILE)
    test_ids = meta["target_id"].tolist()
    name = dataset_name or dataset_dir.name

    rows = []
    for ident in identities:
        tagp = int(round(ident * 100))
        mode0_tsv = dataset_dir / f"clu_{tagp}_cluster.tsv"
        if not mode0_tsv.exists():
            raise FileNotFoundError(
                f"{mode0_tsv} not found; run the audit at {tagp}% first")
        n0, _, shared0 = _shared_and_count(mode0_tsv, test_ids)
        prefix = out_dir / f"greedy_clu_id{tagp}"
        tsv2 = easy_cluster(mmseqs, fasta, prefix, out_dir / f"tmp_greedy_{tagp}",
                            min_seq_id=ident, coverage=coverage, cov_mode=0,
                            cluster_mode=2, threads=threads)
        n2, _, shared2 = _shared_and_count(tsv2, test_ids)
        labels0 = {tid: tid in shared0 for tid in test_ids}
        labels2 = {tid: tid in shared2 for tid in test_ids}
        kappa, n_agree, n_tot = cohen_kappa(labels0, labels2)
        n_test = len(test_ids)
        rows.append({
            "dataset": name, "identity_pct": tagp,
            "setcover_n_clusters": n0, "greedy_n_clusters": n2,
            "setcover_cross_ratio_pct": round(100 * (n_test - len(shared0)) / n_test, 2),
            "greedy_cross_ratio_pct": round(100 * (n_test - len(shared2)) / n_test, 2),
            "label_agreement_pct": round(100 * n_agree / n_tot, 2),
            "cohen_kappa": round(kappa, 4)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cluster_algorithm_comparison.csv", index=False,
              encoding="utf-8-sig")
    return df
