# -*- coding: utf-8 -*-
"""End-to-end homology audit for one dataset or a batch of datasets.

Pipeline (per dataset)
----------------------
1. Collect de-duplicated train/test protein sequences and write FASTA files
   (``sequences_train.fasta``, ``sequences_test.fasta``, merged
   ``sequences_all.fasta``) with deterministic ``train_#### / test_####`` IDs.
2. ``mmseqs easy-search`` test vs train to obtain each test target's nearest
   training identity under bidirectional coverage (``-c 0.8 --cov-mode 0``).
3. ``mmseqs easy-cluster`` on the merged FASTA at every identity threshold
   (default 0.40 / 0.60 / 0.80) and derive, for each test target, whether it
   shares a cluster with any training target.
4. Emit ``per_target_audit.csv`` and dataset-level ``audit_summary.csv`` rows.

"Cross-cluster ratio" = fraction of test targets that share NO cluster with
any training target, i.e. the homology-free (family-remote) fraction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .io_utils import (build_id_maps, read_fasta, sequences_from_csv, write_fasta)
from .labels import cross_cluster_stats, nearest_train_identity, read_clusters
from .mmseqs import easy_cluster, easy_search, resolve_mmseqs

PER_TARGET_FILE = "per_target_audit.csv"
SUMMARY_FILE = "audit_summary.csv"
REPORT_FILE = "audit_report.txt"


def _load_sequences(train_csv, test_csv, train_fasta, test_fasta, seq_column
                    ) -> tuple[List[str], List[str]]:
    if train_csv and test_csv:
        return (sequences_from_csv(Path(train_csv), seq_column),
                sequences_from_csv(Path(test_csv), seq_column))
    if train_fasta and test_fasta:
        tr = [s for _, s in read_fasta(Path(train_fasta))]
        te = [s for _, s in read_fasta(Path(test_fasta))]
        return sorted(set(tr)), sorted(set(te))
    raise ValueError("provide either --train-csv/--test-csv or "
                     "--train-fasta/--test-fasta")


def run_audit(name: str, out_dir: Path,
              thresholds: Sequence[float] = (0.4, 0.6, 0.8),
              coverage: float = 0.8, cov_mode: int = 0, cluster_mode: int = 0,
              threads: int = 8, mmseqs_path: Optional[Path] = None,
              train_csv: Optional[Path] = None, test_csv: Optional[Path] = None,
              train_fasta: Optional[Path] = None, test_fasta: Optional[Path] = None,
              seq_column: str = "target_sequence", verbose: bool = True
              ) -> Dict[str, object]:
    """Run the complete audit for one dataset; return summary rows + paths."""
    out_dir = (Path(out_dir) / name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mmseqs = resolve_mmseqs(mmseqs_path)

    train_seqs, test_seqs = _load_sequences(
        train_csv, test_csv, train_fasta, test_fasta, seq_column)
    train_map, test_map = build_id_maps(train_seqs, test_seqs)
    if verbose:
        print(f"[{name}] train targets={len(train_map)} "
              f"test targets={len(test_map)}")

    write_fasta(train_map, out_dir / "sequences_train.fasta")
    write_fasta(test_map, out_dir / "sequences_test.fasta")
    all_fasta = out_dir / "sequences_all.fasta"
    write_fasta({**train_map, **test_map}, all_fasta)

    # --- nearest identity ---------------------------------------------------
    m8 = easy_search(mmseqs, out_dir / "sequences_test.fasta",
                     out_dir / "sequences_train.fasta",
                     out_dir / "search_test_vs_train.m8",
                     out_dir / "tmp_search", coverage=coverage,
                     cov_mode=cov_mode, threads=threads)
    best = nearest_train_identity(m8)

    # id -> sequence (for transparent joins against user predictions)
    test_seq_by_id = {sid: seq for seq, sid in test_map.items()}
    per_row: Dict[str, dict] = {
        sid: {"target_id": sid, "target_sequence": test_seq_by_id[sid],
              "nearest_train_identity_pct": best.get(sid, 0.0)}
        for sid in test_map.values()
    }

    # --- clustering at each identity threshold -----------------------------
    summary_rows: List[dict] = []
    for t in thresholds:
        tag = int(round(t * 100))
        tsv = easy_cluster(mmseqs, all_fasta, out_dir / f"clu_{tag}",
                           out_dir / f"tmp_clu_{tag}", min_seq_id=t,
                           coverage=coverage, cov_mode=cov_mode,
                           cluster_mode=cluster_mode, threads=threads)
        clusters = read_clusters(tsv)
        n_same, n_cross, n_both, shared = cross_cluster_stats(
            clusters, list(test_map.values()))
        n_test = len(test_map)
        for sid in test_map.values():
            per_row[sid][f"same_cluster_{tag}"] = sid in shared
        summary_rows.append({
            "dataset": name, "n_train_targets": len(train_map),
            "n_test_targets": n_test, "coverage": coverage,
            "cov_mode": cov_mode, "cluster_mode": cluster_mode,
            "identity_threshold_pct": tag, "n_clusters": len(clusters),
            "n_shared_train_test_clusters": n_both,
            "n_test_targets_same_cluster": n_same,
            "n_test_targets_cross_cluster": n_cross,
            "same_cluster_ratio_pct": round(100 * n_same / n_test, 2),
            "cross_cluster_ratio_pct": round(100 * n_cross / n_test, 2),
        })
        if verbose:
            print(f"[{name}] id={tag}% clusters={len(clusters)} "
                  f"same-cluster test={n_same}/{n_test} "
                  f"cross-cluster={summary_rows[-1]['cross_cluster_ratio_pct']}%")

    per_target = pd.DataFrame(per_row.values()).sort_values("target_id")
    per_target.to_csv(out_dir / PER_TARGET_FILE, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "audit_summary_dataset.csv", index=False,
                   encoding="utf-8-sig")
    return {"dataset": name, "out_dir": str(out_dir),
            "n_train": len(train_map), "n_test": len(test_map),
            "summary_rows": summary_rows}


def run_batch(data_root: Path, datasets: Sequence[str], out_root: Path,
              thresholds: Sequence[float] = (0.4, 0.6, 0.8),
              coverage: float = 0.8, cov_mode: int = 0, cluster_mode: int = 0,
              threads: int = 8, mmseqs_path: Optional[Path] = None,
              seq_column: str = "target_sequence",
              train_template: str = "{ds}_train.csv",
              test_template: str = "{ds}_test.csv",
              verbose: bool = True) -> pd.DataFrame:
    """Audit every dataset under ``<data_root>/<ds>_{train,test}.csv``.

    The filename templates can be customized (e.g. ``"{ds}.train.fasta"`` is
    not supported here — for FASTA inputs call :func:`run_audit` directly).
    """
    data_root, out_root = Path(data_root).resolve(), Path(out_root).resolve()
    all_rows: List[dict] = []
    for ds in datasets:
        result = run_audit(
            name=ds, out_dir=out_root, thresholds=thresholds,
            coverage=coverage, cov_mode=cov_mode, cluster_mode=cluster_mode,
            threads=threads, mmseqs_path=mmseqs_path,
            train_csv=data_root / train_template.format(ds=ds),
            test_csv=data_root / test_template.format(ds=ds),
            seq_column=seq_column, verbose=verbose)
        all_rows.extend(result["summary_rows"])  # type: ignore[arg-type]
    summary = pd.DataFrame(all_rows)
    out_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_root / SUMMARY_FILE, index=False, encoding="utf-8-sig")
    lines = ["Homology audit — batch summary", "=" * 60,
             "cross_cluster_ratio_pct = homology-free test targets (%)", "",
             summary.to_string(index=False)]
    (out_root / REPORT_FILE).write_text("\n".join(lines), encoding="utf-8")
    if verbose:
        print("\n".join(lines))
    return summary
