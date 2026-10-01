# -*- coding: utf-8 -*-
"""Core label bookkeeping: parse MMseqs2 cluster TSVs and m8 search output."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import pandas as pd

from .io_utils import TEST_PREFIX, TRAIN_PREFIX

M8_COLUMNS = ["query", "target", "pident", "alnlen", "mismatch", "gapopen",
              "qstart", "qend", "tstart", "tend", "evalue", "bits"]


def read_clusters(tsv_path: Path) -> Dict[str, List[str]]:
    """Parse ``*_cluster.tsv`` (representative<TAB>member) into a dict."""
    clusters: Dict[str, List[str]] = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            clusters.setdefault(parts[0], []).append(parts[1])
    return clusters


def shared_test_targets(clusters: Dict[str, List[str]]) -> Set[str]:
    """Test IDs that share at least one cluster with a training ID."""
    shared: Set[str] = set()
    for members in clusters.values():
        ms = set(members)
        has_train = any(m.startswith(TRAIN_PREFIX) for m in ms)
        if has_train:
            shared.update(m for m in ms if m.startswith(TEST_PREFIX))
    return shared


def count_shared_clusters(clusters: Dict[str, List[str]]) -> int:
    """Number of clusters containing both training and test members."""
    n = 0
    for members in clusters.values():
        ms = set(members)
        if any(m.startswith(TRAIN_PREFIX) for m in ms) and \
           any(m.startswith(TEST_PREFIX) for m in ms):
            n += 1
    return n


def nearest_train_identity(m8_path: Path) -> Dict[str, float]:
    """Max percent sequence identity of each query to any training target.

    MMseqs2 m8 ``pident`` is a fraction (0-1); converted to percent here.
    Missing queries (no bidirectionally covered hit) get 0.0.
    """
    best: Dict[str, float] = {}
    if not Path(m8_path).exists() or Path(m8_path).stat().st_size == 0:
        return best
    for chunk in pd.read_csv(m8_path, sep="\t", header=None, names=M8_COLUMNS,
                             chunksize=100_000):
        chunk["pct"] = chunk["pident"].astype(float) * 100.0
        for query, sub in chunk.groupby("query"):
            value = float(sub["pct"].max())
            if value > best.get(query, -1.0):
                best[query] = value
    return best


def cross_cluster_stats(clusters: Dict[str, List[str]], test_ids: Sequence[str]
                        ) -> Tuple[int, int, int, Set[str]]:
    """Return (n_shared_test, n_cross_test, n_shared_clusters, shared_id_set)."""
    shared = shared_test_targets(clusters)
    test_set = set(test_ids)
    shared &= test_set
    n_shared = len(shared)
    return n_shared, len(test_set) - n_shared, count_shared_clusters(clusters), shared
