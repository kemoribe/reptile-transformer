# -*- coding: utf-8 -*-
"""Random-sequence negative control.

Each test target's residues are permuted *within the sequence* (length and
amino-acid composition preserved) for several seeds.  The shuffled sequences
are clustered against the untouched training panel under the same audit
settings; a valid audit must classify essentially all shuffled targets as
cross-cluster (no spurious homology) and report very low nearest identities.

The bidirectional coverage filter (``-c 0.8``) suppresses random local
matches for long proteins, so 0% same-cluster is the expected algorithmic
behaviour, not an artifact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .audit import PER_TARGET_FILE
from .io_utils import read_fasta, write_fasta_pairs
from .labels import nearest_train_identity, read_clusters, shared_test_targets
from .mmseqs import easy_cluster, easy_search, resolve_mmseqs


def shuffle_records(records, seed: int):
    """Per-sequence residue permutation preserving length/composition."""
    rng = np.random.default_rng(seed)
    out = []
    for name, seq in records:
        arr = np.array(list(seq))
        rng.shuffle(arr)
        out.append((name, "".join(arr)))
    return out


def run_negative_control(dataset_dir: Path, out_dir: Path,
                         seeds: Sequence[int] = (0, 1, 2),
                         thresholds: Sequence[float] = (0.4, 0.6, 0.8),
                         coverage: float = 0.8, threads: int = 8,
                         mmseqs_path: Optional[Path] = None,
                         dataset_name: Optional[str] = None) -> pd.DataFrame:
    """Run the shuffled-sequence negative control for one audited dataset."""
    dataset_dir, out_dir = Path(dataset_dir).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mmseqs = resolve_mmseqs(mmseqs_path)

    train = read_fasta(dataset_dir / "sequences_train.fasta")
    test = read_fasta(dataset_dir / "sequences_test.fasta")
    n_test = len(test)
    lengths = np.array([len(s) for _, s in test])
    name = dataset_name or dataset_dir.name

    rows = []
    for seed in seeds:
        shuffled = shuffle_records(test, seed)
        shuf_fasta = out_dir / f"shuffled_test_seed{seed}.fasta"
        all_fasta = out_dir / f"shuffled_all_seed{seed}.fasta"
        write_fasta_pairs(shuffled, shuf_fasta)
        write_fasta_pairs(train + shuffled, all_fasta)

        m8 = easy_search(mmseqs, shuf_fasta, dataset_dir / "sequences_train.fasta",
                         out_dir / f"search_seed{seed}.m8",
                         out_dir / f"tmp_search_seed{seed}", coverage=coverage,
                         threads=threads)
        near = nearest_train_identity(m8)
        near_vals = np.array([near.get(nm, 0.0) for nm, _ in shuffled])

        for t in thresholds:
            tagp = int(round(t * 100))
            prefix = out_dir / f"clu_seed{seed}_id{tagp}"
            tsv = easy_cluster(mmseqs, all_fasta, prefix,
                               out_dir / f"tmp_seed{seed}_id{tagp}",
                               min_seq_id=t, coverage=coverage, cluster_mode=0,
                               threads=threads)
            clusters = read_clusters(tsv)
            shared = shared_test_targets(clusters)
            n_same = len(shared)
            rows.append({
                "dataset": name, "seed": seed, "identity_pct": tagp,
                "n_test": n_test, "n_clusters": len(clusters),
                "n_shuffled_same_cluster": n_same,
                "cross_cluster_ratio_pct": round(100 * (n_test - n_same) / n_test, 2),
                "nearest_id_max_pct": round(float(near_vals.max()), 2),
                "nearest_id_mean_pct": round(float(near_vals.mean()), 2),
                "min_test_len": int(lengths.min()),
                "max_test_len": int(lengths.max())})
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "negative_control_results.csv", index=False,
              encoding="utf-8-sig")

    agg = (df.groupby(["dataset", "identity_pct"])
             .agg(cross_ratio_mean=("cross_cluster_ratio_pct", "mean"),
                  cross_ratio_min=("cross_cluster_ratio_pct", "min"),
                  cross_ratio_max=("cross_cluster_ratio_pct", "max"),
                  nearest_id_max=("nearest_id_max_pct", "max"),
                  nearest_id_mean=("nearest_id_mean_pct", "mean"))
             .reset_index())
    agg.to_csv(out_dir / "negative_control_summary.csv", index=False,
               encoding="utf-8-sig")

    lines = [f"Negative control — {name} (within-sequence residue shuffle)",
             "=" * 64, agg.to_string(index=False, float_format=lambda v: f"{v:.2f}"),
             "", f"Expectation: cross-cluster ≈ 100%, nearest identity well "
             f"below the {int(round(min(thresholds)*100))}% threshold in all "
             f"{len(seeds)} seeds."]
    (out_dir / "negative_control_report.txt").write_text(
        "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return df
