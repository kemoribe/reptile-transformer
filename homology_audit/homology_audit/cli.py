# -*- coding: utf-8 -*-
"""``homology-audit`` command-line interface.

Run ``homology-audit <command> -h`` for command-specific options.  The typical
workflow is::

    homology-audit audit-batch --data-root data/ --datasets davis,kiba \
        --out results/audit/
    homology-audit router --labels-root results/audit/ --datasets davis,kiba \
        --out results/router/
    homology-audit sensitivity --dataset-dir results/audit/davis \
        --reference results/audit/audit_summary.csv --out results/sens/davis
    homology-audit algorithm-check --dataset-dir results/audit/davis \
        --out results/sens/davis
    homology-audit negative-control --dataset-dir results/audit/davis \
        --out results/neg/davis
    homology-audit evaluate --predictions preds.csv --labels-root results/audit/ \
        --datasets davis,kiba --out results/eval/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .audit import run_audit, run_batch
from .band_lodo import run_band_lodo
from .degradation import run_degradation
from .evaluate import run_evaluation
from .io_utils import parse_float_list
from .mirror import run_mirror
from .negative import run_negative_control
from .router import discover_label_paths, run_router_validation
from .sensitivity import run_algorithm_check, run_grid


def _csv_list(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def _thresholds(text: str):
    vals = parse_float_list(text)
    return vals or [0.4, 0.6, 0.8]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="homology-audit",
        description="Sequence-homology audit toolkit for target-disjoint DTA "
                    "benchmarking (MMseqs2-based).")
    p.add_argument("--version", action="version", version=f"homology-audit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--mmseqs", type=Path, default=None,
                        help="path to mmseqs binary (else HOMOLOGY_AUDIT_MMSEQS "
                             "or PATH is used)")
        sp.add_argument("--threads", type=int, default=8)

    # ---- audit (single dataset) ------------------------------------------
    a = sub.add_parser("audit", help="audit one dataset (CSV or FASTA inputs)")
    src = a.add_argument_group("inputs (CSV pair or FASTA pair)")
    src.add_argument("--train-csv", type=Path)
    src.add_argument("--test-csv", type=Path)
    src.add_argument("--train-fasta", type=Path)
    src.add_argument("--test-fasta", type=Path)
    src.add_argument("--seq-column", default="target_sequence",
                     help="sequence column name in CSV inputs")
    a.add_argument("--name", required=True, help="dataset name (output subfolder)")
    a.add_argument("--out", type=Path, required=True, help="output root directory")
    a.add_argument("--thresholds", default="0.4,0.6,0.8",
                   help="comma-separated identity thresholds, e.g. 0.4,0.6,0.8")
    a.add_argument("--coverage", type=float, default=0.8, help="-c bidirectional coverage")
    a.add_argument("--cov-mode", type=int, default=0)
    a.add_argument("--cluster-mode", type=int, default=0)
    add_common(a)

    # ---- audit-batch ------------------------------------------------------
    ab = sub.add_parser("audit-batch",
                        help="audit multiple datasets from <name>_train/test.csv")
    ab.add_argument("--data-root", type=Path, required=True)
    ab.add_argument("--datasets", type=_csv_list, required=True)
    ab.add_argument("--out", type=Path, required=True)
    ab.add_argument("--thresholds", default="0.4,0.6,0.8")
    ab.add_argument("--coverage", type=float, default=0.8)
    ab.add_argument("--cov-mode", type=int, default=0)
    ab.add_argument("--cluster-mode", type=int, default=0)
    ab.add_argument("--seq-column", default="target_sequence")
    ab.add_argument("--train-template", default="{ds}_train.csv")
    ab.add_argument("--test-template", default="{ds}_test.csv")
    add_common(ab)

    # ---- router -----------------------------------------------------------
    r = sub.add_parser("router", help="family-remote router LODO validation")
    r.add_argument("--labels-root", type=Path,
                   help="root containing <dataset>/per_target_audit.csv")
    r.add_argument("--datasets", type=_csv_list,
                   help="dataset subfolder names under --labels-root")
    r.add_argument("--label", action="append", default=[],
                   metavar="NAME=PATH",
                   help="explicit label file, repeatable (alternative to "
                        "--labels-root)")
    r.add_argument("--identity-tag", type=int, default=40)
    r.add_argument("--grid", default="25,60,0.5",
                   help="start,stop,step in percent (default 25,60,0.5)")
    r.add_argument("--prereg", type=float, default=40.0)
    r.add_argument("--out", type=Path, required=True)

    # ---- sensitivity ------------------------------------------------------
    s = sub.add_parser("sensitivity", help="coverage x identity parameter grid")
    s.add_argument("--dataset-dir", type=Path, required=True,
                   help="an audited dataset directory (sequences_all.fasta + "
                        "per_target_audit.csv)")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--coverages", default="0.7,0.8,0.9")
    s.add_argument("--identities", default="0.3,0.4,0.5,0.6,0.7,0.8")
    s.add_argument("--reference", type=Path, default=None,
                   help="batch audit_summary.csv to reproduce-check against")
    s.add_argument("--name", default=None)
    add_common(s)

    # ---- algorithm-check --------------------------------------------------
    ac = sub.add_parser("algorithm-check",
                        help="set-cover (mode 0) vs CD-HIT-style greedy (mode 2)")
    ac.add_argument("--dataset-dir", type=Path, required=True)
    ac.add_argument("--out", type=Path, required=True)
    ac.add_argument("--identities", default="0.4,0.6,0.8")
    ac.add_argument("--coverage", type=float, default=0.8)
    ac.add_argument("--name", default=None)
    add_common(ac)

    # ---- negative-control -------------------------------------------------
    nc = sub.add_parser("negative-control",
                        help="residue-shuffled test sequences control")
    nc.add_argument("--dataset-dir", type=Path, required=True)
    nc.add_argument("--out", type=Path, required=True)
    nc.add_argument("--seeds", type=parse_float_list, default=[0, 1, 2])
    nc.add_argument("--thresholds", default="0.4,0.6,0.8")
    nc.add_argument("--coverage", type=float, default=0.8)
    nc.add_argument("--name", default=None)
    add_common(nc)

    # ---- evaluate ---------------------------------------------------------
    ev = sub.add_parser("evaluate",
                        help="audited vs unaudited metrics, ranks, mis-selection")
    ev.add_argument("--predictions", type=Path, required=True,
                    help="long CSV: dataset,model,target_sequence|target_id,"
                         "y_true,y_pred")
    ev.add_argument("--labels-root", type=Path, required=True)
    ev.add_argument("--datasets", type=_csv_list, required=True)
    ev.add_argument("--identity-tag", type=int, default=40)
    ev.add_argument("--metrics", type=_csv_list, default=["R2", "EF@1%", "ECE"])
    ev.add_argument("--out", type=Path, required=True)

    # ---- band-lodo --------------------------------------------------------
    bl = sub.add_parser("band-lodo",
                        help="LODO check of a 1-D applicability band")
    bl.add_argument("--points-csv", type=Path, default=None,
                    help="columns name,x,y[,effect]; omit to use built-in "
                         "paper Table S3 points")
    bl.add_argument("--out", type=Path, required=True)

    # ---- mirror -----------------------------------------------------------
    mi = sub.add_parser("mirror",
                        help="binding-spectrum mirror audit (ΔΔG "
                             "redundancy: shared-mutation Spearman + KS)")
    mi.add_argument("--data", type=Path, required=True,
                    help="long CSV of mutation-level measurements "
                         "(complex_id, split, mutation, ΔΔG or affinities)")
    mi.add_argument("--out", type=Path, required=True)
    mi.add_argument("--name", default="dataset")
    mi.add_argument("--complex-col", default="complex_id")
    mi.add_argument("--split-col", default="split")
    mi.add_argument("--mutation-col", default="mutation")
    mi.add_argument("--ddg-col", default="ddG",
                    help="ΔΔG column; leave empty to derive from "
                         "--aff-mut-col/--aff-wt-col")
    mi.add_argument("--aff-mut-col", default=None)
    mi.add_argument("--aff-wt-col", default=None)
    mi.add_argument("--temp-col", default=None,
                    help="temperature column (K); default 298.15 K")
    mi.add_argument("--default-temp", type=float, default=298.15)
    mi.add_argument("--min-shared-mutations", type=int, default=3)
    mi.add_argument("--min-sites-ks", type=int, default=3)
    mi.add_argument("--thresholds", default="0.5,0.7")
    mi.add_argument("--per-target-audit", type=Path, default=None,
                    help="per_target_audit.csv for the cross-cluster "
                         "intersection")
    mi.add_argument("--identity-tag", type=int, default=40)
    mi.add_argument("--target-id-col", default=None,
                    help="input column joining rows to audit target_id")

    # ---- degradation ------------------------------------------------------
    dg = sub.add_parser("degradation",
                        help="severity verdicts on audit-induced metric "
                             "degradation (needs evaluate outputs or raw "
                             "predictions)")
    src = dg.add_mutually_exclusive_group(required=True)
    src.add_argument("--evaluation-dir", type=Path,
                     help="directory containing rank_comparison.csv and "
                          "audited_evaluation_per_model.csv")
    src.add_argument("--predictions", type=Path,
                     help="raw predictions CSV; requires --labels-root and "
                          "--datasets (evaluate is re-run internally)")
    dg.add_argument("--labels-root", type=Path)
    dg.add_argument("--datasets", type=_csv_list)
    dg.add_argument("--identity-tag", type=int, default=40)
    dg.add_argument("--metrics", type=_csv_list,
                    default=["R2", "EF@1%", "ECE"])
    dg.add_argument("--warn-cost-pct", type=float, default=5.0)
    dg.add_argument("--fail-cost-pct", type=float, default=10.0)
    dg.add_argument("--out", type=Path, required=True)

    return p


def _label_paths_from_args(args) -> dict:
    paths = {}
    if args.label:
        for item in args.label:
            if "=" not in item:
                raise SystemExit(f"--label expects NAME=PATH, got {item!r}")
            name, path = item.split("=", 1)
            paths[name.strip()] = Path(path.strip())
    if args.labels_root:
        if not args.datasets:
            raise SystemExit("--datasets is required with --labels-root")
        paths.update(discover_label_paths(args.labels_root, args.datasets))
    if not paths:
        raise SystemExit("provide --labels-root/--datasets or one or more --label")
    return paths


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = build_parser().parse_args(argv)

    if args.command == "audit":
        run_audit(
            name=args.name, out_dir=args.out,
            thresholds=parse_float_list(args.thresholds),
            coverage=args.coverage, cov_mode=args.cov_mode,
            cluster_mode=args.cluster_mode, threads=args.threads,
            mmseqs_path=args.mmseqs, train_csv=args.train_csv,
            test_csv=args.test_csv, train_fasta=args.train_fasta,
            test_fasta=args.test_fasta, seq_column=args.seq_column)
    elif args.command == "audit-batch":
        run_batch(
            data_root=args.data_root, datasets=args.datasets, out_root=args.out,
            thresholds=parse_float_list(args.thresholds),
            coverage=args.coverage, cov_mode=args.cov_mode,
            cluster_mode=args.cluster_mode, threads=args.threads,
            mmseqs_path=args.mmseqs, seq_column=args.seq_column,
            train_template=args.train_template, test_template=args.test_template)
    elif args.command == "router":
        start, stop, step = (float(x) for x in args.grid.split(","))
        import numpy as np
        grid = np.round(np.arange(start, stop + 1e-9, step), 3)
        run_router_validation(_label_paths_from_args(args), out_dir=args.out,
                              identity_tag=args.identity_tag, grid=grid,
                              prereg=args.prereg)
    elif args.command == "sensitivity":
        run_grid(dataset_dir=args.dataset_dir, out_dir=args.out,
                 coverages=parse_float_list(args.coverages),
                 identities=parse_float_list(args.identities),
                 threads=args.threads, mmseqs_path=args.mmseqs,
                 reference_summary=args.reference, dataset_name=args.name)
    elif args.command == "algorithm-check":
        run_algorithm_check(dataset_dir=args.dataset_dir, out_dir=args.out,
                            identities=parse_float_list(args.identities),
                            coverage=args.coverage, threads=args.threads,
                            mmseqs_path=args.mmseqs, dataset_name=args.name)
    elif args.command == "negative-control":
        run_negative_control(dataset_dir=args.dataset_dir, out_dir=args.out,
                             seeds=[int(x) for x in args.seeds],
                             thresholds=parse_float_list(args.thresholds),
                             coverage=args.coverage, threads=args.threads,
                             mmseqs_path=args.mmseqs, dataset_name=args.name)
    elif args.command == "evaluate":
        run_evaluation(predictions_csv=args.predictions,
                       labels_root=args.labels_root, datasets=args.datasets,
                       out_dir=args.out, identity_tag=args.identity_tag,
                       metrics=args.metrics)
    elif args.command == "band-lodo":
        run_band_lodo(out_dir=args.out, points_csv=args.points_csv)
    elif args.command == "mirror":
        run_mirror(
            data_csv=args.data, out_dir=args.out, dataset_name=args.name,
            complex_col=args.complex_col, split_col=args.split_col,
            mutation_col=args.mutation_col, ddg_col=args.ddg_col or None,
            aff_mut_col=args.aff_mut_col, aff_wt_col=args.aff_wt_col,
            temp_col=args.temp_col, default_temp=args.default_temp,
            min_shared_mutations=args.min_shared_mutations,
            min_sites_ks=args.min_sites_ks,
            thresholds=parse_float_list(args.thresholds),
            per_target_audit=args.per_target_audit,
            identity_tag=args.identity_tag, target_id_col=args.target_id_col)
    elif args.command == "degradation":
        if args.evaluation_dir is not None:
            run_degradation(
                out_dir=args.out,
                ranking_csv=args.evaluation_dir / "rank_comparison.csv",
                per_model_csv=args.evaluation_dir /
                "audited_evaluation_per_model.csv",
                warn_cost_pct=args.warn_cost_pct,
                fail_cost_pct=args.fail_cost_pct)
        else:
            if not args.labels_root or not args.datasets:
                raise SystemExit("--predictions requires --labels-root and "
                                 "--datasets")
            run_degradation(
                out_dir=args.out, predictions_csv=args.predictions,
                labels_root=args.labels_root, datasets=args.datasets,
                identity_tag=args.identity_tag, metrics=args.metrics,
                warn_cost_pct=args.warn_cost_pct,
                fail_cost_pct=args.fail_cost_pct)
    else:  # pragma: no cover
        raise SystemExit(f"unknown command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
