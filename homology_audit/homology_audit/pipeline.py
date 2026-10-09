# -*- coding: utf-8 -*-
"""``audit-benchmark`` — one-command homology audit pipeline.

Wires the whole toolkit together::

    audit-benchmark --data <path> --output <path> [--config config.yaml]

Steps (each stage can be skipped when its inputs are absent):

1. **prepare**  — adapters normalize ``--data`` into train/test CSVs.
2. **audit**    — MMseqs2 clustering, nearest identity, cross-cluster labels.
3. **mirror**   — binding-spectrum redundancy (auto-enabled when the input
                  carries mutation-level ΔΔG columns).
4. **evaluate** — standard vs audited panel metrics when predictions are
                  given (``--predictions`` or ``evaluation.predictions``).
5. **degrade**  — degradation detection with severity verdicts.

A consolidated ``audit-benchmark_summary.txt`` is written to the output root.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .adapters import prepare_inputs
from .audit import run_audit
from .config import load_config, validate_config
from .degradation import run_degradation
from .mirror import run_mirror

SUMMARY_FILE = "audit-benchmark_summary.txt"

MIRROR_REQUIRED_COLS = ("complex_id", "mutation")   # + one of ddG / affinities


def _mirror_columns_present(data_csv: Path, cfg_mirror: dict) -> bool:
    try:
        head = pd.read_csv(data_csv, nrows=0)
    except Exception:
        return False
    cols = set(head.columns)
    if not all(c in cols for c in MIRROR_REQUIRED_COLS):
        return False
    if cfg_mirror.get("ddg_col") and cfg_mirror["ddg_col"] in cols:
        return True
    return (bool(cfg_mirror.get("aff_mut_col")) and
            cfg_mirror["aff_mut_col"] in cols and
            bool(cfg_mirror.get("aff_wt_col")) and
            cfg_mirror["aff_wt_col"] in cols)


def run_audit_benchmark(data: Path, output: Path,
                        config_path: Optional[Path] = None,
                        overrides: Optional[dict] = None,
                        verbose: bool = True) -> dict:
    """Run the full benchmark audit; return stage paths and key numbers."""
    cfg = load_config(config_path, overrides)
    validate_config(cfg)

    ds = cfg["dataset"]
    name = ds["name"]
    data = Path(data) if data is not None else Path(ds["data"])
    if data is None or not Path(data).exists():
        raise FileNotFoundError(
            f"dataset path not found: {data}; set --data or dataset.data in "
            f"the config")
    output = Path(output)
    prepared_dir, audit_dir = output / "prepared", output / "audit"
    mirror_dir, eval_dir = output / "mirror", output / "evaluation"
    degr_dir = output / "degradation"

    stages: list = []
    summary_lines = [f"audit-benchmark (homology-audit pipeline)",
                     "=" * 62, f"dataset name : {name}",
                     f"data         : {data}",
                     f"output       : {output}", ""]

    # ---- 1. prepare --------------------------------------------------------
    prep = prepare_inputs(
        data=data, out_dir=prepared_dir, name=name,
        seq_column=ds.get("seq_column"), split_col=ds.get("split_column"),
        test_data=ds.get("test_data"), test_targets=ds.get("test_targets"),
        targets_file=ds.get("targets_file"),
        sequences_fasta=ds.get("sequences_fasta"),
        affinity_key=ds.get("affinity_key", "affinity"), verbose=verbose)
    stages.append("prepare")
    summary_lines += [f"[prepare] format={prep['format']}  "
                      f"train={prep['train_csv'].name}  "
                      f"test={prep['test_csv'].name}"]

    # ---- 2. audit ----------------------------------------------------------
    audit_cfg = cfg["audit"]
    audit_res = run_audit(
        name=name, out_dir=audit_dir,
        thresholds=audit_cfg["thresholds"], coverage=audit_cfg["coverage"],
        cov_mode=audit_cfg["cov_mode"], cluster_mode=audit_cfg["cluster_mode"],
        threads=audit_cfg["threads"], mmseqs_path=audit_cfg["mmseqs"],
        train_csv=prep["train_csv"], test_csv=prep["test_csv"],
        seq_column="target_sequence", verbose=verbose)
    stages.append("audit")
    best40 = next((r for r in audit_res["summary_rows"]
                   if r["identity_threshold_pct"] == 40),
                  audit_res["summary_rows"][-1])
    summary_lines += [
        f"[audit] train targets={audit_res['n_train']}  "
        f"test targets={audit_res['n_test']}",
        f"[audit] @40% identity: cross-cluster "
        f"{best40['n_test_targets_cross_cluster']}/{best40['n_test_targets']} "
        f"= {best40['cross_cluster_ratio_pct']}%"]

    # ---- 3. mirror (optional) ----------------------------------------------
    mirror_cfg = cfg["mirror"]
    enabled = str(mirror_cfg["enabled"]).lower()
    run_mirror_flag = (enabled == "true") or (
        enabled == "auto" and _mirror_columns_present(Path(data), mirror_cfg))
    if run_mirror_flag:
        target_id_col = mirror_cfg.get("target_id_col")
        if target_id_col is None:
            try:
                target_id_col = ("target_id"
                                 if "target_id" in pd.read_csv(data, nrows=0).columns
                                 else None)
            except Exception:
                target_id_col = None
        mirror_res = run_mirror(
            data_csv=Path(data), out_dir=mirror_dir, dataset_name=name,
            complex_col=mirror_cfg["complex_col"],
            split_col=mirror_cfg["split_col"],
            mutation_col=mirror_cfg["mutation_col"],
            ddg_col=mirror_cfg.get("ddg_col"),
            aff_mut_col=mirror_cfg.get("aff_mut_col"),
            aff_wt_col=mirror_cfg.get("aff_wt_col"),
            temp_col=mirror_cfg.get("temp_col"),
            default_temp=mirror_cfg.get("default_temp", 298.15),
            min_shared_mutations=mirror_cfg["min_shared_mutations"],
            min_sites_ks=mirror_cfg["min_sites_ks"],
            thresholds=mirror_cfg["thresholds"],
            per_target_audit=(audit_dir / name / "per_target_audit.csv"),
            identity_tag=cfg["evaluation"]["identity_tag"],
            target_id_col=target_id_col, verbose=verbose)
        stages.append("mirror")
        row07 = next((r for r in mirror_res["summary"].to_dict("records")
                      if abs(r["threshold"] - 0.7) < 1e-9),
                     mirror_res["summary"].iloc[0].to_dict())
        summary_lines += [
            f"[mirror] KS-redundant test complexes @0.7: "
            f"{row07['n_ks_redundant']}/{row07['n_test_complexes']} "
            f"= {row07['pct_ks_redundant']}%"]
    else:
        summary_lines.append("[mirror] skipped "
                             f"(enabled={mirror_cfg['enabled']})")

    # ---- 4+5. evaluation & degradation (optional) ---------------------------
    preds = cfg["evaluation"]["predictions"]
    if preds is not None:
        from .evaluate import run_evaluation
        eval_res = run_evaluation(
            predictions_csv=preds, labels_root=audit_dir, datasets=[name],
            out_dir=eval_dir, identity_tag=cfg["evaluation"]["identity_tag"],
            metrics=cfg["evaluation"]["metrics"])
        stages.append("evaluate")
        rk = eval_res["ranking"]
        if rk.empty:
            summary_lines.append(
                "[evaluate] no prediction rows matched dataset "
                f"{name!r}; check the 'dataset' column in the predictions CSV")
        else:
            degr_res = run_degradation(
                out_dir=degr_dir,
                ranking_csv=eval_dir / "rank_comparison.csv",
                per_model_csv=eval_dir / "audited_evaluation_per_model.csv",
                warn_cost_pct=cfg["degradation"]["warn_cost_pct"],
                fail_cost_pct=cfg["degradation"]["fail_cost_pct"],
                verbose=verbose)
            stages.append("degradation")
            summary_lines += [
                f"[evaluate] {len(rk)} metric panels evaluated "
                f"(identity_tag={cfg['evaluation']['identity_tag']}%)",
                f"[degradation] winner reversals: "
                f"{int(rk['winner_reversed'].sum())}/{len(rk)}  "
                f"overall verdict: {degr_res['overall']}"]
    else:
        summary_lines.append("[evaluate/degradation] skipped "
                             "(no predictions provided)")

    summary_lines += ["", f"stages run: {' -> '.join(stages)}",
                      f"version: {_version()}"]
    (output / SUMMARY_FILE).write_text("\n".join(summary_lines),
                                       encoding="utf-8")
    if verbose:
        print("\n" + "\n".join(summary_lines))
    return {"stages": stages, "config": cfg,
            "prepared": prep, "audit": audit_res,
            "summary_file": str(output / SUMMARY_FILE)}


def _version() -> str:
    from . import __version__
    return __version__


def main(argv=None) -> int:
    """CLI entry point: ``audit-benchmark --data <path> --output <path>``."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    import argparse
    p = argparse.ArgumentParser(
        prog="audit-benchmark",
        description="One-command homology audit for DTA/PPI benchmarks "
                    "(prepare -> MMseqs2 audit -> mirror -> evaluate -> "
                    "degradation).  All parameters can also come from a "
                    "YAML config (--config).")
    p.add_argument("--data", type=Path,
                   help="dataset path: split directory, single long CSV, "
                        "FASTA, or DAVIS/KIBA affinity .mat")
    p.add_argument("--output", type=Path, required=True,
                   help="output root directory")
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config file (see configs/default_audit.yaml)")
    p.add_argument("--name", default=None,
                   help="dataset name (default: config or 'dataset')")
    p.add_argument("--test-data", type=Path, default=None,
                   help="test CSV/FASTA when --data holds only training data")
    p.add_argument("--test-targets", type=Path, default=None,
                   help="file listing test target sequences or ids "
                        "(one per line)")
    p.add_argument("--seq-column", default=None,
                   help="sequence column name for CSV inputs")
    p.add_argument("--predictions", type=Path, default=None,
                   help="optional predictions CSV (dataset,model,"
                        "target_sequence,y_true,y_pred)")
    p.add_argument("--mmseqs", type=Path, default=None,
                   help="path to the mmseqs binary")
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args(argv)

    if args.data is None and args.config is None:
        p.error("provide --data (and optionally --config), or --config whose "
                "dataset.data points at the data")

    overrides: dict = {}
    dataset = {k: v for k, v in {
        "data": args.data,
        "test_data": args.test_data,
        "test_targets": args.test_targets,
        "seq_column": args.seq_column,
        "name": args.name,
    }.items() if v is not None}
    if dataset:
        overrides["dataset"] = dataset
    audit = {k: v for k, v in {"mmseqs": args.mmseqs,
                               "threads": args.threads}.items()
             if v is not None}
    if audit:
        overrides["audit"] = audit
    if args.predictions is not None:
        overrides.setdefault("evaluation", {})["predictions"] = \
            args.predictions

    if args.name and args.config is None:
        overrides.setdefault("dataset", {})["name"] = args.name

    run_audit_benchmark(data=overrides.get("dataset", {}).get("data"),
                        output=args.output, config_path=args.config,
                        overrides=overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
