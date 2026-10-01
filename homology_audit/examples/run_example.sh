#!/usr/bin/env bash
# Linux / macOS end-to-end demo of homology-audit on the synthetic data.
# Prerequisite: mmseqs on PATH (conda install -c conda-forge mmseqs2),
# or export HOMOLOGY_AUDIT_MMSEQS=/path/to/mmseqs
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=examples/output
python examples/make_example_data.py

python -m homology_audit.cli audit-batch --data-root examples/data \
    --datasets demo,demo2 --out "$OUT/audit" --thresholds 0.4,0.6,0.8
python -m homology_audit.cli router --labels-root "$OUT/audit" \
    --datasets demo,demo2 --out "$OUT/router"
python -m homology_audit.cli sensitivity --dataset-dir "$OUT/audit/demo" \
    --reference "$OUT/audit/audit_summary.csv" --out "$OUT/sens/demo" --name demo
python -m homology_audit.cli algorithm-check --dataset-dir "$OUT/audit/demo" \
    --out "$OUT/sens/demo"
python -m homology_audit.cli negative-control --dataset-dir "$OUT/audit/demo" \
    --out "$OUT/negative/demo"
python -m homology_audit.cli evaluate \
    --predictions examples/predictions/demo_predictions.csv \
    --labels-root "$OUT/audit" --datasets demo --out "$OUT/eval"
python -m homology_audit.cli band-lodo --out "$OUT/band"

echo
echo "All demo outputs are under examples/output (see *_report.txt)."
