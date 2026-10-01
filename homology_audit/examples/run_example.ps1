# Windows PowerShell: end-to-end demo of homology-audit on the synthetic data.
# Usage:
#   .\run_example.ps1 -Mmseqs "C:\path\to\mmseqs.exe"
# or set the environment variable HOMOLOGY_AUDIT_MMSEQS first.
param(
    [string]$Mmseqs = $env:HOMOLOGY_AUDIT_MMSEQS
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if ($Mmseqs) { $env:HOMOLOGY_AUDIT_MMSEQS = $Mmseqs }
python examples\make_example_data.py
$out = "examples\output"

python -m homology_audit.cli audit-batch --data-root examples\data `
    --datasets demo,demo2 --out "$out\audit" `
    --thresholds 0.4,0.6,0.8
python -m homology_audit.cli router --labels-root "$out\audit" `
    --datasets demo,demo2 --out "$out\router"
python -m homology_audit.cli sensitivity --dataset-dir "$out\audit\demo" `
    --reference "$out\audit\audit_summary.csv" --out "$out\sens\demo" --name demo
python -m homology_audit.cli algorithm-check --dataset-dir "$out\audit\demo" `
    --out "$out\sens\demo"
python -m homology_audit.cli negative-control --dataset-dir "$out\audit\demo" `
    --out "$out\negative\demo"
python -m homology_audit.cli evaluate `
    --predictions examples\predictions\demo_predictions.csv `
    --labels-root "$out\audit" --datasets demo --out "$out\eval"
python -m homology_audit.cli band-lodo --out "$out\band"

Write-Host "`nAll demo outputs are under examples\output (see *_report.txt)." -ForegroundColor Green
