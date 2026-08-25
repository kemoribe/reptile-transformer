param(
    [string]$PackageRoot = ".",
    [string]$Version = "v1.0.0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path $PackageRoot).Path
$checksumPath = Join-Path $root "SHA256SUMS.txt"
if (-not (Test-Path $checksumPath -PathType Leaf)) {
    throw "Missing checksum file: $checksumPath"
}

$expectedArchives = @(
    "dta-reptile-code-$Version.zip",
    "data-processed-chembl-$Version.zip",
    "data-processed-davis-$Version.zip",
    "data-processed-kiba-$Version.zip",
    "data-processed-bindingdb-$Version.zip"
)

$supportFiles = @("README-FIRST.txt")
foreach ($name in $supportFiles) {
    if (-not (Test-Path (Join-Path $root $name) -PathType Leaf)) {
        throw "Missing transfer support file: $name"
    }
}

$expectedHashes = @{}
foreach ($line in Get-Content $checksumPath) {
    if ($line -match "^([0-9a-fA-F]{64})\s+(.+)$") {
        $expectedHashes[$Matches[2]] = $Matches[1].ToLowerInvariant()
    }
}

$failed = $false
Write-Host "Checking SHA256:"
foreach ($name in $expectedArchives) {
    $path = Join-Path $root $name
    if (-not (Test-Path $path -PathType Leaf)) {
        Write-Host "  MISSING  $name" -ForegroundColor Red
        $failed = $true
        continue
    }
    if (-not $expectedHashes.ContainsKey($name)) {
        Write-Host "  NO HASH  $name" -ForegroundColor Red
        $failed = $true
        continue
    }

    $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expectedHashes[$name]) {
        Write-Host "  FAILED   $name" -ForegroundColor Red
        $failed = $true
    } else {
        Write-Host "  OK       $name"
    }
}

$tar = Get-Command tar.exe -ErrorAction SilentlyContinue
if (-not $tar) {
    throw "tar.exe is required to inspect archive contents."
}

Write-Host "Checking archive layout:"
$codeArchive = Join-Path $root $expectedArchives[0]
$codeEntries = @(& $tar.Source -tf $codeArchive)
if ($LASTEXITCODE -ne 0) {
    throw "Cannot read code archive: $codeArchive"
}

$forbiddenPattern = '(^|/)(__pycache__|data/processed|data/raw|outputs?|release_packages)(/|$)|\.(pyc|pt|pth|ckpt|model|zip|tar|gz|xz)$'
$forbiddenEntries = @($codeEntries | Where-Object { $_ -match $forbiddenPattern })
$requiredCodeEntries = @(
    "./README.md",
    "./START_HERE.md",
    "./.gitignore",
    "./data_preprocessing.py",
    "./run_baseline_mlp.py",
    "./run_transformer_baseline.py",
    "./run_reptile_transformer.py",
    "./graphdta/models/gcn.py",
    "./graphdta/models/gat.py",
    "./graphdta/models/ginconv.py",
    "./graphdta/models/gat_gcn.py",
    "./scripts/check_esm2_model.py"
)
$missingCodeEntries = @($requiredCodeEntries | Where-Object { $_ -notin $codeEntries })
if ($forbiddenEntries.Count -gt 0 -or $missingCodeEntries.Count -gt 0) {
    if ($forbiddenEntries.Count -gt 0) {
        Write-Host "  FAILED   code archive contains excluded files:" -ForegroundColor Red
        $forbiddenEntries | ForEach-Object { Write-Host "           $_" }
    }
    if ($missingCodeEntries.Count -gt 0) {
        Write-Host "  FAILED   code archive is missing required files:" -ForegroundColor Red
        $missingCodeEntries | ForEach-Object { Write-Host "           $_" }
    }
    $failed = $true
} else {
    Write-Host "  OK       code archive"
}

foreach ($dataset in @("chembl", "davis", "kiba", "bindingdb")) {
    $archive = Join-Path $root "data-processed-$dataset-$Version.zip"
    $entries = @(& $tar.Source -tf $archive)
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot read dataset archive: $archive"
    }

    $outsideRoot = @($entries | Where-Object {
        $_ -ne "$dataset/" -and -not $_.StartsWith("$dataset/")
    })
    $required = @(
        "$dataset/combined_activities.csv",
        "$dataset/train_set/",
        "$dataset/val_set/",
        "$dataset/test_set/"
    )
    $missingRequired = @($required | Where-Object { $_ -notin $entries })

    if ($outsideRoot.Count -gt 0 -or $missingRequired.Count -gt 0) {
        Write-Host "  FAILED   $dataset archive layout" -ForegroundColor Red
        $failed = $true
    } else {
        Write-Host "  OK       $dataset archive"
    }
}

if ($failed) {
    throw "Transfer package verification failed."
}

Write-Host "All release packages passed verification." -ForegroundColor Green
