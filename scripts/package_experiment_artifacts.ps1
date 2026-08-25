param(
    [string]$Version = "v1.0.0",
    [string]$SourceRoot,
    [string]$ResultsRoot,
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $SourceRoot) {
    $SourceRoot = Join-Path $repoRoot ".."
}
if (-not $ResultsRoot) {
    $ResultsRoot = Join-Path $repoRoot "results\tables"
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $repoRoot "..\release_packages"
}

$sourceRootPath = (Resolve-Path $SourceRoot).Path
$resultsRootPath = (Resolve-Path $ResultsRoot).Path
$outputRootPath = [System.IO.Path]::GetFullPath($OutputRoot)
$stagingRoot = Join-Path $outputRootPath ".experiment_artifacts_staging"

New-Item -ItemType Directory -Force $outputRootPath | Out-Null
Remove-Item $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $stagingRoot | Out-Null

$tar = Get-Command tar.exe -ErrorAction SilentlyContinue
if (-not $tar) {
    throw "tar.exe is required. It is included with current Windows 10/11 installations."
}

$modelRoots = [ordered]@{
    "mlp" = @(
        "baseline_output",
        "baseline_output_kiba"
    )
    "transformer" = @(
        "transformer_baseline_output",
        "transformer_baseline_output_kiba",
        "transformer_ablation_output",
        "transformer_ablation_output_chembl",
        "transformer_ablation_output_davis",
        "transformer_ablation_output_kiba",
        "output_bdb_mse",
        "transformer_morgan_mse_output"
    )
    "reptile-transformer" = @(
        "reptile_output",
        "reptile_output_davis",
        "reptile_output_kiba",
        "reptile_ablation_output_chembl",
        "reptile_ablation_output_davis",
        "reptile_ablation_output_kiba",
        "reptile_ablation_output_bindingdb"
    )
}

$resultFileNames = @(
    "final_results.json",
    "predictions.npz",
    "per_target_results.json",
    "training_history.json",
    "target_scaler.json"
)

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$FullPath
    )

    $base = [System.IO.Path]::GetFullPath($BasePath).TrimEnd("\") + "\"
    $full = [System.IO.Path]::GetFullPath($FullPath)
    if (-not $full.StartsWith($base, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside source root: $full"
    }
    return $full.Substring($base.Length)
}

function Copy-Artifact {
    param(
        [Parameter(Mandatory = $true)][System.IO.FileInfo]$Source,
        [Parameter(Mandatory = $true)][string]$StagePath,
        [Parameter(Mandatory = $true)][string]$ArchiveRelativePath
    )

    $destination = Join-Path $StagePath $ArchiveRelativePath
    New-Item -ItemType Directory -Force (Split-Path $destination -Parent) | Out-Null
    Copy-Item $Source.FullName $destination -Force
    $hash = Get-FileHash $Source.FullName -Algorithm SHA256

    [PSCustomObject]@{
        source_path = (Get-RelativePath $sourceRootPath $Source.FullName).Replace("\", "/")
        archive_path = $ArchiveRelativePath.Replace("\", "/")
        bytes = $Source.Length
        sha256 = $hash.Hash.ToLowerInvariant()
    }
}

function Write-ArtifactManifest {
    param(
        [Parameter(Mandatory = $true)][string]$StagePath,
        [Parameter(Mandatory = $true)][array]$Rows
    )

    $manifestPath = Join-Path $StagePath "MANIFEST.csv"
    $Rows |
        Sort-Object archive_path |
        Export-Csv $manifestPath -NoTypeInformation -Encoding UTF8
}

function New-ArtifactArchive {
    param(
        [Parameter(Mandatory = $true)][string]$StagePath,
        [Parameter(Mandatory = $true)][string]$ArchiveName
    )

    $archivePath = Join-Path $outputRootPath $ArchiveName
    Remove-Item $archivePath -Force -ErrorAction SilentlyContinue
    & $tar.Source -a -cf $archivePath -C $StagePath .
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create archive: $archivePath"
    }
    return Get-Item $archivePath
}

$archives = @()

# Package approved tables plus lightweight metrics and predictions.
$resultsStage = Join-Path $stagingRoot "results"
New-Item -ItemType Directory -Force $resultsStage | Out-Null
$resultManifest = @()

$tableFiles = Get-ChildItem $resultsRootPath -File -Filter "*.xlsx"
if ($tableFiles.Count -eq 0) {
    throw "No approved XLSX result tables found in $resultsRootPath"
}
$tableArchiveNamesByHash = @{
    "081d61d4c374839f3884677108e8593889a3dfc2a16a1659dcb1a5cdaca94069" = "feature-combination-results.xlsx"
    "01d3c81fd40bc4c3c72360b612546e9d327d5a69632e91d61f1c76b9affd5b0c" = "internal-external-ablation-results.xlsx"
    "a3ac683d563bcf7b587acb7b928a29c9e03dd04e4b1f1726a39bbfb3919bf9ba" = "model-performance-results.xlsx"
}
foreach ($file in $tableFiles) {
    $tableHash = (Get-FileHash $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not $tableArchiveNamesByHash.ContainsKey($tableHash)) {
        throw "No release filename is configured for result table: $($file.Name)"
    }
    $archiveName = $tableArchiveNamesByHash[$tableHash]
    $resultManifest += Copy-Artifact $file $resultsStage (Join-Path "tables" $archiveName)
}

$combinedCsv = Join-Path $repoRoot "results\metrics_all_tables.csv"
if (Test-Path $combinedCsv -PathType Leaf) {
    $resultManifest += Copy-Artifact (Get-Item $combinedCsv) $resultsStage "tables\metrics_all_tables.csv"
}

foreach ($family in $modelRoots.GetEnumerator()) {
    foreach ($relativeRoot in $family.Value) {
        $root = Join-Path $sourceRootPath $relativeRoot
        if (-not (Test-Path $root -PathType Container)) {
            continue
        }
        $files = Get-ChildItem $root -Recurse -File |
            Where-Object { $resultFileNames -contains $_.Name }
        foreach ($file in $files) {
            $sourceRelative = Get-RelativePath $sourceRootPath $file.FullName
            $resultManifest += Copy-Artifact $file $resultsStage (Join-Path "metrics" $sourceRelative)
        }
    }
}

$graphRoot = Join-Path $sourceRootPath "GrapthDTA"
if (Test-Path $graphRoot -PathType Container) {
    $graphResults = Get-ChildItem $graphRoot -File |
        Where-Object {
            $_.Name -like "result_*.json" -or
            $_.Name -like "*_training_summary.json"
        }
    foreach ($file in $graphResults) {
        $sourceRelative = Get-RelativePath $sourceRootPath $file.FullName
        $resultManifest += Copy-Artifact $file $resultsStage (Join-Path "metrics" $sourceRelative)
    }
}

$resultsReadme = @"
Experiment results $Version
===========================

The project owner designated the three XLSX workbooks in tables/ as the final
result summaries. metrics_all_tables.csv is their machine-readable export.

The metrics/ directory preserves lightweight JSON, prediction, per-target,
history, and scaler files from the corresponding experiment output folders.
Large feature caches and per-epoch checkpoints are intentionally excluded.
"@
$resultsReadme | Set-Content (Join-Path $resultsStage "README.txt") -Encoding UTF8
Write-ArtifactManifest $resultsStage $resultManifest
$archives += New-ArtifactArchive $resultsStage "experiment-results-$Version.zip"
Remove-Item $resultsStage -Recurse -Force

# Package non-empty best weights by model family.
foreach ($family in $modelRoots.GetEnumerator()) {
    $familyStage = Join-Path $stagingRoot ("weights-" + $family.Key)
    New-Item -ItemType Directory -Force $familyStage | Out-Null
    $weightManifest = @()

    foreach ($relativeRoot in $family.Value) {
        $root = Join-Path $sourceRootPath $relativeRoot
        if (-not (Test-Path $root -PathType Container)) {
            continue
        }
        $weightFiles = Get-ChildItem $root -Recurse -File |
            Where-Object {
                $_.Length -gt 0 -and
                $_.Name -in @("best_model.pt", "best_model.pth")
            }
        foreach ($file in $weightFiles) {
            $sourceRelative = Get-RelativePath $sourceRootPath $file.FullName
            $weightManifest += Copy-Artifact $file $familyStage (Join-Path "weights" $sourceRelative)
        }
    }

    $weightReadme = @"
Model weights: $($family.Key) ($Version)
========================================

Only non-empty best_model files are included. Feature caches, optimizer
checkpoints, and per-epoch checkpoints are excluded. MANIFEST.csv records the
original relative path, archive path, size, and SHA256 for every weight.
"@
    $weightReadme | Set-Content (Join-Path $familyStage "README.txt") -Encoding UTF8
    Write-ArtifactManifest $familyStage $weightManifest
    $archiveName = "model-weights-$($family.Key)-$Version.zip"
    $archives += New-ArtifactArchive $familyStage $archiveName
    Remove-Item $familyStage -Recurse -Force
}

# GraphDTA uses model_*.model naming rather than best_model.pt.
$graphStage = Join-Path $stagingRoot "weights-graphdta"
New-Item -ItemType Directory -Force $graphStage | Out-Null
$graphManifest = @()
if (Test-Path $graphRoot -PathType Container) {
    $graphWeights = Get-ChildItem $graphRoot -File -Filter "model_*.model" |
        Where-Object { $_.Length -gt 0 }
    foreach ($file in $graphWeights) {
        $sourceRelative = Get-RelativePath $sourceRootPath $file.FullName
        $graphManifest += Copy-Artifact $file $graphStage (Join-Path "weights" $sourceRelative)
    }
}

$graphReadme = @"
Model weights: GraphDTA ($Version)
=================================

This archive includes all non-empty GraphDTA model_*.model files. Zero-byte
placeholders are excluded. MANIFEST.csv records each included file.
"@
$graphReadme | Set-Content (Join-Path $graphStage "README.txt") -Encoding UTF8
Write-ArtifactManifest $graphStage $graphManifest
$archives += New-ArtifactArchive $graphStage "model-weights-graphdta-$Version.zip"
Remove-Item $graphStage -Recurse -Force

$checksumPath = Join-Path $outputRootPath "SHA256SUMS-EXPERIMENT-ARTIFACTS.txt"
$checksumLines = foreach ($archive in $archives) {
    $hash = Get-FileHash $archive.FullName -Algorithm SHA256
    "$($hash.Hash.ToLowerInvariant())  $($archive.Name)"
}
$checksumLines | Set-Content $checksumPath -Encoding ascii

$uploadGuide = Join-Path $repoRoot "docs\EXPERIMENT_ARTIFACTS_UPLOAD.md"
if (Test-Path $uploadGuide -PathType Leaf) {
    Copy-Item $uploadGuide (Join-Path $outputRootPath "EXPERIMENT-ARTIFACTS-UPLOAD.md") -Force
}

Remove-Item $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Experiment artifact packages:"
foreach ($archive in $archives) {
    Write-Host ("  {0,-62} {1,9:N2} MiB" -f $archive.Name, ($archive.Length / 1MB))
}
Write-Host "  SHA256SUMS-EXPERIMENT-ARTIFACTS.txt"
