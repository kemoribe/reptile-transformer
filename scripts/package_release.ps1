param(
    [string]$Version = "v1.0.0",
    [string]$DataRoot,
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $DataRoot) {
    $DataRoot = Join-Path $repoRoot "..\github_release_data\processed"
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $repoRoot "..\release_packages"
}

$dataRootPath = (Resolve-Path $DataRoot).Path
$outputRootPath = [System.IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force $outputRootPath | Out-Null

$tar = Get-Command tar.exe -ErrorAction SilentlyContinue
if (-not $tar) {
    throw "tar.exe is required. It is included with current Windows 10/11 installations."
}

$datasets = @("chembl", "davis", "kiba", "bindingdb")
foreach ($dataset in $datasets) {
    $datasetPath = Join-Path $dataRootPath $dataset
    if (-not (Test-Path $datasetPath -PathType Container)) {
        throw "Missing processed dataset: $datasetPath"
    }
}

$archives = @()
$codeArchive = Join-Path $outputRootPath "dta-reptile-code-$Version.zip"
Remove-Item $codeArchive -Force -ErrorAction SilentlyContinue
& $tar.Source -a -cf $codeArchive `
    --exclude=".git" `
    --exclude="__pycache__" `
    --exclude="*.pyc" `
    --exclude="*_output*" `
    --exclude="outputs" `
    --exclude="data/processed" `
    --exclude="data/raw" `
    --exclude="release_packages" `
    -C $repoRoot .
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create code archive."
}
$archives += Get-Item $codeArchive

foreach ($dataset in $datasets) {
    $archive = Join-Path $outputRootPath "data-processed-$dataset-$Version.zip"
    Remove-Item $archive -Force -ErrorAction SilentlyContinue
    & $tar.Source -a -cf $archive -C $dataRootPath $dataset
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create archive for $dataset."
    }
    $archives += Get-Item $archive
}

$checksumPath = Join-Path $outputRootPath "SHA256SUMS.txt"
$checksumLines = foreach ($archive in $archives) {
    $hash = Get-FileHash $archive.FullName -Algorithm SHA256
    "$($hash.Hash.ToLowerInvariant())  $($archive.Name)"
}
$checksumLines | Set-Content $checksumPath -Encoding ascii

$verifyScript = Join-Path $PSScriptRoot "verify_transfer.ps1"
Copy-Item $verifyScript (Join-Path $outputRootPath "verify_transfer.ps1") -Force

$transferReadme = Join-Path $repoRoot "docs\TRANSFER_README.txt"
if (-not (Test-Path $transferReadme -PathType Leaf)) {
    throw "Missing transfer instructions: $transferReadme"
}
Copy-Item $transferReadme (Join-Path $outputRootPath "README-FIRST.txt") -Force

Write-Host "Release packages:"
foreach ($archive in $archives) {
    Write-Host ("  {0,-52} {1,8:N2} MiB" -f $archive.Name, ($archive.Length / 1MB))
}
Write-Host "  SHA256SUMS.txt"
Write-Host "  verify_transfer.ps1"
Write-Host "  README-FIRST.txt"
