param(
    [string]$OutputDir = "",
    [ValidateSet("auto", "stanford", "github")]
    [string]$Source = "auto"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
    $OutputDir = Join-Path $ProjectRoot "shared\data\raw"
}

$Downloader = Join-Path $PSScriptRoot "download_iwslt15.py"
python $Downloader --output-dir $OutputDir --source $Source
if ($LASTEXITCODE -ne 0) {
    throw "IWSLT downloader failed with exit code $LASTEXITCODE"
}

