<#
.SYNOPSIS
    Run RAHGH experiments using official HGB splits.
.DESCRIPTION
    Uses the HGB/venv virtual environment (torch, scipy, sklearn pre-installed)
    and runs hgb_final/main.py with all passed arguments.

    Set $env:HGB_QUICK = "1" for a fast 2-combo × 2-fold smoke test.
.EXAMPLE
    .\run.ps1 --dataset dblp --task nc --seeds 2
    .\run.ps1 --dataset lastfm --task rec --seeds 5
    $env:HGB_QUICK = "1"; .\run.ps1 --dataset dblp --task nc --seeds 1
    .\run.ps1   (runs all supported datasets and tasks)
#>

$ProjectRoot   = Split-Path -Parent $PSScriptRoot
$VenvPython    = Join-Path $ProjectRoot "HGB\venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Python not found at $VenvPython"
    Write-Error "The HGB/venv environment is required."
    exit 1
}

$MainPy = Join-Path $PSScriptRoot "main.py"
& $VenvPython $MainPy @args
exit $LASTEXITCODE
