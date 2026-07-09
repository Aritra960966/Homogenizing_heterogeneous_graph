$ErrorActionPreference = "Stop"
$HGB_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_PYTHON = Join-Path $HGB_DIR "venv\Scripts\python.exe"
$PROJECT_DIR = Join-Path $HGB_DIR ".."

if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "ERROR: venv not found. Run setup_venv.ps1 first." -ForegroundColor Red
    exit 1
}

$DATASETS = @("dblp", "imdb", "freebase", "pubmed")
$SEEDS = 10

Write-Host "=== HGB Link Prediction (n=$SEEDS seeds) ===" -ForegroundColor Cyan
Write-Host ""

& $VENV_PYTHON "$HGB_DIR\generate_splits.py" --task lp
if ($LASTEXITCODE) { throw "Split generation failed" }

foreach ($ds in $DATASETS) {
    Write-Host ""
    Write-Host ">> Running $ds LP ..." -ForegroundColor Green
    $OUT_DIR = "$HGB_DIR\results\lp\$ds"
    New-Item -ItemType Directory -Path $OUT_DIR -Force | Out-Null

    & $VENV_PYTHON -c "
import sys, os, numpy as np
sys.path.insert(0, '$PROJECT_DIR'.replace('\\', '/'))
import RAHGH.src.train as train_mod

train_mod.N_SEEDS = $SEEDS
ds_name = '$ds'
out_dir = '$OUT_DIR'
train_mod.run_lp(ds_name, out_dir)
" 2>&1

    if ($LASTEXITCODE) {
        Write-Host "  [ERROR] $ds LP failed" -ForegroundColor Red
    } else {
        Write-Host "  [DONE] $ds LP" -ForegroundColor Green
    }
}
