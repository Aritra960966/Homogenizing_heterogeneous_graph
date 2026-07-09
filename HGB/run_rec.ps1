$ErrorActionPreference = "Stop"
$HGB_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_PYTHON = Join-Path $HGB_DIR "venv\Scripts\python.exe"
$PROJECT_DIR = Join-Path $HGB_DIR ".."

if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "ERROR: venv not found. Run setup_venv.ps1 first." -ForegroundColor Red
    exit 1
}

$DATASETS = @("amazon", "lastfm", "dblp")
$SEEDS = 10

Write-Host "=== HGB Recommendation (n=$SEEDS seeds) ===" -ForegroundColor Cyan
Write-Host ""

& $VENV_PYTHON "$HGB_DIR\generate_splits.py" --task rec
if ($LASTEXITCODE) { throw "Split generation failed" }

foreach ($ds in $DATASETS) {
    Write-Host ""
    Write-Host ">> Running $ds REC ..." -ForegroundColor Green
    $OUT_DIR = "$HGB_DIR\results\rec\$ds"
    New-Item -ItemType Directory -Path $OUT_DIR -Force | Out-Null

    & $VENV_PYTHON -c "
import sys, os
sys.path.insert(0, '$PROJECT_DIR'.replace('\\', '/'))
import RAHGH.src.train as train_mod
train_mod.N_SEEDS = $SEEDS
train_mod.run_rec('$ds', '$OUT_DIR')
" 2>&1

    if ($LASTEXITCODE) {
        Write-Host "  [ERROR] $ds REC failed" -ForegroundColor Red
    } else {
        Write-Host "  [DONE] $ds REC" -ForegroundColor Green
    }
}
