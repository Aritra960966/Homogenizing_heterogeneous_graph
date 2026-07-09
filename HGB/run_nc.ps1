$ErrorActionPreference = "Stop"
$HGB_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_PYTHON = Join-Path $HGB_DIR "venv\Scripts\python.exe"
$PROJECT_DIR = Join-Path $HGB_DIR ".."

if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "ERROR: venv not found. Run setup_venv.ps1 first." -ForegroundColor Red
    exit 1
}

$DATASETS = @("dblp", "acm", "imdb", "pubmed")
$SEEDS = 10

Write-Host "=== HGB Node Classification (n=$SEEDS seeds) ===" -ForegroundColor Cyan
Write-Host ""

# Generate splits first
Write-Host ">> Generating splits..." -ForegroundColor Yellow
& $VENV_PYTHON "$HGB_DIR\generate_splits.py" --task nc
if ($LASTEXITCODE) { throw "Split generation failed" }

# Run each dataset
foreach ($ds in $DATASETS) {
    Write-Host ""
    Write-Host ">> Running $ds NC ..." -ForegroundColor Green
    $OUT_DIR = "$HGB_DIR\results\nc\$ds"
    New-Item -ItemType Directory -Path $OUT_DIR -Force | Out-Null

    & $VENV_PYTHON -c "
import sys, os, json, numpy as np
sys.path.insert(0, '$PROJECT_DIR'.replace('\\', '/'))
from sklearn.model_selection import train_test_split

# Load predefined splits
train_idx = np.load('$HGB_DIR\\splits\\nc\\$ds\\train_indices.npy')
test_idx  = np.load('$HGB_DIR\\splits\\nc\\$ds\\test_indices.npy')

# Monkey-patch the loader to inject predefined splits
import RAHGH.src.train as train_mod

# Set up the training
train_mod.N_SEEDS = $SEEDS
out_dir = '$OUT_DIR'
ds_name = '$ds'

# Load data
data = train_mod._get_loader(ds_name)
data['name'] = ds_name
data['train_indices'] = train_idx
data['test_indices'] = test_idx

# Run
train_mod.run_nc(ds_name, out_dir)
" 2>&1

    if ($LASTEXITCODE) {
        Write-Host "  [ERROR] $ds NC failed" -ForegroundColor Red
    } else {
        Write-Host "  [DONE] $ds NC" -ForegroundColor Green
    }
}
