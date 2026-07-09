param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('nc','lp','cl','rec')]
    [string]$Task,

    [Parameter(Mandatory=$true)]
    [string]$Dataset,

    [int]$Seeds = 10,

    [ValidateSet('gcn','gat')]
    [string]$Head = 'gcn'
)

$ErrorActionPreference = "Stop"
$HGB_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_DIR = (Resolve-Path (Join-Path $HGB_DIR "..")).Path
$VENV_PY = Join-Path $HGB_DIR "venv\Scripts\python.exe"

if (-not (Test-Path $VENV_PY)) {
    Write-Host "ERROR: venv not found at $VENV_PY" -ForegroundColor Red
    Write-Host "Run: .\HGB\setup_venv.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host ">>> HGB Experiment: $Task on $Dataset (seeds=$Seeds, head=$Head)" -ForegroundColor Cyan

# Generate splits for this task/dataset
& $VENV_PY "$HGB_DIR\generate_splits.py" --task $Task --dataset $Dataset

$OUT_DIR = "$HGB_DIR\results\$Task\$Dataset"
New-Item -ItemType Directory -Path $OUT_DIR -Force | Out-Null

& $VENV_PY -c "
import sys, os, numpy as np
sys.path.insert(0, '$PROJECT_DIR'.replace('\\', '/') + '/RAHGH/src')
sys.path.insert(0, '$PROJECT_DIR'.replace('\\', '/'))

import RAHGH.src.train as train_mod
train_mod.N_SEEDS = $Seeds

data = train_mod._get_loader('$Dataset')
data['name'] = '$Dataset'

split_dir = '$HGB_DIR\\splits\\$Task\\$Dataset'
if os.path.exists(split_dir):
    tr = np.load(os.path.join(split_dir, 'train_indices.npy'))
    va = np.load(os.path.join(split_dir, 'val_indices.npy'))
    te = np.load(os.path.join(split_dir, 'test_indices.npy'))
    data['train_indices'] = np.concatenate([tr, va])
    data['test_indices'] = te
    print(f'  HGB split: {len(tr)} train + {len(va)} val + {len(te)} test')

fn = train_mod.TASK_FNS['$Task']
fn('$Dataset', '$OUT_DIR')
" 2>&1

if ($LASTEXITCODE) {
    Write-Host "  [FAIL] Exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
