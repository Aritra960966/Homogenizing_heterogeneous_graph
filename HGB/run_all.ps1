$ErrorActionPreference = "Continue"
$HGB_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_PYTHON = Join-Path $HGB_DIR "venv\Scripts\python.exe"

$start = Get-Date

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RAHGH — Full HGB Benchmark Suite"       -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 0: Generate all splits
Write-Host ">>> Generating HGB-compatible splits..." -ForegroundColor Yellow
& $VENV_PYTHON "$HGB_DIR\generate_splits.py"
if ($LASTEXITCODE) { Write-Host "  [WARN] Some splits may have failed" -ForegroundColor Yellow }

# Step 1: Node Classification
Write-Host ""
Write-Host ">>> Node Classification..." -ForegroundColor Green
& $VENV_PYTHON "$HGB_DIR\run_nc.ps1"

# Step 2: Link Prediction
Write-Host ""
Write-Host ">>> Link Prediction..." -ForegroundColor Green
& $VENV_PYTHON "$HGB_DIR\run_lp.ps1"

# Step 3: Graph Clustering
Write-Host ""
Write-Host ">>> Graph Clustering..." -ForegroundColor Green
& $VENV_PYTHON "$HGB_DIR\run_cl.ps1"

# Step 4: Recommendation
Write-Host ""
Write-Host ">>> Recommendation..." -ForegroundColor Green
& $VENV_PYTHON "$HGB_DIR\run_rec.ps1"

$elapsed = (Get-Date) - $start
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  All experiments complete!"               -ForegroundColor Cyan
Write-Host "  Elapsed: $($elapsed.Hours)h $($elapsed.Minutes)m $($elapsed.Seconds)s" -ForegroundColor Cyan
Write-Host "  Results: $HGB_DIR\results\"              -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
