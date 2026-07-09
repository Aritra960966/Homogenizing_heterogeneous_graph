$ErrorActionPreference = "Stop"
$HGB_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_DIR = Join-Path $HGB_DIR "venv"

Write-Host "=== Setting up Python venv for HGB experiments ===" -ForegroundColor Cyan

# Create venv
if (-not (Test-Path $VENV_DIR)) {
    python -m venv $VENV_DIR
    Write-Host "  Created venv at $VENV_DIR" -ForegroundColor Green
} else {
    Write-Host "  venv already exists at $VENV_DIR" -ForegroundColor Yellow
}

# Activate and install
$PIP = if ($IsWindows -or $env:OS) {
    Join-Path $VENV_DIR "Scripts\pip.exe"
} else {
    Join-Path $VENV_DIR "bin\pip"
}

& $PIP install --upgrade pip setuptools wheel
if ($LASTEXITCODE) { throw "pip upgrade failed" }

# Install core deps first
& $PIP install numpy scipy pandas scikit-learn tqdm pyyaml matplotlib seaborn nltk
if ($LASTEXITCODE) { throw "core deps install failed" }

# Install PyTorch (CPU or CUDA — adjust as needed)
& $PIP install torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE) { throw "torch install failed" }

# Install PyG
& $PIP install torch_geometric
if ($LASTEXITCODE) { throw "pyg install failed" }

# Install remaining deps
& $PIP install ogb wandb requests jupyter ipykernel
if ($LASTEXITCODE) { throw "remaining deps install failed" }

Write-Host ""
Write-Host "=== venv setup complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate:"
Write-Host "  .\HGB\venv\Scripts\Activate.ps1  (PowerShell)"
Write-Host "  HGB\venv\Scripts\activate.bat    (CMD)"
Write-Host ""
Write-Host "To run experiments:"
Write-Host "  python HGB\generate_splits.py"
Write-Host "  python -m RAHGH.src.train --dataset dblp --task nc --seeds 10"
