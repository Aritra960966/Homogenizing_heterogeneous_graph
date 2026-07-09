#!/usr/bin/env bash
# install_env.sh
# Sets up the full Python environment for RAHGH experiments.
# Detects CUDA version automatically and installs the right PyG wheels.
#
# Usage:
#   bash scripts/install_env.sh              # auto-detect CUDA
#   bash scripts/install_env.sh --cpu        # force CPU-only
#   bash scripts/install_env.sh --cuda 12.1  # force CUDA 12.1

set -e

FORCE_CPU=0
FORCE_CUDA=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cpu)        FORCE_CPU=1 ;;
        --cuda)       FORCE_CUDA="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

detect_cuda() {
    if [[ $FORCE_CPU -eq 1 ]]; then
        echo "cpu"
        return
    fi
    if [[ -n "$FORCE_CUDA" ]]; then
        echo "cu$(echo $FORCE_CUDA | tr -d '.')"
        return
    fi

    if command -v nvcc &>/dev/null; then
        VER=$(nvcc --version | grep -oP "release \K[\d.]+" | head -1)
        MAJOR=$(echo $VER | cut -d. -f1)
        MINOR=$(echo $VER | cut -d. -f2)
        if   [[ $MAJOR -eq 12 && $MINOR -ge 1 ]]; then echo "cu121"
        elif [[ $MAJOR -eq 11 && $MINOR -ge 8 ]]; then echo "cu118"
        elif [[ $MAJOR -eq 11 && $MINOR -ge 7 ]]; then echo "cu117"
        else echo "cpu"
        fi
    else
        echo "cpu"
    fi
}

CUDA_TAG=$(detect_cuda)
TORCH_VER="2.1.0"

echo "────────────────────────────────────────"
echo "  RAHGH environment install"
echo "  PyTorch : ${TORCH_VER}"
echo "  Backend : ${CUDA_TAG}"
echo "────────────────────────────────────────"

if [[ "$CUDA_TAG" == "cpu" ]]; then
    pip install torch==${TORCH_VER} torchvision --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch==${TORCH_VER} torchvision --index-url https://download.pytorch.org/whl/${CUDA_TAG}
fi

PYG_WHEEL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
echo "  Installing PyG wheels from: $PYG_WHEEL"
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f $PYG_WHEEL
pip install torch-geometric>=2.4.0

pip install ogb requests tqdm numpy scipy pandas scikit-learn \
            pyyaml matplotlib seaborn jupyter ipykernel

echo ""
echo "✓ Environment ready."
echo ""
echo "Next step:"
echo "  python scripts/download_datasets.py"
echo "  python scripts/download_datasets.py --skip ogbn_mag yelp freebase  # quick start"
