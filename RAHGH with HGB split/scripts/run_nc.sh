#!/usr/bin/env bash
set -e
for DATASET in dblp acm imdb; do
    echo "=============================="
    echo "Node Classification: $DATASET"
    echo "=============================="
    python -m src.train \
        --dataset $DATASET --task nc \
        --out-dir results \
        --seeds 10
done
echo "All NC runs complete."
