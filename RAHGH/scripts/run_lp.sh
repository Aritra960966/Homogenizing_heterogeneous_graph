#!/usr/bin/env bash
set -e
for DATASET in dblp acm imdb; do
    echo "=============================="
    echo "Link Prediction: $DATASET"
    echo "=============================="
    python -m src.train \
        --dataset $DATASET --task lp \
        --out results/logs/${DATASET}_lp.csv \
        --seeds 10
done
echo "All LP runs complete."
