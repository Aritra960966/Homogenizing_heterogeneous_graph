#!/usr/bin/env bash
set -e
for DATASET in dblp acm imdb; do
    echo "=============================="
    echo "Recommendation: $DATASET"
    echo "=============================="
    python -m src.train --dataset $DATASET --task rec --seeds 10
done
echo "Recommendation complete."
