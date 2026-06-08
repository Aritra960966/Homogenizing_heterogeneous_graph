#!/usr/bin/env bash
set -e
for DATASET in dblp acm imdb; do
    echo "=============================="
    echo "Graph Clustering: $DATASET"
    echo "=============================="
    python -m src.train --dataset $DATASET --task cl --seeds 10
done
echo "Clustering complete."
