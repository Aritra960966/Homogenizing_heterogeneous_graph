# How to Run Experiments (5-Fold CV → Test → Mean ± Std)

## Prerequisites

Activate the right Python venv from the project root:

```powershell
# For GPU (CUDA 12.4, RTX 3050):
graph_cu\Scripts\Activate.ps1

# For CPU only:
graph\Scripts\Activate.ps1
```

All commands run from the `RAHGH/` directory:

```powershell
cd RAHGH
```

## Pipeline Overview

For each dataset–task combination, the script automatically runs:

1. **80/20 stratified split** — target nodes (NC) or edges (LP) split into training pool (80%) and held-out test set (20%)
2. **5-fold CV + exhaustive grid search** — all 960 combos (3×5×2×2×2×2×4), each evaluated with 5-fold StratifiedKFold on the 80% pool; best params selected by mean validation Macro-F1 (NC) / AUC (LP)
3. **Final training** — train on full 80% pool with best hyperparameters
4. **Test evaluation** — report metrics on held-out 20%
5. **Repeat with N seeds** — steps 2–4 repeated N times with different random seeds → report **mean ± std**

## Commands (10 seeds, exhaustive grid)

### Node Classification (NC)

```powershellall 
python -m src.train --dataset dblp --task nc --out results/logs/dblp_nc.csv --seeds 10
python -m src.train --dataset acm  --task nc --out results/logs/acm_nc.csv  --seeds 10
python -m src.train --dataset imdb --task nc --out results/logs/imdb_nc.csv --seeds 10
```

Reports: `Macro-F1`, `Micro-F1`, `Accuracy` (mean ± std over 10 seeds).

### Link Prediction (LP)

```powershell
python -m src.train --dataset dblp --task lp --out results/logs/dblp_lp.csv --seeds 10
python -m src.train --dataset acm  --task lp --out results/logs/acm_lp.csv  --seeds 10
python -m src.train --dataset imdb --task lp --out results/logs/imdb_lp.csv --seeds 10
```

Reports: `AUC`, `AP` (mean ± std over 10 seeds).

### Run All at Once (PowerShell)

```powershell
foreach ($ds in @('dblp','acm','imdb')) {
  python -m src.train --dataset $ds --task nc --out "results/logs/${ds}_nc.csv" --seeds 10
  python -m src.train --dataset $ds --task lp --out "results/logs/${ds}_lp.csv" --seeds 10
}
```

## Hyperparameter Search Space (Exhaustive Grid)

| Param     | Values                          |
|-----------|---------------------------------|
| d         | 64, 128, 256                    |
| K         | 2, 3, 4, 5, 6                   |
| dropout   | 0.3, 0.5                        |
| lr        | 0.001, 0.005                    |
| wd        | 1e-4, 1e-3                      |
| gcn_hidden| 64, 128                         |
| epochs    | 100, 300, 500, 700              |

Full grid: 960 combos × 5 folds = 4800 train runs per CV phase (run once, then final train/test repeated 10 seeds).

## Output

- Per-seed results saved to `RAHGH/results/logs/{dataset}_{task}.csv`
- Learned α (relation importances) and β (hop weights) per run
- Best checkpoints saved in `RAHGH/results/checkpoints/`
- Final report printed to console (mean ± std)

## Notes

- LP uses negative sampling (ratio 5:1 negatives to positives)
- LP masks training edges from the graph to prevent label leakage
- Transductive setting: full graph is used for message passing; only label/edge masks change per fold
