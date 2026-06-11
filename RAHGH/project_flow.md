# RAHGH — Project Flow

## Overview
RAHGH (Relation-Aware Homogenization for Heterogeneous Graphs) is a heterogeneous graph learning framework that homogenizes heterogeneous graphs via relation-aware polynomial diffusion + α-weighted GCN.

**Base:** EXPERIMENT_GUIDE_v3.md (final paper-conformant version with CV protocol)

---

## Project Structure

```
RAHGH/
├── data/
│   ├── raw/
│   │   ├── DBLP/              ← 7 text files (author_label, paper_author, paper, paper_term, paper_conf, term, conf)
│   │   ├── ACM/               ← ACM.mat
│   │   └── IMDB/              ← movie_metadata.csv
│   └── processed/             ← generated .pt files
├── src/
│   ├── data/
│   │   ├── dblp_loader.py     ← loads DBLP text files (11 files in data/raw/DBLP/), builds 6 adjacency matrices, BoW features
│   │   ├── acm_loader.py      ← loads ACM.mat, filters 3-class papers, SVD-64 author features
│   │   └── imdb_loader.py     ← loads movie_metadata.csv, builds 6 relations, keyword BoW
│   ├── model/
│   │   ├── projector.py       ← TypeSpecificProjector: maps X_dict → shared d-dim space
│   │   ├── diffusion.py       ← bipartite correction + polynomial diffusion + relation fusion + build_A_struct
│   │   ├── fusion.py          ← ResidualFusion: MLP([H0 || Z]) with d=d (v3: removed d_prime)
│   │   └── rahgh.py           ← RAHGH pipeline + GCN head with α-weighted A_hat (v3: LP fix, no d_prime)
│   ├── tasks/
│   │   ├── hparam_search.py   ← v3: 5-fold CV + exhaustive grid search (960 combos, d∈[64,128,256], epochs∈[100,300,500,700])
│   │   ├── node_classification.py  ← run_single_nc (ablation) + run_final_nc (CV→final→test)
│   │   └── link_prediction.py      ← run_single_lp (ablation) + run_final_lp + _run_fold_lp + masked operators
│   └── train.py               ← v3: hparam_search → final train → held-out test, mean±std report
├── configs/
│   ├── dblp_nc.yaml  acm_nc.yaml  imdb_nc.yaml    ← v3: full grid ranges, n_seeds=10
│   └── dblp_lp.yaml  acm_lp.yaml  imdb_lp.yaml    ← v3: full grid ranges, n_seeds=10
├── scripts/
│   ├── run_nc.sh              ← v3: --seeds 10 (updated)
│   ├── run_lp.sh              ← v3: --seeds 10 (updated)
│   ├── setup_data.py          ← copies available ACM.mat + movie_metadata.csv into place
│   ├── download_datasets.py   ← from files_unzipped: PyG/OGB/HGB dataset downloader (8 datasets)
│   ├── install_env.sh         ← from files_unzipped: CUDA-aware env setup
│   └── verify_downloads.py    ← from files_unzipped: download validation
├── results/
│   ├── logs/                  ← per-dataset CSV results
│   ├── hparam_logs/           ← CV fold scores per combo (v3)
│   └── checkpoints/
├── requirements.txt           ← expanded with files_unzipped deps (ogb, requests, tqdm, wandb, jupyter)
└── project_flow.md            ← this file
```

---

## v3 Changes (vs v2)

| # | File | Change |
|---|------|--------|
| 1 | `rahgh.py` | **LP bug fixed** — `forward()` returns GCN output as 4th value (not Z_final); removed `d_prime` param |
| 2 | `fusion.py` | Removed `d_prime` parameter; output dim = d (=64, paper-matched) |
| 3 | `hparam_search.py` | **New file** — random search (N_ITER=50) × 5-fold StratifiedKFold CV on 80% training pool |
| 4 | `node_classification.py` | Added `run_final_nc()` + `_evaluate()` helper; `run_single_nc` uses GPU-first `.to(device)`; 80/20 split |
| 5 | `link_prediction.py` | Added `_build_masked_operators()` (train-edge-only graph), `_run_fold_lp()`, `run_final_lp()`; LP fix uses `emb, *_ = backbone(...)` (position [0]) |
| 6 | `train.py` | Complete rewrite: `run_experiment()` orchestrates CV → best params → final train → held-out test; mean ± std over n_seeds |
| 7 | Configs | Replaced sweep arrays with single default + CV fields (`cv_folds:5`, `test_frac:0.20`, `n_iter:50`, `n_seeds:5`) |
| 8 | Scripts | `--seeds 5` flag added to python commands |

### v3.1 Changes (post-v3)

| # | File | Change |
|---|------|--------|
| 1 | `hparam_search.py` | **Exhaustive grid** — replaced random 50 combos with full 960-combo grid search; added `d=256` and `epochs=100,700` |
| 2 | Configs | Updated to full grid ranges (`d: [64,128,256]`, `epochs: [100,300,500,700]`); removed `n_iter`; set `n_seeds: 10` |
| 3 | `train.py` | Updated default CFG comment |
| 4 | Scripts | `--seeds 10` (updated from 5) |

### v2 fixes retained (from EXPERIMENT_GUIDE_v2.md)
| # | File | Change |
|---|------|--------|
| 1 | `diffusion.py` | `build_A_struct()` builds **α-weighted** `A_struct = Σᵣ αᵣ Aᵣ` for downstream GCN |
| 2 | `rahgh.py` | RAHGH stores `A_list_sp` and `N`; builds `A_hat` dynamically from learned α each forward |
| 3 | `node_classification.py` | No fixed `A_hat` pre-computation; RAHGH builds it internally |
| 4 | `link_prediction.py` | Same — RAHGH builds `A_hat` internally |

---

## Six-Stage Pipeline

1. **TypeSpecificProjector** — ReLU(W_{type} x + b) per type → ℝ^{N×d}, d=64
2. **Bipartite Correction** — P̃_r = P_r @ P_r^T (bipartite) / P_r (homogeneous)
3. **Polynomial Diffusion** — Z_r = Σ_{k=0}^{K} β_{r,k} P̃_r^k H0
4. **Relation Fusion** — Z = Σ_r α_r Z_r with learnable α
5. **Residual Fusion** — Z_final = MLP([H0 || Z]) ∈ ℝ^{N×d}
6. **GCN Head** — on α-weighted A_hat = normalize(Σ_r α_r A_r + I)

---

## Dimension Flow (paper-matched, v3 enforced)

```
X_i ∈ ℝ^{d_φ(i)} → Projector → H^(0) ∈ ℝ^{N×64}
P_r ∈ ℝ^{N×N} → Bipartite Correction → P̃_r ∈ ℝ^{N×N}
Z_r = Σ_k β_{r,k} P̃_r^k H^(0) ∈ ℝ^{N×64}
Z  = Σ_r α_r Z_r ∈ ℝ^{N×64}
Z_final = MLP(2d→d→d) ∈ ℝ^{N×64}    (d_prime = d = 64 enforced)
A_hat = normalize(Σ_r α_r A_r + I) ∈ ℝ^{N×N}
GCN(A_hat, Z_final) ∈ ℝ^{N × out_dim}   (NC: n_classes, LP: d=64)
```

---

## Datasets

| Dataset | File(s) | Location | Status |
|---------|---------|----------|--------|
| DBLP | 11 text files (author_label.txt, author.txt, conf_label.txt, conf.txt, paper_author.txt, paper_conf.txt, paper_label.txt, paper_term.txt, paper.txt, readme.txt, term.txt) | `RAHGH/data/raw/DBLP/` | ✅ Available |
| ACM | ACM.mat | `RAHGH/data/raw/ACM/` | ✅ Available |
| IMDB | movie_metadata.csv | `RAHGH/data/raw/IMDB/` | ✅ Available |

### Automated downloads (from `scripts/download_datasets.py`)
| Dataset | Source | Size |
|---------|--------|------|
| DBLP | PyG / HGB | ~26k nodes |
| ACM | PyG / HGB | ~11k nodes |
| IMDB | PyG / HGB | ~12k nodes |
| OGBN-MAG | OGB | ~1.9M nodes |
| Amazon | PyG / HGB | ~49k nodes |
| LastFM | HGB | ~27k nodes |
| Yelp | HGB | variable |
| Freebase | HGB | variable |

### Available Raw Data (unzipped from drive downloads)
- **ACM, DBLP, IMDB** (HGB format .dat files): `drive-download-20260603T190845Z-3-001/`
- **Amazon, LastFM, PubMed, YouTube** and variants: `drive-download-20260603T191009Z-3-001/`
- `acmdata.ipynb` — Colab notebook with GCN/HAN/HGT experiments on ACM

---

## How to Run

### Install
```bash
bash scripts/install_env.sh               # CUDA-aware env setup
# OR
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Setup Data
```bash
python scripts/setup_data.py              # copy ACM.mat + movie_metadata.csv
# OR
python scripts/download_datasets.py --only dblp acm imdb   # automated download via PyG/HGB
```

Then manually place DBLP text files in `data/raw/DBLP/` (or let download_datasets.py fetch them).

### Run Experiments
```bash
# Node Classification (exhaustive grid CV → final → test, 10 seeds)
python -m src.train --dataset acm --task nc --out results/logs/acm_nc.csv --seeds 10

# Link Prediction
python -m src.train --dataset acm --task lp --out results/logs/acm_lp.csv --seeds 10
```

### Verify Downloads
```bash
python scripts/verify_downloads.py
```

---

## Experiment Protocol (v3.1)

1. **80/20 stratified split** of target nodes/edges
2. **5-fold CV** on the 80% pool with **exhaustive grid search** (960 combos: d∈[64,128,256], K∈[2,3,4,5,6], dropout∈[0.3,0.5], lr∈[0.001,0.005], wd∈[1e-4,1e-3], hidden∈[64,128], epochs∈[100,300,500,700])
3. **Best params** selected by mean validation Macro-F1 (NC) / AUC (LP)
4. **Final training** on full 80% with best params
5. **Evaluation** on held-out 20%
6. **Repeat** 10× with different seeds → report mean ± std

### GNN-specific design
- Full graph (all N nodes, all edges) is always used for **message passing**
- Only **label masks** change per fold (transductive: no neighbor-label leakage)
- For LP: val/test edges are **removed** from the graph to prevent leakage
- `P_list` operators are pre-built once and reused across folds

---

## Hyperparameter Search Space (Exhaustive Grid)

| Param | Values |
|-------|--------|
| d | 64, 128, 256 |
| K | 2, 3, 4, 5, 6 |
| dropout | 0.3, 0.5 |
| lr | 0.001, 0.005 |
| wd | 1e-4, 1e-3 |
| hidden | 64, 128 |
| epochs | 100, 300, 500, 700 |

Full grid: 960 combos × 5 folds = 4800 train runs per CV phase (run once, then final train/test repeated 10 seeds).

---

## Outputs
- Per-run results streamed to `results/logs/{dataset}_{task}.csv`
- Learned α (relation importances) and β (hop weights) per run
- Best checkpoint saved on validation improvement
- Final report: mean ± std over seeds

---

## files_unzipped Integration

### `scripts/download_datasets.py`
- Downloads 8 heterogeneous datasets (DBLP, ACM, IMDB, OGBN-MAG, Amazon, LastFM, Yelp, Freebase)
- Tries PyG datasets first, falls back to HGB benchmark zip downloads
- Uses `.done` marker files for idempotent re-runs
- Stores in lowercase directories (dblp/, acm/, imdb/) under data/raw/

### `scripts/install_env.sh`
- Auto-detects CUDA version via nvcc
- Installs correct PyTorch + PyG wheels for the detected CUDA version
- Also installs ogb, requests, tqdm, matplotlib, seaborn, jupyter, ipykernel
- Supports `--cpu` and `--cuda X.Y` flags

### `scripts/verify_downloads.py`
- Checks `.done` markers, directory sizes, and file counts
- Attempts PyG/OGB data load to confirm integrity
- Reports node types and relation counts per dataset

### Differences from EXPERIMENT_GUIDE_v3.md loaders
- The `download_datasets.py` downloads PyG-native data (not the custom text-based format)
- The v3 guide loaders (`dblp_loader.py`, `acm_loader.py`, `imdb_loader.py`) expect text files / .mat / .csv
- The `files_unzipped/README.md` outlines a separate project layout with unified `loader.py`, baselines/, etc.
- `files_unzipped` is a **supplementary/side resource** — not required for the v3 experiment protocol

---

## Common Errors and Fixes

**`mat1 and mat2 shapes cannot be multiplied`**
→ `in_dims` order does not match `X_dict.values()` order. Check the dimension flow table.

**`sparse tensor indices out of range`**
→ Node offset wrong in loader. Print `max(A.tocoo().col)` and confirm it equals `N-1`.

**`CUDA out of memory` on bipartite correction**
→ For N > 50k use sparse version: `torch.sparse.mm(P_r, P_r.t())` instead of dense.

**`RAHGH() missing keyword arguments`**
→ v3 requires `A_list_sp`, `N`, `device`; `d_prime` parameter is removed.

**LP decoder AUC ≈ 0.5**
→ `emb, *_ = backbone(...)` — position [0] is the GCN output. v3 returns `(gcn_out, alpha, beta, gcn_out)`.

**Z_final dimension mismatch**
→ v3 fusion MLP outputs dim d (=64), not a separate d_prime. GCN input dim = d.

---

## Change Log

| Date | File | Change |
|------|------|--------|
| 2026-06-05 | All | **v3 upgrade** — CV protocol, hparam_search.py, run_final_*, LP fix, d_prime removal, GPU-first |
| 2026-06-05 | `scripts/` | Added download_datasets.py, install_env.sh, verify_downloads.py from files_unzipped |
| 2026-06-05 | `requirements.txt` | Expanded with ogb, requests, tqdm, wandb, jupyter, ipykernel |
| 2026-06-05 | `train.py` | Rewrote from v2 sweep to v3 run_experiment protocol |
| 2026-06-05 | `train.py:71` | Fixed `--` → `──` (Unicode em dash) to match guide |
| 2026-06-05 | `train.py:77` | Added docstring to `_get_lp_edges()` |
| 2026-06-05 | All v2 files | Initial implementation from `EXPERIMENT_GUIDE_v2.md` |
| 2026-06-05 | Venv `graph` | Created at `E:\Homogenizning Heterogeneous Graphs\graph\` — Python 3.14.5, torch 2.12.0+cpu (CPU-only, no CUDA wheels for Python 3.14) |
| 2026-06-05 | Venv `graph_cu` | Created at `E:\Homogenizning Heterogeneous Graphs\graph_cu\` — Python 3.11, torch 2.6.0+cu124 (CUDA 12.4, RTX 3050 6GB detected) |
| 2026-06-05 | Dependencies | Installed torch 2.12.0, torchvision, torch-geometric 2.7.0, ogb, requests, tqdm, scipy, pandas, scikit-learn into both venvs |
| 2026-06-05 | `download_datasets.py` | Ran from files_unzipped — PyG ACM/DBLP/IMDB dataset classes unavailable in tg 2.7.0 (deprecated); HGB fallback URL returned non-zip content |
| 2026-06-05 | `verify_downloads.py` | Ran from files_unzipped — confirmed ACM download incomplete (0 MB, no .done marker) |
| 2026-06-05 | `install_env.sh` | Not runnable on Windows (bash script); equivalent deps installed via pip |
| 2026-06-06 | `hparam_search.py` | **v3.1** — exhaustive grid (960 combos: added d=256, epochs 100/700); removed random sampling |
| 2026-06-06 | Configs | Updated to full grid ranges; removed `n_iter`; set `n_seeds: 10` |
| 2026-06-06 | `how_to_run.md` | Created with full instructions for exhaustive grid + 10 seeds |
| 2026-06-06 | `project_progress.md` | Created tracking data availability status |
