# Project Progress

## DBLP Raw Data — Status

| Item | Status | Location |
|------|--------|----------|
| Text files (11 files) | ✅ Ready | `RAHGH/data/raw/DBLP/` |
| Files include | ✅ | author_label.txt, author.txt, conf_label.txt, conf.txt, paper_author.txt, paper_conf.txt, paper_label.txt, paper_term.txt, paper.txt, readme.txt, term.txt |
| HGB `.dat` files (drive download) | ✅ Also uploaded | `drive-download-20260603T190845Z-3-001/DBLP/DBLP/` |

## Other Datasets

| Dataset | Status | Location |
|---------|--------|----------|
| **ACM** | ✅ Ready | `RAHGH/data/raw/ACM/ACM.mat` |
| **IMDB** | ✅ Ready | `RAHGH/data/raw/IMDB/movie_metadata.csv` |
| **Amazon, LastFM, PubMed, YouTube, Freebase** | ✅ Available (drive download) | `drive-download-20260603T191009Z-3-001/` |

## Model Specification — Status

| Item | Status | Location |
|------|--------|----------|
| `rahgh_model_spec.md` — RAHGHClassifier docstring updated | ✅ Done | `rahgh_model_spec.md:896` |
| `rahgh_model_spec.md` — Two-Stage Message Passing Philosophy section added | ✅ Done | `rahgh_model_spec.md:1049` |

## Experiment Tasks — v4 Source Files Status

| Item | Status | Location |
|------|--------|----------|
| Node Classification (`node_classification.py`) | ✅ Existing, no changes needed | `src/tasks/node_classification.py` |
| Link Prediction (`link_prediction.py`) | ✅ Existing, no changes needed | `src/tasks/link_prediction.py` |
| Graph Clustering (`graph_clustering.py`) | ✅ **NEW** — unsupervised training + K-Means eval (NMI, ARI, ACC) | `src/tasks/graph_clustering.py` |
| Recommendation (`recommendation.py`) | ✅ **NEW** — BPR loss + top-K metrics (Recall@K, NDCG@K, Hit@K, Precision@K, MRR) | `src/tasks/recommendation.py` |
| HParam Search (`hparam_search.py`) | ✅ **Updated** — random search (50 combos), cl/rec CV wrappers, CSV + best_params.json logging | `src/tasks/hparam_search.py` |
| Train Orchestrator (`train.py`) | ✅ **Updated** — all 4 tasks, N_SEEDS=10, per-run + summary CSVs per task dir | `src/train.py` |

## Config Files — Status

| File | Task | Dataset |
|------|------|---------|
| `configs/dblp_nc.yaml` | ✅ NC | DBLP |
| `configs/acm_nc.yaml` | ✅ NC | ACM |
| `configs/imdb_nc.yaml` | ✅ NC | IMDB |
| `configs/dblp_lp.yaml` | ✅ LP | DBLP |
| `configs/acm_lp.yaml` | ✅ LP | ACM |
| `configs/imdb_lp.yaml` | ✅ LP | IMDB |
| `configs/dblp_cl.yaml` | ✅ **NEW** Cluster | DBLP |
| `configs/acm_cl.yaml` | ✅ **NEW** Cluster | ACM |
| `configs/imdb_cl.yaml` | ✅ **NEW** Cluster | IMDB |
| `configs/dblp_rec.yaml` | ✅ **NEW** Rec | DBLP |
| `configs/acm_rec.yaml` | ✅ **NEW** Rec | ACM |
| `configs/imdb_rec.yaml` | ✅ **NEW** Rec | IMDB |

## Run Scripts — Status

| Script | Purpose |
|--------|---------|
| `scripts/run_nc.sh` | Node classification (DBLP, ACM, IMDB) |
| `scripts/run_lp.sh` | Link prediction (DBLP, ACM, IMDB) |
| `scripts/run_cl.sh` | **NEW** Graph clustering (DBLP, ACM, IMDB) |
| `scripts/run_rec.sh` | **NEW** Recommendation (DBLP, ACM, IMDB) |

## Results Directories — Status

| Directory | Contents |
|-----------|----------|
| `results/nc/` | ✅ cv_fold_scores.csv, per_run_results.csv, summary.csv, epoch_logs/ |
| `results/lp/` | ✅ cv_fold_scores.csv, per_run_results.csv, summary.csv, epoch_logs/ |
| `results/clustering/` | ✅ cv_fold_scores.csv, per_run_results.csv, summary.csv, epoch_logs/ |
| `results/recommendation/` | ✅ cv_fold_scores.csv, per_run_results.csv, summary.csv, epoch_logs/ |

All dirs created with `best_params.json` support via `_save_best_params()`.

## Metrics Summary

| Task | Reported Metrics | CV Selection |
|------|-----------------|--------------|
| Node Classification | Accuracy ± std, Macro-F1 ± std, Micro-F1 ± std | Macro-F1 |
| Link Prediction | AUC-ROC ± std, AP ± std | AUC-ROC |
| Graph Clustering | NMI ± std, ARI ± std, ACC ± std | NMI |
| Recommendation | Recall@K, NDCG@K, Hit@K, Precision@K, MRR (K ∈ {10,20,50}) | Recall@K |

All means & std computed over **10 seeds** on the held-out 20 % test set.

## Command Reference

```bash
python -m src.train --dataset dblp --task nc   --seeds 10
python -m src.train --dataset acm  --task cl   --seeds 10
python -m src.train --dataset imdb --task rec  --seeds 10
python -m src.train --dataset dblp --task lp   --seeds 10
```

## Git

| Item | Status |
|------|--------|
| `.gitignore` | ✅ Created (pycache, graph_cu, __MACOSX, drive-download-*, results) |
| Repo initialized | ✅ `main` branch |
| Remote | ✅ `origin/main` pushed to `https://github.com/Aritra960966/Homogenizing_heterogeneous_graph.git` |
| v4 changes committed | ✅ 13 files, commit `4679edc` |

**Note:** Remote `master` branch still exists as default on GitHub. Change default to `main` at Settings → Branches, then `git push origin --delete master`.
