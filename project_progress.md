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

Change: Explicitly documents linear polynomial diffusion (Stage 4) as purely spectral filtering vs. non-linear GCN/GAT backbone as the weight-matrix learner, preventing future "optimization" that would merge or remove either phase.

## Experiment Tasks & Source Files — Status

| Item | Status | Location |
|------|--------|----------|
| Task: Node Classification | ✅ Spec'd in `EXPERIMENT_GUIDE_v4.md` | Steps 1–6 |
| Task: Link Prediction | ✅ Spec'd in `EXPERIMENT_GUIDE_v4.md` | Steps 7–12 |
| Task: Graph Clustering (`graph_clustering.py`) | ✅ Added | `src/tasks/graph_clustering.py` — Step 13 |
| Task: Recommendation (`recommendation.py`) | ✅ Added | `src/tasks/recommendation.py` — Step 14 |
| Extended hparam_search (`hparam_search_cluster`, `hparam_search_rec`) | ✅ Added | `src/tasks/hparam_search.py` — Step 15 |
| CSV storage specification (4-task folder structure) | ✅ Documented | `EXPERIMENT_GUIDE_v4.md` — Step 16 |
| `collect_results.py` (LaTeX-ready summary) | ✅ Added | `scripts/collect_results.py` — Step 16 |

## Config Files — Status

| File | Task | Dataset |
|------|------|---------|
| `configs/dblp_cluster.yaml` | ✅ Cluster | DBLP |
| `configs/acm_cluster.yaml` | ✅ Cluster | ACM |
| `configs/amazon_rec.yaml` | ✅ Rec | Amazon |
| `configs/lastfm_rec.yaml` | ✅ Rec | LastFM |
| `configs/dblp_rec.yaml` | ✅ Rec | DBLP (author→paper) |

All defined in `EXPERIMENT_GUIDE_v4.md` — Step 17.

## Run Scripts — Status

| Script | Purpose |
|--------|---------|
| `scripts/run_clustering.sh` | Graph clustering (DBLP, ACM, IMDB) — Step 18 |
| `scripts/run_rec.sh` | Recommendation (Amazon, LastFM, DBLP) — Step 18 |
| `scripts/run_all.sh` | Full experiment suite (NC → LP → Cluster → Rec) — Step 18 |

## Metrics Summary

| Task | Reported Metrics | CV Selection |
|------|-----------------|--------------|
| Node Classification | Accuracy ± std, Macro-F1 ± std, Micro-F1 ± std | Macro-F1 |
| Link Prediction | AUC-ROC ± std, AP ± std | AUC-ROC |
| Graph Clustering | NMI ± std, ARI ± std, ACC ± std, Silhouette ± std | NMI |
| Recommendation | Hits@1/5/10/20, NDCG@10, Recall@10, MRR | Hits@10 |

All means & std computed over **10 seeds** on the held-out 20 % test set.

## Training Output Cleanup

| Change | Status | Files |
|--------|--------|-------|
| Removed all per-stage `print()` statements from model forward passes | ✅ Done | `projector.py`, `diffusion.py`, `fusion.py`, `rahgh.py` |
| Removed all per-stage `print()` from training/evaluation code | ✅ Done | `node_classification.py`, `link_prediction.py`, `hparam_search.py`, `train.py` |
| Added live epoch metrics via `tqdm.set_description()` | ✅ Done | `_run_fold_nc`, `run_single_nc`, `run_final_nc`, `_run_fold_lp`, `run_single_lp`, `run_final_lp` |
| Verified all CUDA device placement | ✅ Done | All models `.to(device)`, all tensors `device=device`, operators built on correct device |

Remaining output: experiment header, final results summary, and tqdm progress bars — no intermediate stage noise.

## Next Steps

1. Run `python RAHGH/scripts/setup_data.py` if needed to verify paths
2. Proceed with experiments per `RAHGH/project_flow.md` — start with `scripts/run_all.sh`
