"""
Run all 6 HGB experiments: DBLP NC, ACM NC, IMDB NC, LastFM LP, Amazon LP, PubMed LP.
Uses HGB splits for NC, full PARAM_GRID_BASE with N_ITER=100, macro_f1 for selection.
"""
import sys, os, numpy as np, torch, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAHGH = os.path.join(_HERE, '..', 'RAHGH')
sys.path.insert(0, _RAHGH)
os.chdir(_RAHGH)

print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

from src.tasks import hparam_search
hparam_search.N_ITER = 100
hparam_search.N_FOLDS = 5
hparam_search.PATIENCE = 50
# Restore full grids (in case a prior script overrode them)
hparam_search.PARAM_GRID_BASE['epochs'] = [200, 300, 400, 500, 600, 700, 1000]
hparam_search.PARAM_GRID_BASE['d'] = [32, 64, 128]
hparam_search.PARAM_GRID_BASE['d_prime'] = [32, 64, 128]
hparam_search.PARAM_GRID_BASE['K'] = [1, 2, 3, 4, 5, 6]

from src.train import _get_loader, run_nc, run_lp
from src.tasks.hparam_search import _save_best_params

results_root = os.path.join(_HERE, 'results')

EXPERIMENTS = []

# ── NC experiments with HGB splits ───────────────────────────────────────
NC_SPLITS_DIR = os.path.join(_HERE, 'splits', 'nc')
for ds_name in ['dblp', 'acm', 'imdb']:
    out = os.path.join(results_root, 'nc', ds_name)
    os.makedirs(out, exist_ok=True)
    data = _get_loader(ds_name)
    data['name'] = ds_name
    data['train_indices'] = torch.from_numpy(np.load(os.path.join(NC_SPLITS_DIR, ds_name, 'train_indices.npy')))
    data['test_indices']  = torch.from_numpy(np.load(os.path.join(NC_SPLITS_DIR, ds_name, 'test_indices.npy')))
    EXPERIMENTS.append(('nc', ds_name, data, out))

# ── LP experiments ──────────────────────────────────────────────────────
import scipy.sparse as sp
TARGET_REL_IDX = {'dblp': 0, 'acm': 0, 'imdb': 2, 'lastfm': 0, 'amazon': 0, 'pubmed': 4}
for ds_name in ['lastfm', 'amazon', 'pubmed']:
    out = os.path.join(results_root, 'lp', ds_name)
    os.makedirs(out, exist_ok=True)
    data = _get_loader(ds_name)
    data['name'] = ds_name
    rel_idx = data.get('target_relation_idx', TARGET_REL_IDX.get(ds_name, 0))
    A = data['A_list_sp'][rel_idx].tocoo()
    data['target_edges'] = np.column_stack([A.row, A.col])
    EXPERIMENTS.append(('lp', ds_name, data, out))

# ── Run all experiments sequentially ────────────────────────────────────
grand_start = time.time()
for task, ds_name, data, out_dir in EXPERIMENTS:
    print(f"\n{'='*70}")
    print(f"  START: {ds_name.upper()} {task.upper()}  ->  {out_dir}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        if task == 'nc':
            run_nc(ds_name, out_dir)
        elif task == 'lp':
            run_lp(ds_name, out_dir)
        elapsed = time.time() - t0
        print(f"\n  COMPLETE: {ds_name} {task}  [{elapsed/60:.1f} min]")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  FAILED: {ds_name} {task}  —  {e}")

grand_elapsed = time.time() - grand_start
print(f"\n{'='*70}")
print(f"  ALL EXPERIMENTS COMPLETE  [total: {grand_elapsed/60:.1f} min]")
print(f"{'='*70}")
