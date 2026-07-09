"""
Quick DBLP NC test with per-fold printing.
Run from project root: python HGB/scripts/_test_nc.py
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.join(_HERE, '..', '..')
_RAHGH = os.path.join(_PROJ, 'RAHGH')
sys.path.insert(0, _RAHGH)
os.chdir(_RAHGH)

import numpy as np
from src.tasks import hparam_search
from sklearn.model_selection import train_test_split

# Fast test settings
hparam_search.N_ITER = 1
hparam_search.N_FOLDS = 3
hparam_search.PATIENCE = 15
hparam_search.PARAM_GRID_BASE['epochs'] = [100]
hparam_search.PARAM_GRID_BASE['d'] = [64]
hparam_search.PARAM_GRID_BASE['d_prime'] = [64]

from src.train import _get_loader

data = _get_loader('dblp')
data['name'] = 'dblp'
data['train_indices'] = np.load('../HGB/splits/nc/dblp/train_indices.npy')
data['test_indices'] = np.load('../HGB/splits/nc/dblp/test_indices.npy')

tr80 = data['train_indices']
te20 = data['test_indices']
lbl_np = data['labels'].numpy()

best_params = {'d': 64, 'd_prime': 64, 'K': 3, 'dropout': 0.5, 'dropout_gnn': 0.5,
               'lr': 0.005, 'wd': 0.0001, 'epochs': 100, 'hidden': 64,
               'label_smoothing': 0.1, 'warmup': 0}

from src.tasks.node_classification import run_final_nc

print('--- HGB DBLP NC Test (3 seeds) ---')
for seed in range(3):
    print(f'\n  Seed {seed}: training ...', flush=True)
    r = run_final_nc(data, best_params, tr80, te20, seed=seed,
                     out_dir='../HGB/results/nc/dblp_test')
    print(f'  Seed {seed} RESULTS: test_macro={r["test_macro"]:.4f}  '
          f'test_micro={r["test_micro"]:.4f}  '
          f'test_acc={r["test_acc"]:.4f}  '
          f'test_auc={r["test_auc"]:.4f}')
