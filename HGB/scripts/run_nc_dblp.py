"""
DBLP NC — HGB splits, per-fold printing, GPU-optimized settings.
Run: python HGB/scripts/run_nc_dblp.py
"""
import sys, os, numpy as np, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAHGH = os.path.join(_HERE, '..', '..', 'RAHGH')
sys.path.insert(0, _RAHGH)
os.chdir(_RAHGH)

# Verify GPU
assert torch.cuda.is_available(), "GPU not available — this script is GPU-optimized"
torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# GPU-optimized hyperparams
from src.tasks import hparam_search
hparam_search.N_ITER = 10
hparam_search.N_FOLDS = 5
hparam_search.PATIENCE = 50
hparam_search.PARAM_GRID_BASE['epochs'] = [500]
hparam_search.PARAM_GRID_BASE['d'] = [128, 256]
hparam_search.PARAM_GRID_BASE['d_prime'] = [128, 256]
hparam_search.PARAM_GRID_BASE['K'] = [2, 3, 4, 5]

from src.train import _get_loader, N_SEEDS

data = _get_loader('dblp')
data['name'] = 'dblp'
data['train_indices'] = np.load(os.path.join(_HERE, '..', 'splits', 'nc', 'dblp', 'train_indices.npy'))
data['test_indices'] = np.load(os.path.join(_HERE, '..', 'splits', 'nc', 'dblp', 'test_indices.npy'))

from src.train import run_nc
run_nc('dblp', os.path.join(_HERE, '..', 'results', 'nc', 'dblp'))
