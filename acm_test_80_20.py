import sys, os, numpy as np, torch, time, json, copy, warnings
sys.path.insert(0, r'D:\Aritra\graph\Homogenizing_heterogeneous_graph\RAHGH')
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from src.data.acm_loader import load_acm
from src.model.rahgh import build_rahgh_classifier, build_edge_index_dict, build_node_type_indices, compile_model
from src.tasks.hparam_search import _run_fold_nc, _random_combos, PARAM_GRID_BASE, N_ITER, N_FOLDS
import torch.nn.functional as F
from torch.optim import AdamW

data = load_acm(mat_path=r'D:\Aritra\graph\Homogenizing_heterogeneous_graph\ACM.mat', seed=42)
data['name'] = 'acm'
Nt = data['target_size']
labels = data['labels'].numpy()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use 80/20 split (RAHGH original protocol)
tr80, te20 = train_test_split(np.arange(Nt), test_size=0.2, random_state=42, stratify=labels)
print(f'80/20 split: {len(tr80)} train, {len(te20)} test')

# Quick hparam search (2 combos, 2 folds)
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
combos = _random_combos(PARAM_GRID_BASE, seed=42, n=2)
print(f'Testing {len(combos)} combos with 2 folds')

best_params, best_mean = None, 0.0

x_dict_once = {k: v.to(device) for k, v in data['X_dict'].items()}
edge_index_dict_once = build_edge_index_dict(data, device)
node_type_indices_once = {k: v.to(device) for k, v in build_node_type_indices(data).items()}
labels_once = data['labels'].to(device)

for ci, params in enumerate(combos):
    fold_scores = []
    for fold, (tr_fold, va_fold) in enumerate(skf.split(tr80, labels[tr80])):
        vm = _run_fold_nc(data, params, tr80[tr_fold], tr80[va_fold], device, head='gcn',
                          x_dict=x_dict_once, edge_index_dict=edge_index_dict_once,
                          labels=labels_once, node_type_indices=node_type_indices_once)
        fold_scores.append(vm)
    mean_vm = float(np.mean(fold_scores))
    print(f'Combo {ci}: mean={mean_vm:.4f}')
    if mean_vm > best_mean:
        best_mean, best_params = mean_vm, copy.deepcopy(params)

print(f'Best params: {best_params} (val_macro={best_mean:.4f})')

# Final run with best params
from src.tasks.node_classification import run_final_nc
result = run_final_nc(data, best_params, tr80, te20, seed=42, out_dir=None, head='gcn')
print(f'Final: macro={result["test_macro"]:.4f} micro={result["test_micro"]:.4f} acc={result["test_acc"]:.4f}')
