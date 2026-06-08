import itertools
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import Adam
from tqdm import tqdm

from ..model.rahgh    import RAHGH
from ..model.diffusion import build_operators


PARAM_GRID = {
    'd'         : [32, 64,128],
    'K'         : [1, 2, 3, 4, 5],
    'dropout'   : [0.3, 0.5],
    'lr'        : [0.001, 0.005],
    'wd'        : [0.0, 0.001],
    'gcn_hidden': [64,128],
    'epochs'    : [500, 700, 1000],
}

N_FOLDS   = 5
TEST_FRAC = 0.20


def _all_combos():
    keys = list(PARAM_GRID.keys())
    for vals in itertools.product(*PARAM_GRID.values()):
        yield dict(zip(keys, vals))


def _run_fold_nc(data, P_list, X_list, params, tr_idx, va_idx, device):
    d       = params['d']
    R       = len(data['A_list_sp'])
    Nt      = data['target_size']
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    labels  = data['labels'].to(device)

    model = RAHGH(
        in_dims=in_dims, d=d, R=R, K=params['K'],
        gcn_hidden=params['gcn_hidden'],
        out_dim=data['n_classes'],
        dropout=params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    opt = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    tr_t = torch.tensor(tr_idx, dtype=torch.long, device=device)
    va_t = torch.tensor(va_idx, dtype=torch.long, device=device)

    best_vm, best_sd = 0.0, None
    pbar = tqdm(range(1, params['epochs'] + 1), desc="Fold training", leave=False)
    for ep in pbar:
        model.train()
        opt.zero_grad()
        logits, *_ = model(X_list, P_list)
        F.cross_entropy(logits[:Nt][tr_t], labels[tr_t]).backward()
        opt.step()

        if ep % 50 == 0 or ep == params['epochs']:
            model.eval()
            with torch.no_grad():
                logits, *_ = model(X_list, P_list)
                p = logits[:Nt][va_t].argmax(1).cpu().numpy()
                y = data['labels'][va_idx].numpy()
                vm = f1_score(y, p, average='macro', zero_division=0)
            pbar.set_description(f"Fold tr loss? val_macro={vm:.4f}")
            if vm > best_vm:
                best_vm = vm
                best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    del model
    torch.cuda.empty_cache()
    return best_vm


def hparam_search_nc(data, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Nt     = data['target_size']
    lbl_np = data['labels'].numpy()

    tr80, te20 = train_test_split(
        np.arange(Nt), test_size=TEST_FRAC, random_state=seed,
        stratify=lbl_np
    )

    P_list = build_operators(data['A_list_sp'], data['bipartite_flags'], device)
    X_list = [x.to(device) for x in data['X_dict'].values()]

    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos   = list(_all_combos())
    best_params, best_mean = None, -1.0

    for ci, params in enumerate(tqdm(combos, desc="Hyperparam combos", position=0)):
        fold_scores = []
        for fold, (tr_fold, va_fold) in enumerate(
                tqdm(skf.split(tr80, lbl_np[tr80]), desc=f"Folds for combo {ci+1}", leave=False, total=N_FOLDS)):
            tr_idx = tr80[tr_fold]
            va_idx = tr80[va_fold]
            vm = _run_fold_nc(data, P_list, X_list, params,
                              tr_idx, va_idx, device)
            fold_scores.append(vm)

        mean_vm = float(np.mean(fold_scores))
        if mean_vm > best_mean:
            best_mean   = mean_vm
            best_params = params

    return best_params, tr80, te20


def hparam_search_lp(data, target_edges, seed=42):
    from .link_prediction import _run_fold_lp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_idx    = np.arange(len(target_edges))
    tr80_idx, te20_idx = train_test_split(
        all_idx, test_size=TEST_FRAC, random_state=seed
    )
    tr80_edges = target_edges[tr80_idx]
    te20_edges = target_edges[te20_idx]

    combos     = list(_all_combos())
    best_params, best_mean = None, -1.0
    kf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=seed)
    dummy_labels = np.zeros(len(tr80_edges), dtype=int)

    for ci, params in enumerate(tqdm(combos, desc="Hyperparam combos", position=0)):
        fold_aucs = []
        for fold, (tr_fold, va_fold) in enumerate(
                tqdm(kf.split(tr80_edges, dummy_labels), desc=f"Folds for combo {ci+1}", leave=False, total=N_FOLDS)):
            auc = _run_fold_lp(data, tr80_edges[tr_fold],
                               tr80_edges[va_fold],
                               te20_edges, params, device)
            fold_aucs.append(auc)

        mean_auc = float(np.mean(fold_aucs))
        if mean_auc > best_mean:
            best_mean, best_params = mean_auc, params

    return best_params, tr80_edges, te20_edges
