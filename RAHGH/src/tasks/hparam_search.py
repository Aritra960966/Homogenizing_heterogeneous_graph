"""
src/train/hparam_search.py

Hyperparameter search for RAHGH across four tasks:
  NC  — node classification   (val metric: macro-F1)
  CL  — node clustering       (val metric: NMI)
  REC — recommendation        (val metric: Recall@K)
  LP  — link prediction       (val metric: AUC-ROC)

Each search function:
  1. Holds out 20% as a final test split (never touched during search).
  2. Runs N_FOLDS cross-validation on the remaining 80%.
  3. Picks the combo with the best mean validation metric.
  4. Returns (best_params, tr80_idx, te20_idx).
"""

import csv
import itertools
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.optim import Adam

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):          # silent fallback if tqdm not installed
        return iterable

from ..model.rahgh     import RAHGH, compile_model
from ..model.diffusion import build_operators


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

N_FOLDS    = 5      # CV folds during search
TEST_FRAC  = 0.20   # held-out test fraction
PATIENCE   = 50     # early-stopping patience (epochs with no val improvement)
N_RANDOM   = 100    # combos sampled for CL / REC (grid too large for full sweep)


# ─────────────────────────────────────────────────────────────────────────────
#  Hyperparameter grids
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRID_BASE = {
    'd'         : [64, 128,256],
    'K'         : [1,2, 3,4,5],
    'dropout'   : [0.3, 0.5],
    'lr'        : [0.001, 0.005],
    'wd'        : [1e-4, 1e-3],
    'gcn_hidden': [64, 128],
    'epochs'    : [300, 500,700,1000],
}

# Clustering — extends base with unsupervised loss options (sample N_RANDOM)
PARAM_GRID_CL = {
    **PARAM_GRID_BASE,
    'cl_loss' : ['reconstruction', 'contrastive'],
    'cl_temp' : [0.1, 0.5],
}

# Recommendation — extends base with BPR-specific options (sample N_RANDOM)
PARAM_GRID_REC = {
    **PARAM_GRID_BASE,
    'K_rec'    : [10, 20, 50],
    'neg_ratio': [5, 10],
    'bpr_reg'  : [1e-4, 1e-3],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _all_combos(grid: dict) -> list:
    """Return every combination in the grid as a list of dicts."""
    keys   = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def _random_combos(grid: dict, n: int, seed: int = 0) -> list:
    """Sample n random combinations from the grid without replacement."""
    combos = _all_combos(grid)
    random.seed(seed)
    random.shuffle(combos)
    return combos[:n]


def _save_best_params(best_params: dict, dataset: str, task: str, out_dir: str):
    """Upsert best_params into results/best_params.json under key '{dataset}_{task}'."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'best_params.json')
    try:
        with open(path) as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {}
    existing[f"{dataset}_{task}"] = best_params
    with open(path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"[hparam] best params saved → {path}")


def _write_csv(rows: list, path: str):
    """Write a list of dicts to a CSV file."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"[hparam] CV scores saved → {path}")


def _build_model(data: dict, params: dict, out_dim: int, device: torch.device) -> RAHGH:
    """Construct and compile a RAHGH model from data metadata + params dict."""
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    model   = RAHGH(
        in_dims    = in_dims,
        d          = params['d'],
        R          = len(data['A_list_sp']),
        K          = params['K'],
        gcn_hidden = params['gcn_hidden'],
        out_dim    = out_dim,
        dropout    = params['dropout'],
        A_list_sp  = data['A_list_sp'],
        N          = data['N'],
        device     = device,
    ).to(device)
    return compile_model(model)


# ─────────────────────────────────────────────────────────────────────────────
#  Fold runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_fold_nc(
    data   : dict,
    P_list : list,
    X_list : list,
    params : dict,
    tr_idx : np.ndarray,
    va_idx : np.ndarray,
    device : torch.device,
) -> float:
    """
    Train one NC fold and return the best validation macro-F1.

    Uses AMP + GradScaler on CUDA, early stopping with PATIENCE,
    and evaluates every epoch (cheap: single forward with no_grad).
    """
    Nt     = data['target_size']
    labels = data['labels'].to(device)
    tr_t   = torch.tensor(tr_idx, dtype=torch.long, device=device)
    va_t   = torch.tensor(va_idx, dtype=torch.long, device=device)

    model  = _build_model(data, params, out_dim=data['n_classes'], device=device)
    opt    = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler(device='cuda') if use_amp else None

    best_vm, best_sd, stall = 0.0, None, 0

    for ep in tqdm(range(1, params['epochs'] + 1), desc='NC fold', leave=False):
        # ── train ──
        model.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, *_ = model(X_list, P_list)
            loss = F.cross_entropy(logits[:Nt][tr_t], labels[tr_t])
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        # ── validate ──
        model.eval()
        with torch.no_grad():
            logits, *_ = model(X_list, P_list)
        preds = logits[:Nt][va_t].argmax(1).cpu().numpy()
        truth = data['labels'][va_idx].numpy()
        vm    = f1_score(truth, preds, average='macro', zero_division=0)

        if vm > best_vm:
            best_vm = vm
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            stall   = 0
        else:
            stall += 1
            if stall >= PATIENCE:
                break

    del model
    torch.cuda.empty_cache()
    return best_vm


def _run_fold_cl(
    data   : dict,
    P_list : list,
    X_list : list,
    params : dict,
    tr_idx : np.ndarray,
    va_idx : np.ndarray,
    device : torch.device,
) -> float:
    """
    Train one clustering fold and return the best validation NMI.

    Trains with MSE reconstruction loss on the training set.
    Evaluates NMI via KMeans on the validation embeddings every 50 epochs.
    """
    from sklearn.cluster import KMeans

    Nt      = data['target_size']
    n_cl    = data['n_classes']
    d       = params['d']
    tr_t    = torch.tensor(tr_idx, dtype=torch.long, device=device)

    model   = _build_model(data, params, out_dim=d, device=device)
    decoder = torch.nn.Linear(d, d).to(device)
    opt     = Adam(
        list(model.parameters()) + list(decoder.parameters()),
        lr=params['lr'], weight_decay=params['wd'],
    )

    # Target for reconstruction: concatenated X projected to dim d
    X_cat = torch.cat(X_list, dim=0)[:Nt][tr_t]          # (|tr|, raw_dim)
    if X_cat.shape[1] != d:
        X_cat = X_cat[:, :d] if X_cat.shape[1] > d \
                else F.pad(X_cat, (0, d - X_cat.shape[1]))
    X_cat = X_cat.to(device)

    best_nmi = 0.0

    for ep in tqdm(range(1, params['epochs'] + 1), desc='CL fold', leave=False):
        model.train(); decoder.train()
        opt.zero_grad()
        emb, *_ = model(X_list, P_list)
        loss    = F.mse_loss(decoder(emb[:Nt][tr_t]), X_cat)
        loss.backward()
        opt.step()

        if ep % 50 == 0 or ep == params['epochs']:
            model.eval()
            with torch.no_grad():
                emb_v, *_ = model(X_list, P_list)
            emb_np = emb_v[:Nt][va_idx].cpu().numpy()
            pred   = KMeans(n_clusters=n_cl, n_init=10, random_state=0).fit_predict(emb_np)
            nmi    = normalized_mutual_info_score(data['labels'][va_idx].numpy(), pred)
            best_nmi = max(best_nmi, nmi)

    del model, decoder
    torch.cuda.empty_cache()
    return best_nmi


def _run_fold_rec(
    data      : dict,
    tr_edges  : np.ndarray,
    va_edges  : np.ndarray,
    params    : dict,
    device    : torch.device,
) -> float:
    """
    Train one recommendation fold and return the best validation Recall@K.

    Uses BPR (Bayesian Personalised Ranking) loss with uniform negative sampling.
    K_rec defaults to params['K_rec'] (default 20).
    """
    from .recommendation import bpr_loss, recall_at_k

    d      = params['d']
    K_rec  = params.get('K_rec', 20)
    X_list = [x.to(device) for x in data['X_dict'].values()]
    P_list = build_operators(data['A_list_sp'], data['bipartite_flags'], device)

    model  = _build_model(data, params, out_dim=d, device=device)
    opt    = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    all_items = np.unique(tr_edges[:, 1])
    user_pos  = {}
    for u, i in tr_edges:
        user_pos.setdefault(u, set()).add(i)

    rng      = np.random.default_rng(0)
    best_rec = 0.0

    for ep in tqdm(range(1, params['epochs'] + 1), desc='REC fold', leave=False):
        model.train()
        opt.zero_grad()
        emb, *_   = model(X_list, P_list)
        users      = tr_edges[:, 0]
        pos_items  = tr_edges[:, 1]
        neg_items  = rng.choice(all_items, size=len(users))
        loss = bpr_loss(
            emb, users, pos_items, neg_items, device,
            reg=params.get('bpr_reg', 1e-4),
        )
        loss.backward()
        opt.step()

        if ep % 50 == 0 or ep == params['epochs']:
            model.eval()
            with torch.no_grad():
                emb_v, *_ = model(X_list, P_list)
            rec = recall_at_k(emb_v, va_edges, user_pos, all_items, K_rec, device)
            best_rec = max(best_rec, rec)

    del model
    torch.cuda.empty_cache()
    return best_rec


# ─────────────────────────────────────────────────────────────────────────────
#  Search functions — one per task
# ─────────────────────────────────────────────────────────────────────────────

def hparam_search_nc(
    data    : dict,
    seed    : int  = 42,
    out_dir : str  = 'results/nc',
) -> tuple:
    """
    Node classification hyperparameter search.

    Args:
        data    : preprocessed dataset dict (keys: X_dict, A_list_sp,
                  bipartite_flags, labels, target_size, n_classes, N, name)
        seed    : random seed for reproducibility
        out_dir : directory for CV scores CSV and best_params.json

    Returns:
        best_params : dict
        tr80_idx    : np.ndarray  — 80% indices for final training
        te20_idx    : np.ndarray  — 20% held-out test indices
    """
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Nt      = data['target_size']
    lbl_np  = data['labels'].numpy()

    tr80, te20 = train_test_split(
        np.arange(Nt), test_size=TEST_FRAC, random_state=seed, stratify=lbl_np,
    )
    P_list = build_operators(data['A_list_sp'], data['bipartite_flags'], device)
    X_list = [x.to(device) for x in data['X_dict'].values()]

    combos = _random_combos(PARAM_GRID_BASE, n=N_RANDOM, seed=seed)
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    cv_rows, best_params, best_mean = [], None, -1.0

    for ci, params in enumerate(combos):
        scores = []
        for fold, (tr_fold, va_fold) in enumerate(skf.split(tr80, lbl_np[tr80])):
            vm = _run_fold_nc(
                data, P_list, X_list, params,
                tr80[tr_fold], tr80[va_fold], device,
            )
            scores.append(vm)
            cv_rows.append({
                'combo': ci, 'fold': fold, 'val_macro_f1': round(vm, 4),
                **{f'hp_{k}': v for k, v in params.items()},
            })
            print(f"  [NC] combo {ci+1}/{len(combos)}  fold {fold+1}  val_macro={vm:.4f}")

        mean_vm = float(np.mean(scores))
        print(f"  [NC] combo {ci+1}/{len(combos)}  mean_macro_f1={mean_vm:.4f}  {params}")
        if mean_vm > best_mean:
            best_mean, best_params = mean_vm, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name', ''), 'nc', out_dir)
    print(f"[NC] best_val_macro_f1={best_mean:.4f}  params={best_params}")
    return best_params, tr80, te20


def hparam_search_cl(
    data    : dict,
    seed    : int  = 42,
    out_dir : str  = 'results/clustering',
) -> tuple:
    """
    Node clustering hyperparameter search.

    Args / Returns: same structure as hparam_search_nc.
    Val metric: NMI (higher is better).
    """
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Nt      = data['target_size']
    lbl_np  = data['labels'].numpy()

    tr80, te20 = train_test_split(
        np.arange(Nt), test_size=TEST_FRAC, random_state=seed, stratify=lbl_np,
    )
    P_list = build_operators(data['A_list_sp'], data['bipartite_flags'], device)
    X_list = [x.to(device) for x in data['X_dict'].values()]

    combos = _random_combos(PARAM_GRID_CL, n=N_RANDOM, seed=seed)  # grid too large
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    cv_rows, best_params, best_mean = [], None, -1.0

    for ci, params in enumerate(combos):
        scores = []
        for fold, (tr_fold, va_fold) in enumerate(skf.split(tr80, lbl_np[tr80])):
            nmi = _run_fold_cl(
                data, P_list, X_list, params,
                tr80[tr_fold], tr80[va_fold], device,
            )
            scores.append(nmi)
            cv_rows.append({
                'combo': ci, 'fold': fold, 'val_nmi': round(nmi, 4),
                **{f'hp_{k}': v for k, v in params.items()},
            })

        mean_nmi = float(np.mean(scores))
        print(f"[CL] combo {ci+1:3d}/{len(combos)}  mean_nmi={mean_nmi:.4f}  {params}")
        if mean_nmi > best_mean:
            best_mean, best_params = mean_nmi, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name', ''), 'cl', out_dir)
    print(f"[CL] best_val_nmi={best_mean:.4f}  params={best_params}")
    return best_params, tr80, te20


def hparam_search_rec(
    data         : dict,
    target_edges : np.ndarray,
    seed         : int  = 42,
    out_dir      : str  = 'results/recommendation',
) -> tuple:
    """
    Recommendation hyperparameter search.

    Args:
        data         : preprocessed dataset dict
        target_edges : (E, 2) array of (user, item) interaction edges
        seed         : random seed
        out_dir      : output directory

    Returns:
        best_params   : dict
        tr80_edges    : np.ndarray  — 80% edges for final training
        te20_edges    : np.ndarray  — 20% held-out test edges
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_idx              = np.arange(len(target_edges))
    tr80_idx, te20_idx   = train_test_split(all_idx, test_size=TEST_FRAC, random_state=seed)
    tr80_edges           = target_edges[tr80_idx]
    te20_edges           = target_edges[te20_idx]

    combos = _random_combos(PARAM_GRID_REC, n=N_RANDOM, seed=seed)
    kf     = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    cv_rows, best_params, best_mean = [], None, -1.0

    for ci, params in enumerate(combos):
        scores = []
        for fold, (tr_fold, va_fold) in enumerate(kf.split(tr80_edges)):
            rec = _run_fold_rec(
                data, tr80_edges[tr_fold], tr80_edges[va_fold], params, device,
            )
            scores.append(rec)
            cv_rows.append({
                'combo': ci, 'fold': fold,
                'val_recall_at_k': round(rec, 4),
                **{f'hp_{k}': v for k, v in params.items()},
            })

        mean_rec = float(np.mean(scores))
        print(f"[REC] combo {ci+1:3d}/{len(combos)}  mean_recall@K={mean_rec:.4f}  {params}")
        if mean_rec > best_mean:
            best_mean, best_params = mean_rec, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name', ''), 'rec', out_dir)
    print(f"[REC] best_val_recall@K={best_mean:.4f}  params={best_params}")
    return best_params, tr80_edges, te20_edges


def hparam_search_lp(
    data         : dict,
    target_edges : np.ndarray,
    seed         : int  = 42,
    out_dir      : str  = 'results/lp',
) -> tuple:
    """
    Link prediction hyperparameter search.

    Args:
        data         : preprocessed dataset dict
        target_edges : (E, 2) array of positive edges
        seed         : random seed
        out_dir      : output directory

    Returns:
        best_params   : dict
        tr80_edges    : np.ndarray
        te20_edges    : np.ndarray
    """
    from .link_prediction import _run_fold_lp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_idx            = np.arange(len(target_edges))
    tr80_idx, te20_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=seed)
    tr80_edges         = target_edges[tr80_idx]
    te20_edges         = target_edges[te20_idx]

    combos = _random_combos(PARAM_GRID_BASE, n=N_RANDOM, seed=seed)
    kf     = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    cv_rows, best_params, best_mean = [], None, -1.0

    for ci, params in enumerate(combos):
        scores = []
        for fold, (tr_fold, va_fold) in enumerate(kf.split(tr80_edges)):
            auc = _run_fold_lp(
                data, tr80_edges[tr_fold], tr80_edges[va_fold],
                te20_edges, params, device,
            )
            scores.append(auc)
            cv_rows.append({
                'combo': ci, 'fold': fold, 'val_auc': round(auc, 4),
                **{f'hp_{k}': v for k, v in params.items()},
            })
            print(f"  [LP] combo {ci+1}/{len(combos)}  fold {fold+1}  val_auc={auc:.4f}")

        mean_auc = float(np.mean(scores))
        print(f"  [LP] combo {ci+1}/{len(combos)}  mean_auc={mean_auc:.4f}  {params}")
        if mean_auc > best_mean:
            best_mean, best_params = mean_auc, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name', ''), 'lp', out_dir)
    print(f"[LP] best_val_auc={best_mean:.4f}  params={best_params}")
    return best_params, tr80_edges, te20_edges