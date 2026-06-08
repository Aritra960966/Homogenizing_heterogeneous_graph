import itertools, random, time, json, os
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from ..model.rahgh    import RAHGH
from ..model.diffusion import build_operators
import torch.nn.functional as F
from torch.optim import Adam


PARAM_GRID_BASE = {
    'd'         : [64, 128],
    'K'         : [2, 3, 4, 5, 6],
    'dropout'   : [0.3, 0.5],
    'lr'        : [0.001, 0.005],
    'wd'        : [1e-4, 1e-3],
    'gcn_hidden': [64, 128],
    'epochs'    : [300, 500],
}

PARAM_GRID_CLUSTERING = {
    **PARAM_GRID_BASE,
    'cl_loss' : ['reconstruction', 'contrastive'],
    'cl_temp' : [0.1, 0.5],
}

PARAM_GRID_REC = {
    **PARAM_GRID_BASE,
    'K_rec'   : [10, 20, 50],
    'neg_ratio': [5, 10],
    'bpr_reg' : [1e-4, 1e-3],
}

N_ITER    = 50
N_FOLDS   = 5
TEST_FRAC = 0.20


def _random_combos(grid, seed=0, n=N_ITER):
    keys = list(grid.keys())
    all_c = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    random.seed(seed); random.shuffle(all_c)
    return all_c[:n]


def _save_best_params(best_params, dataset, task, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'best_params.json')
    try:
        with open(path) as f: existing = json.load(f)
    except FileNotFoundError:
        existing = {}
    existing[f"{dataset}_{task}"] = best_params
    with open(path, 'w') as f: json.dump(existing, f, indent=2)
    print(f"[hparam] Best params saved -> {path}")


def _write_csv(rows, path):
    import csv
    if not rows: return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  Saved -> {path}")


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
    for ep in range(1, params['epochs'] + 1):
        model.train(); opt.zero_grad()
        logits, *_ = model(X_list, P_list)
        F.cross_entropy(logits[:Nt][tr_t], labels[tr_t]).backward()
        opt.step()

        if ep % 50 == 0 or ep == params['epochs']:
            model.eval()
            with torch.no_grad():
                logits, *_ = model(X_list, P_list)
                p  = logits[:Nt][va_t].argmax(1).cpu().numpy()
                y  = data['labels'][va_idx].numpy()
                vm = f1_score(y, p, average='macro', zero_division=0)
            if vm > best_vm:
                best_vm = vm
                best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    del model; torch.cuda.empty_cache()
    return best_vm


def _run_fold_cl(data, P_list, X_list, params, tr_idx, va_idx, device):
    from sklearn.cluster import KMeans
    from sklearn.metrics  import normalized_mutual_info_score

    d       = params['d']
    R       = len(data['A_list_sp'])
    Nt      = data['target_size']
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    n_cl    = data['n_classes']

    model = RAHGH(
        in_dims=in_dims, d=d, R=R, K=params['K'],
        gcn_hidden=params['gcn_hidden'],
        out_dim=d,
        dropout=params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)

    decoder = torch.nn.Linear(d, d).to(device)
    opt = Adam(list(model.parameters()) + list(decoder.parameters()),
               lr=params['lr'], weight_decay=params['wd'])

    tr_t = torch.tensor(tr_idx, dtype=torch.long, device=device)

    best_nmi, best_sd = 0.0, None
    for ep in range(1, params['epochs'] + 1):
        model.train(); decoder.train(); opt.zero_grad()
        emb, *_ = model(X_list, P_list)
        recon = decoder(emb[:Nt][tr_t])
        target = torch.cat([x for x in X_list], dim=0)[:Nt][tr_t]
        if target.shape[1] != d:
            target = target[:, :d] if target.shape[1] > d \
                     else F.pad(target, (0, d - target.shape[1]))
        loss = F.mse_loss(recon, target.to(device))
        loss.backward(); opt.step()

        if ep % 50 == 0 or ep == params['epochs']:
            model.eval()
            with torch.no_grad():
                emb_v, *_ = model(X_list, P_list)
                emb_np = emb_v[:Nt][va_idx].cpu().numpy()
                km     = KMeans(n_clusters=n_cl, n_init=10, random_state=0)
                pred   = km.fit_predict(emb_np)
                y      = data['labels'][va_idx].numpy()
                nmi    = normalized_mutual_info_score(y, pred)
            if nmi > best_nmi:
                best_nmi = nmi
                best_sd  = {k: v.clone() for k, v in model.state_dict().items()}

    del model, decoder; torch.cuda.empty_cache()
    return best_nmi


def _run_fold_rec(data, tr_edges, va_edges, params, device, K_rec=20):
    from .recommendation import bpr_loss, recall_at_k

    d       = params['d']
    R       = len(data['A_list_sp'])
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    X_list  = [x.to(device) for x in data['X_dict'].values()]
    N       = data['N']

    P_list  = build_operators(data['A_list_sp'], data['bipartite_flags'], device)

    model = RAHGH(
        in_dims=in_dims, d=d, R=R, K=params['K'],
        gcn_hidden=params['gcn_hidden'],
        out_dim=d, dropout=params['dropout'],
        A_list_sp=data['A_list_sp'], N=N, device=device,
    ).to(device)
    opt = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    all_items = np.unique(tr_edges[:, 1])
    user_pos  = {}
    for u, i in tr_edges: user_pos.setdefault(u, set()).add(i)

    best_rec, best_sd = 0.0, None
    rng = np.random.default_rng(0)

    for ep in range(1, params['epochs'] + 1):
        model.train(); opt.zero_grad()
        emb, *_ = model(X_list, P_list)
        users = tr_edges[:, 0]; pos_items = tr_edges[:, 1]
        neg_items = rng.choice(all_items, size=len(users))
        loss = bpr_loss(emb, users, pos_items, neg_items,
                        device, reg=params.get('bpr_reg', 1e-4))
        loss.backward(); opt.step()

        if ep % 50 == 0 or ep == params['epochs']:
            model.eval()
            with torch.no_grad():
                emb_v, *_ = model(X_list, P_list)
                rec = recall_at_k(emb_v, va_edges, user_pos, all_items, K_rec, device)
            if rec > best_rec:
                best_rec = rec
                best_sd  = {k: v.clone() for k, v in model.state_dict().items()}

    del model; torch.cuda.empty_cache()
    return best_rec


def hparam_search_nc(data, seed=42, out_dir='results/nc'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Nt     = data['target_size']
    lbl_np = data['labels'].numpy()

    tr80, te20 = train_test_split(np.arange(Nt), test_size=TEST_FRAC,
                                   random_state=seed, stratify=lbl_np)
    P_list = build_operators(data['A_list_sp'], data['bipartite_flags'], device)
    X_list = [x.to(device) for x in data['X_dict'].values()]

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_BASE, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows  = []
    best_params, best_mean = None, 0.0

    for ci, params in enumerate(combos):
        fold_scores = []
        for fold, (tr_fold, va_fold) in enumerate(skf.split(tr80, lbl_np[tr80])):
            vm = _run_fold_nc(data, P_list, X_list, params,
                              tr80[tr_fold], tr80[va_fold], device)
            fold_scores.append(vm)
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_macro': round(vm, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
        mean_vm = float(np.mean(fold_scores))
        if mean_vm > best_mean: best_mean, best_params = mean_vm, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name',''), 'nc', out_dir)
    print(f"[NC hparam] best_val_macro={best_mean:.4f}  params={best_params}")
    return best_params, tr80, te20


def hparam_search_cl(data, seed=42, out_dir='results/clustering'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Nt     = data['target_size']
    lbl_np = data['labels'].numpy()

    tr80, te20 = train_test_split(np.arange(Nt), test_size=TEST_FRAC,
                                   random_state=seed, stratify=lbl_np)
    P_list = build_operators(data['A_list_sp'], data['bipartite_flags'], device)
    X_list = [x.to(device) for x in data['X_dict'].values()]

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_CLUSTERING, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows  = []
    best_params, best_mean = None, 0.0

    for ci, params in enumerate(combos):
        fold_nmis = []
        for fold, (tr_fold, va_fold) in enumerate(skf.split(tr80, lbl_np[tr80])):
            nmi = _run_fold_cl(data, P_list, X_list, params,
                               tr80[tr_fold], tr80[va_fold], device)
            fold_nmis.append(nmi)
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_nmi': round(nmi, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
        mean_nmi = float(np.mean(fold_nmis))
        if mean_nmi > best_mean: best_mean, best_params = mean_nmi, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name',''), 'cl', out_dir)
    print(f"[CL hparam] best_val_nmi={best_mean:.4f}  params={best_params}")
    return best_params, tr80, te20


def hparam_search_rec(data, target_edges, seed=42, out_dir='results/recommendation'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_idx    = np.arange(len(target_edges))
    tr80_idx, te20_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=seed)
    tr80_edges = target_edges[tr80_idx]
    te20_edges = target_edges[te20_idx]

    kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos  = _random_combos(PARAM_GRID_REC, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows  = []
    best_params, best_mean = None, 0.0

    for ci, params in enumerate(combos):
        fold_recs = []
        for fold, (tr_fold, va_fold) in enumerate(kf.split(tr80_edges)):
            rec = _run_fold_rec(data, tr80_edges[tr_fold], tr80_edges[va_fold],
                                params, device, K_rec=params.get('K_rec', 20))
            fold_recs.append(rec)
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_recall': round(rec, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
        mean_rec = float(np.mean(fold_recs))
        if mean_rec > best_mean: best_mean, best_params = mean_rec, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name',''), 'rec', out_dir)
    print(f"[REC hparam] best_val_recall@K={best_mean:.4f}  params={best_params}")
    return best_params, tr80_edges, te20_edges


def hparam_search_lp(data, target_edges, seed=42, out_dir='results/lp'):
    from .link_prediction import _run_fold_lp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_idx = np.arange(len(target_edges))
    tr80_idx, te20_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=seed)
    tr80_edges = target_edges[tr80_idx]
    te20_edges = target_edges[te20_idx]

    kf     = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_BASE, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows  = []
    best_params, best_mean = None, 0.0

    for ci, params in enumerate(combos):
        fold_aucs = []
        for fold, (tr_fold, va_fold) in enumerate(kf.split(tr80_edges)):
            auc = _run_fold_lp(data, tr80_edges[tr_fold], tr80_edges[va_fold],
                               te20_edges, params, device)
            fold_aucs.append(auc)
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_auc': round(auc, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
        mean_auc = float(np.mean(fold_aucs))
        if mean_auc > best_mean: best_mean, best_params = mean_auc, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name',''), 'lp', out_dir)
    print(f"[LP hparam] best_val_auc={best_mean:.4f}  params={best_params}")
    return best_params, tr80_edges, te20_edges
