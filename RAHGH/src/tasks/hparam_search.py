import itertools, random, time, json, os, sys
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from ..model.rahgh import (
    build_rahgh_classifier, build_edge_index_dict, build_node_type_indices,
    compile_model,
)


PARAM_GRID_BASE = {
    'd'         : [64],
    'K'         : [2, 3, 4, 5, 6],
    'dropout'   : [0.3, 0.5],
    'lr'        : [0.001, 0.005],
    'wd'        : [1e-4, 1e-3],
    'gcn_hidden': [64, 128],
    'epochs'    : [500, 700, 1000],
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

N_ITER    = 100
N_FOLDS   = 5
TEST_FRAC = 0.20
PATIENCE  = 50


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


def _build_model(data, params, out_dim, device, head='gcn'):
    model = build_rahgh_classifier(
        data, hidden_dim=params['d'], num_classes=out_dim,
        K=params['K'], head=head,
        dropout_homo=params['dropout'], dropout_gnn=params['dropout'],
    ).to(device)
    return compile_model(model)


def _run_fold_nc(data, params, tr_idx, va_idx, device, head='gcn',
                 x_dict=None, edge_index_dict=None, labels=None,
                 node_type_indices=None):
    Nt = data['target_size']
    tr_t = torch.tensor(tr_idx, dtype=torch.long, device=device)
    va_t = torch.tensor(va_idx, dtype=torch.long, device=device)

    model = _build_model(data, params, out_dim=data['n_classes'], device=device, head=head)
    opt = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device='cuda') if use_amp else None

    best_vm, best_sd, stall = 0.0, None, 0

    for ep in range(1, params['epochs'] + 1):
        model.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, *_ = model(x_dict, edge_index_dict, node_type_indices)
            loss = F.cross_entropy(logits[:Nt][tr_t], labels[tr_t])
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits, *_ = model(x_dict, edge_index_dict, node_type_indices)
        preds = logits[:Nt][va_t].argmax(1).cpu().numpy()
        truth = data['labels'][va_idx].numpy()
        vm = f1_score(truth, preds, average='macro', zero_division=0)

        if vm > best_vm:
            best_vm = vm
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= PATIENCE:
                break

    del model
    return best_vm


def _run_fold_cl(data, params, tr_idx, va_idx, device, head='gcn',
                 x_dict=None, edge_index_dict=None, node_type_indices=None):
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    Nt = data['target_size']
    n_cl = data['n_classes']
    d = params['d']

    model = _build_model(data, params, out_dim=d, device=device, head=head)
    decoder = torch.nn.Linear(d, d).to(device)
    opt = Adam(
        list(model.parameters()) + list(decoder.parameters()),
        lr=params['lr'], weight_decay=params['wd'],
    )

    tr_t = torch.tensor(tr_idx, dtype=torch.long, device=device)
    X_cat = torch.cat(list(x_dict.values()), dim=0)[:Nt][tr_t]
    if X_cat.shape[1] != d:
        X_cat = X_cat[:, :d] if X_cat.shape[1] > d \
                else F.pad(X_cat, (0, d - X_cat.shape[1]))

    best_nmi = 0.0

    for ep in range(1, params['epochs'] + 1):
        model.train(); decoder.train()
        opt.zero_grad()
        emb, *_ = model(x_dict, edge_index_dict, node_type_indices)
        loss = F.mse_loss(decoder(emb[:Nt][tr_t]), X_cat)
        loss.backward()
        opt.step()

        if ep % 50 == 0 or ep == params['epochs']:
            model.eval()
            with torch.no_grad():
                emb_v, *_ = model(x_dict, edge_index_dict, node_type_indices)
            emb_np = emb_v[:Nt][va_idx].cpu().numpy()
            pred = KMeans(n_clusters=n_cl, n_init=10, random_state=0).fit_predict(emb_np)
            nmi = normalized_mutual_info_score(data['labels'][va_idx].numpy(), pred)
            best_nmi = max(best_nmi, nmi)

    del model, decoder
    return best_nmi


def _run_fold_rec(data, tr_edges, va_edges, params, device, head='gcn', K_rec=20,
                  x_dict=None, edge_index_dict=None, node_type_indices=None):
    from .recommendation import bpr_loss, recall_at_k

    d = params['d']

    model = _build_model(data, params, out_dim=d, device=device, head=head)
    opt = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    all_items = np.unique(tr_edges[:, 1])
    user_pos = {}
    for u, i in tr_edges: user_pos.setdefault(u, set()).add(i)

    best_rec, best_sd = 0.0, None
    rng = np.random.default_rng(0)

    for ep in range(1, params['epochs'] + 1):
        model.train()
        opt.zero_grad()
        emb, *_ = model(x_dict, edge_index_dict, node_type_indices)
        users = tr_edges[:, 0]; pos_items = tr_edges[:, 1]
        neg_items = rng.choice(all_items, size=len(users))
        loss = bpr_loss(emb, users, pos_items, neg_items, device,
                        reg=params.get('bpr_reg', 1e-4))
        loss.backward()
        opt.step()

        if ep % 50 == 0 or ep == params['epochs']:
            model.eval()
            with torch.no_grad():
                emb_v, *_ = model(x_dict, edge_index_dict, node_type_indices)
            rec = recall_at_k(emb_v, va_edges, user_pos, all_items, K_rec, device)
            best_rec = max(best_rec, rec)

    del model
    return best_rec


def _run_fold_lp(data, tr_edges, va_edges, te_edges, params, device, head='gcn', neg_ratio=5,
                 x_dict=None, edge_index_dict=None, node_type_indices=None):
    from .link_prediction import sample_negatives, MLPDecoder

    d = params['d']

    model = _build_model(data, params, out_dim=d, device=device, head=head)
    decoder = MLPDecoder(d).to(device)
    opt = Adam(list(model.parameters()) + list(decoder.parameters()),
               lr=params['lr'], weight_decay=params['wd'])
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device='cuda') if use_amp else None

    all_src = np.unique(tr_edges[:, 0])
    all_dst = np.unique(tr_edges[:, 1])
    tr_neg = sample_negatives(tr_edges, len(tr_edges) * neg_ratio, all_src, all_dst, 0)
    va_neg = sample_negatives(va_edges, len(va_edges) * neg_ratio, all_src, all_dst, 1)

    def tensors(pos, neg):
        e = np.concatenate([pos, neg], 0)
        l = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
        return (torch.tensor(e[:, 0], dtype=torch.long, device=device),
                torch.tensor(e[:, 1], dtype=torch.long, device=device),
                torch.tensor(l, device=device))

    tr_s, tr_d, tr_l = tensors(tr_edges, tr_neg)
    va_s, va_d, va_l = tensors(va_edges, va_neg)

    best_auc, stall = 0.0, 0
    max_epochs = params['epochs']

    for ep in range(1, max_epochs + 1):
        model.train(); decoder.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            emb, *_ = model(x_dict, edge_index_dict, node_type_indices)
            loss = F.binary_cross_entropy_with_logits(decoder(emb, tr_s, tr_d), tr_l)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        model.eval(); decoder.eval()
        with torch.no_grad():
            emb_v, *_ = model(x_dict, edge_index_dict, node_type_indices)
            p = torch.sigmoid(decoder(emb_v, va_s, va_d)).cpu().numpy()
            auc = roc_auc_score(va_l.cpu().numpy(), p)

        if auc > best_auc:
            best_auc = auc
            stall = 0
        else:
            stall += 1
            if stall >= PATIENCE:
                break

    del model, decoder
    return best_auc


def hparam_search_nc(data, seed=42, out_dir='results/nc', head='gcn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Nt = data['target_size']
    lbl_np = data['labels'].numpy()

    tr80, te20 = train_test_split(np.arange(Nt), test_size=TEST_FRAC,
                                   random_state=seed, stratify=lbl_np)

    # Prepare on-device data once before any fold
    x_dict_once = {k: v.to(device) for k, v in data['X_dict'].items()}
    edge_index_dict_once = build_edge_index_dict(data, device)
    node_type_indices_once = {k: v.to(device) for k, v in build_node_type_indices(data).items()}
    labels_once = data['labels'].to(device)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_BASE, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows = []
    best_params, best_mean = None, 0.0

    t0_hp = time.time()
    n_total = len(combos) * N_FOLDS
    print(f"\n  Hyperparameter search: {len(combos)} combos × {N_FOLDS} folds = {n_total} runs", flush=True)
    for ci, params in enumerate(combos):
        t_combo = time.time()
        print(f"\n  combination {ci+1}({params})", flush=True)
        fold_scores = []
        fold_iter = tqdm(skf.split(tr80, lbl_np[tr80]), desc=f"    fold", total=N_FOLDS, leave=False)
        for fold, (tr_fold, va_fold) in enumerate(fold_iter):
            vm = _run_fold_nc(data, params, tr80[tr_fold], tr80[va_fold], device, head=head,
                              x_dict=x_dict_once, edge_index_dict=edge_index_dict_once,
                              labels=labels_once, node_type_indices=node_type_indices_once)
            fold_scores.append(vm)
            fold_iter.set_postfix(macro_f1=f"{vm:.4f}")
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_macro': round(vm, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
        mean_vm = float(np.mean(fold_scores))
        elapsed = time.time() - t_combo
        print(f"    fold_scores={[round(s, 4) for s in fold_scores]}")
        print(f"    mean_macro_f1={mean_vm:.4f}  [{elapsed:.0f}s]", flush=True)
        if mean_vm > best_mean: best_mean, best_params = mean_vm, params

    total_hp = time.time() - t0_hp
    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name', ''), 'nc', out_dir)
    print(f"[NC hparam] best_val_macro={best_mean:.4f}  params={best_params}  total={total_hp:.0f}s", flush=True)
    return best_params, tr80, te20


def hparam_search_cl(data, seed=42, out_dir='results/clustering', head='gcn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Nt = data['target_size']
    lbl_np = data['labels'].numpy()

    tr80, te20 = train_test_split(np.arange(Nt), test_size=TEST_FRAC,
                                   random_state=seed, stratify=lbl_np)

    # Prepare on-device data once before any fold
    x_dict_once = {k: v.to(device) for k, v in data['X_dict'].items()}
    edge_index_dict_once = build_edge_index_dict(data, device)
    node_type_indices_once = {k: v.to(device) for k, v in build_node_type_indices(data).items()}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_CLUSTERING, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows = []
    best_params, best_mean = None, 0.0

    for ci, params in enumerate(combos):
        fold_nmis = []
        for fold, (tr_fold, va_fold) in enumerate(skf.split(tr80, lbl_np[tr80])):
            nmi = _run_fold_cl(data, params, tr80[tr_fold], tr80[va_fold], device, head=head,
                               x_dict=x_dict_once, edge_index_dict=edge_index_dict_once,
                               node_type_indices=node_type_indices_once)
            fold_nmis.append(nmi)
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_nmi': round(nmi, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
        mean_nmi = float(np.mean(fold_nmis))
        if mean_nmi > best_mean: best_mean, best_params = mean_nmi, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name', ''), 'cl', out_dir)
    print(f"[CL hparam] best_val_nmi={best_mean:.4f}  params={best_params}")
    return best_params, tr80, te20


def hparam_search_rec(data, target_edges, seed=42, out_dir='results/recommendation', head='gcn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_idx = np.arange(len(target_edges))
    tr80_idx, te20_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=seed)
    tr80_edges = target_edges[tr80_idx]
    te20_edges = target_edges[te20_idx]

    # Prepare on-device data once before any fold
    x_dict_once = {k: v.to(device) for k, v in data['X_dict'].items()}
    edge_index_dict_once = build_edge_index_dict(data, device)
    node_type_indices_once = {k: v.to(device) for k, v in build_node_type_indices(data).items()}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_REC, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows = []
    best_params, best_mean = None, 0.0

    for ci, params in enumerate(combos):
        fold_recs = []
        for fold, (tr_fold, va_fold) in enumerate(kf.split(tr80_edges)):
            rec = _run_fold_rec(data, tr80_edges[tr_fold], tr80_edges[va_fold],
                                params, device, head=head, K_rec=params.get('K_rec', 20),
                                x_dict=x_dict_once, edge_index_dict=edge_index_dict_once,
                                node_type_indices=node_type_indices_once)
            fold_recs.append(rec)
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_recall': round(rec, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
        mean_rec = float(np.mean(fold_recs))
        if mean_rec > best_mean: best_mean, best_params = mean_rec, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name', ''), 'rec', out_dir)
    print(f"[REC hparam] best_val_recall@K={best_mean:.4f}  params={best_params}")
    return best_params, tr80_edges, te20_edges


def hparam_search_lp(data, target_edges, seed=42, out_dir='results/lp', head='gcn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_idx = np.arange(len(target_edges))
    tr80_idx, te20_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=seed)
    tr80_edges = target_edges[tr80_idx]
    te20_edges = target_edges[te20_idx]

    # Prepare on-device data once before any fold
    x_dict_once = {k: v.to(device) for k, v in data['X_dict'].items()}
    edge_index_dict_once = build_edge_index_dict(data, device)
    node_type_indices_once = {k: v.to(device) for k, v in build_node_type_indices(data).items()}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_BASE, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows = []
    best_params, best_mean = None, 0.0

    for ci, params in enumerate(combos):
        fold_aucs = []
        for fold, (tr_fold, va_fold) in enumerate(kf.split(tr80_edges)):
            auc = _run_fold_lp(data, tr80_edges[tr_fold], tr80_edges[va_fold],
                               te20_edges, params, device, head=head,
                               x_dict=x_dict_once, edge_index_dict=edge_index_dict_once,
                               node_type_indices=node_type_indices_once)
            fold_aucs.append(auc)
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_auc': round(auc, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
        mean_auc = float(np.mean(fold_aucs))
        if mean_auc > best_mean: best_mean, best_params = mean_auc, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name', ''), 'lp', out_dir)
    print(f"[LP hparam] best_val_auc={best_mean:.4f}  params={best_params}")
    return best_params, tr80_edges, te20_edges
