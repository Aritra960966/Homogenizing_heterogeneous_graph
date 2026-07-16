import itertools, random, time, json, os, sys, copy, warnings
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from tqdm import tqdm

from ..model.rahgh import (
    build_rahgh_classifier, build_encoder,
    build_edge_index_dict, build_node_type_indices,
    compile_model,
)
from ..data.lastfm_loader import rebuild_user_features
from .link_prediction import _build_masked_edge_index
from .node_clustering import run_fold_clustering


PARAM_GRID_BASE = {
    'd'              : [32,64,128],
    'd_prime'        : [32,64,128],
    'K'              : [1,2, 3, 4, 5, 6],
    'dropout'        : [0.5,0.1],
    'dropout_gnn'    : [0.5,0.1],
    'lr'             : [0.005,0.001],
    'wd'             : [1e-4],
    'epochs'         : [200, 300, 400, 500, 600, 700,1000],
    'hidden'         : [32,64,128],
    'label_smoothing': [0.0, 0.1],
    'warmup'         : [0, 10],
}

PARAM_GRID_CLUSTERING = {
    'd'          : [64],
    'd_prime'    : [32],
    'K'          : [2, 3, 4, 5, 6],
    'dropout'    : [0.5],
    'lr'         : [0.005],
    'wd'         : [1e-4],
    'epochs'     : [200, 300, 400, 500, 600, 700],
    # Contrastive losses (SOTA upgrade)
    'cl_temp'    : [0.3, 0.4, 0.5],
    'lam_recon'  : [1.0],
    'lam_cr'     : [0.3, 0.5, 1.0],
    'lam_align'  : [0.3, 0.5],
    'mask_rate'  : [0.1, 0.2],
    'batch_size' : [512],
}

PARAM_GRID_REC = {
    **PARAM_GRID_BASE,
    'K_rec'   : [10, 20, 50],
    'neg_ratio': [5, 10],
    'bpr_reg' : [1e-4, 1e-3],
}

# Fast mode: override these via env for quick experiments
if os.environ.get('RAHGH_FAST'):
    for key in list(PARAM_GRID_BASE.keys()):
        if key == 'd':              PARAM_GRID_BASE[key] = [64]
        elif key == 'd_prime':      PARAM_GRID_BASE[key] = [32]
        elif key == 'K':            PARAM_GRID_BASE[key] = [2, 3]
        elif key == 'lr':           PARAM_GRID_BASE[key] = [0.005]
        elif key == 'wd':           PARAM_GRID_BASE[key] = [1e-4]
        elif key == 'epochs':       PARAM_GRID_BASE[key] = [100]
        elif key == 'dropout':      PARAM_GRID_BASE[key] = [0.5]
        elif key == 'dropout_gnn':  PARAM_GRID_BASE[key] = [0.5]
        elif key == 'hidden':       PARAM_GRID_BASE[key] = [32]
        elif key == 'label_smoothing': PARAM_GRID_BASE[key] = [0.0]
        elif key == 'warmup':       PARAM_GRID_BASE[key] = [0]
    for key in list(PARAM_GRID_REC.keys()):
        if key == 'K_rec':   PARAM_GRID_REC[key] = [20]
        elif key == 'neg_ratio': PARAM_GRID_REC[key] = [5]
        elif key == 'bpr_reg': PARAM_GRID_REC[key] = [1e-4]
    N_ITER = 5

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
        dropout_homo=params['dropout'],
        dropout_gnn=params.get('dropout_gnn', params['dropout']),
        gnn_hidden_dim=params.get('d_prime', params.get('hidden', params['d'])),
    ).to(device)
    return compile_model(model)


def _run_fold_nc(data, params, tr_idx, va_idx, device, head='gcn',
                 x_dict=None, edge_index_dict=None, labels=None,
                 node_type_indices=None):
    Nt = data['target_size']
    tr_t = torch.tensor(tr_idx, dtype=torch.long, device=device)
    va_t = torch.tensor(va_idx, dtype=torch.long, device=device)

    model = _build_model(data, params, out_dim=data['n_classes'], device=device, head=head)
    opt = AdamW(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    warmup_epochs = params.get('warmup', 0)
    if warmup_epochs > 0:
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=warmup_epochs
        )
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=params['epochs'], eta_min=params['lr'] * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sched, main_sched], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=params['epochs'], eta_min=params['lr'] * 0.01,
        )
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device='cuda') if use_amp else None

    best_vm, best_sd, stall = 0.0, None, 0

    for ep in range(1, params['epochs'] + 1):
        model.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, *_ = model(x_dict, edge_index_dict, node_type_indices)
            loss = F.cross_entropy(logits[:Nt][tr_t], labels[tr_t],
                                    label_smoothing=params.get('label_smoothing', 0.1))
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        scheduler.step()

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
    """
    Delegate to graph_clustering.run_fold_clustering.
    Extra args (head, x_dict, etc.) are accepted for backward compatibility.
    """
    return run_fold_clustering(data, params, tr_idx, va_idx, device)


def _run_fold_rec(data, tr_edges, va_edges, params, device, head='gcn', K_rec=20,
                  x_dict=None, edge_index_dict=None, node_type_indices=None):
    from .recommendation import bpr_loss, recall_at_k, compute_rec_metrics, sample_bpr_negatives

    # Build inputs using only training edges to prevent leakage
    fold_x_dict = rebuild_user_features(data, tr_edges, device)
    fold_edge_index_dict = _build_masked_edge_index(data, tr_edges, device)
    fold_node_type_indices = {k: v.to(device)
                              for k, v in build_node_type_indices(data).items()}

    model = build_encoder(data, params, device)
    opt = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    # Pre-compute propagation operators once — cache in model for all epochs
    homogenizer = getattr(model, 'homogenizer', model)
    homogenizer.clear_propagation_cache()
    homogenizer._build_operators(fold_edge_index_dict)  # populates _prop_cache

    all_items = np.unique(tr_edges[:, 1])
    user_pos = {}
    for u, i in tr_edges: user_pos.setdefault(u, set()).add(i)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device='cuda') if use_amp else None
    best_score, best_metrics = 0.0, {}
    best_sd, stall = None, 0
    rng = np.random.default_rng(0)
    max_epochs = params['epochs']
    K_ALL = [10, 20, 50]
    pbar = tqdm(range(1, max_epochs + 1), desc="    REC fold", leave=False)

    for ep in pbar:
        model.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            emb, *_ = model(fold_x_dict, fold_edge_index_dict, fold_node_type_indices)
            users = tr_edges[:, 0]; pos_items = tr_edges[:, 1]
            neg_ratio = params.get('neg_ratio', 5)
            if neg_ratio > 1:
                users_rep = np.repeat(users, neg_ratio)
                pos_rep = np.repeat(pos_items, neg_ratio)
                neg_batches = [sample_bpr_negatives(users, all_items, user_pos, rng)
                              for _ in range(neg_ratio)]
                neg_items = np.concatenate(neg_batches)
                loss = bpr_loss(emb, users_rep, pos_rep, neg_items, device,
                                reg=params.get('bpr_reg', 1e-4))
            else:
                neg_items = sample_bpr_negatives(users, all_items, user_pos, rng)
                loss = bpr_loss(emb, users, pos_items, neg_items, device,
                                reg=params.get('bpr_reg', 1e-4))
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        if ep % 100 == 0 or ep == max_epochs:
            model.eval()
            with torch.no_grad():
                emb_v, *_ = model(fold_x_dict, fold_edge_index_dict, fold_node_type_indices)
            agg = compute_rec_metrics(emb_v, va_edges, user_pos, all_items, K_ALL, device)
            r20 = agg.get('recall@20', 0.0)
            n20 = agg.get('ndcg@20', 0.0)
            score = (r20 + n20) / 2
            pbar.set_postfix(loss=f"{loss.item():.4f}", r10=f"{agg.get('recall@10',0):.3f}",
                             r20=f"{r20:.3f}", r50=f"{agg.get('recall@50',0):.3f}")
            if score > best_score:
                best_score = score
                best_metrics = agg.copy()
                stall = 0
            else:
                stall += 1
                if stall >= PATIENCE:
                    pbar.set_description(f"    REC fold early stop @{ep}/{max_epochs} best_r20={r20:.4f} n20={n20:.4f}")
                    break

    del model
    return best_score, best_metrics


def _run_fold_lp(data, tr_edges, va_edges, te_edges, params, device, head='gcn', neg_ratio=5,
                 x_dict=None, edge_index_dict=None, node_type_indices=None):
    from .link_prediction import sample_negatives, MLPDecoder

    d = params['d']

    # Build inputs using only training edges to prevent leakage
    fold_x_dict = rebuild_user_features(data, tr_edges, device)
    fold_edge_index_dict = _build_masked_edge_index(data, tr_edges, device)
    fold_node_type_indices = {k: v.to(device)
                              for k, v in build_node_type_indices(data).items()}

    model = build_encoder(data, params, device)
    decoder = MLPDecoder(d).to(device)
    opt = Adam(list(model.parameters()) + list(decoder.parameters()),
               lr=params['lr'], weight_decay=params['wd'])
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device='cuda') if use_amp else None

    all_src = np.unique(tr_edges[:, 0])
    all_dst = np.unique(tr_edges[:, 1])
    all_targets = data.get('all_target_edges')
    va_neg = sample_negatives(va_edges, len(va_edges) * neg_ratio,
                              all_src, all_dst, 1,
                              all_positives=all_targets)

    def tensors(pos, neg):
        e = np.concatenate([pos, neg], 0)
        l = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
        return (torch.tensor(e[:, 0], dtype=torch.long, device=device),
                torch.tensor(e[:, 1], dtype=torch.long, device=device),
                torch.tensor(l, device=device))

    va_s, va_d, va_l = tensors(va_edges, va_neg)

    tr_s = torch.tensor(tr_edges[:, 0], dtype=torch.long, device=device)
    tr_d = torch.tensor(tr_edges[:, 1], dtype=torch.long, device=device)
    all_pos_set = set(map(tuple, tr_edges))
    if all_targets is not None:
        all_pos_set |= set(map(tuple, all_targets))
    rng = np.random.default_rng(0)

    best_auc, stall = 0.0, 0
    max_epochs = params['epochs']
    pbar = tqdm(range(1, max_epochs + 1), desc="LP fold", leave=False)
    for ep in pbar:
        model.train(); decoder.train()
        opt.zero_grad()

        neg_dst = rng.choice(all_dst, size=len(tr_edges))
        for i in range(len(neg_dst)):
            while (int(tr_s[i]), int(neg_dst[i])) in all_pos_set:
                neg_dst[i] = rng.choice(all_dst)
        neg_dst_t = torch.tensor(neg_dst, dtype=torch.long, device=device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            emb, *_ = model(fold_x_dict, fold_edge_index_dict, fold_node_type_indices)
            pos_score = decoder(emb, tr_s, tr_d)
            neg_score = decoder(emb, tr_s, neg_dst_t)
            loss = -F.logsigmoid(pos_score - neg_score).mean()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        model.eval(); decoder.eval()
        with torch.no_grad():
            emb_v, *_ = model(fold_x_dict, fold_edge_index_dict, fold_node_type_indices)
            p = torch.sigmoid(decoder(emb_v, va_s, va_d)).cpu().numpy()
            auc = roc_auc_score(va_l.cpu().numpy(), p)

        pbar.set_description(f"LP fold loss={loss.item():.4f} val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            stall = 0
        else:
            stall += 1
            if stall >= PATIENCE:
                pbar.set_description(f"LP fold early stop @{ep}/{max_epochs} best={best_auc:.4f}")
                break

    del model, decoder
    return best_auc


def hparam_search_nc(data, seed=42, out_dir='results/nc', head='gcn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'train_indices' in data:
        tr80 = data['train_indices'].numpy()
        te20 = data['test_indices'].numpy()
        lbl_np = data['labels'].numpy()
        print(f"  Using predefined split: {len(tr80)} train / {len(te20)} test", flush=True)
    else:
        Nt = data['target_size']
        lbl_np = data['labels'].numpy()
        tr80, te20 = train_test_split(np.arange(Nt), test_size=0.70,
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
        if mean_vm > best_mean: best_mean, best_params = mean_vm, copy.deepcopy(params)

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

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_CLUSTERING, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows = []
    best_params, best_mean = None, 0.0

    n_total = len(combos) * N_FOLDS
    print(f"\n  Hyperparameter search: {len(combos)} combos × {N_FOLDS} folds = {n_total} runs", flush=True)
    t0_hp = time.time()
    for ci, params in enumerate(combos):
        t_combo = time.time()
        print(f"\n  combination {ci+1} {params}", flush=True)
        fold_nmis = []
        fold_iter = tqdm(skf.split(tr80, lbl_np[tr80]), desc=f"    fold", total=N_FOLDS, leave=False)
        for fold, (tr_fold, va_fold) in enumerate(fold_iter):
            nmi = _run_fold_cl(data, params, tr80[tr_fold], tr80[va_fold], device)
            fold_nmis.append(nmi)
            fold_iter.set_postfix(nmi=f"{nmi:.4f}")
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_nmi': round(nmi, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
        mean_nmi = float(np.mean(fold_nmis))
        elapsed = time.time() - t_combo
        print(f"    fold_nmis={[round(s, 4) for s in fold_nmis]}")
        print(f"    mean_nmi={mean_nmi:.4f}  [{elapsed:.0f}s]", flush=True)
        if mean_nmi > best_mean: best_mean, best_params = mean_nmi, copy.deepcopy(params)

    total_hp = time.time() - t0_hp
    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name', ''), 'cl', out_dir)
    print(f"[CL hparam] best_val_nmi={best_mean:.4f}  params={best_params}  total={total_hp:.0f}s", flush=True)
    return best_params, tr80, te20


def hparam_search_rec(data, target_edges, seed=42, out_dir='results/recommendation', head='gcn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_idx = np.arange(len(target_edges))
    tr80_idx, te20_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=seed)
    tr80_edges = target_edges[tr80_idx]
    te20_edges = target_edges[te20_idx]

    data['all_target_edges'] = target_edges

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_REC, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows = []
    best_params, best_mean = None, 0.0

    n_total = len(combos) * N_FOLDS
    print(f"\n  Hyperparameter search: {len(combos)} combos × {N_FOLDS} folds = {n_total} runs", flush=True)
    t0_hp = time.time()
    for ci, params in enumerate(combos):
        t_combo = time.time()
        print(f"\n  combination {ci+1} {params}", flush=True)
        fold_scores = []
        fold_iter = tqdm(kf.split(tr80_edges), desc=f"    fold", total=N_FOLDS, leave=False)
        for fold, (tr_fold, va_fold) in enumerate(fold_iter):
            score, metrics = _run_fold_rec(data, tr80_edges[tr_fold], tr80_edges[va_fold],
                                           params, device, head=head,
                                           K_rec=params.get('K_rec', 20))
            fold_scores.append(score)
            r10 = metrics.get('recall@10', 0); n10 = metrics.get('ndcg@10', 0)
            r20 = metrics.get('recall@20', 0); n20 = metrics.get('ndcg@20', 0)
            r50 = metrics.get('recall@50', 0); n50 = metrics.get('ndcg@50', 0)
            fold_iter.set_postfix(r10=f"{r10:.3f}", r20=f"{r20:.3f}", r50=f"{r50:.3f}")
            cv_rows.append({'combo_id': ci, 'fold': fold,
                            'val_recall@10': round(r10, 4), 'val_ndcg@10': round(n10, 4),
                            'val_recall@20': round(r20, 4), 'val_ndcg@20': round(n20, 4),
                            'val_recall@50': round(r50, 4), 'val_ndcg@50': round(n50, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
        mean_score = float(np.mean(fold_scores))
        elapsed = time.time() - t_combo
        combo_rows = [r for r in cv_rows if r['combo_id'] == ci]
        if combo_rows:
            mr10 = np.mean([r['val_recall@10'] for r in combo_rows])
            mn10 = np.mean([r['val_ndcg@10'] for r in combo_rows])
            mr20 = np.mean([r['val_recall@20'] for r in combo_rows])
            mn20 = np.mean([r['val_ndcg@20'] for r in combo_rows])
            mr50 = np.mean([r['val_recall@50'] for r in combo_rows])
            mn50 = np.mean([r['val_ndcg@50'] for r in combo_rows])
            print(f"    fold_scores={[round(s, 4) for s in fold_scores]}")
            print(f"    mean_score={mean_score:.4f}  r10={mr10:.4f} n10={mn10:.4f}  r20={mr20:.4f} n20={mn20:.4f}  r50={mr50:.4f} n50={mn50:.4f}  [{elapsed:.0f}s]", flush=True)
        if mean_score > best_mean: best_mean, best_params = mean_score, copy.deepcopy(params)

    total_hp = time.time() - t0_hp
    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name', ''), 'rec', out_dir)
    print(f"[REC hparam] best_val_recall@K={best_mean:.4f}  params={best_params}  total={total_hp:.0f}s", flush=True)
    return best_params, tr80_edges, te20_edges


def hparam_search_lp(data, target_edges, seed=42, out_dir='results/lp', head='gcn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_idx = np.arange(len(target_edges))
    tr80_idx, te20_idx = train_test_split(all_idx, test_size=0.10, random_state=seed)
    tr80_edges = target_edges[tr80_idx]
    te20_edges = target_edges[te20_idx]

    # Store for negative sampling across all splits
    data['all_target_edges'] = target_edges

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_BASE, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows = []
    best_params, best_mean = None, 0.0

    n_total = len(combos) * N_FOLDS
    print(f"\n  Hyperparameter search: {len(combos)} combos × {N_FOLDS} folds = {n_total} runs", flush=True)
    t0_hp = time.time()
    for ci, params in enumerate(combos):
        t_combo = time.time()
        print(f"\n  combination {ci+1}({params})", flush=True)
        fold_aucs = []
        fold_iter = tqdm(kf.split(tr80_edges), desc=f"    fold", total=N_FOLDS, leave=False)
        for fold, (tr_fold, va_fold) in enumerate(fold_iter):
            auc = _run_fold_lp(data, tr80_edges[tr_fold], tr80_edges[va_fold],
                               te20_edges, params, device, head=head)
            fold_aucs.append(auc)
            fold_iter.set_postfix(auc=f"{auc:.4f}")
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_auc': round(auc, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
        mean_auc = float(np.mean(fold_aucs))
        elapsed = time.time() - t_combo
        print(f"    fold_aucs={[round(s, 4) for s in fold_aucs]}")
        print(f"    mean_auc={mean_auc:.4f}  [{elapsed:.0f}s]", flush=True)
        if mean_auc > best_mean: best_mean, best_params = mean_auc, copy.deepcopy(params)

    total_hp = time.time() - t0_hp
    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name', ''), 'lp', out_dir)
    print(f"[LP hparam] best_val_auc={best_mean:.4f}  params={best_params}  total={total_hp:.0f}s", flush=True)
    return best_params, tr80_edges, te20_edges
