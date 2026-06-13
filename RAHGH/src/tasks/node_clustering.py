import os, csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import linear_sum_assignment

from ..model.rahgh import (
    RAHGH,
    build_edge_index_dict,
    build_node_type_indices,
    compile_model,
)


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────
def clustering_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n      = max(y_true.max(), y_pred.max()) + 1
    D      = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        D[t, p] += 1
    r, c = linear_sum_assignment(-D)
    return float(D[r, c].sum()) / len(y_true)


def eval_clustering(Z, y, n_clusters, n_runs=10, seed=0):
    Z_norm = sk_normalize(Z, norm='l2')
    nmis, aris, accs = [], [], []
    for i in range(n_runs):
        pred = KMeans(n_clusters=n_clusters, n_init=1,
                      random_state=seed + i,
                      max_iter=300).fit_predict(Z_norm)
        nmis.append(normalized_mutual_info_score(y, pred))
        aris.append(adjusted_rand_score(y, pred))
        accs.append(clustering_accuracy(y, pred))
    return dict(
        nmi=float(np.mean(nmis)), nmi_std=float(np.std(nmis)),
        ari=float(np.mean(aris)), ari_std=float(np.std(aris)),
        acc=float(np.mean(accs)), acc_std=float(np.std(accs)),
    )


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────
def _build_model(data, params, device):
    node_type_dims = {k: v.shape[1] for k, v in data['X_dict'].items()}

    if 'relation_info' in data:
        relation_info = data['relation_info']
    else:
        rel_names = data.get('relation_names',
                             [f'rel_{i}'
                              for i in range(len(data['A_list_sp']))])
        if 'bipartite_flags' in data:
            types       = list(data['X_dict'].keys())
            target_type = data.get('target_type', types[0])
            relation_info = {}
            for i, rname in enumerate(rel_names):
                if data['bipartite_flags'][i]:
                    other = next(
                        (t for t in types if t != target_type), types[-1])
                    src_t, dst_t = ((target_type, other) if i % 2 == 0
                                    else (other, target_type))
                else:
                    src_t = dst_t = target_type
                relation_info[rname] = (src_t, dst_t)
        else:
            relation_info = {
                r: (data.get('target_type', 'node'),
                    data.get('target_type', 'node'))
                for r in rel_names
            }

    model = RAHGH(
        node_type_dims=node_type_dims,
        relation_info=relation_info,
        num_nodes=data['N'],
        hidden_dim=params['d'],
        output_dim=params['d'],
        K=params['K'],
        dropout=params['dropout'],
        directed=False,
    ).to(device)
    return compile_model(model)


# ─────────────────────────────────────────────────────────────────────
# Reconstruction target
# ─────────────────────────────────────────────────────────────────────
def _make_target(X_primary, d, device):
    X_np     = X_primary.cpu().numpy()
    N, F     = X_np.shape
    d_target = min(F, max(d * 4, 256))

    if F <= d_target:
        return X_primary.to(device), F

    svd    = TruncatedSVD(n_components=d_target, random_state=0, n_iter=7)
    X_comp = svd.fit_transform(X_np).astype(np.float32)
    var    = svd.explained_variance_ratio_.sum()
    print(f'    target: ({N},{F})->({N},{d_target})  var={var:.3f}')
    return torch.from_numpy(X_comp).to(device), d_target


# ─────────────────────────────────────────────────────────────────────
# Training — two losses, nothing more
# ─────────────────────────────────────────────────────────────────────
def _train(model, x_dict, edge_index_dict, node_type_indices,
           data, params, device, patience=50, min_epochs=200):
    n_epochs  = params['epochs']
    Nt        = data['target_size']
    d         = params['d']
    temp      = params.get('cl_temp',   0.5)
    mask_rate = params.get('mask_rate', 0.3)
    lam       = params.get('lam',       0.5)

    X_target, d_tgt = _make_target(
        list(x_dict.values())[0], d, device)

    decoder = nn.Sequential(
        nn.Linear(d, max(d, d_tgt // 2)),
        nn.ReLU(),
        nn.Linear(max(d, d_tgt // 2), d_tgt),
    ).to(device)

    opt = Adam(
        list(model.parameters()) + list(decoder.parameters()),
        lr=params['lr'], weight_decay=params['wd'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=params['lr'] * 0.01)

    best_loss, best_sd = float('inf'), None
    no_improve, last_l = 0, float('inf')

    for ep in range(1, n_epochs + 1):
        model.train(); decoder.train()
        opt.zero_grad()

        # View 1: original
        Z1, alpha = model(x_dict, edge_index_dict, node_type_indices)
        Z1_nt     = Z1[:Nt]

        loss_recon = F.mse_loss(
            decoder(F.normalize(Z1_nt, dim=-1)), X_target)

        # View 2: masked features
        x_aug = {
            ntype: X * torch.bernoulli(
                torch.full(X.shape, 1.0 - mask_rate, device=device))
            for ntype, X in x_dict.items()
        }
        Z2, _ = model(x_aug, edge_index_dict, node_type_indices)
        Z2_nt = Z2[:Nt]

        # NT-Xent alignment (mini-batch)
        B   = min(512, Nt)
        idx = torch.randperm(Nt, device=device)[:B]
        h1  = F.normalize(Z1_nt[idx], dim=-1)
        h2  = F.normalize(Z2_nt[idx], dim=-1)
        sim = torch.mm(h1, h2.t()) / temp
        lbl = torch.arange(B, device=device)
        loss_align = (F.cross_entropy(sim, lbl) +
                      F.cross_entropy(sim.t(), lbl)) / 2

        loss = loss_recon + lam * loss_align

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sch.step()

        l = loss.item()
        if l < best_loss:
            best_loss = l
            best_sd   = {k: v.clone()
                         for k, v in model.state_dict().items()}

        if ep % 10 == 0:
            if (last_l - l) < 1e-5 and ep >= min_epochs:
                no_improve += 1
            else:
                no_improve = 0
            last_l = l
            if no_improve >= patience:
                print(f'    early stop ep={ep}  loss={l:.4f}',
                      flush=True)
                break

        if ep % 100 == 0 or ep == 1:
            print(f'    ep={ep:4d}/{n_epochs}  '
                  f'loss={l:.4f}  '
                  f'recon={loss_recon.item():.4f}  '
                  f'align={loss_align.item():.4f}',
                  flush=True)

    model.load_state_dict(best_sd)
    return model, alpha


# ─────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _embed(model, x_dict, edge_index_dict, node_type_indices, Nt):
    model.eval()
    Z, alpha = model(x_dict, edge_index_dict, node_type_indices)
    return Z[:Nt].cpu().numpy(), alpha


# ─────────────────────────────────────────────────────────────────────
# CV fold
# ─────────────────────────────────────────────────────────────────────
def run_fold_clustering(data, params, tr_idx, va_idx, device):
    x_dict = {k: v.to(device) for k, v in data['X_dict'].items()}
    ei     = build_edge_index_dict(data, device)
    nti    = {k: v.to(device)
              for k, v in build_node_type_indices(data).items()}

    model = _build_model(data, params, device)
    model, _ = _train(model, x_dict, ei, nti, data, params, device,
                      patience=25, min_epochs=100)

    Z_np, _ = _embed(model, x_dict, ei, nti, data['target_size'])
    y_np    = data['labels'].numpy()

    pred = KMeans(n_clusters=data['n_classes'], n_init=5,
                  random_state=0).fit_predict(
        sk_normalize(Z_np, norm='l2'))
    nmi = normalized_mutual_info_score(y_np, pred)

    del model; torch.cuda.empty_cache()
    return nmi


# ─────────────────────────────────────────────────────────────────────
# Final run
# ─────────────────────────────────────────────────────────────────────
def run_final_clustering(data, best_params, tr80_idx=None,
                          te20_idx=None, seed=42,
                          out_dir='results/clustering',
                          sdcn_iters=0, head=None):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Nt         = data['target_size']
    n_clusters = data['n_classes']
    y_all      = data['labels'].numpy()

    x_dict = {k: v.to(device) for k, v in data['X_dict'].items()}
    ei     = build_edge_index_dict(data, device)
    nti    = {k: v.to(device)
              for k, v in build_node_type_indices(data).items()}

    t0    = time.time()
    model = _build_model(data, best_params, device)
    model, alpha = _train(model, x_dict, ei, nti, data,
                          best_params, device,
                          patience=80, min_epochs=200)

    Z_np, alpha = _embed(model, x_dict, ei, nti, Nt)
    res         = eval_clustering(Z_np, y_all, n_clusters,
                                  n_runs=10, seed=seed)

    print(f'\n  [CL seed={seed}]  '
          f'NMI={res["nmi"]:.4f}+-{res["nmi_std"]:.4f}  '
          f'ARI={res["ari"]:.4f}  '
          f'ACC={res["acc"]:.4f}  '
          f'time={time.time() - t0:.1f}s')

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        alpha_np  = alpha.detach().cpu().numpy()
        rel_names = [n.replace('\u2192', '->')
                     for n in data.get(
                         'relation_names',
                         [f'rel_{i}' for i in range(len(alpha_np))])]
        _save_alpha(alpha_np, rel_names, seed,
                    os.path.join(out_dir, 'alpha_weights.csv'))

    return dict(nmi=res['nmi'], nmi_std=res['nmi_std'],
                ari=res['ari'], acc=res['acc'],
                alpha=alpha.detach().cpu().numpy(),
                time_sec=time.time() - t0)


def _save_alpha(alpha_np, rel_names, seed, path):
    row = {'seed': seed}
    row.update({f'alpha_{n}': round(float(v), 6)
                for n, v in zip(rel_names, alpha_np)})
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists: w.writeheader()
        w.writerow(row)
