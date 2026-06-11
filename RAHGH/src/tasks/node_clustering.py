import numpy as np
import torch
import torch.nn.functional as F
import time, os, csv
from sklearn.cluster  import KMeans
from sklearn.metrics  import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize   import linear_sum_assignment
from torch.optim      import Adam

from ..model.rahgh import (
    RAHGH, build_edge_index_dict, build_node_type_indices, compile_model,
)


def clustering_accuracy(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    n_classes = max(y_true.max(), y_pred.max()) + 1
    D = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        D[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-D)
    return D[row_ind, col_ind].sum() / len(y_true)


def _build_rahgh_model(data, params, device):
    node_type_dims = {k: v.shape[1] for k, v in data['X_dict'].items()}
    if 'relation_info' in data:
        relation_info = data['relation_info']
    elif 'bipartite_flags' in data:
        types = list(data['X_dict'].keys())
        target_type = data.get('target_type', types[0])
        rel_names = data.get('relation_names', [f'rel_{i}' for i in range(len(data['A_list_sp']))])
        relation_info = {}
        for i, rname in enumerate(rel_names):
            is_bip = data['bipartite_flags'][i]
            if is_bip:
                other_type = next((t for t in types if t != target_type), types[-1])
                src_type, dst_type = (target_type, other_type) if i % 2 == 0 else (other_type, target_type)
            else:
                src_type = dst_type = target_type
            relation_info[rname] = (src_type, dst_type)
    else:
        rel_names = data.get('relation_names', [f'rel_{i}' for i in range(len(data['A_list_sp']))])
        relation_info = {r: (data.get('target_type', 'node'), data.get('target_type', 'node'))
                         for r in rel_names}
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


def _reconstruction_loss(model, x_dict, edge_index_dict, node_type_indices,
                         tr_idx, device):
    Z, _ = model(x_dict, edge_index_dict, node_type_indices)
    Z_norm = F.normalize(Z, dim=1)
    sim = Z_norm @ Z_norm.T

    loss = 0.0
    n_rel = len(edge_index_dict)
    for rel_name, ei in edge_index_dict.items():
        s, t = ei[0], ei[1]
        pos_sim = sim[s, t]
        pos_loss = F.binary_cross_entropy_with_logits(pos_sim, torch.ones_like(pos_sim))
        neg_src = torch.randint(0, Z.size(0), (ei.size(1),), device=device)
        neg_dst = torch.randint(0, Z.size(0), (ei.size(1),), device=device)
        neg_sim = sim[neg_src, neg_dst]
        neg_loss = F.binary_cross_entropy_with_logits(neg_sim, torch.zeros_like(neg_sim))
        loss += pos_loss + neg_loss
    return loss / n_rel


def _train_self_supervised(model, x_dict, edge_index_dict, node_type_indices,
                           tr_idx, params, device):
    opt = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    n_epochs = min(params['epochs'], 300)

    model.train()
    for ep in range(1, n_epochs + 1):
        opt.zero_grad()
        loss = _reconstruction_loss(model, x_dict, edge_index_dict,
                                    node_type_indices, tr_idx, device)
        loss.backward()
        opt.step()


def run_final_clustering(data, best_params, tr80_idx, te20_idx,
                          seed=42, out_dir='results/clustering',
                          head='gcn'):
    torch.manual_seed(seed); np.random.seed(seed)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d       = best_params['d']
    Nt      = data['target_size']
    n_cl    = data['n_classes']

    x_dict = {k: v.to(device) for k, v in data['X_dict'].items()}
    edge_index_dict = build_edge_index_dict(data, device)
    node_type_indices = {k: v.to(device) for k, v in build_node_type_indices(data).items()}

    model = _build_rahgh_model(data, best_params, device)

    t0 = time.time()

    # Self-supervised training on the 80% split
    print(f"    Self-supervised training ...", flush=True)
    _train_self_supervised(model, x_dict, edge_index_dict, node_type_indices,
                           tr80_idx, best_params, device)

    # Extract latent embeddings from RAHGH (returns Z_final, alpha)
    model.eval()
    with torch.no_grad():
        Z, alpha = model(x_dict, edge_index_dict, node_type_indices)
    emb_np = Z[:Nt][te20_idx].cpu().numpy()
    y      = data['labels'][te20_idx].numpy()

    print(f"    Running KMeans (n_clusters={n_cl}, n_init=20) on {len(emb_np)} test nodes...", flush=True)
    km   = KMeans(n_clusters=n_cl, n_init=20, random_state=seed)
    pred = km.fit_predict(emb_np)
    nmi  = normalized_mutual_info_score(y, pred)
    ari  = adjusted_rand_score(y, pred)
    acc  = clustering_accuracy(y, pred)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    alpha_np = alpha.detach().cpu().numpy()
    alpha_row = {'seed': seed}
    for i, rname in enumerate(model.relation_names):
        alpha_row[f'alpha_{rname}'] = round(float(alpha_np[i]), 6)
    alpha_path = os.path.join(out_dir, 'alpha_weights.csv')
    file_exists = os.path.exists(alpha_path)
    with open(alpha_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=alpha_row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(alpha_row)

    return dict(nmi=nmi, ari=ari, acc=acc,
                alpha=alpha_np,
                time_sec=time.time() - t0)


def _write_csv(rows, path):
    if not rows: return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
