import numpy as np
import torch
import time, os, csv
from sklearn.cluster  import KMeans
from sklearn.metrics  import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize   import linear_sum_assignment

from ..model.rahgh import (
    build_rahgh_classifier, build_edge_index_dict,
    build_node_type_indices,
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

    model = build_rahgh_classifier(
        data, hidden_dim=d, num_classes=n_cl, K=best_params['K'],
        head=head,
        dropout_homo=best_params['dropout'], dropout_gnn=best_params['dropout'],
        gnn_hidden_dim=best_params.get('hidden', d),
    ).to(device)

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        logits, alpha = model(x_dict, edge_index_dict, node_type_indices)
    emb_np = logits[:Nt].cpu().numpy()
    y      = data['labels'].numpy()

    print(f"    Running KMeans (n_clusters={n_cl}, n_init=20) on {len(emb_np)} nodes...", flush=True)
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
                time_sec=time.time()-t0)


def _write_csv(rows, path):
    if not rows: return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
