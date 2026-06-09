import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import time, os
from tqdm import tqdm

from ..model.rahgh import (
    build_rahgh_classifier, build_edge_index_dict, build_node_type_indices,
    compile_model,
)


def clustering_accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    D = max(int(y_true.max()), int(y_pred.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_true)):
        w[int(y_true[i]), int(y_pred[i])] += 1
    row_ind, col_ind = linear_sum_assignment(-w)
    total = sum(w[row_ind[i], col_ind[i]] for i in range(len(row_ind)))
    return total / len(y_true)


def _build_model(data, params, out_dim, device, head='gcn'):
    model = build_rahgh_classifier(
        data, hidden_dim=params['d'], num_classes=out_dim,
        K=params['K'], head=head,
        dropout_homo=params['dropout'], dropout_gnn=params['dropout'],
    ).to(device)
    return compile_model(model)


def run_final_cluster(data, best_params, tr80_idx, te20_idx,
                      seed=42, out_dir=None, head='gcn'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = best_params['d']
    Nt = data['target_size']
    n_cl = data['n_classes']

    x_dict = {k: v.to(device) for k, v in data['X_dict'].items()}
    edge_index_dict = build_edge_index_dict(data, device)
    node_type_indices = {k: v.to(device) for k, v in build_node_type_indices(data).items()}

    model = _build_model(data, best_params, out_dim=d, device=device, head=head)
    decoder = torch.nn.Linear(d, d).to(device)
    opt = Adam(
        list(model.parameters()) + list(decoder.parameters()),
        lr=best_params['lr'], weight_decay=best_params['wd'],
    )

    tr_t = torch.tensor(tr80_idx, dtype=torch.long, device=device)
    X_cat = torch.cat(list(x_dict.values()), dim=0)[:Nt][tr_t]
    if X_cat.shape[1] != d:
        X_cat = X_cat[:, :d] if X_cat.shape[1] > d \
                else F.pad(X_cat, (0, d - X_cat.shape[1]))
    X_cat = X_cat.to(device)
    t0 = time.time()

    epoch_rows = []
    pbar = tqdm(range(1, best_params['epochs'] + 1),
                desc="Final Cluster training")
    for ep in pbar:
        model.train()
        decoder.train()
        opt.zero_grad()
        emb, *_ = model(x_dict, edge_index_dict, node_type_indices)
        loss = F.mse_loss(decoder(emb[:Nt][tr_t]), X_cat)
        loss.backward()
        opt.step()
        epoch_rows.append({'epoch': ep, 'loss': loss.item()})
        if ep % 100 == 0 or ep == best_params['epochs']:
            pbar.set_description(f"loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        emb, *_ = model(x_dict, edge_index_dict, node_type_indices)
    te_emb = emb[:Nt][te20_idx].cpu().numpy()
    te_labels = data['labels'][te20_idx].numpy()

    kmeans = KMeans(n_clusters=n_cl, random_state=seed, n_init=10).fit(te_emb)
    te_pred = kmeans.labels_

    nmi = normalized_mutual_info_score(te_labels, te_pred)
    ari = adjusted_rand_score(te_labels, te_pred)
    ca = clustering_accuracy(te_labels, te_pred)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        pt_path = os.path.join(out_dir, f'final_model_seed{seed}.pt')
        torch.save(model.state_dict(), pt_path)

        import csv
        from pathlib import Path
        ep_path = Path(out_dir) / f'epoch_metrics_seed{seed}.csv'
        with open(ep_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['epoch', 'loss'])
            w.writeheader()
            w.writerows(epoch_rows)

    return dict(
        nmi=nmi, ari=ari, clustering_acc=ca,
        time_sec=time.time() - t0,
    )
