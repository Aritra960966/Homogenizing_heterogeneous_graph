import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, os, csv
from torch.optim import Adam
from sklearn.cluster  import KMeans
from sklearn.metrics  import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize   import linear_sum_assignment

from ..model.rahgh    import RAHGH
from ..model.diffusion import build_operators


def clustering_accuracy(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    n_classes = max(y_true.max(), y_pred.max()) + 1
    D = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        D[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-D)
    return D[row_ind, col_ind].sum() / len(y_true)


class ReconDecoder(nn.Module):
    def __init__(self, d: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, out_dim))
    def forward(self, emb):
        return self.mlp(emb)


def run_final_clustering(data, best_params, tr80_idx, te20_idx,
                          seed=42, out_dir='results/clustering'):
    torch.manual_seed(seed); np.random.seed(seed)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d       = best_params['d']
    R       = len(data['A_list_sp'])
    Nt      = data['target_size']
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    n_cl    = data['n_classes']

    P_list  = build_operators(data['A_list_sp'], data['bipartite_flags'], device)
    X_list  = [x.to(device) for x in data['X_dict'].values()]

    model   = RAHGH(
        in_dims=in_dims, d=d, R=R, K=best_params['K'],
        gcn_hidden=best_params['gcn_hidden'],
        out_dim=d, dropout=best_params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)

    d_feat  = data['X_dict'][list(data['X_dict'].keys())[0]].shape[1]
    decoder = ReconDecoder(d, min(d_feat, d)).to(device)

    opt  = Adam(list(model.parameters()) + list(decoder.parameters()),
                lr=best_params['lr'], weight_decay=best_params['wd'])
    tr_t = torch.tensor(tr80_idx, dtype=torch.long, device=device)

    epoch_rows = []
    best_nmi, best_sd = 0.0, None
    t0 = time.time()

    for ep in range(1, best_params['epochs'] + 1):
        model.train(); decoder.train(); opt.zero_grad()
        emb, *_ = model(X_list, P_list)

        recon   = decoder(emb[:Nt][tr_t])
        X_tgt   = X_list[0][:Nt][tr_t]
        if X_tgt.shape[1] != recon.shape[1]:
            X_tgt = X_tgt[:, :recon.shape[1]]
        loss    = F.mse_loss(recon, X_tgt.detach())
        loss.backward(); opt.step()

        epoch_rows.append({'epoch': ep, 'recon_loss': round(loss.item(), 6)})

        if ep % 100 == 0 or ep == best_params['epochs']:
            model.eval()
            with torch.no_grad():
                emb_v, *_ = model(X_list, P_list)
                emb_np    = emb_v[:Nt].cpu().numpy()
                km        = KMeans(n_clusters=n_cl, n_init=10, random_state=0)
                pred      = km.fit_predict(emb_np)
                y         = data['labels'].numpy()
                nmi       = normalized_mutual_info_score(y, pred)
                ari       = adjusted_rand_score(y, pred)
                acc       = clustering_accuracy(y, pred)
            if nmi > best_nmi:
                best_nmi = nmi
                best_sd  = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd); model.eval()
    with torch.no_grad():
        emb_f, alpha, beta, _ = model(X_list, P_list)
        emb_np = emb_f[:Nt].cpu().numpy()
        km     = KMeans(n_clusters=n_cl, n_init=20, random_state=seed)
        pred   = km.fit_predict(emb_np)
        y      = data['labels'].numpy()
        nmi    = normalized_mutual_info_score(y, pred)
        ari    = adjusted_rand_score(y, pred)
        acc    = clustering_accuracy(y, pred)

    os.makedirs(os.path.join(out_dir, 'epoch_logs'), exist_ok=True)
    _write_csv(epoch_rows,
               os.path.join(out_dir, 'epoch_logs', f'seed{seed}_epochs.csv'))

    return dict(nmi=nmi, ari=ari, acc=acc,
                alpha=alpha.detach().cpu().numpy(),
                beta=beta.detach().cpu().numpy(),
                time_sec=time.time()-t0)


def _write_csv(rows, path):
    if not rows: return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
