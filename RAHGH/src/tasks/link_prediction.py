import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import time
from tqdm import tqdm

from ..model.rahgh    import RAHGH, compile_model
from ..model.diffusion import build_operators, normalize_plain


def split_edges(edges, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(edges))
    n = len(edges)
    n_tr = int(0.8 * n); n_va = int(0.1 * n)
    return edges[idx[:n_tr]], edges[idx[n_tr:n_tr + n_va]], \
        edges[idx[n_tr + n_va:]]


def sample_negatives(pos, n_neg, all_src, all_dst, seed=0):
    rng = np.random.default_rng(seed)
    pos_set = set(map(tuple, pos))
    negs = []
    while len(negs) < n_neg:
        s = rng.choice(all_src, size=n_neg * 2)
        d = rng.choice(all_dst, size=n_neg * 2)
        for si, di in zip(s, d):
            if (si, di) not in pos_set:
                negs.append((si, di))
            if len(negs) >= n_neg:
                break
    return np.array(negs[:n_neg])


class MLPDecoder(nn.Module):
    def __init__(self, emb_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, emb, src, dst):
        return self.mlp(
            torch.cat([emb[src], emb[dst]], dim=1)).squeeze(-1)


def _build_masked_operators(data, train_edges, device):
    N = data['N']
    tr_r, tr_c = train_edges[:, 0], train_edges[:, 1]
    A_train = sp.coo_matrix(
        (np.ones(len(tr_r)), (tr_r, tr_c)), shape=(N, N)).tocsr()
    A_list_masked = list(data['A_list_sp'])
    A_list_masked[0] = A_train
    A_list_masked[1] = A_train.T.tocsr()
    return build_operators(A_list_masked, data['bipartite_flags'], device)


PATIENCE = 100


def _run_fold_lp(data, tr_edges, va_edges, te_edges, params,
                 device, neg_ratio=5):
    torch.manual_seed(0)
    np.random.seed(0)
    d       = params['d']
    R       = len(data['A_list_sp'])
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    X_list  = [x.to(device) for x in data['X_dict'].values()]

    P_list = _build_masked_operators(data, tr_edges, device)

    backbone = RAHGH(
        in_dims=in_dims, d=d, R=R, K=params['K'],
        gcn_hidden=params['gcn_hidden'],
        out_dim=d,
        dropout=params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    backbone = compile_model(backbone, verbose=True)
    decoder = MLPDecoder(d).to(device)

    opt = Adam(list(backbone.parameters()) + list(decoder.parameters()),
               lr=params['lr'], weight_decay=params['wd'])
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    all_src = np.unique(tr_edges[:, 0])
    all_dst = np.unique(tr_edges[:, 1])
    tr_neg  = sample_negatives(tr_edges, len(tr_edges) * neg_ratio,
                               all_src, all_dst, 0)
    va_neg  = sample_negatives(va_edges, len(va_edges) * neg_ratio,
                               all_src, all_dst, 1)

    def tensors(pos, neg):
        e = np.concatenate([pos, neg], 0)
        l = np.concatenate(
            [np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
        return (torch.tensor(e[:, 0], dtype=torch.long, device=device),
                torch.tensor(e[:, 1], dtype=torch.long, device=device),
                torch.tensor(l, device=device))

    tr_s, tr_d, tr_l = tensors(tr_edges, tr_neg)
    va_s, va_d, va_l = tensors(va_edges, va_neg)

    best_auc, stall = 0.0, 0
    max_epochs = params['epochs']
    pbar = tqdm(range(1, max_epochs + 1), desc="LP fold training", leave=False)
    for ep in pbar:
        backbone.train()
        decoder.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            emb, *_ = backbone(X_list, P_list)
            loss = F.binary_cross_entropy_with_logits(
                decoder(emb, tr_s, tr_d), tr_l)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        backbone.eval()
        decoder.eval()
        with torch.no_grad():
            emb_v, *_ = backbone(X_list, P_list)
            p = torch.sigmoid(decoder(emb_v, va_s, va_d)).cpu().numpy()
            auc = roc_auc_score(va_l.cpu().numpy(), p)

        pbar.set_description(f"loss={loss.item():.4f} val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            stall = 0
        else:
            stall += 1
            if stall >= PATIENCE:
                pbar.set_description(f"Early stop @{ep}/{max_epochs} best={best_auc:.4f}")
                break

    del backbone, decoder
    torch.cuda.empty_cache()
    return best_auc


def run_single_lp(data, target_edges, K, epochs, seed, cfg, neg_ratio=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d      = cfg['d']

    P_list  = build_operators(data['A_list_sp'], data['bipartite_flags'],
                              device)
    X_list  = [x.to(device) for x in data['X_dict'].values()]
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    R       = len(data['A_list_sp'])

    backbone = RAHGH(
        in_dims=in_dims, d=d, R=R, K=K,
        gcn_hidden=cfg['gcn_hidden'],
        out_dim=d,
        dropout=cfg['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    backbone = compile_model(backbone, verbose=True)

    decoder   = MLPDecoder(d).to(device)
    optimizer = Adam(
        list(backbone.parameters()) + list(decoder.parameters()),
        lr=cfg['lr'], weight_decay=cfg['wd'])
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    tr_e, va_e, te_e = split_edges(target_edges, seed=seed)
    all_src = np.unique(target_edges[:, 0])
    all_dst = np.unique(target_edges[:, 1])
    tr_neg  = sample_negatives(tr_e, len(tr_e) * neg_ratio,
                               all_src, all_dst, 0)
    va_neg  = sample_negatives(va_e, len(va_e) * neg_ratio,
                               all_src, all_dst, 1)
    te_neg  = sample_negatives(te_e, len(te_e) * neg_ratio,
                               all_src, all_dst, 2)

    def to_tensors(pos, neg):
        edges = np.concatenate([pos, neg], 0)
        lbl   = np.concatenate(
            [np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
        return (torch.tensor(edges[:, 0], dtype=torch.long, device=device),
                torch.tensor(edges[:, 1], dtype=torch.long, device=device),
                torch.tensor(lbl, device=device))

    tr_s, tr_d, tr_l = to_tensors(tr_e, tr_neg)
    va_s, va_d, va_l = to_tensors(va_e, va_neg)
    te_s, te_d, te_l = to_tensors(te_e, te_neg)

    def metrics(logits, lbl):
        p = torch.sigmoid(logits).detach().cpu().numpy()
        y = lbl.cpu().numpy()
        return roc_auc_score(y, p), average_precision_score(y, p)

    best_auc, best_sd_bb, best_sd_dec = 0.0, None, None
    t0 = time.time()

    pbar = tqdm(range(1, epochs + 1), desc="LP training", leave=False)
    for ep in pbar:
        backbone.train()
        decoder.train()
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            emb, *_ = backbone(X_list, P_list)
            loss = F.binary_cross_entropy_with_logits(
                decoder(emb, tr_s, tr_d), tr_l)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        backbone.eval()
        decoder.eval()
        with torch.no_grad():
            emb_v, *_ = backbone(X_list, P_list)
            auc_v, _  = metrics(decoder(emb_v, va_s, va_d), va_l)
        pbar.set_description(f"loss={loss.item():.4f} val_auc={auc_v:.4f}")
        if auc_v > best_auc:
            best_auc    = auc_v
            best_sd_bb  = {k: v.clone()
                           for k, v in backbone.state_dict().items()}
            best_sd_dec = {k: v.clone()
                           for k, v in decoder.state_dict().items()}

    backbone.load_state_dict(best_sd_bb)
    decoder.load_state_dict(best_sd_dec)
    backbone.eval()
    decoder.eval()
    with torch.no_grad():
        emb_te, *_ = backbone(X_list, P_list)
        auc_te, ap_te = metrics(decoder(emb_te, te_s, te_d), te_l)

    return dict(auc=auc_te, ap=ap_te, best_val_auc=best_auc,
                time_sec=time.time() - t0)


def run_final_lp(data, best_params, tr80_edges, te20_edges,
                 seed=42, neg_ratio=5, out_dir=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d      = best_params['d']
    R      = len(data['A_list_sp'])
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    X_list = [x.to(device) for x in data['X_dict'].values()]

    P_list = _build_masked_operators(data, tr80_edges, device)

    backbone = RAHGH(
        in_dims=in_dims, d=d, R=R, K=best_params['K'],
        gcn_hidden=best_params['gcn_hidden'],
        out_dim=d,
        dropout=best_params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    backbone = compile_model(backbone, verbose=True)
    decoder = MLPDecoder(d).to(device)
    opt = Adam(list(backbone.parameters()) + list(decoder.parameters()),
               lr=best_params['lr'], weight_decay=best_params['wd'])
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    all_src = np.unique(tr80_edges[:, 0])
    all_dst = np.unique(tr80_edges[:, 1])
    tr_neg  = sample_negatives(tr80_edges, len(tr80_edges) * neg_ratio,
                               all_src, all_dst, 0)
    te_neg  = sample_negatives(te20_edges, len(te20_edges) * neg_ratio,
                               all_src, all_dst, 2)

    def tensors(pos, neg):
        e = np.concatenate([pos, neg], 0)
        l = np.concatenate(
            [np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
        return (torch.tensor(e[:, 0], dtype=torch.long, device=device),
                torch.tensor(e[:, 1], dtype=torch.long, device=device),
                torch.tensor(l, device=device))

    tr_s, tr_d, tr_l = tensors(tr80_edges, tr_neg)
    te_s, te_d, te_l = tensors(te20_edges, te_neg)

    best_sd_b, best_sd_d, best_auc = None, None, 0.0
    t0 = time.time()

    epoch_rows = []
    pbar = tqdm(range(1, best_params['epochs'] + 1), desc="Final LP training")
    for ep in pbar:
        backbone.train()
        decoder.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            emb, *_ = backbone(X_list, P_list)
            loss = F.binary_cross_entropy_with_logits(
                decoder(emb, tr_s, tr_d), tr_l)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        # Track training AUC every epoch
        backbone.eval()
        decoder.eval()
        with torch.no_grad():
            emb_v, a, b, _ = backbone(X_list, P_list)
            pv = torch.sigmoid(decoder(emb_v, tr_s, tr_d)).cpu().numpy()
            train_auc = roc_auc_score(tr_l.cpu().numpy(), pv)
        epoch_rows.append({'epoch': ep, 'loss': loss.item(),
                           'train_auc': float(train_auc)})

        if ep % 50 == 0 or ep == best_params['epochs']:
            pbar.set_description(f"loss={loss.item():.4f} train_auc={train_auc:.4f}")
            if train_auc > best_auc:
                best_auc  = train_auc
                best_sd_b = {k: v.clone()
                             for k, v in backbone.state_dict().items()}
                best_sd_d = {k: v.clone()
                             for k, v in decoder.state_dict().items()}

    backbone.load_state_dict(best_sd_b)
    decoder.load_state_dict(best_sd_d)
    backbone.eval()
    decoder.eval()
    with torch.no_grad():
        emb_te, alpha, beta, _ = backbone(X_list, P_list)
        p = torch.sigmoid(decoder(emb_te, te_s, te_d)).cpu().numpy()
        auc = roc_auc_score(te_l.cpu().numpy(), p)
        ap  = average_precision_score(te_l.cpu().numpy(), p)

    # Save epoch metrics
    if out_dir is not None:
        import csv
        from pathlib import Path
        ep_path = Path(out_dir) / f'epoch_metrics_seed{seed}.csv'
        with open(ep_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'train_auc'])
            w.writeheader()
            w.writerows(epoch_rows)

    return dict(auc=auc, ap=ap,
                alpha=alpha.detach().cpu().numpy(),
                beta=beta.detach().cpu().numpy(),
                time_sec=time.time() - t0)
