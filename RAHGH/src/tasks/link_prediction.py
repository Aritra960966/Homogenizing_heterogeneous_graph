import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import time, os
from tqdm import tqdm

from ..model.rahgh import (
    RAHGHClassifier, compile_model,
    build_rahgh_classifier, build_edge_index_dict, build_node_type_indices,
)


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


def _build_masked_edge_index(data, train_edges, device):
    """Build edge_index_dict from data but mask the first relation with train_edges."""
    N = data['N']
    tr_r, tr_c = train_edges[:, 0], train_edges[:, 1]
    A_train = sp.coo_matrix(
        (np.ones(len(tr_r)), (tr_r, tr_c)), shape=(N, N)).tocsr()

    rel_names = data.get('relation_names', [f'rel_{i}' for i in range(len(data['A_list_sp']))])
    edge_dict = {}
    for i, (A_sp, rname) in enumerate(zip(data['A_list_sp'], rel_names)):
        A_use = A_train if i == 0 else A_sp
        A_coo = A_use.tocoo()
        ei = np.vstack([A_coo.row, A_coo.col])
        edge_dict[rname] = torch.tensor(ei, dtype=torch.long, device=device)
    return edge_dict


PATIENCE = 100


def _run_fold_lp(data, tr_edges, va_edges, te_edges, params,
                 device, neg_ratio=5, head='gcn'):
    torch.manual_seed(0)
    np.random.seed(0)
    d = params['d']
    x_dict = {k: v.to(device) for k, v in data['X_dict'].items()}
    edge_index_dict = build_edge_index_dict(data, device)
    node_type_indices = {k: v.to(device) for k, v in build_node_type_indices(data).items()}

    model = build_rahgh_classifier(
        data, hidden_dim=d, num_classes=d, K=params['K'],
        head=head,
        dropout_homo=params['dropout'], dropout_gnn=params['dropout'],
    ).to(device)
    model = compile_model(model)
    decoder = MLPDecoder(d).to(device)

    opt = Adam(list(model.parameters()) + list(decoder.parameters()),
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
        model.train()
        decoder.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            emb, *_ = model(x_dict, edge_index_dict, node_type_indices)
            loss = F.binary_cross_entropy_with_logits(
                decoder(emb, tr_s, tr_d), tr_l)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        model.eval()
        decoder.eval()
        with torch.no_grad():
            emb_v, *_ = model(x_dict, edge_index_dict, node_type_indices)
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

    del model, decoder
    return best_auc


def run_single_lp(data, target_edges, K, epochs, seed, cfg, neg_ratio=5,
                  head='gcn'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = cfg['d']

    x_dict = {k: v.to(device) for k, v in data['X_dict'].items()}
    edge_index_dict = build_edge_index_dict(data, device)
    node_type_indices = {k: v.to(device) for k, v in build_node_type_indices(data).items()}

    model = build_rahgh_classifier(
        data, hidden_dim=d, num_classes=d, K=K,
        head=head,
        dropout_homo=cfg['dropout'], dropout_gnn=cfg['dropout'],
    ).to(device)
    model = compile_model(model)

    decoder   = MLPDecoder(d).to(device)
    optimizer = Adam(
        list(model.parameters()) + list(decoder.parameters()),
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

    best_auc, best_sd_h, best_sd_dec = 0.0, None, None
    t0 = time.time()

    pbar = tqdm(range(1, epochs + 1), desc="LP training", leave=False)
    for ep in pbar:
        model.train()
        decoder.train()
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            emb, *_ = model(x_dict, edge_index_dict, node_type_indices)
            loss = F.binary_cross_entropy_with_logits(
                decoder(emb, tr_s, tr_d), tr_l)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        model.eval()
        decoder.eval()
        with torch.no_grad():
            emb_v, *_ = model(x_dict, edge_index_dict, node_type_indices)
            auc_v, _  = metrics(decoder(emb_v, va_s, va_d), va_l)
        pbar.set_description(f"loss={loss.item():.4f} val_auc={auc_v:.4f}")
        if auc_v > best_auc:
            best_auc    = auc_v
            best_sd_h  = {k: v.clone()
                          for k, v in model.state_dict().items()}
            best_sd_dec = {k: v.clone()
                           for k, v in decoder.state_dict().items()}

    model.load_state_dict(best_sd_h)
    decoder.load_state_dict(best_sd_dec)
    model.eval()
    decoder.eval()
    with torch.no_grad():
        emb_te, *_ = model(x_dict, edge_index_dict, node_type_indices)
        auc_te, ap_te = metrics(decoder(emb_te, te_s, te_d), te_l)

    return dict(auc=auc_te, ap=ap_te, best_val_auc=best_auc,
                time_sec=time.time() - t0)


def run_final_lp(data, best_params, tr80_edges, te20_edges,
                 seed=42, neg_ratio=5, out_dir=None, head='gcn'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = best_params['d']

    x_dict = {k: v.to(device) for k, v in data['X_dict'].items()}
    edge_index_dict = build_edge_index_dict(data, device)
    node_type_indices = {k: v.to(device) for k, v in build_node_type_indices(data).items()}

    model = build_rahgh_classifier(
        data, hidden_dim=d, num_classes=d, K=best_params['K'],
        head=head,
        dropout_homo=best_params['dropout'], dropout_gnn=best_params['dropout'],
    ).to(device)
    model = compile_model(model)
    decoder = MLPDecoder(d).to(device)
    opt = Adam(list(model.parameters()) + list(decoder.parameters()),
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

    best_sd_h, best_sd_d, best_auc = None, None, 0.0
    t0 = time.time()

    epoch_rows = []
    pbar = tqdm(range(1, best_params['epochs'] + 1), desc="Final LP training")
    for ep in pbar:
        model.train()
        decoder.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            emb, *_ = model(x_dict, edge_index_dict, node_type_indices)
            loss = F.binary_cross_entropy_with_logits(
                decoder(emb, tr_s, tr_d), tr_l)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        model.eval()
        decoder.eval()
        with torch.no_grad():
            emb_v, *_ = model(x_dict, edge_index_dict, node_type_indices)
            pv = torch.sigmoid(decoder(emb_v, tr_s, tr_d)).cpu().numpy()
            train_auc = roc_auc_score(tr_l.cpu().numpy(), pv)
        epoch_rows.append({'epoch': ep, 'loss': loss.item(),
                           'train_auc': float(train_auc)})

        if ep % 50 == 0 or ep == best_params['epochs']:
            pbar.set_description(f"loss={loss.item():.4f} train_auc={train_auc:.4f}")
            if train_auc > best_auc:
                best_auc  = train_auc
                best_sd_h = {k: v.clone()
                             for k, v in model.state_dict().items()}
                best_sd_d = {k: v.clone()
                             for k, v in decoder.state_dict().items()}

    model.load_state_dict(best_sd_h)
    decoder.load_state_dict(best_sd_d)
    model.eval()
    decoder.eval()
    with torch.no_grad():
        emb_te, alpha = model(x_dict, edge_index_dict, node_type_indices)
        p = torch.sigmoid(decoder(emb_te, te_s, te_d)).cpu().numpy()
        auc = roc_auc_score(te_l.cpu().numpy(), p)
        ap  = average_precision_score(te_l.cpu().numpy(), p)

    # Save final model
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        pt_path = os.path.join(out_dir, f'final_model_seed{seed}.pt')
        torch.save(model.state_dict(), pt_path)
        print(f"  Model saved → {pt_path}")

    # Save epoch metrics
    if out_dir is not None:
        import csv
        from pathlib import Path
        ep_path = Path(out_dir) / f'epoch_metrics_seed{seed}.csv'
        write_header = not ep_path.exists()
        with open(ep_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'train_auc'])
            if write_header:
                w.writeheader()
            w.writerows(epoch_rows)

    return dict(auc=auc, ap=ap,
                alpha=alpha.detach().cpu().numpy(),
                time_sec=time.time() - t0)
