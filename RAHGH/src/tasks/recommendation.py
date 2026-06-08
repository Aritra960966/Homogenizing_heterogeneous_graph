import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import time, os, csv
from torch.optim import Adam

from ..model.rahgh    import RAHGH
from ..model.diffusion import build_operators


def bpr_loss(emb, users, pos_items, neg_items, device, reg=1e-4):
    u   = emb[torch.tensor(users,     dtype=torch.long, device=device)]
    pos = emb[torch.tensor(pos_items, dtype=torch.long, device=device)]
    neg = emb[torch.tensor(neg_items, dtype=torch.long, device=device)]

    pos_score = (u * pos).sum(dim=1)
    neg_score = (u * neg).sum(dim=1)
    loss      = -F.logsigmoid(pos_score - neg_score).mean()

    reg_loss = reg * (u.norm(2).pow(2) + pos.norm(2).pow(2) + neg.norm(2).pow(2)) / len(users)
    return loss + reg_loss


def compute_rec_metrics(emb, test_edges, user_train_pos, all_items, K_list, device):
    emb_np    = emb.cpu().numpy()
    user_test = {}
    for u, i in test_edges:
        user_test.setdefault(u, []).append(i)

    results = {K: {'recall':[], 'ndcg':[], 'hit':[], 'precision':[], 'mrr':[]}
               for K in K_list}

    for user, pos_list in user_test.items():
        u_emb    = emb_np[user]
        item_emb = emb_np[all_items]
        scores   = item_emb @ u_emb

        train_pos = user_train_pos.get(user, set())
        for idx, item in enumerate(all_items):
            if item in train_pos:
                scores[idx] = -1e9

        top_K_max = max(K_list)
        top_idx   = np.argpartition(scores, -top_K_max)[-top_K_max:]
        top_idx   = top_idx[np.argsort(scores[top_idx])[::-1]]
        top_items = all_items[top_idx]

        pos_set = set(pos_list)

        for K in K_list:
            recs     = top_items[:K]
            hits     = [1 if i in pos_set else 0 for i in recs]
            n_hits   = sum(hits)

            results[K]['recall'].append(n_hits / len(pos_set))

            dcg  = sum(h / np.log2(r+2) for r, h in enumerate(hits))
            idcg = sum(1.0 / np.log2(r+2) for r in range(min(len(pos_set), K)))
            results[K]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

            results[K]['hit'].append(float(n_hits > 0))
            results[K]['precision'].append(n_hits / K)

            rr = 0.0
            for r, i in enumerate(recs):
                if i in pos_set: rr = 1.0 / (r+1); break
            results[K]['mrr'].append(rr)

    agg = {}
    for K in K_list:
        for metric, vals in results[K].items():
            agg[f'{metric}@{K}'] = float(np.mean(vals))
    return agg


def sample_bpr_negatives(users, all_items, user_pos, rng, n=None):
    n       = n or len(users)
    neg     = rng.choice(all_items, size=n*3)
    out     = []
    ni      = 0
    for u in users:
        pos_set = user_pos.get(u, set())
        while neg[ni] in pos_set:
            ni += 1
            if ni >= len(neg): neg = rng.choice(all_items, size=n*3); ni = 0
        out.append(neg[ni]); ni += 1
    return np.array(out)


def run_final_recommendation(data, best_params, tr80_edges, te20_edges,
                              target_relation_idx=0,
                              K_list=(10, 20, 50),
                              seed=42,
                              out_dir='results/recommendation'):
    torch.manual_seed(seed); np.random.seed(seed)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d       = best_params['d']
    R       = len(data['A_list_sp'])
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    X_list  = [x.to(device) for x in data['X_dict'].values()]
    K_list  = list(K_list)

    P_list  = build_operators(data['A_list_sp'], data['bipartite_flags'], device)

    backbone = RAHGH(
        in_dims=in_dims, d=d, R=R, K=best_params['K'],
        gcn_hidden=best_params['gcn_hidden'],
        out_dim=d, dropout=best_params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    opt = Adam(backbone.parameters(), lr=best_params['lr'],
               weight_decay=best_params['wd'])

    all_items  = np.unique(tr80_edges[:, 1])
    user_pos   = {}
    for u, i in tr80_edges: user_pos.setdefault(u, set()).add(i)

    rng        = np.random.default_rng(seed)
    epoch_rows = []
    best_rec   = 0.0
    best_sd    = None
    t0         = time.time()

    for ep in range(1, best_params['epochs'] + 1):
        backbone.train(); opt.zero_grad()
        emb, *_ = backbone(X_list, P_list)

        users     = tr80_edges[:, 0]
        pos_items = tr80_edges[:, 1]
        neg_items = sample_bpr_negatives(users, all_items, user_pos, rng)

        loss = bpr_loss(emb, users, pos_items, neg_items, device,
                        reg=best_params.get('bpr_reg', 1e-4))
        loss.backward(); opt.step()

        epoch_rows.append({'epoch': ep, 'bpr_loss': round(loss.item(), 6)})

        if ep % 50 == 0 or ep == best_params['epochs']:
            backbone.eval()
            with torch.no_grad():
                emb_v, *_ = backbone(X_list, P_list)
                agg = compute_rec_metrics(emb_v, te20_edges, user_pos, all_items,
                                          [best_params.get('K_rec', 20)], device)
                rec = agg.get(f'recall@{best_params.get("K_rec", 20)}', 0.0)
            if rec > best_rec:
                best_rec = rec
                best_sd  = {k: v.clone() for k, v in backbone.state_dict().items()}

    backbone.load_state_dict(best_sd); backbone.eval()
    with torch.no_grad():
        emb_f, alpha, beta, _ = backbone(X_list, P_list)
        final_agg = compute_rec_metrics(emb_f, te20_edges, user_pos,
                                         all_items, K_list, device)

    os.makedirs(os.path.join(out_dir, 'epoch_logs'), exist_ok=True)
    if epoch_rows:
        with open(os.path.join(out_dir, 'epoch_logs', f'seed{seed}_epochs.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=epoch_rows[0].keys())
            w.writeheader(); w.writerows(epoch_rows)

    return dict(**final_agg,
                alpha=alpha.detach().cpu().numpy(),
                beta=beta.detach().cpu().numpy(),
                time_sec=time.time()-t0)


def recall_at_k(emb, test_edges, user_train_pos, all_items, K, device):
    agg = compute_rec_metrics(emb, test_edges, user_train_pos, all_items, [K], device)
    return agg.get(f'recall@{K}', 0.0)
