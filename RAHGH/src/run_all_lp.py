"""
Standalone LP evaluation for ALL datasets using RAHGH encoder
with HGB-compatible evaluation (per-relation-type AUC + MRR).

Two modes:
  HGB mode   — uses link.dat.test as ground truth, 90/10 train/val, 5 runs
  Legacy mode — 80/10/10 split, n_seeds runs (for LastFM etc.)

Usage:
    python -m src.run_all_lp
    python -m src.run_all_lp --datasets amazon lastfm pubmed
"""

import argparse, json, os, sys, time, csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from .model.rahgh import build_encoder, build_edge_index_dict, build_node_type_indices

# ── Decoder ──────────────────────────────────────────────────────────────────────

class MLPDecoder(nn.Module):
    def __init__(self, emb_dim, hidden=None, dropout=0.3):
        super().__init__()
        h = hidden or emb_dim
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h // 2, 1),
        )
    def forward(self, emb, src, dst):
        return self.net(torch.cat([emb[src], emb[dst]], dim=1)).squeeze(-1)


class BilinearDecoder(nn.Module):
    """
    Bilinear decoder: score = emb[src] @ W @ emb[dst]^T
    W is a learned (d, d) matrix — more expressive than dot product
    but far fewer params than MLP (d² vs ~4d² for MLP).
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(emb_dim, emb_dim) * 0.01)
    def forward(self, emb, src, dst):
        s = emb[src]
        d = emb[dst]
        return torch.sum(s * (d @ self.W.T), dim=1)


# ── HGB evaluation helpers ───────────────────────────────────────────────────────

def compute_mrr_per_head(edge_list, confidence, labels):
    conf = np.asarray(confidence)
    lbl = np.asarray(labels)
    t_dict, l_dict, c_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, h_id in enumerate(edge_list[0]):
        t_dict[h_id].append(edge_list[1][i])
        l_dict[h_id].append(lbl[i])
        c_dict[h_id].append(conf[i])
    mrr_list = []
    for h_id in t_dict:
        c_arr = np.array(c_dict[h_id])
        rank = np.argsort(-c_arr)
        sorted_lbl = np.array(l_dict[h_id])[rank]
        pos_idx = np.where(sorted_lbl == 1)[0]
        if len(pos_idx) == 0:
            continue
        mrr_list.append(1.0 / (1 + pos_idx[0]))
    return float(np.mean(mrr_list)) if mrr_list else 0.0


def hgb_evaluate_rel(edge_list, confidence, labels):
    """Per-relation-type evaluation: AUC + per-head MRR."""
    auc = roc_auc_score(np.asarray(labels), np.asarray(confidence))
    mrr = compute_mrr_per_head(edge_list, confidence, labels)
    return {'auc': auc, 'mrr': mrr}


def compute_hits_at_k(emb, decoder, test_edges, all_dst,
                      n_neg=99, ks=(1, 3, 10), device='cpu',
                      all_positives=None):
    n_test = len(test_edges)
    src_t = torch.tensor(test_edges[:, 0], dtype=torch.long, device=device)
    dst_t = torch.tensor(test_edges[:, 1], dtype=torch.long, device=device)
    all_dst_t = torch.tensor(all_dst, dtype=torch.long, device=device)
    N = max(int(test_edges[:, 0].max()), int(all_dst.max())) + 1

    pos_pairs = set(map(tuple, test_edges))
    if all_positives is not None:
        pos_pairs |= set(map(tuple, all_positives))
    pos_enc = torch.tensor([s * N + d for s, d in pos_pairs],
                           dtype=torch.long, device=device)

    decoder.eval()
    with torch.no_grad():
        pos_scores = decoder(emb, src_t, dst_t)
        n_cands = n_neg * 3
        neg_dst = all_dst_t[torch.randint(0, len(all_dst),
                                          size=(n_test, n_cands), device=device)]
        mask = neg_dst != dst_t.unsqueeze(1)
        pair_enc = src_t.unsqueeze(1) * N + neg_dst
        mask &= ~torch.isin(pair_enc, pos_enc)
        order = mask.to(torch.int64).argsort(dim=1, descending=True)
        valid_dst = torch.gather(neg_dst, 1, order)[:, :n_neg]
        src_exp = src_t.unsqueeze(1).expand(-1, n_neg).reshape(-1)
        dst_exp = valid_dst.reshape(-1)
        neg_scores = decoder(emb, src_exp, dst_exp).reshape(n_test, n_neg)
        ranks = 1 + torch.sum(neg_scores > pos_scores.unsqueeze(1), dim=1)
    ranks_np = ranks.cpu().numpy()
    result = {f'hits@{k}': float(np.mean(ranks_np <= k)) for k in ks}
    result['mrr'] = float(np.mean(1.0 / ranks_np))
    return result


# ── Edge masking ─────────────────────────────────────────────────────────────────

def build_masked_edge_index(data, train_edges, target_rel_idx, device,
                             rev_rel_idx=None):
    import scipy.sparse as sp
    N = data['N']
    tr_r, tr_c = train_edges[:, 0], train_edges[:, 1]
    A_train = sp.coo_matrix(
        (np.ones(len(tr_r)), (tr_r, tr_c)), shape=(N, N)).tocsr()
    rel_names = data.get('relation_names',
                         [f'rel_{i}' for i in range(len(data['A_list_sp']))])
    indices_to_mask = {target_rel_idx}
    if rev_rel_idx is not None:
        indices_to_mask.add(rev_rel_idx)
    # Also detect automatic reverse (rname like paper->author, next is author->paper)
    if '→' in rel_names[target_rel_idx]:
        parts = rel_names[target_rel_idx].split('→')
        rev_str = f'{parts[1]}→{parts[0]}'
        for i, rn in enumerate(rel_names):
            if rn == rev_str:
                indices_to_mask.add(i)
    edge_dict = {}
    for i, (A_sp, rname) in enumerate(zip(data['A_list_sp'], rel_names)):
        if i in indices_to_mask:
            if i == target_rel_idx:
                A_use = A_train
            else:
                # Reverse relation: build from training edges reversed
                A_use = sp.coo_matrix(
                    (np.ones(len(tr_c)), (tr_c, tr_r)), shape=(N, N)).tocsr()
        else:
            A_use = A_sp
        A_coo = A_use.tocoo()
        ei = np.vstack([A_coo.row, A_coo.col])
        edge_dict[rname] = torch.tensor(ei, dtype=torch.long, device=device)
    return edge_dict


# ── HGB-compatible LP training/eval ─────────────────────────────────────────────

def _sample_neg_per_head(pos_edges, all_pos_set, node_type_ids, seed):
    """For each (src, dst) in pos_edges, sample one neg from same dst type range."""
    if len(pos_edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    dst_pool = np.array(node_type_ids, dtype=np.int64)
    if len(dst_pool) == 0:
        return np.empty((0, 2), dtype=np.int64)
    rng = np.random.default_rng(seed)
    negs = []
    for s, d in pos_edges:
        for _ in range(100):
            nd = int(rng.choice(dst_pool))
            if (int(s), nd) not in all_pos_set:
                negs.append((int(s), nd))
                break
    return np.array(negs, dtype=np.int64) if negs else np.empty((0, 2), dtype=np.int64)


def run_lp_hgb(data, dataset_name, params, device, out_dir, n_runs=5):
    """
    HGB-compatible LP evaluation.
    - Trains on 90/10 split of link.dat edges
    - Tests on link.dat.test (from lp_test_edges) with per-relation-type AUC + MRR
    - 5 runs with different train/val splits
    """
    import scipy.sparse as sp
    node_type_indices_map = build_node_type_indices(data)
    types_list = list(data['X_dict'].keys())

    # Load test edges from lp_test_edges
    lp_test = data.get('lp_test_edges', None)
    has_hgb_test = lp_test is not None and len(lp_test) > 0

    # Determine target relation index from test relation type
    rel_names = data['relation_names']
    if has_hgb_test:
        # Use the first test relation's name to find matching training relation
        test_rname = next(iter(lp_test.keys()))
        src_t, dst_t = lp_test[test_rname]['src_type'], lp_test[test_rname]['dst_type']
        train_rel_candidates = [
            (i, rn) for i, rn in enumerate(rel_names)
            if data.get('relation_info', {}).get(rn, ('', '')) == (src_t, dst_t)
        ]
        if not train_rel_candidates:
            # Fallback: use target_relation_idx
            rel_idx = data.get('target_relation_idx', 0)
            target_rname = rel_names[rel_idx]
        else:
            rel_idx = train_rel_candidates[0][0]
            target_rname = train_rel_candidates[0][1]
            print(f'  Training on {target_rname.replace(chr(8594), "->")} (idx={rel_idx}) for test rel {test_rname.replace(chr(8594), "->")}')
    else:
        rel_idx = data.get('target_relation_idx', 0)
        target_rname = rel_names[rel_idx] if rel_idx < len(rel_names) else rel_names[0]

    # Extract all training edges from A_list_sp for the target relation
    A = data['A_list_sp'][rel_idx].tocoo()
    all_link_edges = np.column_stack([A.row, A.col])

    all_results = []
    for run in range(n_runs):
        seed_base = run * 10

        # ── Train/val split ─────────────────────────────────────────────────
        rng = np.random.default_rng(seed_base)
        idx = rng.permutation(len(all_link_edges))
        n_all = len(all_link_edges)
        if has_hgb_test:
            # HGB: 90/10 train/val, test on link.dat.test
            n_tr = int(0.9 * n_all)
            tr_edges = all_link_edges[idx[:n_tr]]
            va_edges = all_link_edges[idx[n_tr:]]
        else:
            # No link.dat.test: 80/10/10 split
            n_tr = int(0.8 * n_all)
            n_va = int(0.1 * n_all)
            tr_edges = all_link_edges[idx[:n_tr]]
            va_edges = all_link_edges[idx[n_tr:n_tr + n_va]]
            te_edges_fallback = all_link_edges[idx[n_tr + n_va:]]

        # ── Negative sampling for validation ─────────────────────────────────
        all_pos_train_set = set(map(tuple, all_link_edges))
        if has_hgb_test:
            target_dst_type = lp_test[next(iter(lp_test.keys()))]['dst_type']
        else:
            # Infer dst_type from relation_info
            rel_info = data.get('relation_info', {})
            rn = rel_names[rel_idx] if rel_idx < len(rel_names) else rel_names[0]
            src_type, target_dst_type = rel_info.get(rn, (types_list[0], types_list[-1]))
        dst_type_ids = node_type_indices_map.get(target_dst_type, node_type_indices_map[types_list[-1]])
        va_neg = _sample_neg_per_head(va_edges, all_pos_train_set,
                                       dst_type_ids, seed_base + 1)

        def to_tensors(pos, neg):
            if len(neg) == 0:
                e = pos
                l = np.ones(len(pos), dtype=np.float32)
            else:
                e = np.concatenate([pos, neg], 0)
                l = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
            return (torch.tensor(e[:, 0], dtype=torch.long, device=device),
                    torch.tensor(e[:, 1], dtype=torch.long, device=device),
                    torch.tensor(l, device=device))

        va_s, va_d, va_l = to_tensors(va_edges, va_neg)
        tr_s = torch.tensor(tr_edges[:, 0], dtype=torch.long, device=device)
        tr_d = torch.tensor(tr_edges[:, 1], dtype=torch.long, device=device)

        # Pool of dst IDs of the target type (for in-batch negative sampling)
        all_dst_t = dst_type_ids.to(device=device)

        # ── Train ─────────────────────────────────────────────────────────────
        torch.manual_seed(seed_base)
        np.random.seed(seed_base)

        # Also mask the reverse relation if it exists
        rev_rel_idx = None
        if has_hgb_test:
            rev_rname = f'{target_rname.split(chr(8594))[1]}{chr(8594)}{target_rname.split(chr(8594))[0]}' \
                if chr(8594) in target_rname else None
            if rev_rname and rev_rname in rel_names:
                rev_rel_idx = rel_names.index(rev_rname)

        edge_index_dict = build_masked_edge_index(data, tr_edges, rel_idx, device,
                                                   rev_rel_idx=rev_rel_idx)
        node_type_indices = {k: v.to(device)
                             for k, v in node_type_indices_map.items()}
        from .data.lastfm_loader import rebuild_user_features
        x_dict = rebuild_user_features(data, tr_edges, device)

        model = build_encoder(data, params, device, head=params.get('head', 'none'))

        # Choose decoder
        dec_type = params.get('decoder', 'mlp')
        if dec_type == 'dot':
            decoder = None  # no params, use dot product
            opt = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
        elif dec_type == 'bilinear':
            decoder = BilinearDecoder(params['d']).to(device)
            opt = Adam(list(model.parameters()) + list(decoder.parameters()),
                       lr=params['lr'], weight_decay=params['wd'])
        else:
            decoder = MLPDecoder(params['d'], dropout=params.get('decoder_dropout', 0.3)).to(device)
            opt = Adam(list(model.parameters()) + list(decoder.parameters()),
                       lr=params['lr'], weight_decay=params['wd'])

        n_neg = params.get('n_neg', 1)
        edge_drop = params.get('edge_drop', 0.0)

        def score_edges(emb, src, dst):
            if dec_type == 'dot':
                return torch.sum(emb[src] * emb[dst], dim=1)
            return decoder(emb, src, dst)

        best_auc = 0.0
        best_sd_h = best_sd_d = None
        stall = 0
        patience = params.get('patience', 30)
        val_every = 5
        t0 = time.time()

        pbar = tqdm(range(1, params['epochs'] + 1),
                    desc=f"{dataset_name} run={run}", leave=False)
        for ep in pbar:
            model.train(); opt.zero_grad()
            if decoder is not None:
                decoder.train()
            # Sample n_neg negatives per positive
            neg_dst = all_dst_t[torch.randint(0, len(all_dst_t),
                                              size=(len(tr_edges) * n_neg,), device=device)]
            tr_s_rep = tr_s.repeat(n_neg)
            with torch.amp.autocast(device_type=device.type,
                                    enabled=device.type == "cuda"):
                if edge_drop > 0 and model.training:
                    # Edge dropout on homogeneous adjacency
                    pass  # implemented via RAHGH if supported
                emb, *_ = model(x_dict, edge_index_dict, node_type_indices)
                pos = score_edges(emb, tr_s, tr_d)
                neg = score_edges(emb, tr_s_rep, neg_dst)
                loss = -F.logsigmoid(pos.unsqueeze(1).expand(-1, n_neg).reshape(-1) - neg).mean()
            loss.backward()
            params_list = list(model.parameters())
            if decoder is not None:
                params_list += list(decoder.parameters())
            torch.nn.utils.clip_grad_norm_(params_list, 5.0)
            opt.step()

            if ep % val_every == 0 or ep == 1:
                model.eval()
                if decoder is not None:
                    decoder.eval()
                with torch.no_grad():
                    emb_v, *_ = model(x_dict, edge_index_dict, node_type_indices)
                    p = torch.sigmoid(score_edges(emb_v, va_s, va_d)).cpu().numpy()
                    auc = roc_auc_score(va_l.cpu().numpy(), p)
                pbar.set_postfix(loss=f"{loss.item():.3f}", auc=f"{auc:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    best_sd_h = {k: v.clone() for k, v in model.state_dict().items()}
                    if decoder is not None:
                        best_sd_d = {k: v.clone() for k, v in decoder.state_dict().items()}
                    stall = 0
                else:
                    stall += val_every
                    if stall >= patience:
                        break

        model.load_state_dict(best_sd_h)
        if decoder is not None:
            decoder.load_state_dict(best_sd_d)
            decoder.eval()
        model.eval()

        with torch.no_grad():
            emb_te, *_ = model(x_dict, edge_index_dict, node_type_indices)

        # ── HGB test evaluation ──────────────────────────────────────────────
        if has_hgb_test:
            # Per-relation-type evaluation on link.dat.test
            rel_metrics = {}
            all_test_src, all_test_dst, all_test_conf, all_test_lbl = [], [], [], []
            neg_seed = seed_base + 100

            for rname, tedata in lp_test.items():
                te_pos = tedata['edges']
                dst_type = tedata['dst_type']
                dst_ids = node_type_indices_map.get(dst_type, [])
                te_neg = _sample_neg_per_head(te_pos, all_pos_train_set,
                                               dst_ids, neg_seed)
                neg_seed += 1

                # Predict
                te_s_t = torch.tensor(te_pos[:, 0], dtype=torch.long, device=device)
                te_d_t = torch.tensor(te_pos[:, 1], dtype=torch.long, device=device)
                with torch.no_grad():
                    pos_conf = torch.sigmoid(score_edges(emb_te, te_s_t, te_d_t)).cpu().numpy()
                neg_conf = np.array([])
                if len(te_neg) > 0:
                    te_ns_t = torch.tensor(te_neg[:, 0], dtype=torch.long, device=device)
                    te_nd_t = torch.tensor(te_neg[:, 1], dtype=torch.long, device=device)
                    with torch.no_grad():
                        neg_conf = torch.sigmoid(score_edges(emb_te, te_ns_t, te_nd_t)).cpu().numpy()

                edge_list = np.concatenate([te_pos, te_neg], 0) if len(te_neg) > 0 else te_pos
                conf = np.concatenate([pos_conf, neg_conf]) if len(te_neg) > 0 else pos_conf
                lbl = np.concatenate([np.ones(len(te_pos)), np.zeros(len(te_neg))])

                m = hgb_evaluate_rel(edge_list.T, conf, lbl)
                rel_metrics[rname] = m

                all_test_src.append(edge_list[:, 0])
                all_test_dst.append(edge_list[:, 1])
                all_test_conf.append(conf)
                all_test_lbl.append(lbl)

            # Average across relation types
            auc_mean = float(np.mean([m['auc'] for m in rel_metrics.values()]))
            mrr_mean = float(np.mean([m['mrr'] for m in rel_metrics.values()]))

            r = dict(dataset=dataset_name, run=run, auc=round(auc_mean, 4),
                     mrr=round(mrr_mean, 4),
                     auc_per_rel=json.dumps({k: round(v['auc'], 4)
                                             for k, v in rel_metrics.items()}),
                     mrr_per_rel=json.dumps({k: round(v['mrr'], 4)
                                             for k, v in rel_metrics.items()}),
                     time_sec=round(time.time() - t0, 2),
                     **{f'hp_{k}': v for k, v in params.items()})
        else:
            # Fallback: use our 10% holdout from link.dat
            te_neg = _sample_neg_per_head(te_edges_fallback,
                                           all_pos_train_set,
                                           dst_type_ids,
                                           seed_base + 200)
            te_edges = te_edges_fallback
            te_s_t = torch.tensor(te_edges[:, 0], dtype=torch.long, device=device)
            te_d_t = torch.tensor(te_edges[:, 1], dtype=torch.long, device=device)
            with torch.no_grad():
                pos_conf = torch.sigmoid(score_edges(emb_te, te_s_t, te_d_t)).cpu().numpy()
            neg_conf = np.array([])
            if len(te_neg) > 0:
                te_ns_t = torch.tensor(te_neg[:, 0], dtype=torch.long, device=device)
                te_nd_t = torch.tensor(te_neg[:, 1], dtype=torch.long, device=device)
                with torch.no_grad():
                    neg_conf = torch.sigmoid(score_edges(emb_te, te_ns_t, te_nd_t)).cpu().numpy()
            te_all = np.concatenate([te_edges, te_neg], 0) if len(te_neg) > 0 else te_edges
            conf = np.concatenate([pos_conf, neg_conf]) if len(te_neg) > 0 else pos_conf
            lbl = np.concatenate([np.ones(len(te_edges)), np.zeros(len(te_neg))])
            m = hgb_evaluate_rel(te_all.T, conf, lbl)
            r = dict(dataset=dataset_name, run=run, auc=round(m['auc'], 4),
                     mrr=round(m['mrr'], 4),
                     time_sec=round(time.time() - t0, 2),
                     **{f'hp_{k}': v for k, v in params.items()})

        all_results.append(r)
        print(f"  {dataset_name} run={run}: AUC={r['auc']:.4f}, MRR={r['mrr']:.4f}")

    return all_results


# ── Main ───────────────────────────────────────────────────────────────────────

ALLOWED_DATASETS = ['amazon', 'amazon_ini', 'lastfm', 'pubmed', 'pubmed_ini']

LP_PARAMS = {
    'd'        : 128,
    'K'        : 5,
    'dropout'  : 0.3,
    'lr'       : 0.001,
    'wd'       : 1e-3,
    'epochs'   : 400,
    'patience' : 30,
    'head'     : 'none',
    'n_neg'    : 5,
    'decoder'  : 'bilinear',
}


def collect_results(all_results):
    if not all_results:
        return {}
    by_ds = defaultdict(list)
    for r in all_results:
        by_ds[r['dataset']].append(r)
    summary = {}
    for ds, rows in sorted(by_ds.items()):
        aucs = [r['auc'] for r in rows]
        mrrs = [r['mrr'] for r in rows]
        s = {'auc': f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
             'mrr': f"{np.mean(mrrs):.4f} ± {np.std(mrrs):.4f}"}
        if 'auc_per_rel' in rows[0]:
            rel_names = list(json.loads(rows[0]['auc_per_rel']).keys())
            for rn in rel_names:
                rauc = [json.loads(r['auc_per_rel'])[rn] for r in rows]
                rmrr = [json.loads(r['mrr_per_rel'])[rn] for r in rows]
                s[f'auc_{rn}'] = f"{np.mean(rauc):.4f} ± {np.std(rauc):.4f}"
                s[f'mrr_{rn}'] = f"{np.mean(rmrr):.4f} ± {np.std(rmrr):.4f}"
        summary[ds] = s
    return summary


def write_results(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'lp_all_results.csv')
    with open(path, 'w', newline='') as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
    print(f"\nResults saved -> {path}")

    summary = collect_results(all_results)
    spath = os.path.join(out_dir, 'lp_summary.json')
    with open(spath, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved -> {spath}")

    print(f"\n{'='*70}")
    print(f"  LP Results — All Datasets")
    print(f"{'='*70}")
    print(f"  {'Dataset':<15} {'AUC':<22} {'MRR':<22}")
    print(f"  {'-'*60}")
    for ds, s in sorted(summary.items()):
        print(f"  {ds:<15} {s['auc']:<22} {s['mrr']:<22}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['amazon_ini', 'lastfm', 'pubmed_ini'],
                        choices=ALLOWED_DATASETS)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--out', type=str, default='results/lp_all')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    all_results = []
    for dname in args.datasets:
        print(f"\n{'='*60}")
        print(f"  Loading {dname}...")
        t0 = time.time()
        from .train import _get_loader
        data = _get_loader(dname)
        data['name'] = dname
        has_test = data.get('lp_test_edges') is not None
        print(f"  Loaded in {time.time()-t0:.1f}s — N={data['N']}, "
              f"types={list(data['X_dict'].keys())}, "
              f"has_link.dat.test={has_test}")
        params = LP_PARAMS.copy()
        results = run_lp_hgb(data, dname, params, device, args.out,
                             n_runs=args.runs)
        all_results.extend(results)

    write_results(all_results, args.out)


if __name__ == '__main__':
    main()
