import json
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
from collections import defaultdict


def load_lastfm(root="data/raw"):
    """
    Load LastFM dataset.
    Supports:
      - Legacy format: user_artist.dat, artist_tag.dat, user_user*.dat, artist_artist*.dat
      - HGB unified format: node.dat, link.dat, info.dat
    """
    candidates = [
        Path(root) / 'LastFM',         Path(root) / 'LastFM' / 'LastFM',
        Path(root) / 'LastFM_ini',     Path(root) / 'LastFM_ini' / 'LastFM_ini',
        Path(root) / 'LastFM_magnn',   Path(root) / 'LastFM_magnn' / 'LastFM_magnn',
    ]
    folder = None
    for c in candidates:
        if c.exists():
            folder = c; break
    assert folder is not None, f"LastFM folder not found under {root}"

    # ── Detect format ────────────────────────────────────────────────────────
    if (folder / 'node.dat').exists():
        return _load_lastfm_hgb(folder)
    else:
        return _load_lastfm_legacy(folder)


def _load_lastfm_legacy(folder: Path) -> dict:
    """
    Load LastFM legacy format:
      user_artist.dat       : user_id \t artist_id \t weight
      artist_tag.dat        : artist_id \t tag_id
      user_user(knn).dat    : user_id \t user_id \t similarity
      user_user(original).dat : user_id \t user_id
      artist_artist(knn).dat: artist_id \t artist_id \t similarity
      train_val_test_idx.npz: pre-split indices (train_idx, val_idx, test_idx)

    All IDs are 1-indexed.  Layout: users [0, Nu), artists [Nu, Nu+Na), tags [Nu+Na, N)
    """
    # ── Parse user_artist.dat ────────────────────────────────────────────────
    ua_path = folder / 'user_artist.dat'
    assert ua_path.exists(), f"Missing user_artist.dat in {folder}"

    ua_data = []
    users_set = set(); artists_set = set()
    with open(ua_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                u = int(parts[0]); a = int(parts[1]); w = float(parts[2])
                users_set.add(u); artists_set.add(a)
                ua_data.append((u, a, w))

    # ── Parse artist_tag.dat ─────────────────────────────────────────────────
    at_path = folder / 'artist_tag.dat'
    at_data = []
    tags_set = set()
    if at_path.exists():
        with open(at_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    a = int(parts[0]); t = int(parts[1])
                    artists_set.add(a); tags_set.add(t)
                    at_data.append((a, t))

    # ── Parse user_user edges ────────────────────────────────────────────────
    uu_data = []
    for fname in ['user_user(original).dat', 'user_user(knn).dat']:
        fp = folder / fname
        if fp.exists():
            with open(fp) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        u1, u2 = int(parts[0]), int(parts[1])
                        w = float(parts[2]) if len(parts) >= 3 else 1.0
                        uu_data.append((u1, u2, w))

    # ── Parse artist_artist edges ────────────────────────────────────────────
    aa_data = []
    for fname in ['artist_artist(knn).dat']:
        fp = folder / fname
        if fp.exists():
            with open(fp) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        a1, a2, w = int(parts[0]), int(parts[1]), float(parts[2])
                        aa_data.append((a1, a2, w))

    # ── Build sorted lists ───────────────────────────────────────────────────
    user_list   = sorted(users_set)
    artist_list = sorted(artists_set)
    tag_list    = sorted(tags_set)

    # 1-indexed → 0-indexed local IDs
    u2l = {uid: i for i, uid in enumerate(user_list)}
    a2l = {aid: i for i, aid in enumerate(artist_list)}
    t2l = {tid: i for i, tid in enumerate(tag_list)}

    Nu = len(user_list)
    Na = len(artist_list)
    Nt = len(tag_list)
    N  = Nu + Na + Nt  # contiguous layout: users | artists | tags

    # ── Build adjacency matrices ─────────────────────────────────────────────
    def build_mat(rows, cols, vals=None):
        if vals is None:
            vals = np.ones(len(rows), dtype=np.float32)
        return sp.coo_matrix(
            (vals, (rows, cols)), shape=(N, N)).tocsr()

    A_list_sp = []
    relation_names = []
    bipartite_flags = []
    ua_weights = {}

    # UA: user → artist
    ua_rows = [u2l[u]          for u, a, w in ua_data]
    ua_cols = [a2l[a] + Nu     for u, a, w in ua_data]
    ua_vals = np.array([w       for u, a, w in ua_data], dtype=np.float32)
    UA = build_mat(ua_rows, ua_cols, ua_vals)
    AU = UA.T.tocsr()
    A_list_sp.extend([UA, AU])
    relation_names.extend(['user→artist', 'artist→user'])
    bipartite_flags.extend([True, True])

    for (u, a, w) in ua_data:
        ua_weights[(u, a)] = w

    # AT: artist → tag
    if at_data:
        at_rows = [a2l[a] + Nu     for a, t in at_data]
        at_cols = [t2l[t] + Nu + Na for a, t in at_data]
        AT = build_mat(at_rows, at_cols)
        TA = AT.T.tocsr()
        A_list_sp.extend([AT, TA])
        relation_names.extend(['artist→tag', 'tag→artist'])
        bipartite_flags.extend([True, True])

    # UU: user → user
    if uu_data:
        uu_rows = [u2l[u1]          for u1, u2, w in uu_data]
        uu_cols = [u2l[u2]          for u1, u2, w in uu_data]
        uu_vals = np.array([w       for u1, u2, w in uu_data], dtype=np.float32)
        UU = build_mat(uu_rows, uu_cols, uu_vals)
        UU = UU.maximum(UU.T.tocsr()).tocsr()  # symmetrize
        A_list_sp.append(UU)
        relation_names.append('user→user')
        bipartite_flags.append(False)

    # AA: artist → artist
    if aa_data:
        aa_rows = [a2l[a1] + Nu     for a1, a2, w in aa_data]
        aa_cols = [a2l[a2] + Nu     for a1, a2, w in aa_data]
        aa_vals = np.array([w       for a1, a2, w in aa_data], dtype=np.float32)
        AA = build_mat(aa_rows, aa_cols, aa_vals)
        AA = AA.maximum(AA.T.tocsr()).tocsr()
        A_list_sp.append(AA)
        relation_names.append('artist→artist')
        bipartite_flags.append(False)

    # ── Features ─────────────────────────────────────────────────────────────
    # Tag features: random (no tag attributes in legacy format)
    tag_dim = min(max(Nt, 1), 64)
    X_tag = torch.randn(Nt, tag_dim, dtype=torch.float32) if Nt > 0 else torch.zeros(1, tag_dim)

    # Artist features: mean of connected tag features
    X_artist = torch.zeros(Na, tag_dim, dtype=torch.float32)
    artist_count = torch.zeros(Na, dtype=torch.float32)
    for a, t in at_data:
        li_a, li_t = a2l.get(a), t2l.get(t)
        if li_a is not None and li_t is not None:
            X_artist[li_a] += X_tag[li_t]
            artist_count[li_a] += 1.0
    mask = artist_count > 0
    if mask.any():
        X_artist[mask] = X_artist[mask] / artist_count[mask, None].clamp(min=1.0)

    # User features: identity (avoid label leakage from UA edges)
    user_dim = min(max(Nu, 1), 64)
    X_user = torch.eye(Nu, user_dim, dtype=torch.float32)

    X_dict = {}
    if Nu > 0: X_dict['user'] = X_user
    if Na > 0: X_dict['artist'] = X_artist
    if Nt > 0: X_dict['tag'] = X_tag

    # ── Labels & split indices ──────────────────────────────────────────────
    labels = torch.zeros(max(Nu, 1), dtype=torch.long)

    # Load pre-split indices if available
    npz_path = folder / 'train_val_test_idx.npz'
    train_indices = None
    val_indices = None
    test_indices = None
    if npz_path.exists():
        npz = np.load(npz_path)
        train_indices = torch.tensor(npz['train_idx'], dtype=torch.long)
        val_indices   = torch.tensor(npz['val_idx'],   dtype=torch.long)
        test_indices  = torch.tensor(npz['test_idx'],  dtype=torch.long)

    target_relation_idx = 0  # user→artist is first in A_list_sp

    # ── Build global node type indices ────────────────────────────────────────
    node_type_global_ids = {
        'user':   list(range(Nu)),
        'artist': list(range(Nu, Nu + Na)),
        'tag':    list(range(Nu + Na, N)),
    }

    return dict(
        A_list_sp=A_list_sp,
        relation_names=relation_names,
        bipartite_flags=bipartite_flags,
        X_dict=X_dict,
        labels=labels,
        Nu=Nu, Na=Na, Nt=Nt, N=N,
        target_type='user',
        target_size=Nu,
        n_classes=0,
        target_relation_idx=target_relation_idx,
        ua_weights=ua_weights,
        node_type_global_ids=node_type_global_ids,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        relation_info={
            'user→artist':   ('user', 'artist'),
            'artist→user':   ('artist', 'user'),
            'artist→tag':    ('artist', 'tag'),
            'tag→artist':    ('tag', 'artist'),
            'user→user':     ('user', 'user'),
            'artist→artist': ('artist', 'artist'),
        },
    )


def _load_lastfm_hgb(folder: Path) -> dict:
    """Load LastFM from HGB unified format (node.dat, link.dat, info.dat)."""
    assert (folder / 'link.dat').exists(), f"link.dat missing in {folder}"

    info = json.loads((folder / 'info.dat').read_text())
    type_names = {int(k): v for k, v in info['node.dat'].items()}
    rel_map = info['link.dat']

    edges_by_rel = defaultdict(list)
    with open(folder / 'link.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                src, dst, rid, w = int(parts[0]), int(parts[1]), parts[2], float(parts[3])
                edges_by_rel[rid].append((src, dst, w))

    type_id_sets = defaultdict(set)
    for rid, edges in edges_by_rel.items():
        if rid not in rel_map:
            continue
        ri = rel_map[rid]
        src_type = int(ri['start']); dst_type = int(ri['end'])
        for s, d, w in edges:
            type_id_sets[src_type].add(s)
            type_id_sets[dst_type].add(d)

    type_global_ids = {}
    for tid in sorted(type_id_sets.keys()):
        tname = type_names[tid]
        type_global_ids[tname] = sorted(type_id_sets[tid])

    Nu = len(type_global_ids.get('user', []))
    Na = len(type_global_ids.get('artist', []))
    Nt = len(type_global_ids.get('tag', []))
    N = max(max(ids) for ids in type_id_sets.values()) + 1 if type_id_sets else 0

    global_to_local = {}
    for tname, gids in type_global_ids.items():
        for li, gid in enumerate(gids):
            global_to_local[(tname, gid)] = li

    def build_sp(rows, cols, N):
        return sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(N, N)).tocsr()

    A_list_sp = []; relation_names = []; bipartite_flags = []

    for rid in sorted(edges_by_rel.keys(), key=int):
        edges = edges_by_rel[rid]
        if rid not in rel_map:
            continue
        ri = rel_map[rid]
        st = type_names[int(ri['start'])]; dt = type_names[int(ri['end'])]
        rows = [s for s, d, w in edges]
        cols = [d for s, d, w in edges]

        if st == 'user' and dt == 'artist':
            UA = build_sp(rows, cols, N); AU = UA.T.tocsr()
            A_list_sp.extend([UA, AU])
            relation_names.extend(['user→artist', 'artist→user'])
            bipartite_flags.extend([True, True])
        elif st == 'user' and dt == 'user':
            UU = build_sp(rows, cols, N)
            UU = UU.maximum(UU.T.tocsr()).tocsr()
            A_list_sp.append(UU)
            relation_names.append('user→user')
            bipartite_flags.append(False)
        elif st == 'artist' and dt == 'tag':
            AT = build_sp(rows, cols, N); TA = AT.T.tocsr()
            A_list_sp.extend([AT, TA])
            relation_names.extend(['artist→tag', 'tag→artist'])
            bipartite_flags.extend([True, True])
        else:
            A = build_sp(rows, cols, N)
            A_list_sp.append(A)
            relation_names.append(ri.get('meaning', f'rel_{rid}'))
            bipartite_flags.append(st != dt)

    tag_dim = min(max(Nt, 1), 64)
    X_tag = torch.randn(Nt, tag_dim, dtype=torch.float32) if Nt > 0 else torch.zeros(1, tag_dim)

    X_artist = torch.zeros(Na, tag_dim, dtype=torch.float32)
    artist_count = torch.zeros(Na, dtype=torch.float32)
    for rid, edges in edges_by_rel.items():
        if rid not in rel_map: continue
        ri = rel_map[rid]
        if type_names[int(ri['start'])] != 'artist' or type_names[int(ri['end'])] != 'tag':
            continue
        for s, d, w in edges:
            li_a = global_to_local.get(('artist', s))
            li_t = global_to_local.get(('tag', d))
            if li_a is not None and li_t is not None:
                X_artist[li_a] += X_tag[li_t]
                artist_count[li_a] += 1.0
        break
    mask = artist_count > 0
    if mask.any():
        X_artist[mask] = X_artist[mask] / artist_count[mask, None].clamp(min=1.0)

    user_dim = min(max(Nu, 1), 64)
    X_user = torch.eye(Nu, user_dim, dtype=torch.float32)

    X_dict = {}
    if Nu > 0: X_dict['user'] = X_user
    if Na > 0: X_dict['artist'] = X_artist
    if Nt > 0: X_dict['tag'] = X_tag

    ua_weights = {}
    for rid, edges in edges_by_rel.items():
        if rid not in rel_map: continue
        ri = rel_map[rid]
        if type_names[int(ri['start'])] != 'user' or type_names[int(ri['end'])] != 'artist':
            continue
        for s, d, w in edges:
            ua_weights[(s, d)] = w
        break

    labels = torch.zeros(max(Nu, 1), dtype=torch.long)
    target_relation_idx = next(
        (i for i, r in enumerate(relation_names) if r == 'user→artist'), 0)

    return dict(
        A_list_sp=A_list_sp,
        relation_names=relation_names,
        bipartite_flags=bipartite_flags,
        X_dict=X_dict,
        labels=labels,
        Nu=Nu, Na=Na, Nt=Nt, N=N,
        target_type='user',
        target_size=Nu,
        n_classes=0,
        target_relation_idx=target_relation_idx,
        ua_weights=ua_weights,
        node_type_global_ids=type_global_ids,
        relation_info={
            'user→artist':   ('user', 'artist'),
            'artist→user':   ('artist', 'user'),
            'artist→tag':    ('artist', 'tag'),
            'tag→artist':    ('tag', 'artist'),
            'user→user':     ('user', 'user'),
        },
    )


def rebuild_user_features(data, tr_edges, device):
    """
    Rebuild X_user using ONLY training user-artist edges.
    This prevents label leakage from test/val edges into node features.

    Only effective for datasets with 'ua_weights' in data dict.
    Returns updated x_dict.
    """
    if 'ua_weights' not in data or 'user' not in data['X_dict']:
        return {k: v.to(device) for k, v in data['X_dict'].items()}

    weights = data['ua_weights']
    user_g2l = {gid: li for li, gid in enumerate(data['node_type_global_ids']['user'])}
    artist_g2l = {gid: li for li, gid in enumerate(data['node_type_global_ids']['artist'])}
    x_dict = {k: v.to(device) for k, v in data['X_dict'].items()}
    X_artist = x_dict['artist']
    d = X_artist.shape[1]

    X_user = torch.zeros(data['Nu'], d, dtype=torch.float32, device=device)
    weight_sum = torch.zeros(data['Nu'], dtype=torch.float32, device=device)

    for u, a in tr_edges:
        ui = user_g2l.get(int(u))
        ai = artist_g2l.get(int(a))
        if ui is None or ai is None:
            continue
        w = weights.get((int(u), int(a)), 1.0)
        X_user[ui] += X_artist[ai] * w
        weight_sum[ui] += w

    weight_sum = weight_sum.clamp(min=1.0)
    X_user = X_user / weight_sum.unsqueeze(1)

    x_dict['user'] = X_user
    return x_dict
