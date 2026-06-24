import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path


def load_lastfm(root="data/raw/LastFM"):
    folder = Path(root)

    # ── User-Artist edges ──────────────────────────────────────────────────
    users, artists = set(), set()
    ua_edges = []
    with open(folder / 'user_artist.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                u, a = parts[0], parts[1]
                users.add(u); artists.add(a)
                ua_edges.append((u, a))

    user_list   = sorted(users, key=int)
    artist_list = sorted(artists, key=int)
    u2i = {u: i for i, u in enumerate(user_list)}
    a2i = {a: i for i, a in enumerate(artist_list)}
    Nu = len(user_list)
    Na = len(artist_list)

    # ── Artist-Tag edges (deduplicated) ────────────────────────────────────
    tags = set()
    at_pairs = set()
    with open(folder / 'artist_tag.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                a, t = parts[0], parts[1]
                if a in a2i and (a, t) not in at_pairs:
                    at_pairs.add((a, t))
                    tags.add(t)
    at_edges = list(at_pairs)

    tag_list = sorted(tags, key=int)
    t2i = {t: i for i, t in enumerate(tag_list)}
    Nt = len(tag_list)
    N = Nu + Na + Nt

    # ── User-User (original) edges ─────────────────────────────────────────
    uu_edges = []
    with open(folder / 'user_user(original).dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                u1, u2 = parts[0], parts[1]
                if u1 in u2i and u2 in u2i:
                    uu_edges.append((u1, u2))

    # ── Artist-Artist (KNN) edges ──────────────────────────────────────────
    aa_edges = []
    with open(folder / 'artist_artist(knn).dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                a1, a2 = parts[0], parts[1]
                if a1 in a2i and a2 in a2i:
                    aa_edges.append((a1, a2))

    # ── Build adjacency matrices ───────────────────────────────────────────
    def build_coo(rows, cols):
        return sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(N, N)).tocsr()

    ua_r = np.array([u2i[u] for u, a in ua_edges], dtype=np.int64)
    ua_c = np.array([a2i[a] + Nu for u, a in ua_edges], dtype=np.int64)
    UA = build_coo(ua_r, ua_c)
    AU = UA.T.tocsr()

    at_r = np.array([a2i[a] + Nu for a, t in at_edges], dtype=np.int64)
    at_c = np.array([t2i[t] + Nu + Na for a, t in at_edges], dtype=np.int64)
    AT = build_coo(at_r, at_c)
    TA = AT.T.tocsr()

    uu_r = np.array([u2i[u1] for u1, u2 in uu_edges], dtype=np.int64)
    uu_c = np.array([u2i[u2] for u1, u2 in uu_edges], dtype=np.int64)
    UU = build_coo(uu_r, uu_c)

    aa_r = np.array([a2i[a1] + Nu for a1, a2 in aa_edges], dtype=np.int64)
    aa_c = np.array([a2i[a2] + Nu for a1, a2 in aa_edges], dtype=np.int64)
    AA = build_coo(aa_r, aa_c)

    A_list_sp = [UA, AU, AT, TA, UU, AA]
    relation_names = [
        'user→artist', 'artist→user',
        'artist→tag', 'tag→artist',
        'user→user', 'artist→artist',
    ]
    bipartite_flags = [True, True, True, True, False, False]

    # ── Features (relational aggregation) ──────────────────────────────────
    # Tag features: identity
    d = min(Nt, 256)
    X_tag = torch.eye(Nt, d, dtype=torch.float32) if Nt > 0 else torch.zeros(1, 1)

    # Artist features: aggregate tag embeddings (average of tags per artist)
    X_artist = torch.zeros(Na, d, dtype=torch.float32)
    artist_tag_count = torch.zeros(Na, dtype=torch.float32)
    for a, t in at_edges:
        ai = a2i[a]; ti = t2i[t]
        X_artist[ai] += X_tag[ti]
        artist_tag_count[ai] += 1.0
    artist_tag_count = artist_tag_count.clamp(min=1.0)
    X_artist = X_artist / artist_tag_count.unsqueeze(1)

    # User features: aggregate artist embeddings (weighted by playcount)
    X_user = torch.zeros(Nu, d, dtype=torch.float32)
    user_artist_weight = torch.zeros(Nu, dtype=torch.float32)
    with open(folder / 'user_artist.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                u, a, w = parts[0], parts[1], float(parts[2])
                if u in u2i and a in a2i:
                    ui = u2i[u]; ai = a2i[a]
                    X_user[ui] += X_artist[ai] * w
                    user_artist_weight[ui] += w
    user_artist_weight = user_artist_weight.clamp(min=1.0)
    X_user = X_user / user_artist_weight.unsqueeze(1)

    X_dict = {'user': X_user, 'artist': X_artist, 'tag': X_tag}

    # ── Labels (placeholder for LP — not used) ────────────────────────────
    labels = torch.zeros(Nu, dtype=torch.long)

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
        target_relation_idx=0,
        relation_info={
            'user→artist':   ('user', 'artist'),
            'artist→user':   ('artist', 'user'),
            'artist→tag':    ('artist', 'tag'),
            'tag→artist':    ('tag', 'artist'),
            'user→user':     ('user', 'user'),
            'artist→artist': ('artist', 'artist'),
        },
    )
