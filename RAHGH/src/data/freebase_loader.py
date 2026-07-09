"""
Freebase / KG triples loader for HGB link prediction.
"""
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
from collections import defaultdict


def load_freebase(root: str = "data/raw", named: bool = True) -> dict:
    folder = Path(root) / ('Freebase' if named else 'Freebase_no_name')
    if not folder.exists():
        folder = Path(root) / 'Freebase' / 'Freebase'
    if not folder.exists():
        folder = Path(root) / 'Freebase_no_name' / 'Freebase_no_name'

    ent2id, rel2id = {}, {}

    e2id_path = folder / 'entity2id.txt'
    if e2id_path.exists():
        with open(e2id_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if '\t' in line:
                    parts = line.split('\t')
                    ent2id[parts[0]] = int(parts[1])
                else:
                    if line.isdigit():
                        continue
                    ent2id[line] = len(ent2id)

    r2id_path = folder / 'relation2id.txt'
    if r2id_path.exists():
        with open(r2id_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if '\t' in line:
                    parts = line.split('\t')
                    rel2id[parts[0]] = int(parts[1])

    # Auto-build from triple files if maps empty
    edges_by_rel = defaultdict(lambda: ([], []))
    for split in ['train.txt', 'valid.txt', 'test.txt']:
        fp = folder / split
        if not fp.exists(): continue
        with open(fp) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    h, r, t = parts
                    if h not in ent2id: ent2id[h] = len(ent2id)
                    if t not in ent2id: ent2id[t] = len(ent2id)
                    if r not in rel2id: rel2id[r] = len(rel2id)
                    hi, ri, ti = ent2id[h], rel2id[r], ent2id[t]
                    edges_by_rel[ri][0].append(hi)
                    edges_by_rel[ri][1].append(ti)

    N = len(ent2id) if ent2id else 0
    if N == 0 and edges_by_rel:
        N = max(max(max(e[0]) if e[0] else 0, max(e[1]) if e[1] else 0) for e in edges_by_rel.values()) + 1

    A_list_sp = []
    relation_names = []
    bipartite_flags = []

    for rid in sorted(edges_by_rel.keys()):
        rows, cols = edges_by_rel[rid]
        A = sp.coo_matrix((np.ones(len(rows), np.float32),
                            (np.array(rows, np.int64), np.array(cols, np.int64))),
                           shape=(N, N)).tocsr()
        A_list_sp.append(A)
        rname = {v: k for k, v in rel2id.items()}.get(rid, f'rel_{rid}')
        relation_names.append(rname)
        bipartite_flags.append(False)

    feat_dim = min(N, 512)
    X = torch.eye(N, feat_dim, dtype=torch.float32)

    return dict(
        A_list_sp=A_list_sp,
        bipartite_flags=bipartite_flags,
        relation_names=relation_names,
        X_dict={'entity': X},
        labels=torch.zeros(N, dtype=torch.long),
        N=N, target_type='entity', target_size=N,
        n_classes=0, target_relation_idx=0,
        node_type_global_ids={'entity': list(range(N))},
        relation_info={rn: ('entity', 'entity') for rn in relation_names},
    )
