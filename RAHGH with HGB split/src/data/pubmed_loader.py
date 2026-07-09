import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
from collections import defaultdict


def _find_pubmed_folder(root, variant='PubMed'):
    """Find PubMed dataset folder. variant='PubMed' or 'PubMed_ini'."""
    candidates = [Path(root) / variant / variant, Path(root) / variant]
    for c in candidates:
        if (c / 'node.dat').exists():
            return c
    return None


def load_pubmed(root="data/raw"):
    folder = _find_pubmed_folder(root, 'PubMed')
    if folder is None:
        folder = _find_pubmed_folder(root, 'PubMed_ini')
    assert folder is not None, f"PubMed folder not found under {root}"
    return _load_pubmed_from_folder(folder)


def load_pubmed_ini(root="data/raw"):
    folder = _find_pubmed_folder(root, 'PubMed_ini')
    assert folder is not None, f"PubMed_ini folder not found under {root}"
    return _load_pubmed_from_folder(folder)


def _load_pubmed_from_folder(folder):

    # type_id -> name mapping
    type_id_to_name = {0: 'term', 1: 'author', 2: 'paper', 3: 'venue'}

    # ── Parse node.dat (first pass: collect all nodes grouped by type) ──────
    # Format: node_id \t name \t type_id \t feat1,feat2,...
    nodes_raw = defaultdict(list)  # type_name -> list of (orig_id, feat_vec)
    with open(folder / 'node.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            nid = int(parts[0])
            tid = int(parts[2])
            tname = type_id_to_name.get(tid, f'type_{tid}')
            feats = np.array(parts[3].split(','), dtype=np.float32) if len(parts) > 3 else None
            nodes_raw[tname].append((nid, feats))

    # Assign new contiguous IDs: sort by type order (term→author→paper→venue),
    # then within each type sort by original ID for reproducibility
    type_order = ['term', 'author', 'paper', 'venue']
    old_to_new = {}
    X_dict = {}
    node_type_global_ids = {}
    offset = 0
    for tname in type_order:
        node_list = nodes_raw.get(tname, [])
        node_list.sort(key=lambda x: x[0])  # sort by original ID
        ids = []
        feats_list = []
        for orig_id, feats in node_list:
            old_to_new[orig_id] = offset
            ids.append(offset)
            feats_list.append(feats)
            offset += 1
        node_type_global_ids[tname] = ids
        if feats_list and feats_list[0] is not None:
            feat_mat = np.stack(feats_list, axis=0)
        else:
            feat_mat = np.eye(len(ids), min(len(ids), 512), dtype=np.float32)
        X_dict[tname] = torch.tensor(feat_mat)

    N = offset
    Nt = len(node_type_global_ids.get('term', []))
    Na = len(node_type_global_ids.get('author', []))
    Np = len(node_type_global_ids.get('paper', []))
    Nv = len(node_type_global_ids.get('venue', []))

    # ── Parse link.dat ────────────────────────────────────────────────────────
    # Format: src \t dst \t rel_type \t weight
    rel_names = {
        0: 'paper→term', 1: 'term→paper',
        2: 'paper→author', 3: 'author→paper',
        4: 'paper→paper', 5: 'paper→venue',
        6: 'venue→paper', 7: 'author→author',
        8: 'author→term', 9: 'term→author',
    }

    rel_info_map = {
        'paper→term': ('paper', 'term'),
        'term→paper': ('term', 'paper'),
        'paper→author': ('paper', 'author'),
        'author→paper': ('author', 'paper'),
        'paper→paper': ('paper', 'paper'),
        'paper→venue': ('paper', 'venue'),
        'venue→paper': ('venue', 'paper'),
        'author→author': ('author', 'author'),
        'author→term': ('author', 'term'),
        'term→author': ('term', 'author'),
    }

    edges_by_rel = defaultdict(list)
    with open(folder / 'link.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                src, dst, rid = int(parts[0]), int(parts[1]), int(parts[2])
                weight = float(parts[3]) if len(parts) >= 4 else 1.0
                edges_by_rel[rid].append((src, dst, weight))

    def build_mat(rows, cols, vals=None, N=N):
        if vals is None:
            vals = np.ones(len(rows), dtype=np.float32)
        return sp.coo_matrix((vals, (rows, cols)),
                              shape=(N, N)).tocsr()

    A_list_sp = []
    relation_names = []
    bipartite_flags = []
    relation_info = {}

    for rid in sorted(edges_by_rel.keys()):
        edges = edges_by_rel[rid]
        rname = rel_names.get(rid, f'rel_{rid}')
        # Remap using new contiguous IDs
        rows = [old_to_new.get(e[0], 0) for e in edges]
        cols = [old_to_new.get(e[1], 0) for e in edges]
        vals = np.array([e[2] for e in edges], dtype=np.float32)
        A = build_mat(rows, cols, vals)
        A_list_sp.append(A)
        relation_names.append(rname)

        src_type, dst_type = rel_info_map.get(rname, ('paper', 'paper'))
        bipartite_flags.append(src_type != dst_type)
        relation_info[rname] = (src_type, dst_type)

    # Target relation for LP: paper→paper (rel 4)
    target_relation_idx = next((i for i, r in enumerate(relation_names) if r == 'paper→paper'), 0)

    # ── Labels (LP task — dummy) ──────────────────────────────────────────────
    labels = torch.zeros(max(Np, 1), dtype=torch.long)

    # ── Parse link.dat.test for HGB-compatible LP evaluation ──────────────────
    test_fp = folder / 'link.dat.test'
    lp_test_edges = None
    if test_fp.exists():
        test_edges_by_rel = defaultdict(list)
        with open(test_fp) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    src = int(parts[0])
                    dst = int(parts[1])
                    rid = int(parts[2])
                    new_src = old_to_new.get(src, 0)
                    new_dst = old_to_new.get(dst, 0)
                    test_edges_by_rel[rid].append((new_src, new_dst))
        lp_test_edges = {}
        for rid, edges in test_edges_by_rel.items():
            arr = np.array(edges, dtype=np.int64)
            rname = rel_names.get(rid, f'rel_{rid}')
            src_type, dst_type = rel_info_map.get(rname, ('paper', 'paper'))
            lp_test_edges[rname] = {
                'edges': arr,
                'src_type': src_type,
                'dst_type': dst_type,
                'rel_id': rid,
            }

    return dict(
        A_list_sp=A_list_sp,
        relation_names=relation_names,
        bipartite_flags=bipartite_flags,
        X_dict=X_dict,
        labels=labels,
        N=N, Np=Np, Na=Na, Nt=Nt, Nv=Nv,
        target_type='paper',
        target_size=Np,
        n_classes=0,
        target_relation_idx=target_relation_idx,
        node_type_global_ids=node_type_global_ids,
        relation_info=relation_info,
        lp_test_edges=lp_test_edges,
    )
