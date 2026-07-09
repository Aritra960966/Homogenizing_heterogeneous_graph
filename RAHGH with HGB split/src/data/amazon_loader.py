import json
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
from collections import defaultdict


def _find_amazon_folder(root, variant='amazon'):
    """Find Amazon dataset folder. variant='amazon' or 'amazon_ini'."""
    candidates = [
        Path(root) / variant / variant,
        Path(root) / variant,
    ]
    for c in candidates:
        if (c / 'node.dat').exists():
            return c
    return None


def load_amazon(root="data/raw"):
    folder = _find_amazon_folder(root, 'amazon')
    if folder is None:
        folder = _find_amazon_folder(root, 'amazon_ini')
    assert folder is not None, f"Amazon folder not found under {root}"
    return _load_from_folder(folder)


def load_amazon_ini(root="data/raw"):
    """Load from the amazon_ini/ folder (alternative format)."""
    folder = _find_amazon_folder(root, 'amazon_ini')
    assert folder is not None, f"amazon_ini folder not found under {root}"
    return _load_from_folder(folder)


def _load_from_folder(folder):
    """Core loading logic shared by load_amazon and load_amazon_ini."""
    import json, numpy as np, scipy.sparse as sp, torch
    from pathlib import Path
    from collections import defaultdict

    # ── Parse info.dat ────────────────────────────────────────────────────────
    info = {}
    info_path = folder / 'info.dat'
    if info_path.exists():
        info = json.loads(info_path.read_text())

    type_names = {int(k): v for k, v in info.get('node.dat', {'0': 'product'}).items()}

    # ── Parse node.dat ────────────────────────────────────────────────────────
    nodes_by_type = defaultdict(list)
    max_node_id = 0
    with open(folder / 'node.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            nid = int(parts[0])
            tid = int(parts[2])
            tname = type_names.get(tid, f'type_{tid}')
            feats = np.array(parts[3].split(','), dtype=np.float32) if len(parts) > 3 else None
            nodes_by_type[tname].append((nid, feats))
            max_node_id = max(max_node_id, nid)

    N = max_node_id + 1
    node_type_global_ids = {}
    X_dict = {}
    for tname, node_list in nodes_by_type.items():
        ids = [n[0] for n in node_list]
        feats = [n[1] for n in node_list]
        node_type_global_ids[tname] = sorted(ids)
        if feats[0] is not None:
            feat_mat = np.stack(feats, axis=0)
        else:
            dim = min(len(node_list), 512)
            feat_mat = np.eye(len(node_list), dim, dtype=np.float32)
        X_dict[tname] = torch.tensor(feat_mat)

    # ── Parse link.dat ────────────────────────────────────────────────────────
    edges_by_rel = defaultdict(list)
    with open(folder / 'link.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                src, dst, rid = int(parts[0]), int(parts[1]), parts[2]
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
    for rid in sorted(edges_by_rel.keys(), key=int):
        edges = edges_by_rel[rid]
        rows = [e[0] for e in edges]
        cols = [e[1] for e in edges]
        vals = np.array([e[2] for e in edges], dtype=np.float32)
        A = build_mat(rows, cols, vals)
        A_list_sp.append(A)
        relation_names.append(f'product→product_{rid}')
        bipartite_flags.append(False)

    target_relation_idx = 0
    labels = torch.zeros(max(len(node_type_global_ids.get('product', [])), 1), dtype=torch.long)
    Nu = len(node_type_global_ids.get('product', []))

    # ── Parse link.dat.test for HGB-compatible LP evaluation ──────────────────
    test_fp = folder / 'link.dat.test'
    lp_test_edges = None
    if test_fp.exists():
        test_edges_by_rel = defaultdict(list)
        with open(test_fp) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    src, dst, rid = int(parts[0]), int(parts[1]), parts[2]
                    test_edges_by_rel[rid].append((src, dst))
        # Build per-relation-type arrays and source/target type mapping
        lp_test_edges = {}
        for rid, edges in test_edges_by_rel.items():
            arr = np.array(edges, dtype=np.int64)  # (E, 2)
            src_t = type_names.get(info.get('link.dat', {}).get(rid, {}).get('start', '0'), 'product')
            dst_t = type_names.get(info.get('link.dat', {}).get(rid, {}).get('end', '0'), 'product')
            rname = relation_names[int(rid)] if int(rid) < len(relation_names) else f'rel_{rid}'
            lp_test_edges[rname] = {
                'edges': arr,
                'src_type': src_t,
                'dst_type': dst_t,
                'rel_id': rid,
            }

    return dict(
        A_list_sp=A_list_sp,
        relation_names=relation_names,
        bipartite_flags=bipartite_flags,
        X_dict=X_dict,
        labels=labels,
        N=N, Nu=Nu,
        target_type='product',
        target_size=Nu,
        n_classes=0,
        target_relation_idx=target_relation_idx,
        node_type_global_ids=node_type_global_ids,
        relation_info={r: ('product', 'product') for r in relation_names},
        lp_test_edges=lp_test_edges,
    )
