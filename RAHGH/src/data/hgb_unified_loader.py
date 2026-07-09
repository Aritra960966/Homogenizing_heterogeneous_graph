"""
HGB unified format loader — reads node.dat, link.dat, label.dat, info.dat.

Used by datasets: YouTube, PubMed (HGB unified), and as fallback for any dataset.
"""
import json, numpy as np
import torch, scipy.sparse as sp
from pathlib import Path
from collections import defaultdict


def load_hgb_unified(dataset_name: str, root: str = "data/raw") -> dict:
    folder = Path(root) / dataset_name
    if not folder.exists():
        folder = Path(root) / dataset_name / dataset_name

    # ── Parse info.dat ──────────────────────────────────────────────────────
    info_path = folder / 'info.dat'
    info_raw = {}
    if info_path.exists():
        raw = open(info_path).read().strip()
        try:
            info_raw = json.loads(raw)
        except Exception:
            pass

    type_id_to_name = info_raw.get('type_id_to_name', {})
    if not type_id_to_name and 'node.dat' in info_raw:
        type_id_to_name = {str(k): v['name'] if isinstance(v, dict) else v
                          for k, v in info_raw.get('node.dat', {}).items()}

    rel_info_map = info_raw.get('link.dat', info_raw.get('rel_id_to_info', {}))

    # ── Parse node.dat ──────────────────────────────────────────────────────
    nodes_by_type = defaultdict(list)
    max_node_id = 0
    node_path = folder / 'node.dat'
    if node_path.exists():
        with open(node_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                nid = int(parts[0])
                tid = parts[1]
                tname = type_id_to_name.get(tid, f'type_{tid}')
                feats = np.array(parts[2:], dtype=np.float32) if len(parts) > 2 else None
                nodes_by_type[tname].append((nid, feats))
                max_node_id = max(max_node_id, nid)

    N = max_node_id + 1 if nodes_by_type else 0
    if N == 0:
        N = info_raw.get('num_nodes', sum(info_raw.get('node_type_counts', {}).values()))

    node_type_indices = {}
    X_dict = {}
    node_type_dims = {}

    for tname, node_list in nodes_by_type.items():
        ids = np.array([n[0] for n in node_list], dtype=np.int64)
        feats = [n[1] for n in node_list]
        node_type_indices[tname] = torch.tensor(ids)
        if feats[0] is not None:
            feat_mat = np.stack(feats, axis=0)
        else:
            dim = min(len(node_list), 512)
            feat_mat = np.eye(len(node_list), dim, dtype=np.float32)
        X_dict[tname] = torch.tensor(feat_mat)
        node_type_dims[tname] = feat_mat.shape[1]

    if not X_dict:
        feat_dim = min(N, 512)
        X_dict = {'node': torch.eye(N, feat_dim, dtype=torch.float32)}
        node_type_indices = {'node': torch.arange(N)}
        node_type_dims = {'node': feat_dim}

    # ── Parse link.dat ──────────────────────────────────────────────────────
    edges_by_rel = defaultdict(lambda: ([], []))
    link_path = folder / 'link.dat'
    if link_path.exists():
        with open(link_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                src, dst, rid = int(parts[0]), int(parts[1]), parts[2]
                edges_by_rel[rid][0].append(src)
                edges_by_rel[rid][1].append(dst)

    # Ensure N covers all node IDs in edges
    for rid, (rows, cols) in edges_by_rel.items():
        if rows: N = max(N, max(rows) + 1)
        if cols: N = max(N, max(cols) + 1)

    A_list_sp = []
    bipartite_flags = []
    relation_names = []
    edge_index_dict = {}
    relation_info = {}

    for rid, (rows, cols) in sorted(edges_by_rel.items(), key=lambda x: x[0] if isinstance(x[0], int) else 0):
        rname = str(rid)
        if isinstance(rel_info_map, dict) and rid in rel_info_map:
            ri = rel_info_map[rid]
            if isinstance(ri, dict):
                src_type = ri.get('name', ri.get('start', '?'))
                dst_type = ri.get('name_end', ri.get('end', '?'))
            elif isinstance(ri, (list, tuple)):
                src_type, dst_type = ri[0], ri[1]
            else:
                src_type = dst_type = '?'
            rname = ri.get('meaning', ri.get('name', f'rel_{rid}')) if isinstance(ri, dict) else f'rel_{rid}'
        else:
            src_type = dst_type = '?'

        is_bip = (src_type != dst_type)
        r = np.array(rows, dtype=np.int64)
        c = np.array(cols, dtype=np.int64)
        A = sp.coo_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                           shape=(N, N)).tocsr()
        A_list_sp.append(A)
        bipartite_flags.append(is_bip)
        relation_names.append(rname)
        edge_index_dict[rname] = torch.tensor(np.stack([r, c], axis=0))
        relation_info[rname] = (src_type, dst_type)

    # ── Parse label.dat ─────────────────────────────────────────────────────
    labels = torch.zeros(max(1, N), dtype=torch.long)
    n_classes = 0
    label_path = folder / 'label.dat'
    if label_path.exists():
        labeled_ids, raw_labels = [], []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    labeled_ids.append(int(parts[0]))
                    raw_labels.append(int(parts[2]))
        if raw_labels:
            unique = sorted(set(raw_labels))
            lbl_map = {v: i for i, v in enumerate(unique)}
            n_classes = len(unique)
            for nid, rl in zip(labeled_ids, raw_labels):
                if nid < N:
                    labels[nid] = lbl_map[rl]

    target_type = list(X_dict.keys())[0] if X_dict else 'node'
    target_size = len(node_type_indices.get(target_type, [])) if node_type_indices else N
    if target_size == 0:
        target_size = N

    return dict(
        A_list_sp=A_list_sp,
        bipartite_flags=bipartite_flags,
        relation_names=relation_names,
        X_dict=X_dict,
        node_type_indices=node_type_indices,
        node_type_dims=node_type_dims,
        relation_info=relation_info,
        edge_index_dict=edge_index_dict,
        labels=labels,
        N=N,
        target_type=target_type,
        target_size=target_size,
        n_classes=n_classes,
    )
