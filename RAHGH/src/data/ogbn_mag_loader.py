import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path


def load_ogbn_mag(root: str = "data/raw") -> dict:
    from ogb.nodeproppred import NodePropPredDataset

    folder = Path(root) / "ogbn_mag"
    ds = NodePropPredDataset(name="ogbn-mag", root=str(folder))
    raw_data, info = ds[0]
    split_idx = ds.get_idx_split()

    type_order = ["paper", "author", "institution", "field_of_study"]
    num_nodes_dict = raw_data["num_nodes_dict"]
    N_types = {t: num_nodes_dict[t] for t in type_order}

    offsets = {}
    offset = 0
    for t in type_order:
        offsets[t] = offset
        offset += N_types[t]
    N_total = offset

    # ── Features ──
    node_feat_dict = raw_data["node_feat_dict"]
    X_dict = {}
    for t in type_order:
        if t in node_feat_dict and node_feat_dict[t] is not None:
            X_dict[t] = torch.tensor(node_feat_dict[t], dtype=torch.float32)
        else:
            nt = N_types[t]
            dim = min(nt, 128)
            X_dict[t] = torch.eye(nt, dim, dtype=torch.float32)

    # ── Edge index ──
    edge_index_raw = raw_data["edge_index_dict"]
    relation_names = []
    bipartite_flags = []
    A_list_sp = []
    relation_info = {}
    edge_index_dict = {}

    for (src_type, rel_type, dst_type), ei_np in edge_index_raw.items():
        ei_np = np.asarray(ei_np)
        src_global = ei_np[0] + offsets[src_type]
        dst_global = ei_np[1] + offsets[dst_type]

        rname = f"{src_type}→{dst_type}"
        relation_names.append(rname)
        relation_info[rname] = (src_type, dst_type)
        bipartite_flags.append(src_type != dst_type)

        edge_index_dict[rname] = torch.stack([
            torch.tensor(src_global, dtype=torch.long),
            torch.tensor(dst_global, dtype=torch.long),
        ], dim=0)

        r = src_global.astype(np.int64)
        c = dst_global.astype(np.int64)
        A = sp.coo_matrix(
            (np.ones(len(r), dtype=np.float32), (r, c)),
            shape=(N_total, N_total)
        ).tocsr()
        A_list_sp.append(A)

    # ── Labels ──
    labels_dict = ds.labels
    paper_labels = np.asarray(labels_dict["paper"]).squeeze(-1)

    train_idx = np.asarray(split_idx["train"]["paper"])
    valid_idx = np.asarray(split_idx["valid"]["paper"])
    test_idx = np.asarray(split_idx["test"]["paper"])

    labeled = np.where(paper_labels >= 0)[0]
    n_classes = int(paper_labels[labeled].max()) + 1 if len(labeled) > 0 else 0

    labels_full = torch.full((N_types["paper"],), -1, dtype=torch.long)
    labels_full[labeled] = torch.tensor(paper_labels[labeled], dtype=torch.long)
    labeled_mask = labels_full >= 0
    labels_labeled = labels_full[labeled_mask]

    # ── Node type indices ──
    node_type_indices = {}
    for t in type_order:
        off = offsets[t]
        sz = N_types[t]
        node_type_indices[t] = torch.arange(off, off + sz, dtype=torch.long)

    node_type_dims = {t: X_dict[t].shape[1] for t in type_order}

    return dict(
        A_list_sp=A_list_sp,
        bipartite_flags=bipartite_flags,
        relation_names=relation_names,
        X_dict=X_dict,
        node_type_indices=node_type_indices,
        node_type_dims=node_type_dims,
        relation_info=relation_info,
        edge_index_dict=edge_index_dict,
        labels=labels_labeled,
        labels_full=labels_full,
        labeled_mask=labeled_mask,
        N=N_total,
        target_type="paper",
        target_size=N_types["paper"],
        n_classes=n_classes,
        train_indices=torch.tensor(train_idx, dtype=torch.long),
        valid_indices=torch.tensor(valid_idx, dtype=torch.long),
        test_indices=torch.tensor(test_idx, dtype=torch.long),
    )
