from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import scipy.sparse as sp

from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader


def to_pyg_data(
    data_dict: dict,
    relation_name: Optional[str] = None,
    relation_idx: Optional[int] = None,
    include_all_node_types: bool = True,
) -> Data:
    """
    Convert an RAHGH data dict to a PyG ``Data`` object.

    If ``relation_name`` or ``relation_idx`` is given, only that relation's
    edges are used.  Otherwise *all* relations are fused into a single
    undirected ``edge_index`` (duplicates removed).

    Node features from all types are concatenated in global-ID order, so
    ``x.shape == (N, total_feat_dim)``.

    Args:
        data_dict: RAHGH data dict (keys: ``A_list_sp``, ``X_dict``,
            ``labels``, ``N``, ``target_type``, …).
        relation_name:  pick this named relation from ``edge_index_dict``.
        relation_idx:   pick this positional relation from ``A_list_sp``.
        include_all_node_types:
            If ``True`` (default), ``x`` covers all N nodes using the
            per-type tensors in ``X_dict``.  If ``False``, only the target
            type is included (for homogeneous downstream use).

    Returns:
        ``Data(x, edge_index, y, …)``
    """
    N = data_dict['N']
    device = torch.device('cpu')

    # ── edge_index ──────────────────────────────────────────────────────────
    if relation_name is not None:
        # Use pre-built edge_index_dict
        ei = data_dict.get('edge_index_dict', {}).get(relation_name)
        if ei is not None:
            edge_index = ei.clone()
        else:
            # Fall back to building from A_list_sp using relation_names
            rel_names = data_dict.get('relation_names', [])
            if relation_name in rel_names:
                idx = rel_names.index(relation_name)
                edge_index = _sp_to_edge_index(data_dict['A_list_sp'][idx])
            else:
                raise KeyError(
                    f"relation '{relation_name}' not found. "
                    f"Available: {rel_names}"
                )
    elif relation_idx is not None:
        edge_index = _sp_to_edge_index(data_dict['A_list_sp'][relation_idx])
    else:
        # Fuse all relations
        all_src, all_dst = [], []
        for A_sp in data_dict['A_list_sp']:
            A_coo = A_sp.tocoo()
            all_src.append(A_coo.row)
            all_dst.append(A_coo.col)
        src = np.concatenate(all_src)
        dst = np.concatenate(all_dst)
        # Symmetrize + deduplicate
        edges_np = np.unique(
            np.sort(np.stack([src, dst], axis=1), axis=1), axis=0
        )
        # Remove self-loops
        edges_np = edges_np[edges_np[:, 0] != edges_np[:, 1]]
        edge_index = torch.tensor(edges_np.T, dtype=torch.long)

    # ── node features ───────────────────────────────────────────────────────
    if include_all_node_types:
        node_type_indices = data_dict.get(
            'node_type_indices',
            _build_node_type_indices(data_dict),
        )
        # Build full x tensor by scattering per-type features into column blocks
        x_dim = sum(
            data_dict['X_dict'][t].shape[1]
            for t in data_dict['X_dict']
        )
        x = torch.zeros(N, x_dim, dtype=torch.float32)
        offset = 0
        for tname, feats in data_dict['X_dict'].items():
            d_t = feats.shape[1]
            idx = node_type_indices[tname]  # global IDs
            x[idx, offset:offset + d_t] = feats
            offset += d_t
    else:
        # Target type only
        target_type = data_dict.get('target_type', list(data_dict['X_dict'].keys())[0])
        x = data_dict['X_dict'][target_type]

    # ── labels ──────────────────────────────────────────────────────────────
    y = data_dict.get('labels_full', data_dict.get('labels', None))
    if y is not None and isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.long)

    # ── build Data ──────────────────────────────────────────────────────────
    pyg_data = Data(x=x, edge_index=edge_index, y=y)

    # Attach metadata from the RAHGH data dict
    pyg_data.num_nodes = N
    pyg_data.target_type = data_dict.get('target_type', None)
    pyg_data.target_size = data_dict.get('target_size', N)
    pyg_data.n_classes = data_dict.get('n_classes', 0)
    pyg_data.dataset_name = data_dict.get('name', '')

    if 'train_mask' in data_dict or 'val_mask' in data_dict or 'test_mask' in data_dict:
        pyg_data.train_mask = data_dict.get('train_mask')
        pyg_data.val_mask = data_dict.get('val_mask')
        pyg_data.test_mask = data_dict.get('test_mask')

    return pyg_data


def to_pyg_data_list(
    data_dict: dict,
    include_all_node_types: bool = True,
) -> List[Tuple[str, Data]]:
    """
    Convert each relation in the RAHGH data dict to a separate PyG ``Data``.

    Returns:
        ``[(rel_name, Data), …]`` — one entry per relation in
        ``A_list_sp`` / ``relation_names``.
    """
    rel_names = data_dict.get(
        'relation_names',
        [f'rel_{i}' for i in range(len(data_dict['A_list_sp']))],
    )
    result = []
    for i, rname in enumerate(rel_names):
        d = to_pyg_data(
            data_dict,
            relation_idx=i,
            include_all_node_types=include_all_node_types,
        )
        d.relation_name = rname
        d.relation_idx = i
        result.append((rname, d))
    return result


def build_cluster_dataloader(
    data_dict: dict,
    num_parts: int = 10,
    batch_size: int = 1,
    shuffle: bool = True,
    relation_idx: Optional[int] = None,
    **cluster_kwargs,
) -> ClusterLoader:
    """
    Build a ``ClusterLoader`` from the RAHGH data dict.

    The graph is first converted to a homogeneous PyG ``Data`` (fused
    adjacency), then partitioned with ``ClusterData``, and wrapped in a
    ``ClusterLoader``.

    This is useful for mini-batch training in the clustering / self-supervised
    task.

    Args:
        data_dict: RAHGH data dict.
        num_parts: Number of METIS partitions for ``ClusterData``.
        batch_size: Batch size for ``ClusterLoader``.
        shuffle: Whether to shuffle subgraphs in the loader.
        relation_idx: If given, use only this relation instead of fusing all.
        **cluster_kwargs: Extra kwargs forwarded to ``ClusterData``
            (e.g. ``recursive=True``, ``save_space=True``).

    Returns:
        ``ClusterLoader`` instance.
    """
    if relation_idx is not None:
        pyg_data = to_pyg_data(data_dict, relation_idx=relation_idx)
    else:
        pyg_data = to_pyg_data(data_dict)  # fused adjacency

    cluster_data = ClusterData(
        pyg_data,
        num_parts=num_parts,
        **cluster_kwargs,
    )
    loader = ClusterLoader(
        cluster_data,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader


def homogeneous_to_pyg_data(
    Z_final: torch.Tensor,
    A_homo: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Data:
    """
    Wrap the RAHGH homogenised output + induced adjacency into a PyG ``Data``.

    This is the natural point to call after ``rahgh.forward()`` when you
    want to use PyG utilities (``ClusterData``, ``ClusterLoader``,
    ``NeighborLoader``, etc.) on the unified graph.

    Args:
        Z_final: ``(N, d)`` — RAHGH output embedding.
        A_homo:  ``(N, N)`` sparse COO tensor — weighted homogeneous
            adjacency from ``build_homo_adjacency()``.
        labels:  Optional ``(N,)`` or ``(N_labeled,)`` label tensor.
        num_nodes: Total number of nodes (defaults to ``Z_final.size(0)``).

    Returns:
        ``Data(x=Z_final, edge_index=A_homo.indices(), y=labels)``
    """
    N = num_nodes or Z_final.size(0)
    edge_index = A_homo.indices()
    return Data(x=Z_final, edge_index=edge_index, y=labels, num_nodes=N)


# ── Internal helpers ─────────────────────────────────────────────────────────


def _sp_to_edge_index(A_sp: sp.spmatrix) -> torch.Tensor:
    """Convert a single scipy sparse matrix to ``(2, E)`` edge index."""
    A_coo = A_sp.tocoo()
    ei = np.vstack([A_coo.row, A_coo.col])
    return torch.tensor(ei, dtype=torch.long)


def _build_node_type_indices(data_dict: dict) -> Dict[str, torch.Tensor]:
    """
    Build ``node_type_indices`` if the data dict does not already have them.

    Tries ``N_*`` keys (``Na``, ``Np``, …), falls back to ``X_dict`` shapes.
    """
    types = list(data_dict['X_dict'].keys())
    size_keys = {
        k.lower(): k for k in data_dict.keys()
        if k.startswith('N') and len(k) == 2
    }
    type_to_size = {}
    for t in types:
        key = f'N{t[0]}'
        if key in data_dict:
            type_to_size[t] = data_dict[key]
        elif key.lower() in size_keys:
            type_to_size[t] = data_dict[size_keys[key.lower()]]
        else:
            type_to_size[t] = data_dict['X_dict'][t].shape[0]

    offset = 0
    indices = {}
    for t in types:
        sz = type_to_size[t]
        indices[t] = torch.arange(offset, offset + sz, dtype=torch.long)
        offset += sz
    return indices
