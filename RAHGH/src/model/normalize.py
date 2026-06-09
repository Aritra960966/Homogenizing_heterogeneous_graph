import torch
from typing import Optional


def symmetrize_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Add reverse edges and remove duplicates, making the graph undirected.

    Args:
        edge_index : (2, E)  directed edge list (global node ids)

    Returns:
        (2, E')  symmetrized and de-duplicated edge list  (E' >= E)
    """
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    combined = torch.cat([edge_index, rev], dim=1)
    return torch.unique(combined, dim=1)


@torch._dynamo.disable
def build_propagation_operator(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = False,
    add_self_loops: bool = True,
    symmetrize: bool = True,
) -> torch.Tensor:
    """
    Stage 2: Build the normalized propagation operator P_r for one relation.

    Undirected (symmetric normalization):
        P_r = D^{-1/2} (A_r + I) D^{-1/2}
        Spectral radius bounded in [0, 1]; prevents gradient explosion / vanishing.

    Directed (row / asymmetric normalization):
        P_r = D^{-1} (A_r + I)
        Used for inherently directed relations (e.g. OGBN-MAG citations).

    Args:
        edge_index    : (2, E)  edge list with global node ids
        num_nodes     : int     total N (all node types combined)
        directed      : bool    if True use asymmetric D^{-1} normalisation
        add_self_loops: bool    if True prepend I to A_r before normalising
        symmetrize    : bool    if True add reverse edges first (ignored when directed=True)

    Returns:
        P_r : torch.sparse_coo_tensor  shape (num_nodes, num_nodes)
              Stays sparse; do NOT call .to_dense() on large graphs.
    """
    device = edge_index.device

    if (not directed) and symmetrize:
        edge_index = symmetrize_edges(edge_index)

    row, col = edge_index[0], edge_index[1]

    if add_self_loops:
        sl = torch.arange(num_nodes, device=device)
        row = torch.cat([row, sl])
        col = torch.cat([col, sl])

    vals = torch.ones(row.size(0), dtype=torch.float32, device=device)

    deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    deg.scatter_add_(0, row, vals)

    if directed:
        deg_inv = deg.clamp(min=1e-8).pow(-1.0)
        norm_vals = deg_inv[row] * vals
    else:
        deg_inv_sqrt = deg.clamp(min=1e-8).pow(-0.5)
        norm_vals = deg_inv_sqrt[row] * vals * deg_inv_sqrt[col]

    P_r = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0),
        norm_vals,
        (num_nodes, num_nodes),
        dtype=torch.float32,
        device=device,
    ).coalesce()

    return P_r
