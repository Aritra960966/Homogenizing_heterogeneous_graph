import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .projection import TypeSpecificProjection
from .normalize import build_propagation_operator
from .bipartite_correction import BipartiteCorrector
from .poly_diffusion import RelationPolynomialDiffusion
from .fusion import AdaptiveRelationFusion
from .residual import ResidualMLP


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: build weighted homogeneous adjacency from all relations
# ─────────────────────────────────────────────────────────────────────────────

@torch._dynamo.disable
def build_homo_adjacency(
    edge_index_dict: Dict[str, torch.Tensor],
    alpha: torch.Tensor,
    num_nodes: int,
    relation_names: List[str],
) -> torch.Tensor:
    """
    Assemble the induced weighted homogeneous adjacency A_homo.

    Each relation r contributes its edges (symmetrised) weighted by alpha_r.
    All relations are merged into a single (N, N) sparse matrix whose
    non-zero pattern drives the downstream GNN backbone.

        A_homo = Sigma_r  alpha_r * (A_r + A_r^T)        (symmetrised, coalesced)

    Args:
        edge_index_dict : {rel_key: (2, E)}  edge lists in global node ids
        alpha           : (R,)  relation fusion weights from AdaptiveRelationFusion
        num_nodes       : int   total N
        relation_names  : list[str]  ordered relation names matching alpha

    Returns:
        A_homo : torch.sparse_coo_tensor  (N, N)  coalesced
    """
    device = alpha.device
    src_list, dst_list, w_list = [], [], []

    for i, r in enumerate(relation_names):
        ei = edge_index_dict[r].to(device)
        s, t = ei[0], ei[1]
        w = alpha[i].detach().expand(s.size(0))
        src_list.append(s);  dst_list.append(t);  w_list.append(w)
        src_list.append(t);  dst_list.append(s);  w_list.append(w)

    src  = torch.cat(src_list)
    dst  = torch.cat(dst_list)
    vals = torch.cat(w_list)

    A_homo = torch.sparse_coo_tensor(
        torch.stack([src, dst], dim=0),
        vals,
        (num_nodes, num_nodes),
        dtype=torch.float32,
        device=device,
    ).coalesce()

    return A_homo


# ─────────────────────────────────────────────────────────────────────────────
#  RAHGH -- Full Homogenization Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAHGH(nn.Module):
    """
    Relation-Aware Heterogeneous Graph Homogenization (RAHGH).

    Transforms a heterogeneous graph into a single unified latent matrix
    Z_final in R^{N x d} through six tightly coupled stages:

        Stage 1 -- TypeSpecificProjection
                  H^(0)_i = W_{phi(i)} x_i + b_{phi(i)}

        Stage 2 -- build_propagation_operator  (no learnable params)
                  P_r = D^{-1/2}(A_r + I)D^{-1/2}

        Stage 3 -- BipartiteCorrector          (no learnable params)
                  P~_r H = P_r(P_r^T H)  for bipartite relations
                  P~_r H = P_r H          for homogeneous relations

        Stage 4 -- RelationPolynomialDiffusion
                  Z_r = Sigma_{k=0}^{K}  beta_{r,k} P~_r^k H^(0)
                  beta_{r,k} = softmax(psi_{r,k})

        Stage 5 -- AdaptiveRelationFusion
                  Z = Sigma_r  alpha_r Z_r,   alpha = softmax(theta)

        Stage 6 -- ResidualMLP
                  Z_final = MLP([H^(0) || Z])

    Z_final is directly consumable by any standard homogeneous GNN backbone
    (SimpleGCN, SimpleGAT, ...) without further modification.

    Args:
        node_type_dims : dict[str, int]                {node_type: raw feature dim}
        relation_info  : dict[str, tuple[str, str]]    {rel_key: (src_type, dst_type)}
        num_nodes      : int                           total nodes (global id space)
        hidden_dim     : int   = 64                    shared latent dim d
        output_dim     : int   = 64                    Z_final output dim
        K              : int   = 3                     max diffusion depth
        dropout        : float = 0.2                   dropout inside ResidualMLP
        directed       : bool  = False                 asymmetric normalization flag
    """

    def __init__(
        self,
        node_type_dims: Dict[str, int],
        relation_info: Dict[str, Tuple[str, str]],
        num_nodes: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        K: int = 3,
        dropout: float = 0.2,
        directed: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.relation_names = list(relation_info.keys())
        self.relation_info = relation_info
        self.directed = directed

        # Derive bipartite flag per relation: cross-type iff src_type != dst_type
        bipartite_flags: Dict[str, bool] = {
            r: (src != dst) for r, (src, dst) in relation_info.items()
        }

        # Stage 1
        self.projector = TypeSpecificProjection(node_type_dims, hidden_dim)
        # Stage 3 (no params; used for standalone unit-tests)
        self.corrector = BipartiteCorrector(relation_info)
        # Stage 4 -- passes bipartite flags so it can call apply_corrected_propagation
        self.diffuser = RelationPolynomialDiffusion(
            self.relation_names, hidden_dim, K, bipartite_flags
        )
        # Stage 5
        self.fusioner = AdaptiveRelationFusion(self.relation_names)
        # Stage 6
        self.residual = ResidualMLP(hidden_dim, output_dim, dropout)

    # ------------------------------------------------------------------
    # Stage 2 helper -- build all P_r operators (no learnable params)
    # Called once per forward pass; for transductive settings consider
    # pre-computing and caching outside the model.
    # ------------------------------------------------------------------
    def _build_operators(
        self,
        edge_index_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {
            r: build_propagation_operator(
                edge_index_dict[r], self.num_nodes, directed=self.directed
            )
            for r in self.relation_names
        }

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        node_type_indices: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_dict            : {node_type: (N_t, d_t)}  raw node features
            edge_index_dict   : {rel_key: (2, E)}         edge lists, global node ids
            node_type_indices : {node_type: (N_t,)}       global index tensors

        Returns:
            Z_final : (N_total, output_dim)   unified homogeneous embedding
            alpha   : (R,)                    relation fusion weights (for logging)

        Note: all inputs are expected to already be on the correct device.
        The training setup moves them once; this method does NOT call .to().
        """
        # Stage 1 -- project heterogeneous features into shared space
        H0 = self.projector(x_dict, node_type_indices, self.num_nodes)  # (N, d)

        # Stage 2 -- build normalised propagation operators
        P_dict = self._build_operators(edge_index_dict)

        # Stages 3+4 -- bipartite-corrected polynomial diffusion
        Z_dict = self.diffuser(H0, P_dict)                # {rel: (N, d)}

        # Stage 5 -- adaptive relation fusion
        Z, alpha = self.fusioner(Z_dict)                  # (N, d), (R,)

        # Stage 6 -- residual MLP
        Z_final = self.residual(H0, Z)                    # (N, output_dim)

        return Z_final, alpha

    # ------------------------------------------------------------------
    # Introspection helpers (use in ablation.py / visualize.ipynb)
    # ------------------------------------------------------------------
    def get_diffusion_weights(self) -> Dict[str, torch.Tensor]:
        """Per-relation beta_{r,k} hop weights, shape (K+1,) each."""
        return self.diffuser.get_diffusion_weights()

    def get_relation_weights(self) -> Dict[str, float]:
        """Per-relation alpha_r fusion weights as a plain Python dict."""
        return self.fusioner.get_relation_weights()


# ─────────────────────────────────────────────────────────────────────────────
#  Homogeneous GNN Backbones
# ─────────────────────────────────────────────────────────────────────────────

class SimpleGCN(nn.Module):
    """
    Two-layer GCN backbone that operates on RAHGH's unified embedding.

    Architecture:
        H1     = ReLU( row_norm(A_homo) @ Z_final @ W1 )   + Dropout
        logits = row_norm(A_homo) @ H1 @ W2

    Matches the SimpleGCN from the IMDB baseline code:  D^{-1} A row
    normalisation applied before each linear layer.

    Args:
        in_dim     : int    input dim  = output_dim of RAHGH
        hidden_dim : int    GCN hidden dim
        out_dim    : int    number of classes
        dropout    : float  dropout after first layer  (default 0.5)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lin1    = nn.Linear(in_dim, hidden_dim)
        self.lin2    = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    @staticmethod
    def _row_normalize(A: torch.Tensor) -> torch.Tensor:
        """D^{-1} A row normalization for a sparse COO tensor."""
        indices = A.indices()
        values  = A.values().float()
        row = indices[0]
        N   = A.size(0)
        deg = torch.zeros(N, dtype=torch.float32, device=values.device)
        deg.scatter_add_(0, row, values.abs())
        deg = deg.clamp(min=1e-8)
        norm_values = values / deg[row]
        return torch.sparse_coo_tensor(indices, norm_values, A.size()).coalesce()

    @torch._dynamo.disable
    def forward(self, Z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        orig_dtype = Z.dtype
        device_type = "cuda" if Z.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            Z_f32 = Z.to(dtype=torch.float32)
            A_norm = self._row_normalize(A)
            H = torch.sparse.mm(A_norm, Z_f32)
            H = F.relu(self.lin1(H))
            H = F.dropout(H, p=self.dropout, training=self.training)
            H = torch.sparse.mm(A_norm, H)
            return self.lin2(H).to(dtype=orig_dtype)


class SimpleGAT(nn.Module):
    """
    Two-layer GAT backbone operating on RAHGH's unified embedding.

    Uses the sparsity pattern of A_homo as the attention graph.
    Requires torch_geometric.

    Architecture:
        H1     = ELU( GATConv(Z_final, heads=heads) )   + Dropout
        logits = GATConv(H1, heads=1)

    Args:
        in_dim     : int    input dim  = output_dim of RAHGH
        hidden_dim : int    per-head feature dim
        out_dim    : int    number of classes
        heads      : int    attention heads in layer 1  (default 4)
        dropout    : float  attention + feature dropout  (default 0.5)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.conv1   = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2   = GATConv(hidden_dim * heads, out_dim, heads=1,
                               concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(
        self,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            Z          : (N, in_dim)    RAHGH output  Z_final
            edge_index : (2, E)         induced homogeneous edge list
                         (use A_homo.indices() to obtain from A_homo)

        Returns:
            logits : (N, out_dim)
        """
        Z = F.dropout(Z, p=self.dropout, training=self.training)
        Z = F.elu(self.conv1(Z, edge_index))
        Z = F.dropout(Z, p=self.dropout, training=self.training)
        return self.conv2(Z, edge_index)


# ─────────────────────────────────────────────────────────────────────────────
#  RAHGHClassifier -- End-to-End Model (Homogenizer + Head)
# ─────────────────────────────────────────────────────────────────────────────

class RAHGHClassifier(nn.Module):
    """
    End-to-end node classification model.

    Wires together linear homogenization with non-linear downstream message passing:
        1.  RAHGH (Linear/Structural Primer)  ->  Z_final (N, hidden_dim),  alpha (R,)
        2.  build_homo_adjacency(alpha)       ->  A_homo (N, N) sparse
        3.  GNN head (Non-Linear Learner) ->  logits (N, num_classes)

    By separating these stages, the polynomial diffusion acts as a purely linear
    spectral filter, while the downstream GCN/GAT introduces the weight matrices
    and non-linearities (e.g., ReLU) required to learn complex decision boundaries.

    Args:
        node_type_dims : dict[str, int]
        relation_info  : dict[str, tuple[str, str]]
        num_nodes      : int    total nodes (all types, global id space)
        hidden_dim     : int    RAHGH latent dim and head hidden dim    (default 64)
        num_classes    : int    number of output classes
        K              : int    max diffusion depth                      (default 3)
        head           : str    "gcn" or "gat"                          (default "gcn")
        dropout_homo   : float  dropout inside RAHGH ResidualMLP        (default 0.2)
        dropout_gnn    : float  dropout inside GNN head                 (default 0.5)
        directed       : bool   use asymmetric normalisation              (default False)
    """

    def __init__(
        self,
        node_type_dims: Dict[str, int],
        relation_info: Dict[str, Tuple[str, str]],
        num_nodes: int,
        hidden_dim: int = 64,
        num_classes: int = 3,
        K: int = 3,
        head: str = "gcn",
        dropout_homo: float = 0.2,
        dropout_gnn: float = 0.5,
        directed: bool = False,
        gnn_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_nodes      = num_nodes
        self.relation_names = list(relation_info.keys())
        self.head_type      = head
        gnn_hidden_dim = gnn_hidden_dim or hidden_dim

        # RAHGH homogenizer (Stages 1-6)
        self.homogenizer = RAHGH(
            node_type_dims=node_type_dims,
            relation_info=relation_info,
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            K=K,
            dropout=dropout_homo,
            directed=directed,
        )

        # Homogeneous GNN head
        if head == "gcn":
            self.head = SimpleGCN(
                in_dim=hidden_dim,
                hidden_dim=gnn_hidden_dim,
                out_dim=num_classes,
                dropout=dropout_gnn,
            )
        elif head == "gat":
            self.head = SimpleGAT(
                in_dim=hidden_dim,
                hidden_dim=gnn_hidden_dim // 4,
                out_dim=num_classes,
                heads=4,
                dropout=dropout_gnn,
            )
        else:
            raise ValueError(f"Unknown head '{head}'. Choose 'gcn' or 'gat'.")

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        node_type_indices: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_dict            : {node_type: (N_t, d_t)}  raw node features
            edge_index_dict   : {rel_key: (2, E)}         edge lists, global node ids
            node_type_indices : {node_type: (N_t,)}       global index tensors

        Returns:
            logits : (N_total, num_classes)
            alpha  : (R,)  relation fusion weights (for logging)
        """
        # Step 1: Homogenize
        Z_final, alpha = self.homogenizer(x_dict, edge_index_dict, node_type_indices)

        # Step 2: Build induced homogeneous adjacency weighted by learned alpha
        # edge_index_dict is already on-device (moved once in training setup)
        A_homo = build_homo_adjacency(
            edge_index_dict, alpha, self.num_nodes, self.relation_names
        )

        # Step 3: GNN head on the homogeneous graph
        if self.head_type == "gcn":
            logits = self.head(Z_final, A_homo)
        else:
            edge_index_homo = A_homo.indices()
            logits = self.head(Z_final, edge_index_homo)

        return logits, alpha


# ─────────────────────────────────────────────────────────────────────────────
#  Backward-compatible compile_model (kept from original rahgh.py)
# ─────────────────────────────────────────────────────────────────────────────

def compile_model(model: nn.Module, verbose: bool = False) -> nn.Module:
    """
    torch.compile is a no-op for RAHGH.

    Reason: the dominant cost is torch.sparse.mm inside
    apply_corrected_propagation, which is marked @torch._dynamo.disable.
    - Windows / aot_eager: Dynamo traces the graph but emits eager kernels —
      overhead with zero optimization.
    - Linux  / inductor:   sparse ops trigger graph breaks; the compiled
      subgraphs cover only the cheap linear layers.
    In both cases compilation adds latency on the first forward pass and
    provides no throughput improvement.
    """
    if verbose:
        print("[compile_model] sparse-op codebase — skipping torch.compile")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Backward-compatible builder (bridges old data dict format to new API)
# ─────────────────────────────────────────────────────────────────────────────

def build_rahgh_classifier(
    data: dict,
    hidden_dim: int = 64,
    num_classes: int = 3,
    K: int = 3,
    head: str = "gcn",
    dropout_homo: float = 0.2,
    dropout_gnn: float = 0.5,
    directed: bool = False,
    gnn_hidden_dim: Optional[int] = None,
) -> RAHGHClassifier:
    """
    Build an RAHGHClassifier from an old-format data dict.

    Old-format keys expected:
        X_dict         : {type_name: Tensor(N_t, d_t)}
        A_list_sp      : list[scipy.sparse.csr_matrix]
        relation_names : list[str]
        N              : int
        target_size    : int

    Derives:
        node_type_dims  from X_dict
        edge_index_dict from A_list_sp
        node_type_indices from node id offsets (inferred from data)

    Args:
        gnn_hidden_dim : GNN head hidden dim (defaults to hidden_dim if None)

    Usage:
        model = build_rahgh_classifier(data, hidden_dim=64, num_classes=3,
                                        K=3, head='gcn')
        logits, alpha = model(x_dict, edge_index_dict, node_type_indices)
    """
    node_type_dims = {k: v.shape[1] for k, v in data['X_dict'].items()}

    rel_names = data.get('relation_names', [f'rel_{i}' for i in range(len(data['A_list_sp']))])

    # Prefer explicit relation_info from the loader; fall back to bipartite_flags heuristic
    if 'relation_info' in data:
        relation_info = data['relation_info']
    elif 'bipartite_flags' in data:
        types = list(data['X_dict'].keys())
        target_type = data.get('target_type', types[0])
        relation_info = {}
        for i, rname in enumerate(rel_names):
            is_bip = data['bipartite_flags'][i]
            if is_bip:
                other_type = next((t for t in types if t != target_type), types[-1])
                # Alternate src/dst per relation to cover forward/backward pairs
                if i % 2 == 0:
                    src_type, dst_type = target_type, other_type
                else:
                    src_type, dst_type = other_type, target_type
            else:
                src_type = dst_type = target_type
            relation_info[rname] = (src_type, dst_type)
    else:
        relation_info = {r: (data.get('target_type', 'node'), data.get('target_type', 'node'))
                         for r in rel_names}

    return RAHGHClassifier(
        node_type_dims=node_type_dims,
        relation_info=relation_info,
        num_nodes=data['N'],
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        K=K,
        head=head,
        dropout_homo=dropout_homo,
        dropout_gnn=dropout_gnn,
        directed=directed,
        gnn_hidden_dim=gnn_hidden_dim,
    )


def build_edge_index_dict(
    data: dict,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert old-format A_list_sp to edge_index_dict for new model.

    Each sparse CSR matrix is converted to (2, E) edge index tensor.
    """
    import numpy as np
    import scipy.sparse as sp
    rel_names = data.get('relation_names', [f'rel_{i}' for i in range(len(data['A_list_sp']))])
    edge_dict = {}
    for i, (A_sp, rname) in enumerate(zip(data['A_list_sp'], rel_names)):
        A_coo = A_sp.tocoo()
        ei = np.vstack([A_coo.row, A_coo.col])
        edge_dict[rname] = torch.tensor(ei, dtype=torch.long)
        if device is not None:
            edge_dict[rname] = edge_dict[rname].to(device)
    return edge_dict


def build_node_type_indices(
    data: dict,
) -> Dict[str, torch.Tensor]:
    """
    Build node_type_indices from old-format data dict.

    Tries to infer offsets from N_* keys (Na, Np, Nt, etc.) or
    allocates contiguous blocks from X_dict.
    """
    types = list(data['X_dict'].keys())
    # Try to get per-type counts from data keys like 'Na', 'Np', etc.
    size_keys = {k.lower(): k for k in data.keys() if k.startswith('N') and len(k) == 2}
    type_to_size = {}
    for t in types:
        key = f'N{t[0]}'  # e.g. 'Na' for 'author', 'Np' for 'paper'
        if key in data:
            type_to_size[t] = data[key]
        elif key.lower() in size_keys:
            type_to_size[t] = data[size_keys[key.lower()]]
        else:
            type_to_size[t] = data['X_dict'][t].shape[0]

    offset = 0
    indices = {}
    for t in types:
        sz = type_to_size[t]
        indices[t] = torch.arange(offset, offset + sz, dtype=torch.long)
        offset += sz
    return indices
