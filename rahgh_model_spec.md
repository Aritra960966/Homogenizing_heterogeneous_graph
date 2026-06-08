# RAHGH — Full Model Implementation

Complete code for `src/model/`. Seven files, no stubs.
Feed this file to your coding agent; each section maps to one source file.

---

## `src/model/__init__.py`

```python
from .projection import TypeSpecificProjection
from .normalize import build_propagation_operator, symmetrize_edges
from .bipartite_correction import BipartiteCorrector, apply_corrected_propagation
from .poly_diffusion import RelationPolynomialDiffusion
from .fusion import AdaptiveRelationFusion
from .residual import ResidualMLP
from .rahgh import RAHGH, RAHGHClassifier, SimpleGCN, SimpleGAT, build_homo_adjacency

__all__ = [
    "TypeSpecificProjection",
    "build_propagation_operator",
    "symmetrize_edges",
    "BipartiteCorrector",
    "apply_corrected_propagation",
    "RelationPolynomialDiffusion",
    "AdaptiveRelationFusion",
    "ResidualMLP",
    "RAHGH",
    "RAHGHClassifier",
    "SimpleGCN",
    "SimpleGAT",
    "build_homo_adjacency",
]
```

---

## `src/model/projection.py`

Stage 1 — Type-specific feature projection.
Equation: `H^(0)_i = W_{φ(i)} x_i + b_{φ(i)}`

```python
import torch
import torch.nn as nn
from typing import Dict


class TypeSpecificProjection(nn.Module):
    """
    Stage 1: Project each node type's raw features into a shared latent
    space of dimension hidden_dim.

        H^(0)_i = W_{phi(i)} x_i + b_{phi(i)}

    Different node types may have different raw feature dimensions
    (e.g. paper=334, author=128, term=1902 on ACM/DBLP).  After
    projection every node lives in the same R^d geometry, while
    type-specific regions of that space remain semantically distinct.

    Args:
        node_type_dims : dict[str, int]  e.g. {"paper": 334, "author": 128}
        hidden_dim     : int             shared output dim d
    """

    def __init__(self, node_type_dims: Dict[str, int], hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projections = nn.ModuleDict({
            node_type: nn.Linear(in_dim, hidden_dim)
            for node_type, in_dim in node_type_dims.items()
        })

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        node_type_indices: Dict[str, torch.Tensor],
        N_total: int,
    ) -> torch.Tensor:
        """
        Args:
            x_dict            : {node_type: (N_t, d_t)} raw feature tensors
            node_type_indices : {node_type: (N_t,)} global index tensors
                                (offsets applied by loader, all in [0, N_total))
            N_total           : total number of nodes across all types

        Returns:
            H0 : (N_total, hidden_dim)  type-homogeneous initial embedding
        """
        device = next(iter(x_dict.values())).device
        H0 = torch.zeros(N_total, self.hidden_dim, device=device)
        for node_type, idx in node_type_indices.items():
            H0[idx] = self.projections[node_type](x_dict[node_type].to(device))
        return H0
```

---

## `src/model/normalize.py`

Stage 2 — Relation-wise normalization.
Equations:
- Undirected: `P_r = D^{-1/2}(A_r + I)D^{-1/2}` — spectral radius in [0,1]
- Directed:   `P_r = D^{-1}(A_r + I)` — row normalization

```python
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

    # For undirected relations, symmetrize before normalizing so degree
    # reflects actual (undirected) connectivity.
    if (not directed) and symmetrize:
        edge_index = symmetrize_edges(edge_index)

    row, col = edge_index[0], edge_index[1]

    if add_self_loops:
        sl = torch.arange(num_nodes, device=device)
        row = torch.cat([row, sl])
        col = torch.cat([col, sl])

    # All edges have unit weight before normalization
    vals = torch.ones(row.size(0), dtype=torch.float32, device=device)

    # Degree = number of outgoing edges per source node
    # (after symmetrization + self-loops, equals the standard GCN degree)
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    deg.scatter_add_(0, row, vals)

    if directed:
        # P_r = D^{-1} A  (row normalization)
        deg_inv = deg.clamp(min=1e-8).pow(-1.0)
        norm_vals = deg_inv[row] * vals
    else:
        # P_r = D^{-1/2} A D^{-1/2}  (symmetric normalization)
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
```

---

## `src/model/bipartite_correction.py`

Stage 3 — Bipartite correction.
Equation: `P̃_r = P_r P_r^T` (bipartite) or `P̃_r = P_r` (homogeneous).
Key identity: `P̃_r H = P_r (P_r^T H)` — two sparse-dense ops, no dense N×N matrix.

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple


def apply_corrected_propagation(
    H: torch.Tensor,
    P_r: torch.Tensor,
    bipartite: bool,
) -> torch.Tensor:
    """
    Apply the bipartite-corrected operator P̃_r to H.

    Homogeneous relation  (src_type == dst_type):
        P̃_r H = P_r H
        One sparse-dense multiply; standard one-hop neighbourhood aggregation.

    Bipartite relation  (src_type != dst_type):
        P̃_r = P_r P_r^T   ←  second-order co-neighbour similarity operator
        P̃_r[i,j] = Σ_k P_r[i,k] * P_r[j,k]
        Two source nodes receive high similarity iff they share many common
        target neighbours — semantically identical to the length-2 meta-path
        r ∘ r^{-1}, recovered here in closed, differentiable, matrix form.

        Efficient application (no dense N×N materialisation):
            V = P_r^T H          source nodes broadcast to target space
            P̃_r H = P_r V        target space aggregates back to source nodes

    Args:
        H        : (N, d)  current node embedding matrix
        P_r      : torch.sparse_coo_tensor  (N, N)  propagation operator
        bipartite: bool  True if the relation connects nodes of different types

    Returns:
        (N, d) propagated embedding
    """
    if bipartite:
        # Step 1: V = P_r^T H  — each target node aggregates from its source neighbours
        V = torch.sparse.mm(P_r.t(), H)
        # Step 2: P_r V = P_r P_r^T H  — each source node aggregates from target nodes
        return torch.sparse.mm(P_r, V)
    else:
        return torch.sparse.mm(P_r, H)


class BipartiteCorrector(nn.Module):
    """
    Stage 3: Applies bipartite correction to every relation in a dict.

    No learnable parameters — the correction is structurally prescribed
    by the bipartite topology of each relation.

    For each relation r:
        bipartite  (src_type != dst_type):  P̃_r H = P_r (P_r^T H)
        homogeneous (src_type == dst_type): P̃_r H = P_r H

    Args:
        relation_info : dict[str, tuple[str, str]]
                        {rel_key: (src_type, dst_type)}
    """

    def __init__(self, relation_info: Dict[str, Tuple[str, str]]):
        super().__init__()
        self.relation_info = relation_info
        # Pre-compute and store bipartite flags — no forward-pass overhead
        self.bipartite_flags: Dict[str, bool] = {
            r: (src != dst)
            for r, (src, dst) in relation_info.items()
        }

    def forward(
        self,
        H: torch.Tensor,
        P_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Apply one step of corrected propagation for every relation.

        Args:
            H      : (N, d)  current embedding (typically H^(0) from Stage 1)
            P_dict : {rel_key: sparse P_r (N, N)}

        Returns:
            {rel_key: (N, d)}  one propagated embedding per relation
        """
        return {
            r: apply_corrected_propagation(H, P_dict[r], self.bipartite_flags[r])
            for r in P_dict
        }
```

---

## `src/model/poly_diffusion.py`

Stage 4 — Relation-specific polynomial diffusion.
Equation: `Z_r = Σ_{k=0}^{K} β_{r,k} · P̃_r^k H^(0)`,  `β_{r,k} = softmax(ψ_{r,k})`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from .bipartite_correction import apply_corrected_propagation


class RelationPolynomialDiffusion(nn.Module):
    """
    Stage 4: Relation-specific polynomial diffusion with learnable hop weights.

        Z_r = Σ_{k=0}^{K}  β_{r,k} · P̃_r^k H^(0)

        β_{r,k} = softmax(ψ_{r,k}),   ψ_{r,k} ∈ ℝ  (unconstrained, learned)

    Spectral interpretation:
        The sum Σ_k β_{r,k} P̃_r^k defines a degree-K polynomial spectral
        filter applied independently to each relation's corrected operator.
        Initialising ψ to zeros gives uniform β at epoch 0 (all hops equal
        weight), then the model learns the optimal spectral profile per relation:
          - Large β on low k  → low-pass (smoothing) filter
          - Large β on high k → high-pass (sharpening) filter

    Different relations can learn markedly different profiles:
        - A citation relation may benefit from deep multi-hop diffusion
          to capture research-community structure (high K weight).
        - A co-authorship relation may need only shallow local aggregation
          (high k=1 weight).

    Args:
        relation_names  : list[str]        ordered relation keys
        hidden_dim      : int              feature dimensionality d
        K               : int              max diffusion depth  (paper default: 3)
        bipartite_flags : dict[str, bool]  True iff relation is cross-type
    """

    def __init__(
        self,
        relation_names: List[str],
        hidden_dim: int,
        K: int = 3,
        bipartite_flags: Dict[str, bool] = None,
    ):
        super().__init__()
        self.relation_names = relation_names
        self.hidden_dim = hidden_dim
        self.K = K
        self.bipartite_flags = bipartite_flags or {r: False for r in relation_names}

        # ψ_{r,k}: one unconstrained scalar per (relation, hop)
        # Initialised to zeros → uniform β across all K+1 hops at the start
        self.psi = nn.ParameterDict({
            r: nn.Parameter(torch.zeros(K + 1))
            for r in relation_names
        })

    def forward(
        self,
        H0: torch.Tensor,
        P_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute relation-specific diffused embeddings.

        For each relation r:
            powers[0] = H^(0)                      (k=0: no graph diffusion)
            powers[k] = P̃_r^k H^(0)  for k=1..K   (iterative application)
            Z_r = Σ_k β_{r,k} * powers[k]

        Powers are computed iteratively to avoid materialising P̃_r^k explicitly.
        For bipartite relations, each hop costs two sparse-mm ops (P_r^T then P_r).

        Args:
            H0     : (N, d)  type-homogeneous initial embedding from Stage 1
            P_dict : {rel_key: sparse P_r (N, N)}

        Returns:
            Z_dict : {rel_key: (N, d)}  one diffused embedding per relation
        """
        Z_dict = {}

        for r in self.relation_names:
            P_r = P_dict[r]
            bipartite = self.bipartite_flags[r]
            beta = F.softmax(self.psi[r], dim=0)    # (K+1,) — convex weights

            # k=0: identity, preserves original projected features (no aggregation)
            # This acts as a built-in skip connection at the diffusion level.
            powers = [H0]

            # k=1,...,K: iteratively apply P̃_r to the previous power
            H_k = H0
            for _ in range(1, self.K + 1):
                H_k = apply_corrected_propagation(H_k, P_r, bipartite)
                powers.append(H_k)

            # Weighted convex combination: Z_r = Σ_k β_{r,k} P̃_r^k H^(0)
            Z_r = sum(beta[k] * powers[k] for k in range(self.K + 1))
            Z_dict[r] = Z_r

        return Z_dict

    def get_diffusion_weights(self) -> Dict[str, torch.Tensor]:
        """
        Return the softmax-normalised β weights for every relation.
        Shape: (K+1,) per relation.  Use for logging or ablation.
        """
        with torch.no_grad():
            return {r: F.softmax(self.psi[r], dim=0) for r in self.relation_names}
```

---

## `src/model/fusion.py`

Stage 5 — Adaptive relation-wise fusion.
Equation: `α = softmax(θ)`,  `Z = Σ_r α_r · Z_r`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class AdaptiveRelationFusion(nn.Module):
    """
    Stage 5: Collapse R relation-specific embeddings into one via a
    learned convex combination.

        α = softmax(θ),   θ ∈ ℝ^R   (one learnable scalar per relation)
        Z = Σ_r  α_r · Z_r

    Relations that carry task-relevant signal receive large α_r;
    noisy or redundant relations are automatically down-weighted.
    This constitutes semantic homogenisation: R relation-specific latent
    spaces are collapsed into a single unified embedding space.

    Args:
        relation_names : list[str]  ordered list of relation keys
                         (must match the keys used in Z_dict)
    """

    def __init__(self, relation_names: List[str]):
        super().__init__()
        self.relation_names = relation_names
        # θ: one unconstrained scalar per relation
        # Initialised to zeros → uniform α across all relations at epoch 0
        self.theta = nn.Parameter(torch.zeros(len(relation_names)))

    def forward(
        self,
        Z_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Z_dict : {rel_key: (N, d)}  relation-specific embeddings from Stage 4

        Returns:
            Z     : (N, d)  fused embedding
            alpha : (R,)    softmax fusion weights (return for logging / ablation)
        """
        alpha = F.softmax(self.theta, dim=0)    # (R,)  — convex weights

        Z = sum(
            alpha[i] * Z_dict[r]
            for i, r in enumerate(self.relation_names)
        )

        return Z, alpha

    def get_relation_weights(self) -> Dict[str, float]:
        """
        Return {rel_key: weight} as a plain Python dict for logging.
        """
        with torch.no_grad():
            alpha = F.softmax(self.theta, dim=0)
            return {r: alpha[i].item() for i, r in enumerate(self.relation_names)}
```

---

## `src/model/residual.py`

Stage 6 — Residual semantic preservation.
Equation: `Z_final = MLP([H^(0) ‖ Z])`,  `MLP: Linear(2d→d) → ReLU → Dropout → Linear(d→d)`

```python
import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    """
    Stage 6: Prevent over-smoothing by preserving local node identity.

        Z_final = MLP( [H^(0) ‖ Z] )

        MLP architecture:
            Linear(2*hidden_dim  →  hidden_dim)
            ReLU
            Dropout(p=dropout)
            Linear(hidden_dim  →  output_dim)

    Concatenating H^(0) (original projected features, local identity) with
    Z (fused relational embedding, global structure) and passing through an
    MLP lets the model selectively retain or suppress components of each.
    Low-frequency node-level features — which carry local identity and would
    otherwise be diluted by multi-hop aggregation — are preserved here.

    The output Z_final ∈ ℝ^{N×output_dim} is the final unified homogeneous
    latent representation.  It is directly consumable by any standard
    homogeneous GNN or downstream ML model without further modification.

    Args:
        hidden_dim : int    input dimensionality d  (must match Stage 1 output)
        output_dim : int    output dimensionality   (default = hidden_dim)
        dropout    : float  dropout rate inside MLP (paper default: 0.2)
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        output_dim = output_dim if output_dim is not None else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, H0: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H0 : (N, hidden_dim)   original type-projected features (Stage 1 output)
            Z  : (N, hidden_dim)   fused relational embedding (Stage 5 output)

        Returns:
            Z_final : (N, output_dim)  unified homogeneous latent representation
        """
        return self.mlp(torch.cat([H0, Z], dim=1))
```

---

## `src/model/rahgh.py`

Full pipeline wrapper + homogeneous GNN backbones.

```python
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

def build_homo_adjacency(
    edge_index_dict: Dict[str, torch.Tensor],
    alpha: torch.Tensor,
    num_nodes: int,
    relation_names: List[str],
) -> torch.Tensor:
    """
    Assemble the induced weighted homogeneous adjacency A_homo.

    Each relation r contributes its edges (symmetrised) weighted by α_r.
    All relations are merged into a single (N, N) sparse matrix whose
    non-zero pattern drives the downstream GNN backbone.

        A_homo = Σ_r  α_r * (A_r + A_r^T)        (symmetrised, coalesced)

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
        # Add both directions to ensure symmetry
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
#  RAHGH — Full Homogenization Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAHGH(nn.Module):
    """
    Relation-Aware Heterogeneous Graph Homogenization (RAHGH).

    Transforms a heterogeneous graph into a single unified latent matrix
    Z_final ∈ ℝ^{N×d} through six tightly coupled stages:

        Stage 1 — TypeSpecificProjection
                  H^(0)_i = W_{φ(i)} x_i + b_{φ(i)}

        Stage 2 — build_propagation_operator  (no learnable params)
                  P_r = D^{-1/2}(A_r + I)D^{-1/2}

        Stage 3 — BipartiteCorrector          (no learnable params)
                  P̃_r H = P_r(P_r^T H)  for bipartite relations
                  P̃_r H = P_r H          for homogeneous relations

        Stage 4 — RelationPolynomialDiffusion
                  Z_r = Σ_{k=0}^{K}  β_{r,k} P̃_r^k H^(0)
                  β_{r,k} = softmax(ψ_{r,k})

        Stage 5 — AdaptiveRelationFusion
                  Z = Σ_r  α_r Z_r,   α = softmax(θ)

        Stage 6 — ResidualMLP
                  Z_final = MLP([H^(0) ‖ Z])

    Z_final is directly consumable by any standard homogeneous GNN backbone
    (SimpleGCN, SimpleGAT, …) without further modification.

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
        # Stage 4 — passes bipartite flags so it can call apply_corrected_propagation
        self.diffuser = RelationPolynomialDiffusion(
            self.relation_names, hidden_dim, K, bipartite_flags
        )
        # Stage 5
        self.fusioner = AdaptiveRelationFusion(self.relation_names)
        # Stage 6
        self.residual = ResidualMLP(hidden_dim, output_dim, dropout)

    # ------------------------------------------------------------------
    # Stage 2 helper — build all P_r operators (no learnable params)
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
        """
        device = next(iter(x_dict.values())).device
        # Ensure edge indices are on the correct device
        edge_index_dict = {r: ei.to(device) for r, ei in edge_index_dict.items()}

        # Stage 1 — project heterogeneous features into shared space
        H0 = self.projector(x_dict, node_type_indices, self.num_nodes)  # (N, d)

        # Stage 2 — build normalised propagation operators
        P_dict = self._build_operators(edge_index_dict)

        # Stages 3+4 — bipartite-corrected polynomial diffusion
        Z_dict = self.diffuser(H0, P_dict)                # {rel: (N, d)}

        # Stage 5 — adaptive relation fusion
        Z, alpha = self.fusioner(Z_dict)                  # (N, d), (R,)

        # Stage 6 — residual MLP
        Z_final = self.residual(H0, Z)                    # (N, output_dim)

        return Z_final, alpha

    # ------------------------------------------------------------------
    # Introspection helpers (use in ablation.py / visualize.ipynb)
    # ------------------------------------------------------------------
    def get_diffusion_weights(self) -> Dict[str, torch.Tensor]:
        """Per-relation β_{r,k} hop weights, shape (K+1,) each."""
        return self.diffuser.get_diffusion_weights()

    def get_relation_weights(self) -> Dict[str, float]:
        """Per-relation α_r fusion weights as a plain Python dict."""
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
        values  = A.values()
        row = indices[0]
        N   = A.size(0)
        deg = torch.zeros(N, dtype=values.dtype, device=values.device)
        deg.scatter_add_(0, row, values.abs())
        deg = deg.clamp(min=1e-8)
        norm_values = values / deg[row]
        return torch.sparse_coo_tensor(indices, norm_values, A.size()).coalesce()

    def forward(self, Z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z : (N, in_dim)    RAHGH output  Z_final
            A : sparse (N, N)  induced homogeneous adjacency  A_homo

        Returns:
            logits : (N, out_dim)
        """
        A_norm = self._row_normalize(A)
        H = torch.sparse.mm(A_norm, Z)
        H = F.relu(self.lin1(H))
        H = F.dropout(H, p=self.dropout, training=self.training)
        H = torch.sparse.mm(A_norm, H)
        return self.lin2(H)


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
#  RAHGHClassifier — End-to-End Model (Homogenizer + Backbone)
# ─────────────────────────────────────────────────────────────────────────────

class RAHGHClassifier(nn.Module):
    """
    End-to-end node classification model.

    Wires together linear homogenization with non-linear downstream message passing:
        1.  RAHGH (Linear/Structural Primer)  →  Z_final (N, hidden_dim),  alpha (R,)
        2.  build_homo_adjacency(alpha)       →  A_homo (N, N) sparse
        3.  GNN backbone (Non-Linear Learner) →  logits (N, num_classes)

    By separating these stages, the polynomial diffusion acts as a purely linear 
    spectral filter, while the downstream GCN/GAT introduces the weight matrices 
    and non-linearities (e.g., ReLU) required to learn complex decision boundaries.

    Train by slicing logits[target_global_idx][train_mask] and computing
    cross-entropy against the target-type labels.

    Args:
        node_type_dims : dict[str, int]
        relation_info  : dict[str, tuple[str, str]]
        num_nodes      : int    total nodes (all types, global id space)
        hidden_dim     : int    RAHGH latent dim and backbone hidden dim  (default 64)
        num_classes    : int    number of output classes
        K              : int    max diffusion depth                        (default 3)
        backbone       : str    "gcn" or "gat"                            (default "gcn")
        dropout_homo   : float  dropout inside RAHGH ResidualMLP          (default 0.2)
        dropout_gnn    : float  dropout inside GNN backbone               (default 0.5)
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
        backbone: str = "gcn",
        dropout_homo: float = 0.2,
        dropout_gnn: float = 0.5,
        directed: bool = False,
    ):
        super().__init__()
        self.num_nodes      = num_nodes
        self.relation_names = list(relation_info.keys())
        self.backbone_type  = backbone

        # RAHGH homogenizer (Stages 1–6)
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

        # Homogeneous GNN backbone
        if backbone == "gcn":
            self.backbone = SimpleGCN(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                out_dim=num_classes,
                dropout=dropout_gnn,
            )
        elif backbone == "gat":
            self.backbone = SimpleGAT(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim // 4,   # per-head dim; 4 heads → hidden_dim total
                out_dim=num_classes,
                heads=4,
                dropout=dropout_gnn,
            )
        else:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose 'gcn' or 'gat'.")

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
            logits : (N_total, num_classes)   slice by target_global_idx for loss
            alpha  : (R,)                     relation fusion weights (for logging)

        Usage in train loop:
            logits, alpha = model(x_dict, edge_index_dict, node_type_indices)
            target_logits = logits[target_global_idx]        # (N_target, num_classes)
            loss = F.cross_entropy(target_logits[train_mask], labels[train_mask])
        """
        # Step 1: Homogenize
        Z_final, alpha = self.homogenizer(x_dict, edge_index_dict, node_type_indices)

        # Step 2: Build induced homogeneous adjacency weighted by learned α
        device = Z_final.device
        edge_index_dict_dev = {r: ei.to(device) for r, ei in edge_index_dict.items()}
        A_homo = build_homo_adjacency(
            edge_index_dict_dev, alpha, self.num_nodes, self.relation_names
        )

        # Step 3: GNN backbone on the homogeneous graph
        if self.backbone_type == "gcn":
            logits = self.backbone(Z_final, A_homo)
        else:
            # GAT uses edge_index; extract sparsity pattern from A_homo
            edge_index_homo = A_homo.indices()
            logits = self.backbone(Z_final, edge_index_homo)

        return logits, alpha
```

---

## Notes for the coding agent

### Forward-pass data flow

```
x_dict  ──────────────────────────────────────────────────────► H0 (Stage 1)
                                                                   │
edge_index_dict ──► P_dict (Stage 2) ──► Z_dict (Stages 3+4) ──► Z (Stage 5)
                                                                   │
                                            H0 ────────────────► Z_final (Stage 6)
                                                                   │
                            A_homo ◄─── alpha ◄─── Z (Stage 5)   │
                               │                                   │
                               └──────► backbone ◄────────────────┘
                                           │
                                        logits
```

### Tensor shapes at each stage

| Stage | Output | Shape |
|-------|--------|-------|
| 1 — projection | H0 | (N_total, hidden_dim) |
| 2 — normalize | P_dict | {rel: sparse (N_total, N_total)} |
| 3+4 — diffusion | Z_dict | {rel: (N_total, hidden_dim)} |
| 5 — fusion | Z, alpha | (N_total, hidden_dim), (R,) |
| 6 — residual | Z_final | (N_total, output_dim) |
| backbone | logits | (N_total, num_classes) |

### Key invariants to preserve

1. **No densification.** All P_r operators stay as `torch.sparse_coo_tensor`; never call `.to_dense()` on them.
2. **Global node ids everywhere.** `node_type_indices` maps each type to its offset slice in [0, N_total). Every `edge_index` tensor must use these global ids — local-type ids are only inside `x_dict` and `node_type_indices`.
3. **Z_final shape = (N_total, output_dim).** The backbone receives Z_final for all N nodes; slicing by `target_global_idx` happens in the train loop, not inside the model.
4. **alpha is always returned.** `ablation.py` logs it without re-running forward.
5. **Bipartite detection is automatic.** Any relation with `src_type != dst_type` (from `relation_info`) is treated as bipartite — no manual flags in configs.
6. **Iterative powers, never explicit matrix exponentiation.** `P̃_r^k H` is computed as `k` successive applications of `apply_corrected_propagation`, not via `torch.linalg.matrix_power`.

### Two-Stage Message Passing Philosophy (Do Not Optimize Away)

The framework explicitly uses **two distinct phases** of message passing. Do not attempt to merge them or remove the downstream GCN.
1. **Stage 4 (Polynomial Diffusion):** This is **linear message passing**. It passes messages up to $K$ hops away using learnable scalar weights ($\beta$), but applies no neural network weight matrices and no non-linear activation functions. It structurally smooths and aligns the heterogeneous data.
2. **GCN/GAT Backbone:** This is **non-linear message passing**. It takes the fully homogenized $Z_{final}$ and the collapsed topology $A_{homo}$, applying dense linear transformations ($W$) and non-linearities (ReLU) at every hop. This step is strictly required to learn the complex, high-dimensional patterns necessary for accurate node classification.

### Hyperparameter defaults (from paper + code)

| Parameter | Default | Notes |
|-----------|---------|-------|
| hidden_dim | 64 | 512 for OGBN-MAG |
| output_dim | 64 | Must equal backbone in_dim |
| K | 3 | Diffusion depth |
| dropout_homo | 0.2 | ResidualMLP |
| dropout_gnn | 0.5 | GCN / GAT backbone |
| lr | 0.001 | Adam |
| weight_decay | 0.001 | L2 regularisation |
| GAT heads | 4 | hidden_dim // 4 per head |
