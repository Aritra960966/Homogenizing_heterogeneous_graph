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
