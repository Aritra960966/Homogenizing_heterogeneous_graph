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
