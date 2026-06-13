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
            node_type: nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
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
        # Run first projection to determine dtype (may be float16 under AMP autocast)
        first_type = next(iter(node_type_indices.keys()))
        first_proj = self.projections[first_type](x_dict[first_type].to(device))
        H0 = torch.zeros(N_total, self.hidden_dim, device=device, dtype=first_proj.dtype)
        H0[node_type_indices[first_type]] = first_proj
        for node_type, idx in node_type_indices.items():
            if node_type == first_type:
                continue
            H0[idx] = self.projections[node_type](x_dict[node_type].to(device))
        return H0
