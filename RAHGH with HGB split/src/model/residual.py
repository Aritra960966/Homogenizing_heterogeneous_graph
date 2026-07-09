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
    Low-frequency node-level features -- which carry local identity and would
    otherwise be diluted by multi-hop aggregation -- are preserved here.

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
