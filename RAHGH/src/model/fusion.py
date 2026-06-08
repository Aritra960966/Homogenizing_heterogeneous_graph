import torch
import torch.nn as nn


class ResidualFusion(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )

    def forward(self, H0: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        out = self.mlp(torch.cat([H0, Z], dim=1))
        return out
