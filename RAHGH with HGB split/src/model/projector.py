import torch
import torch.nn as nn


class TypeSpecificProjector(nn.Module):
    def __init__(self, in_dims: list[int], d: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_d, d),
                nn.LayerNorm(d),
                nn.ReLU(),
            )
            for in_d in in_dims
        ])

    def forward(self, X_list: list) -> torch.Tensor:
        return torch.cat([
            proj(X)
            for proj, X in zip(self.projections, X_list)
        ], dim=0)
