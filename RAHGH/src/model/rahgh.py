import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .projector import TypeSpecificProjector
from .diffusion  import (RelationSpecificDiffusion,
                         build_operators, build_A_struct,
                         apply_corrected_propagation)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    @torch._dynamo.disable
    def forward(self, A, X):
        return self.W(apply_corrected_propagation(X, A, bipartite=False))


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.l1 = GCNLayer(in_dim, hidden_dim)
        self.l2 = GCNLayer(hidden_dim, out_dim)
        self.drop = dropout

    def forward(self, A, X):
        H = F.relu(self.l1(A, X))
        H = F.dropout(H, p=self.drop, training=self.training)
        out = self.l2(A, H)
        return out


class RAHGH(nn.Module):
    def __init__(self, in_dims, d, R, K,
                 gcn_hidden, out_dim, dropout,
                 A_list_sp, N, device):
        super().__init__()
        self.projector = TypeSpecificProjector(in_dims, d)
        self.diffusion = RelationSpecificDiffusion(R, K)
        self.fusion    = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d, d)
        )
        self.gcn       = GCN(d, gcn_hidden, out_dim, dropout)
        self.A_list_sp = A_list_sp
        self.N_nodes   = N
        self.device    = device

    def forward(self, X_list, P_list):
        H0     = self.projector(X_list)
        Z, alpha, beta = self.diffusion(H0, P_list)
        Z_final = self.fusion(torch.cat([H0, Z], dim=1))
        gcn_out = self._forward_gcn(Z_final)
        return gcn_out, alpha, beta, gcn_out

    @torch._dynamo.disable
    def _forward_gcn(self, Z_final):
        alpha = F.softmax(self.diffusion.theta, dim=0)
        A_hat = build_A_struct(self.A_list_sp,
                                alpha.detach().cpu().numpy(),
                                self.N_nodes,
                                self.device)
        return self.gcn(A_hat, Z_final)


def compile_model(
    model: nn.Module,
    backend: str = None,
    verbose: bool = False,
) -> nn.Module:
    if backend is None:
        backend = "aot_eager" if sys.platform == "win32" else "inductor"
    if verbose:
        print(f"[compile_model] backend='{backend}'")
    try:
        return torch.compile(model, backend=backend, fullgraph=False)
    except Exception as exc:
        import warnings
        warnings.warn(
            f"torch.compile failed with backend='{backend}': {exc}\n"
            f"Falling back to eager mode."
        )
        return model
