import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .projector import TypeSpecificProjector
from .diffusion  import (RelationSpecificDiffusion,
                         build_operators, build_A_struct)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, A, X):
        return self.W(torch.sparse.mm(A, X))


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
        H0                = self.projector(X_list)
        Z, alpha, beta    = self.diffusion(H0, P_list)
        Z_final           = self.fusion(torch.cat([H0, Z], dim=1))
        alpha_np          = alpha.detach().cpu().numpy()
        A_hat             = build_A_struct(self.A_list_sp,
                                            alpha_np,
                                            self.N_nodes,
                                            self.device)
        gcn_out           = self.gcn(A_hat, Z_final)
        return gcn_out, alpha, beta, gcn_out
