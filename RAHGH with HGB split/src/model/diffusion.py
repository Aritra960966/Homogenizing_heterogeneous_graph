import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


@torch._dynamo.disable
def apply_corrected_propagation(
    H: torch.Tensor,
    P_r: torch.Tensor,
    bipartite: bool = False,
) -> torch.Tensor:
    orig_dtype = H.dtype
    device_type = "cuda" if H.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        H_f32 = H.to(dtype=torch.float32)
        if bipartite:
            V = torch.sparse.mm(P_r.t(), H_f32)
            out = torch.sparse.mm(P_r, V)
        else:
            out = torch.sparse.mm(P_r, H_f32)
    return out.to(dtype=orig_dtype)


def normalize_with_selfloops(A_sp, device):
    N       = A_sp.shape[0]
    A_t     = (A_sp + sp.eye(N, format='csr', dtype=np.float32)).tocoo()
    deg     = np.array(A_t.sum(axis=1)).flatten()
    d_inv   = np.where(deg > 0, deg ** -0.5, 0.0)
    data    = A_t.data * d_inv[A_t.row] * d_inv[A_t.col]
    idx     = torch.tensor(np.vstack([A_t.row, A_t.col]), dtype=torch.long)
    val     = torch.tensor(data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (N, N)).coalesce().to(device)


def normalize_plain(A_sp, device):
    A     = A_sp.tocoo().astype(np.float32)
    deg   = np.array(A_sp.sum(axis=1)).flatten()
    d_inv = np.where(deg > 0, deg ** -0.5, 0.0)
    data  = A.data * d_inv[A.row] * d_inv[A.col]
    idx   = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val   = torch.tensor(data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, A_sp.shape).coalesce().to(device)


def build_operators(A_list_sp, bipartite_flags, device):
    P_list = []
    for i, (A_sp, is_bip) in enumerate(zip(A_list_sp, bipartite_flags)):
        P_r = normalize_with_selfloops(A_sp, device)
        if is_bip:
            P_tilde = torch.sparse.mm(P_r, P_r.t()).coalesce()
        else:
            P_tilde = P_r
        P_list.append(P_tilde)
    return P_list


def build_A_struct(A_list_sp, alpha_np, N, device):
    A_struct = sp.csr_matrix((N, N), dtype=np.float32)
    for i, (A_sp, a) in enumerate(zip(A_list_sp, alpha_np)):
        A_struct = A_struct + float(a) * A_sp.astype(np.float32)
    A_struct_I = (A_struct + sp.eye(N, format='csr', dtype=np.float32))
    result = normalize_plain(A_struct_I, device)
    return result


class RelationSpecificDiffusion(nn.Module):
    def __init__(self, R: int, K: int):
        super().__init__()
        self.R   = R
        self.K   = K
        self.phi   = nn.Parameter(torch.zeros(R, K + 1))
        self.theta = nn.Parameter(torch.ones(R))

    def forward(self, H0, P_list):
        beta  = F.softmax(self.phi,   dim=1)
        alpha = F.softmax(self.theta, dim=0)
        Z_list = []
        for r in range(self.R):
            Zr = torch.zeros_like(H0)
            Hk = H0
            for k in range(self.K + 1):
                Zr = Zr + beta[r, k] * Hk
                if k < self.K:
                    Hk = apply_corrected_propagation(Hk, P_list[r], bipartite=False)
            Z_list.append(Zr)
        Z = sum(alpha[r] * Z_list[r] for r in range(self.R))
        return Z, alpha, beta
