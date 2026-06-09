import torch
import torch.nn as nn
from typing import Dict, Tuple


@torch._dynamo.disable
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
        P̃_r = P_r P_r^T   <-  second-order co-neighbour similarity operator
        P̃_r[i,j] = Σ_k P_r[i,k] * P_r[j,k]
        Two source nodes receive high similarity iff they share many common
        target neighbours -- semantically identical to the length-2 meta-path
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


class BipartiteCorrector(nn.Module):
    """
    Stage 3: Applies bipartite correction to every relation in a dict.

    No learnable parameters -- the correction is structurally prescribed
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
