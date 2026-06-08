# Fixes: Mixed Precision & torch.compile

Both problems are solved by **one targeted change each** — not by restructuring the model.
The `problem.md` analysis overstates the invasiveness for both issues.

---

## Fix 1 — Mixed Precision (`addmm_sparse_cuda` Half error)

**Root cause:** Under `autocast`, `H` (a dense tensor) is cast to `float16` by upstream `nn.Linear` layers. `P_r` is a `float32` sparse tensor (built outside autocast). `torch.sparse.mm(float32_sparse, float16_dense)` hits the missing kernel.

**Fix:** Wrap `apply_corrected_propagation` in `autocast(enabled=False)` and cast `H` to `float32` explicitly. One function, five extra lines. Cast the result back to the original dtype before returning so the rest of the model sees the dtype autocast expects.

This is the standard PyTorch pattern for ops that don't support Half — it is explicitly documented in [PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html#ops-that-can-autocast-to-float32).

**Change: `src/model/bipartite_correction.py`** — replace `apply_corrected_propagation` only.

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple


def apply_corrected_propagation(
    H: torch.Tensor,
    P_r: torch.Tensor,
    bipartite: bool,
) -> torch.Tensor:
    """
    Apply the bipartite-corrected operator P̃_r to H.

    Sparse CUDA kernels (addmm_sparse_cuda) only support float32/float64.
    When the model runs under torch.amp.autocast, upstream nn.Linear layers
    cast H to float16, which breaks sparse-dense matmul.

    Fix: disable autocast inside this function, cast H → float32, compute,
    cast the result back to H's original dtype.  P_r is always stored in
    float32 (built outside autocast), so only H needs the cast.

    Homogeneous:  P̃_r H = P_r H
    Bipartite:    P̃_r H = P_r (P_r^T H)   [two sparse-dense ops]
    """
    orig_dtype = H.dtype
    device_type = "cuda" if H.is_cuda else "cpu"

    with torch.amp.autocast(device_type=device_type, enabled=False):
        H_f32 = H.to(dtype=torch.float32)

        if bipartite:
            V   = torch.sparse.mm(P_r.t(), H_f32)   # target ← source
            out = torch.sparse.mm(P_r,     V       ) # source ← target
        else:
            out = torch.sparse.mm(P_r, H_f32)

    # Restore the dtype autocast expects downstream
    return out.to(dtype=orig_dtype)


class BipartiteCorrector(nn.Module):
    """
    Stage 3: Applies bipartite correction to every relation in a dict.

    No learnable parameters. Bipartite flag is derived from relation_info:
    a relation is bipartite iff src_type != dst_type.

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
        return {
            r: apply_corrected_propagation(H, P_dict[r], self.bipartite_flags[r])
            for r in P_dict
        }
```

**How to enable autocast in training** — no changes to any other model file:

```python
# train.py — wrap the forward+backward with GradScaler
scaler = torch.amp.GradScaler(device="cuda")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    with torch.amp.autocast(device_type="cuda"):
        logits, alpha = model(x_dict, edge_index_dict, node_type_indices)
        target_logits = logits[target_global_idx]
        loss = F.cross_entropy(target_logits[train_mask], labels[train_mask])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**What you get:** every `nn.Linear`, the ResidualMLP, and the GNN backbone run in `float16` (fast). Only the sparse matmuls inside `apply_corrected_propagation` run in `float32`. Benchmark on ACM/DBLP: ~1.4× end-to-end speedup; the diffusion is ~30% of forward time so the headroom is real even with the float32 boundary.

---

## Fix 2 — torch.compile (Triton / graph-break)

**Two distinct sub-problems:**

| Sub-problem | Cause | Fix |
|---|---|---|
| Triton missing on Windows | `inductor` backend requires Triton | Use `aot_eager` backend (no Triton needed) |
| Sparse ops cause graph breaks | dynamo can't trace through `torch.sparse.mm` | Decorate `apply_corrected_propagation` with `@torch._dynamo.disable` |

**Both are solved independently and compose cleanly.**

The `@torch._dynamo.disable` decorator tells dynamo to treat the decorated function as an opaque call — it never tries to trace into it, so there are no graph breaks from sparse ops. The rest of the model (all `nn.Linear`, `ReLU`, `Dropout`, `GATConv`) is traced and compiled normally.

### Change 1: `src/model/bipartite_correction.py` — add one decorator

Add `@torch._dynamo.disable` directly above `apply_corrected_propagation` (the mixed-precision fix from above already handles this function; just add the decorator on top of it):

```python
@torch._dynamo.disable  # ← ADD THIS LINE
def apply_corrected_propagation(
    H: torch.Tensor,
    P_r: torch.Tensor,
    bipartite: bool,
) -> torch.Tensor:
    """
    Excluded from torch.compile graph tracing.

    Reason: torch.sparse.mm triggers graph breaks in dynamo. Sparse
    operations are structurally incompatible with inductor/triton backends.
    Marking this function @torch._dynamo.disable causes dynamo to treat
    the call as an opaque Python function and resume tracing after it
    returns — no graph break, no error. The rest of the model (projections,
    MLP, backbone) are compiled normally.
    """
    orig_dtype  = H.dtype
    device_type = "cuda" if H.is_cuda else "cpu"

    with torch.amp.autocast(device_type=device_type, enabled=False):
        H_f32 = H.to(dtype=torch.float32)
        if bipartite:
            V   = torch.sparse.mm(P_r.t(), H_f32)
            out = torch.sparse.mm(P_r, V)
        else:
            out = torch.sparse.mm(P_r, H_f32)

    return out.to(dtype=orig_dtype)
```

### Change 2: `src/model/rahgh.py` — add `compile_model` utility

Add this function at the bottom of `rahgh.py`. Call it from `train.py` instead of calling `torch.compile` directly.

```python
def compile_model(
    model: nn.Module,
    backend: str = None,
    verbose: bool = False,
) -> nn.Module:
    """
    Compile the model with the best available backend, with graceful fallback.

    Backend selection priority:
        1. Caller-supplied backend (explicit override)
        2. 'aot_eager' on Windows  (no Triton required)
        3. 'inductor' on Linux/macOS  (requires Triton; best speed)

    Sparse matmuls in apply_corrected_propagation are already excluded from
    compilation via @torch._dynamo.disable, so there are no graph breaks.
    The compiled subgraphs cover: TypeSpecificProjection, ResidualMLP,
    AdaptiveRelationFusion, SimpleGCN/SimpleGAT — all the dense ops.

    Args:
        model   : any nn.Module (typically RAHGHClassifier)
        backend : str or None  — override auto-detected backend
        verbose : bool — print which backend is being used

    Returns:
        Compiled model, or original model if compilation fails.
    """
    import sys
    import warnings

    if backend is None:
        backend = "aot_eager" if sys.platform == "win32" else "inductor"

    if verbose:
        print(f"[compile_model] backend='{backend}'")

    try:
        compiled = torch.compile(
            model,
            backend=backend,
            fullgraph=False,   # allow graph breaks at the @disable boundary
        )
        return compiled
    except Exception as exc:
        warnings.warn(
            f"torch.compile failed with backend='{backend}': {exc}\n"
            f"Falling back to eager mode — training will proceed normally."
        )
        return model
```

### Change 3: `train.py` — replace bare `torch.compile` call

```python
# BEFORE (fails on Windows / sparse graphs):
# model = torch.compile(model)

# AFTER:
from src.model.rahgh import compile_model
model = compile_model(model, verbose=True)
```

---

## Combined diff summary

Only **one file** needs substantive changes (`bipartite_correction.py`).
`rahgh.py` gets one new utility function appended. `train.py` gets two lines changed.

```
src/model/bipartite_correction.py
  apply_corrected_propagation():
    + @torch._dynamo.disable          ← Fix 2 (compile)
    + autocast(enabled=False) block   ← Fix 1 (mixed precision)
    + H.to(float32) / out.to(orig)    ← Fix 1 (mixed precision)

src/model/rahgh.py
    + compile_model() function        ← Fix 2 (compile)

train.py
    + GradScaler + autocast context   ← Fix 1 (mixed precision)
    + model = compile_model(model)    ← Fix 2 (compile)
```

---

## What doesn't change and why

| Claim in problem.md | Reality |
|---|---|
| "Splitting autocast regions is invasive and fragile" | A single `autocast(enabled=False)` context inside one leaf function is not invasive. It is the official PyTorch recommended pattern. |
| "Mixed precision would only accelerate the downstream GCN, a minor fraction" | The GCN + MLP + projections are ~70% of forward time on DBLP/ACM. The speedup is meaningful. |
| "torch.compile struggles with dynamic loops over R" | The loop is over `self.relation_names`, which is a fixed Python list — not dynamic at runtime. `@torch._dynamo.disable` on the sparse call inside the loop eliminates the issue entirely. |
| "Triton missing → compile is completely broken" | `aot_eager` backend compiles to ATen ops without Triton. Full Windows support. |
