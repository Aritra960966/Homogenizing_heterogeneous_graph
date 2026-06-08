# Problems with Mixed Precision and torch.compile

## 1. Mixed Precision (`torch.amp.autocast`)

**Error:**
```
RuntimeError: "addmm_sparse_cuda" not implemented for 'Half'
```

**Cause:**
`autocast` converts matrix multiplications to `float16` (Half) for speed. The RAHGH diffusion operator performs `P_list[r] @ Hk` where `P_list[r]` is a **sparse** CUDA tensor (`torch.sparse`). PyTorch's `addmm_sparse_cuda` kernel has no Half-precision implementation — it only supports `float32` and `float64`.

**Relevant code in `src/model/diffusion.py`:**
```python
# line 68
Hk = P_list[r] @ Hk   # sparse-dense matmul — fails under Half
```

**Why it can't be easily fixed:**
- Wrapping only dense layers in autocast (excluding the sparse diffusion) would require splitting the forward pass into autocast/non-autocast regions, which is invasive and fragile.
- Converting sparse operators to dense is memory-prohibitive for large graphs.
- The diffusion step is already the dominant cost; mixed precision would only accelerate the downstream GCN, a minor fraction of total compute.

## 2. torch.compile

**Error:**
```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Cannot find a working triton installation.
```

**Cause:**
`torch.compile` with the default `inductor` backend requires **Triton**, which is not available on Windows. Even with `torch._dynamo.config.suppress_errors = True`, the compiler failed again on a subsequent subgraph from the diffusion module with the same Triton error.

**Additional issue (even on Linux):**
The RAHGH model uses dynamic control flow and sparse operations inside `diffusion.forward()`:
```python
# diffusion.py: multiple sparse matmuls in a loop
for r in range(R):
    Hk = P_list[r] @ Hk
```
`torch.compile` struggles with dynamic loops over sparse operations, causing frequent graph breaks that negate any speed benefit. After a graph break, the remaining subgraph still gets passed to inductor, which fails on the sparse ops.

**Why it can't be easily fixed:**
- Triton is not available on Windows (no official build).
- Even on Linux, sparse operations trigger graph breaks, making compilation ineffective.
- The model's loop structure over relation types (`R`) is data-dependent and hard to static-compile.
