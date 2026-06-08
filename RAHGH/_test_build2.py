import torch, time
from src.data.dblp_loader import load_dblp
from src.model.diffusion import build_operators

print("Loading DBLP...")
t0 = time.time()
data = load_dblp()
print(f"Loaded in {time.time()-t0:.1f}s, N={data['N']}")

print("Building operators (sparse)...")
t0 = time.time()
try:
    P_list = build_operators(data["A_list_sp"], data["bipartite_flags"], "cuda")
    print(f"Built {len(P_list)} operators in {time.time()-t0:.1f}s")
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"GPU mem used: {mem:.2f} GB")
    for i, p in enumerate(P_list):
        print(f"  {i}: {p.shape} sparse({p._nnz()}) device={p.device}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"ERROR: {e}")
