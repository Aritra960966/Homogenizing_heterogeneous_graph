import torch, time
from src.data.dblp_loader import load_dblp
from src.model.diffusion import build_operators

print("Loading DBLP...")
t0 = time.time()
data = load_dblp()
print(f"Loaded in {time.time()-t0:.1f}s, N={data['N']}")

print("Building operators...")
t0 = time.time()
try:
    P_list = build_operators(data["A_list_sp"], data["bipartite_flags"], "cuda")
    print(f"Built {len(P_list)} operators in {time.time()-t0:.1f}s")
    for i, p in enumerate(P_list):
        print(f"  {i}: {p.shape} dtype={p.dtype} device={p.device}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"ERROR: {e}")
