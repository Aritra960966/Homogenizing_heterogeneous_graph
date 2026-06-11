import torch, time
from src.data.dblp_loader import load_dblp
from src.model.diffusion import build_operators
from src.model.rahgh import RAHGH

print("Loading...")
data = load_dblp()

device = torch.device('cuda')
print("Building operators...")
P_list = build_operators(data["A_list_sp"], data["bipartite_flags"], device)

X_list = [x.to(device) for x in data["X_dict"].values()]
in_dims = [x.shape[1] for x in data["X_dict"].values()]
R = len(data["A_list_sp"])
N = data["N"]

print(f"in_dims={in_dims}, R={R}, N={N}, n_classes={data['n_classes']}")

print("Creating model...")
t0 = time.time()
model = RAHGH(
    in_dims=in_dims, d=64, R=R, K=3,
    hidden=64,
    out_dim=data["n_classes"],
    dropout=0.5,
    A_list_sp=data["A_list_sp"], N=N, device=device,
).to(device)
print(f"Model created in {time.time()-t0:.1f}s")
print(f"Params: {sum(p.numel() for p in model.parameters())}")

print("Forward pass...")
t0 = time.time()
try:
    with torch.no_grad():
        logits, alpha, beta, _ = model(X_list, P_list)
    print(f"Forward done in {time.time()-t0:.1f}s")
    print(f"logits shape: {logits.shape}, alpha: {alpha}, beta: {beta.shape}")
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"GPU mem: {mem:.2f} GB")
except Exception as e:
    import traceback
    traceback.print_exc()
