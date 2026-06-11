from src.data.acm_loader import load_acm
from src.utils.graph_utils import build_edge_index_dict
import torch, numpy as np

data = load_acm()
eid = build_edge_index_dict(data, 'cpu')

print(f"target_size = {data['target_size']}")
print(f"N = {data['N']}")
print(f"bipartite_flags = {data['bipartite_flags']}")
print()

# What are train indices? (simulate a split like hparam_search does)
from sklearn.model_selection import train_test_split, StratifiedKFold
Nt = data['target_size']
lbl_np = data['labels'].numpy()
tr80, te20 = train_test_split(np.arange(Nt), test_size=0.2, random_state=42, stratify=lbl_np)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold_i, (tr_i, va_i) in enumerate(skf.split(tr80, lbl_np[tr80])):
    tr_idx = tr80[tr_i]
    va_idx = tr80[va_i]
    if fold_i == 0:
        break

tr_t = torch.tensor(tr_idx, dtype=torch.long)
print(f"Fold 0: tr_idx range [{tr_idx.min()}-{tr_idx.max()}], len={len(tr_idx)}")
print()

for rel_name, ei in eid.items():
    s, t = ei[0], ei[1]
    n_edges = s.size(0)
    
    # How many edges pass the source-in-train mask?
    mask_src = torch.isin(s, tr_t)
    n_src_match = mask_src.sum().item()
    
    # How many edges pass if we check EITHER src or dst?
    mask_either = torch.isin(s, tr_t) | torch.isin(t, tr_t)
    n_either = mask_either.sum().item()
    
    print(f"Relation: {rel_name}")
    print(f"  total edges: {n_edges}")
    print(f"  src range: [{s.min().item()}-{s.max().item()}]")
    print(f"  dst range: [{t.min().item()}-{t.max().item()}]")
    print(f"  edges where SOURCE in tr_idx: {n_src_match} ({100*n_src_match/n_edges:.1f}%)")
    print(f"  edges where EITHER in tr_idx: {n_either} ({100*n_either/n_edges:.1f}%)")
    print()
