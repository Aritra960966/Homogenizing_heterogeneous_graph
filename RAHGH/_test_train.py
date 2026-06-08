import torch, time, numpy as np
from src.data.dblp_loader import load_dblp
from src.model.diffusion import build_operators
from src.model.rahgh import RAHGH
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

data = load_dblp()

device = torch.device('cuda')
P_list = build_operators(data["A_list_sp"], data["bipartite_flags"], device)
X_list = [x.to(device) for x in data["X_dict"].values()]

Nt = data['target_size']
lbl_np = data['labels'].numpy()
tr80, te20 = train_test_split(np.arange(Nt), test_size=0.20, random_state=42, stratify=lbl_np)

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

for fold, (tr_fold, va_fold) in enumerate(skf.split(tr80, lbl_np[tr80])):
    tr_idx = tr80[tr_fold]
    va_idx = tr80[va_fold]
    
    in_dims = [x.shape[1] for x in data["X_dict"].values()]
    R = len(data["A_list_sp"])
    
    model = RAHGH(
        in_dims=in_dims, d=64, R=R, K=3,
        gcn_hidden=64,
        out_dim=data['n_classes'],
        dropout=0.5,
        A_list_sp=data["A_list_sp"], N=data['N'], device=device,
    ).to(device)
    
    opt = Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    tr_t = torch.tensor(tr_idx, dtype=torch.long, device=device)
    va_t = torch.tensor(va_idx, dtype=torch.long, device=device)
    labels = data['labels'].to(device)
    
    for ep in tqdm(range(1, 51), desc="Training"):
        model.train()
        opt.zero_grad()
        logits, *_ = model(X_list, P_list)
        loss = F.cross_entropy(logits[:Nt][tr_t], labels[tr_t])
        loss.backward()
        opt.step()
    break  # just one fold for testing
