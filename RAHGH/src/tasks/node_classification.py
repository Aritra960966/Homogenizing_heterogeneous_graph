import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

from ..model.rahgh    import RAHGH, compile_model
from ..model.diffusion import build_operators


def _evaluate(logits, target_size, idx, labels_full):
    p = logits[:target_size][idx].argmax(1).cpu().numpy()
    y = labels_full[idx].numpy()
    return ((p == y).mean(),
            f1_score(y, p, average='macro',  zero_division=0),
            f1_score(y, p, average='micro',  zero_division=0))


def run_single_nc(data, K, epochs, seed, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P_list  = build_operators(data['A_list_sp'], data['bipartite_flags'],
                              device)
    X_list  = [x.to(device) for x in data['X_dict'].values()]
    labels  = data['labels'].to(device)
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    R       = len(data['A_list_sp'])
    Nt      = data['target_size']
    d       = cfg['d']

    model = RAHGH(
        in_dims=in_dims, d=d, R=R, K=K,
        gcn_hidden=cfg['gcn_hidden'],
        out_dim=data['n_classes'],
        dropout=cfg['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    model = compile_model(model, verbose=True)
    opt = Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    lbl_np = data['labels'].numpy()
    tr, te = train_test_split(np.arange(Nt), test_size=0.20,
                               random_state=seed, stratify=lbl_np)
    tr, va = train_test_split(tr, test_size=0.10 / 0.80,
                               random_state=seed, stratify=lbl_np[tr])
    tr_t = torch.tensor(tr, dtype=torch.long, device=device)
    va_t = torch.tensor(va, dtype=torch.long, device=device)
    te_t = torch.tensor(te, dtype=torch.long, device=device)

    best_val, best_alpha, best_beta, best_sd = 0.0, None, None, None
    t0 = time.time()

    pbar = tqdm(range(1, epochs + 1), desc="Training", leave=False)
    for ep in pbar:
        model.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits, a, b, _ = model(X_list, P_list)
            loss = F.cross_entropy(logits[:Nt][tr_t], labels[tr_t])
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits, a, b, _ = model(X_list, P_list)
            _, vm, _ = _evaluate(logits, Nt, va_t.cpu().numpy(),
                                 data['labels'])
        pbar.set_description(f"loss={loss.item():.4f} val_macro={vm:.4f}")
        if vm > best_val:
            best_val   = vm
            best_alpha = a.detach().cpu().numpy().copy()
            best_beta  = b.detach().cpu().numpy().copy()
            best_sd    = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        logits, *_ = model(X_list, P_list)
        acc, macro, micro = _evaluate(logits, Nt, te_t.cpu().numpy(),
                                      data['labels'])

    return dict(test_acc=acc, test_macro=macro, test_micro=micro,
                best_val_macro=best_val, alpha=best_alpha, beta=best_beta,
                time_sec=time.time() - t0)


def run_final_nc(data, best_params, tr80_idx, te20_idx, seed=42,
                 out_dir=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P_list  = build_operators(data['A_list_sp'], data['bipartite_flags'],
                              device)
    X_list  = [x.to(device) for x in data['X_dict'].values()]
    labels  = data['labels'].to(device)
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    R       = len(data['A_list_sp'])
    Nt      = data['target_size']
    d       = best_params['d']

    model = RAHGH(
        in_dims=in_dims, d=d, R=R, K=best_params['K'],
        gcn_hidden=best_params['gcn_hidden'],
        out_dim=data['n_classes'],
        dropout=best_params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    model = compile_model(model, verbose=True)
    opt = Adam(model.parameters(),
               lr=best_params['lr'], weight_decay=best_params['wd'])
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    tr_t = torch.tensor(tr80_idx, dtype=torch.long, device=device)
    te_t = torch.tensor(te20_idx, dtype=torch.long, device=device)
    t0   = time.time()

    epoch_rows = []
    pbar = tqdm(range(1, best_params['epochs'] + 1), desc="Final training")
    for ep in pbar:
        model.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits, *_ = model(X_list, P_list)
            loss = F.cross_entropy(logits[:Nt][tr_t], labels[tr_t])
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        # Track training accuracy
        with torch.no_grad():
            preds = logits[:Nt][tr_t].argmax(1).cpu().numpy()
            tr_acc = (preds == labels[tr_t].cpu().numpy()).mean()
        epoch_rows.append({'epoch': ep, 'loss': loss.item(),
                           'train_acc': float(tr_acc)})
        if ep % 100 == 0 or ep == best_params['epochs']:
            pbar.set_description(f"loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits, alpha, beta, _ = model(X_list, P_list)
        acc, macro, micro = _evaluate(logits, Nt, te20_idx, data['labels'])

    # Save epoch metrics
    if out_dir is not None:
        import csv
        from pathlib import Path
        ep_path = Path(out_dir) / f'epoch_metrics_seed{seed}.csv'
        with open(ep_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'train_acc'])
            w.writeheader()
            w.writerows(epoch_rows)

    return dict(test_acc=acc, test_macro=macro, test_micro=micro,
                alpha=alpha.detach().cpu().numpy(),
                beta=beta.detach().cpu().numpy(),
                time_sec=time.time() - t0)
