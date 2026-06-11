import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import time, os
from tqdm import tqdm

from ..model.rahgh import (
    RAHGHClassifier, compile_model,
    build_rahgh_classifier, build_edge_index_dict, build_node_type_indices,
)


def _evaluate(logits, target_size, idx, labels_full):
    p = logits[:target_size][idx].argmax(1).cpu().numpy()
    y = labels_full[idx].numpy()
    prob = torch.softmax(logits[:target_size][idx], dim=1).cpu().numpy()
    n_classes = prob.shape[1]
    if n_classes == 2:
        auc = roc_auc_score(y, prob[:, 1])
    else:
        auc = roc_auc_score(y, prob, multi_class='ovr')
    return ((p == y).mean(),
            f1_score(y, p, average='macro',  zero_division=0),
            f1_score(y, p, average='micro',  zero_division=0),
            auc)


def run_single_nc(data, K, epochs, seed, cfg, head='gcn'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_dict = {k: v.to(device) for k, v in data['X_dict'].items()}
    edge_index_dict = build_edge_index_dict(data, device)
    node_type_indices = {k: v.to(device) for k, v in build_node_type_indices(data).items()}
    labels = data['labels'].to(device)
    Nt = data['target_size']
    d = cfg['d']

    model = build_rahgh_classifier(
        data, hidden_dim=d, num_classes=data['n_classes'], K=K,
        head=head,
        dropout_homo=cfg['dropout'], dropout_gnn=cfg['dropout'],
        gnn_hidden_dim=cfg.get('hidden', d),
    ).to(device)
    model = compile_model(model)
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

    best_val, best_alpha, best_sd = 0.0, None, None
    t0 = time.time()

    pbar = tqdm(range(1, epochs + 1), desc="Training", leave=False)
    for ep in pbar:
        model.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits, a = model(x_dict, edge_index_dict, node_type_indices)
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
            logits, a = model(x_dict, edge_index_dict, node_type_indices)
            _, vm, _, _ = _evaluate(logits, Nt, va_t.cpu().numpy(), data['labels'])
        pbar.set_description(f"loss={loss.item():.4f} val_macro={vm:.4f}")
        if vm > best_val:
            best_val = vm
            best_alpha = a.detach().cpu().numpy().copy()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        logits, *_ = model(x_dict, edge_index_dict, node_type_indices)
        acc, macro, micro, auc = _evaluate(logits, Nt, te_t.cpu().numpy(), data['labels'])

    return dict(test_acc=acc, test_macro=macro, test_micro=micro, test_auc=auc,
                best_val_macro=best_val, alpha=best_alpha,
                time_sec=time.time() - t0)


def run_final_nc(data, best_params, tr80_idx, te20_idx, seed=42,
                 out_dir=None, head='gcn'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_dict = {k: v.to(device) for k, v in data['X_dict'].items()}
    edge_index_dict = build_edge_index_dict(data, device)
    node_type_indices = {k: v.to(device) for k, v in build_node_type_indices(data).items()}
    labels = data['labels'].to(device)
    Nt = data['target_size']
    d = best_params['d']

    model = build_rahgh_classifier(
        data, hidden_dim=d, num_classes=data['n_classes'],
        K=best_params['K'], head=head,
        dropout_homo=best_params['dropout'], dropout_gnn=best_params['dropout'],
        gnn_hidden_dim=best_params.get('hidden', d),
    ).to(device)
    model = compile_model(model)
    opt = Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=best_params['epochs'], eta_min=best_params['lr'] * 0.01,
    )
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    # Further split tr80 into train (90%) and validation (10%) for early stopping
    from sklearn.model_selection import train_test_split
    lbl_np = data['labels'].numpy()
    tr_idx, va_idx = train_test_split(tr80_idx, test_size=0.125,
                                       random_state=seed, stratify=lbl_np[tr80_idx])
    tr_t = torch.tensor(tr_idx, dtype=torch.long, device=device)
    va_t = torch.tensor(va_idx, dtype=torch.long, device=device)
    t0 = time.time()

    best_val_macro = 0.0
    best_sd = None
    epoch_rows = []
    pbar = tqdm(range(1, best_params['epochs'] + 1), desc="Final training")
    for ep in pbar:
        model.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits, *_ = model(x_dict, edge_index_dict, node_type_indices)
            loss = F.cross_entropy(logits[:Nt][tr_t], labels[tr_t], label_smoothing=0.0)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_ev, _ = model(x_dict, edge_index_dict, node_type_indices)
            preds = logits_ev[:Nt][tr_t].argmax(1).cpu().numpy()
            tr_acc = (preds == labels[tr_t].cpu().numpy()).mean()
            _, vm, _, _ = _evaluate(logits_ev, Nt, va_t.cpu().numpy(), data['labels'])
        if vm > best_val_macro:
            best_val_macro = vm
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        epoch_rows.append({'epoch': ep, 'loss': loss.item(),
                           'train_acc': float(tr_acc)})
        if ep % 100 == 0 or ep == best_params['epochs']:
            pbar.set_description(f"loss={loss.item():.4f} val_macro={vm:.4f}")

    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        logits, alpha = model(x_dict, edge_index_dict, node_type_indices)
        acc, macro, micro, auc = _evaluate(logits, Nt, te20_idx, data['labels'])

    # Save final model
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        pt_path = os.path.join(out_dir, f'final_model_seed{seed}.pt')
        torch.save(model.state_dict(), pt_path)
        print(f"  Model saved → {pt_path}")

    # Save epoch metrics
    if out_dir is not None:
        import csv
        from pathlib import Path
        ep_path = Path(out_dir) / f'epoch_metrics_seed{seed}.csv'
        write_header = not ep_path.exists()
        with open(ep_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'train_acc'])
            if write_header:
                w.writeheader()
            w.writerows(epoch_rows)

    return dict(test_acc=acc, test_macro=macro, test_micro=micro, test_auc=auc,
                alpha=alpha.detach().cpu().numpy(),
                time_sec=time.time() - t0)
