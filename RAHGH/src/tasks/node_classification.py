import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import time, os, warnings
from tqdm import tqdm

from ..model.rahgh import (
    compile_model,
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
        dropout_homo=best_params['dropout'],
        dropout_gnn=best_params.get('dropout_gnn', best_params['dropout']),
        gnn_hidden_dim=best_params.get('hidden', d),
    ).to(device)
    model = compile_model(model)
    opt = AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['wd'])
    warmup_epochs = best_params.get('warmup', 0)
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=warmup_epochs
        )
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=best_params['epochs'], eta_min=best_params['lr'] * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sched, main_sched], milestones=[warmup_epochs]
        )
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
    else:
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
    stall = 0
    patience = 100
    epoch_rows = []
    pbar = tqdm(range(1, best_params['epochs'] + 1), desc="Final NC training")
    for ep in pbar:
        model.train()
        opt.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits, *_ = model(x_dict, edge_index_dict, node_type_indices)
            loss = F.cross_entropy(logits[:Nt][tr_t], labels[tr_t],
                                    label_smoothing=best_params.get('label_smoothing', 0.1))
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_ev, _ = model(x_dict, edge_index_dict, node_type_indices)
            preds = logits_ev[:Nt][tr_t].argmax(1).cpu().numpy()
            tr_acc = (preds == labels[tr_t].cpu().numpy()).mean()
            _, vm, _, _ = _evaluate(logits_ev, Nt, va_t.cpu().numpy(), data['labels'])
        epoch_rows.append({'epoch': ep, 'loss': loss.item(),
                           'val_macro': float(vm)})
        pbar.set_description(f"loss={loss.item():.4f} val_macro={vm:.4f}")
        if vm > best_val_macro:
            best_val_macro = vm
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                pbar.set_description(f"Early stop @{ep}/{best_params['epochs']} best_val_macro={best_val_macro:.4f}")
                break

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
            w = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'val_macro'])
            if write_header:
                w.writeheader()
            w.writerows(epoch_rows)

    return dict(test_acc=acc, test_macro=macro, test_micro=micro, test_auc=auc,
                alpha=alpha.detach().cpu().numpy(),
                time_sec=time.time() - t0)
