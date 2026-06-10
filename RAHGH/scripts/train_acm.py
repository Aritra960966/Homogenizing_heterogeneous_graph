import sys, os, time, json
if '__file__' in dir():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter
from tqdm import tqdm

from src.data.acm_loader import load_acm
from src.model.rahgh import RAHGH, compile_model
from src.model.diffusion import build_operators

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

data = load_acm()
labels = data['labels'].to(device)
lbl_np = data['labels'].numpy()
Nt = data['target_size']
X_list = [x.to(device) for x in data['X_dict'].values()]
P_list = build_operators(data['A_list_sp'], data['bipartite_flags'], device)
in_dims = [x.shape[1] for x in data['X_dict'].values()]
R = len(data['A_list_sp'])

# ── Dataset-level diagnostics (points 3, 7) ──
print(f"Nt={Nt}  n_classes={data['n_classes']}  N={data['N']}  R={R}")
print(f"target_type  = '{data['target_type']}'")
print(f"node types   : {list(data['X_dict'].keys())}")
print(f"Full label dist: {dict(sorted(Counter(lbl_np).items()))}")
assert data['target_type'] == 'paper', f"Expected target_type='paper', got '{data['target_type']}'"

# ── Stratified 80/20 split ──
tr80, te20 = train_test_split(np.arange(Nt), test_size=0.2, random_state=42, stratify=lbl_np)
print(f"Train={len(tr80)}  Test={len(te20)}")
print(f"Train label dist: {dict(sorted(Counter(lbl_np[tr80]).items()))}")
print(f"Test  label dist: {dict(sorted(Counter(lbl_np[te20]).items()))}")

# ── Hyperparameter grid (small, targeted) ──
GRID = [
    {'d': 64,  'K': 1, 'lr': 0.001, 'wd': 1e-4, 'dropout': 0.5, 'gcn_hidden': 64, 'epochs': 300},
    {'d': 64,  'K': 1, 'lr': 0.0005,'wd': 1e-4, 'dropout': 0.3, 'gcn_hidden': 64, 'epochs': 300},
    {'d': 128, 'K': 1, 'lr': 0.001, 'wd': 1e-4, 'dropout': 0.5, 'gcn_hidden': 64, 'epochs': 300},
    {'d': 64,  'K': 2, 'lr': 0.001, 'wd': 1e-4, 'dropout': 0.5, 'gcn_hidden': 64, 'epochs': 300},
    {'d': 64,  'K': 3, 'lr': 0.001, 'wd': 1e-4, 'dropout': 0.5, 'gcn_hidden': 64, 'epochs': 300},
]

PATIENCE = 30

def diagnose_epoch(ep, model, tr_t, va_t, tr_idx, va_idx):
    """Run full diagnostics: pred distribution, logit stats, embedding collapse."""
    model.eval()
    with torch.no_grad():
        logits, alpha, beta, _ = model(X_list, P_list)

    logits_paper = logits[:Nt]
    preds_val = logits_paper[va_t].argmax(1).cpu().numpy()
    preds_all = logits_paper.argmax(1).cpu().numpy()
    truth_val = lbl_np[va_idx]
    truth_train = lbl_np[tr_idx]

    probs = torch.softmax(logits_paper, dim=1)

    # Per-class F1 on val set
    val_macro = f1_score(truth_val, preds_val, average='macro', zero_division=0)
    per_class_f1 = f1_score(truth_val, preds_val, average=None, zero_division=0)

    # AUC (OvR)
    n_classes = len(np.unique(lbl_np))
    try:
        probs_np = probs[va_t].cpu().numpy()
        y_bin    = label_binarize(truth_val, classes=list(range(n_classes)))
        auc      = roc_auc_score(y_bin, probs_np, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    print(f"    [diag ep={ep}]")
    print(f"      true  dist (val)  : {dict(sorted(Counter(truth_val.tolist()).items()))}")
    print(f"      pred  dist (val)  : {dict(sorted(Counter(preds_val.tolist()).items()))}")
    print(f"      pred  dist (ALL)  : {dict(sorted(Counter(preds_all.tolist()).items()))}")
    print(f"      per-class F1 (val): {[f'{v:.4f}' for v in per_class_f1]}")
    print(f"      macro-F1 (val)    : {val_macro:.4f}")
    print(f"      AUC (OvR)   (val) : {auc:.4f}")
    print(f"      logits: mean={logits_paper.mean().item():.4f}  std={logits_paper.std().item():.4f}  "
          f"max={logits_paper.max().item():.4f}  min={logits_paper.min().item():.4f}")
    print(f"      probs : mean={probs.mean().item():.4f}  std={probs.std().item():.4f}")
    print(f"      alpha (rel weights): {F.softmax(model.diffusion.theta, dim=0).detach().cpu().numpy().round(4).tolist()}")

def run_fold(params, tr_idx, va_idx):
    tr_t = torch.tensor(tr_idx, dtype=torch.long, device=device)
    va_t = torch.tensor(va_idx, dtype=torch.long, device=device)

    model = RAHGH(
        in_dims=in_dims, d=params['d'], R=R, K=params['K'],
        gcn_hidden=params['gcn_hidden'], out_dim=data['n_classes'],
        dropout=params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    model = compile_model(model)
    opt = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    best_vm, best_sd, stall = 0.0, None, 0

    for ep in range(1, params['epochs'] + 1):
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

        # ── Diagnose on ep 1, 50, 100, and final ──
        if ep in (1, 50, 100) or ep == params['epochs']:
            diagnose_epoch(ep, model, tr_t, va_t, tr_idx, va_idx)

        model.eval()
        with torch.no_grad():
            logits, *_ = model(X_list, P_list)
        preds = logits[:Nt][va_t].argmax(1).cpu().numpy()
        truth = lbl_np[va_idx]
        vm = f1_score(truth, preds, average='macro', zero_division=0)

        if vm > best_vm:
            best_vm = vm
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= PATIENCE:
                break

    del model
    torch.cuda.empty_cache()
    return best_vm

def final_test(params, tr80_idx, te20_idx):
    tr_t = torch.tensor(tr80_idx, dtype=torch.long, device=device)
    te_t = torch.tensor(te20_idx, dtype=torch.long, device=device)

    model = RAHGH(
        in_dims=in_dims, d=params['d'], R=R, K=params['K'],
        gcn_hidden=params['gcn_hidden'], out_dim=data['n_classes'],
        dropout=params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    model = compile_model(model)
    opt = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    pbar = tqdm(range(1, params['epochs'] + 1), desc="Final", leave=False)
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
        if ep % 50 == 0 or ep == params['epochs']:
            pbar.set_description(f"loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits, alpha, beta, _ = model(X_list, P_list)
    preds = logits[:Nt][te_t].argmax(1).cpu().numpy()
    truth = lbl_np[te20_idx]
    acc  = (preds == truth).mean()
    macro = f1_score(truth, preds, average='macro', zero_division=0)
    micro = f1_score(truth, preds, average='micro', zero_division=0)
    auc   = float('nan')
    try:
        probs = torch.softmax(logits[:Nt][te_t], dim=1).cpu().numpy()
        n_cls = len(np.unique(lbl_np))
        y_bin = label_binarize(truth, classes=list(range(n_cls)))
        auc   = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
    except Exception:
        pass
    print(f"    Test: acc={acc:.4f}  macro={macro:.4f}  micro={micro:.4f}  auc={auc:.4f}")
    print(f"    Pred dist : {dict(sorted(Counter(preds.tolist()).items()))}")
    print(f"    True dist : {dict(sorted(Counter(truth.tolist()).items()))}")
    return acc, macro, micro, auc, alpha, beta

# ── 5-fold CV for each combo ──
print("\n" + "=" * 60)
print("5-Fold Cross-Validation")
print("=" * 60)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for ci, params in enumerate(GRID):
    fold_scores = []
    print(f"\n--- Combo {ci+1}/{len(GRID)}: {params} ---")
    for fold, (tr_fold, va_fold) in enumerate(skf.split(tr80, lbl_np[tr80])):
        vm = run_fold(params, tr80[tr_fold], tr80[va_fold])
        fold_scores.append(vm)
        print(f"  Fold {fold+1}: val_macro={vm:.4f}")
    mean_vm = float(np.mean(fold_scores))
    print(f"  >>> Mean val_macro={mean_vm:.4f}  folds={[f'{v:.4f}' for v in fold_scores]}")
    results.append({'params': params, 'mean_val_macro': mean_vm, 'fold_scores': fold_scores})

# ── Best combo → final test ──
best = max(results, key=lambda r: r['mean_val_macro'])
print("\n" + "=" * 60)
print(f"Best combo: {best['params']}  (mean_val_macro={best['mean_val_macro']:.4f})")
print("=" * 60)

acc, macro, micro, auc, alpha, beta = final_test(best['params'], tr80, te20)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Best CV params       : {json.dumps(best['params'], indent=2)}")
print(f"CV mean val macro-F1 : {best['mean_val_macro']:.4f}")
print(f"Test accuracy        : {acc:.4f}")
print(f"Test macro-F1        : {macro:.4f}")
print(f"Test micro-F1        : {micro:.4f}")
print(f"Test AUC (OvR)       : {auc:.4f}")
print(f"Alpha (rel weights)  : {alpha.round(4).tolist()}")
print(json.dumps({'params': best['params'], 'test_acc': round(acc, 4),
                  'test_macro': round(macro, 4), 'test_micro': round(micro, 4),
                  'test_auc': round(auc, 4)}, indent=2))
