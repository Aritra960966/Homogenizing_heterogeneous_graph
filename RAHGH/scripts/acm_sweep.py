"""
ACM Full Experiment Sweep — Corrected for RAHGH model pipeline v3.

Uses RAHGHClassifier from src/model/rahgh.py with proper 6-stage pipeline:
  Stage 1 — TypeSpecificProjection (Linear + LayerNorm + ReLU)
  Stage 2 — build_propagation_operator  D^{-1/2}(A_r+I)D^{-1/2}
  Stage 3 — BipartiteCorrector
  Stage 4 — RelationPolynomialDiffusion  (Σ β_{r,k} P̃_r^k)
  Stage 5 — AdaptiveRelationFusion        (Σ α_r Z_r)
  Stage 6 — ResidualMLP                   (MLP([H0 ‖ Z]))
  Head   — SimpleGCN or SimpleGAT

7 relations: PA, AP, PT, TP, PV, VP, PP
4 node types: paper (target), author, term, venue
3 classes: Database, Wireless Comm, Data Mining
"""
import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split as sk_split
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.model.rahgh import (
    RAHGHClassifier, compile_model, build_edge_index_dict, build_node_type_indices,
)

# ── Config ────────────────────────────────────────────────────
d        = 64
d_prime  = 32
dropout  = 0.5
lr       = 0.005
wd       = 1e-4
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH_LIST = [200, 300, 400, 500, 600, 700]
K_LIST     = [2, 3, 4, 5, 6]
N_RUNS     = 10
GNN_HEAD   = 'gcn'

total = len(EPOCH_LIST) * len(K_LIST) * N_RUNS
print(f"Device: {device}")
print(f"Total runs: {len(EPOCH_LIST)} x {len(K_LIST)} x {N_RUNS} = {total}")

# ════════════════════════════════════════════════════════════════
# DATA LOADING — keeps 7-relation layout with term + venue types
# ════════════════════════════════════════════════════════════════
mat = scipy.io.loadmat('ACM.mat')

PvsA  = mat['PvsA'].astype(np.float32).tocsr()
PvsC  = mat['PvsC'].astype(np.float32).tocsr()
nPvsT = mat['nPvsT'].astype(np.float32).tocsr()
PvsV  = mat['PvsV'].astype(np.float32).tocsr()
PvsP  = mat['PvsP'].astype(np.float32).tocsr()

C_names = [str(mat['C'][i, 0].flat[0]) for i in range(mat['C'].shape[0])]

area_confs = {
    'Database'      : ['SIGMOD', 'VLDB', 'ICDE', 'PODS'],
    'Wireless_Comm' : ['MobiCOMM', 'SIGCOMM', 'INFOCOM', 'ICNP'],
    'Data_Mining'   : ['KDD', 'WWW', 'ICDM', 'SIGIR', 'CIKM', 'WSDM'],
}
area_label = {area: i for i, area in enumerate(area_confs)}

conf_to_class = {}
for area, confs in area_confs.items():
    for c in confs:
        if c in C_names:
            conf_to_class[c] = area_label[area]

paper_conf_id = np.asarray(PvsC.todense()).argmax(axis=1).flatten()
paper_ids, labels_np = [], []
for pid in range(len(paper_conf_id)):
    cname = C_names[paper_conf_id[pid]]
    if cname in conf_to_class:
        paper_ids.append(pid)
        labels_np.append(conf_to_class[cname])
paper_ids = np.array(paper_ids)
labels_np = np.array(labels_np)
Np = len(paper_ids)
n_classes = 3
labels = torch.tensor(labels_np, dtype=torch.long)

PvsA_sub  = PvsA[paper_ids]
nPvsT_sub = nPvsT[paper_ids]
PvsV_sub  = PvsV[paper_ids]
PvsP_sub  = PvsP[paper_ids][:, paper_ids]

Na = PvsA_sub.shape[1]
Nt = nPvsT_sub.shape[1]
Nv = PvsV_sub.shape[1]
N  = Np + Na + Nt + Nv

p_off = 0
a_off = Np
t_off = Np + Na
v_off = Np + Na + Nt

def embed_bipartite(A_sp, row_offset, col_offset, N):
    A = A_sp.tocoo().astype(np.float32)
    return sp.coo_matrix(
        (A.data, (A.row + row_offset, A.col + col_offset)),
        shape=(N, N)
    ).tocsr()

def embed_homo(A_sp, offset, N):
    A = A_sp.tocoo().astype(np.float32)
    return sp.coo_matrix(
        (A.data, (A.row + offset, A.col + offset)),
        shape=(N, N)
    ).tocsr()

PA_emb = embed_bipartite(PvsA_sub,  p_off, a_off, N)
AP_emb = PA_emb.T.tocsr()
PT_emb = embed_bipartite(nPvsT_sub, p_off, t_off, N)
TP_emb = PT_emb.T.tocsr()
PV_emb = embed_bipartite(PvsV_sub,  p_off, v_off, N)
VP_emb = PV_emb.T.tocsr()
PP_emb = embed_homo(PvsP_sub, p_off, N)

relation_names = ['paper_author', 'author_paper',
                  'paper_term',   'term_paper',
                  'paper_venue',  'venue_paper',
                  'paper_paper']
A_list_sp = [PA_emb, AP_emb, PT_emb, TP_emb, PV_emb, VP_emb, PP_emb]
R = len(A_list_sp)

# ── Features ──────────────────────────────────────────────────
X_paper  = torch.tensor(np.array(nPvsT_sub.todense(), dtype=np.float32))
U, S, Vt = spla.svds(PvsA_sub.T.astype(np.float64), k=64)
X_author = torch.tensor((U * S).astype(np.float32))
X_term   = torch.eye(Nt, dtype=torch.float32)
X_venue  = torch.eye(Nv, dtype=torch.float32)

X_dict = {
    'paper':  X_paper,
    'author': X_author,
    'term':   X_term,
    'venue':  X_venue,
}
node_type_dims = {k: v.shape[1] for k, v in X_dict.items()}

# ── Build data dict in the format expected by RAHGHClassifier ──
relation_info = {
    'paper_author': ('paper', 'author'),
    'author_paper': ('author', 'paper'),
    'paper_term':   ('paper', 'term'),
    'term_paper':   ('term', 'paper'),
    'paper_venue':  ('paper', 'venue'),
    'venue_paper':  ('venue', 'paper'),
    'paper_paper':  ('paper', 'paper'),
}

node_type_indices = {
    'paper':  torch.arange(p_off, p_off + Np),
    'author': torch.arange(a_off, a_off + Na),
    'term':   torch.arange(t_off, t_off + Nt),
    'venue':  torch.arange(v_off, v_off + Nv),
}

data = {
    'X_dict':            X_dict,
    'node_type_dims':    node_type_dims,
    'node_type_indices': node_type_indices,
    'A_list_sp':         A_list_sp,
    'relation_names':    relation_names,
    'relation_info':     relation_info,
    'labels':            labels,
    'N':                 N,
    'target_type':       'paper',
    'target_size':       Np,
    'n_classes':         n_classes,
}

# ── Pre-build edge_index_dict and move to device once ─────────
edge_index_dict = build_edge_index_dict(data, device)
nt_indices_d    = {k: v.to(device) for k, v in node_type_indices.items()}
x_dict_d        = {k: v.to(device) for k, v in X_dict.items()}
labels_d        = labels.to(device)

# ════════════════════════════════════════════════════════════════════
# SINGLE RUN
# ════════════════════════════════════════════════════════════════════
def run_single(K_val, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = RAHGHClassifier(
        node_type_dims=node_type_dims,
        relation_info=relation_info,
        num_nodes=N,
        hidden_dim=d,
        num_classes=n_classes,
        K=K_val,
        head=GNN_HEAD,
        dropout_homo=dropout,
        dropout_gnn=0.5,
        gnn_hidden_dim=d_prime,
    ).to(device)
    model = compile_model(model)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Stratified split
    train_idx_, test_idx_ = sk_split(
        np.arange(Np), test_size=600,
        random_state=seed, stratify=labels_np)
    train_idx_, val_idx_ = sk_split(
        train_idx_, test_size=300,
        random_state=seed, stratify=labels_np[train_idx_])

    tr_t = torch.tensor(np.sort(train_idx_), dtype=torch.long, device=device)
    va_t = torch.tensor(np.sort(val_idx_),   dtype=torch.long, device=device)
    te_t = torch.tensor(np.sort(test_idx_),  dtype=torch.long, device=device)

    best_val_macro = 0.0
    best_sd = None
    stall = 0
    patience = 100

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits, _ = model(x_dict_d, edge_index_dict, nt_indices_d)
        loss = F.cross_entropy(logits[:Np][tr_t], labels_d[tr_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(x_dict_d, edge_index_dict, nt_indices_d)
            preds = logits[:Np][va_t].argmax(1).cpu().numpy()
            truth = labels[va_t.cpu()].numpy()
            val_macro = f1_score(truth, preds, average='macro', zero_division=0)

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        logits, alpha = model(x_dict_d, edge_index_dict, nt_indices_d)
        preds = logits[:Np][te_t].argmax(1).cpu().numpy()
        truth = labels[te_t.cpu()].numpy()
        test_acc   = (preds == truth).mean()
        test_macro = f1_score(truth, preds, average='macro', zero_division=0)
        test_micro = f1_score(truth, preds, average='micro', zero_division=0)

    return {
        'test_acc'      : test_acc,
        'test_macro'    : test_macro,
        'test_micro'    : test_micro,
        'best_val_macro': best_val_macro,
        'alpha'         : alpha.detach().cpu().numpy(),
        'time_sec'      : time.time() - t0,
    }

# ════════════════════════════════════════════════════════════════════
# EXPERIMENT LOOP
# ════════════════════════════════════════════════════════════════════
csv_path = 'acm_experiment_results.csv'
rows = []

fieldnames = [
    'epochs', 'K', 'run', 'seed',
    'test_acc', 'test_macro_f1', 'test_micro_f1', 'best_val_macro',
    'time_sec',
] + [f'alpha_{i}' for i in range(R)]

done = 0
t_global = time.time()

print("\n" + "=" * 70)
print("EXPERIMENT START — ACM Paper Classification (RAHGH Model)")
print(f"  Classes : Database / Wireless Comm / Data Mining")
print(f"  Papers  : {Np}  |  train={Np-900}  val=300  test=600")
print(f"  Runs    : {total}  ({len(EPOCH_LIST)} epochs x {len(K_LIST)} K x {N_RUNS} runs)")
print(f"  Model   : RAHGHClassifier (d={d}, head={GNN_HEAD})")
print("=" * 70)

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for epochs in EPOCH_LIST:
        for K_val in K_LIST:
            run_accs, run_macros, run_micros = [], [], []

            for run in range(1, N_RUNS + 1):
                seed = run * 42
                done += 1

                print(f"  [{done:3d}/{total}] epochs={epochs}  K={K_val}  "
                      f"run={run}/{N_RUNS}  seed={seed} ...",
                      end=' ', flush=True)

                result = run_single(K_val, epochs, seed)

                run_accs.append(result['test_acc'])
                run_macros.append(result['test_macro'])
                run_micros.append(result['test_micro'])

                print(f"acc={result['test_acc']:.4f}  "
                      f"macro={result['test_macro']:.4f}  "
                      f"micro={result['test_micro']:.4f}  "
                      f"({result['time_sec']:.1f}s)")

                row = {
                    'epochs'        : epochs,
                    'K'             : K_val,
                    'run'           : run,
                    'seed'          : seed,
                    'test_acc'      : round(result['test_acc'],        4),
                    'test_macro_f1' : round(result['test_macro'],      4),
                    'test_micro_f1' : round(result['test_micro'],      4),
                    'best_val_macro': round(result['best_val_macro'],  4),
                    'time_sec'      : round(result['time_sec'],        2),
                }
                for i in range(R):
                    row[f'alpha_{i}'] = round(float(result['alpha'][i]), 4)

                rows.append(row)
                writer.writerow(row)
                f.flush()

            print(f"\n  --- Summary  epochs={epochs}  K={K_val} ---")
            print(f"     Acc   : mean={np.mean(run_accs):.4f}  "
                  f"std={np.std(run_accs):.4f}  "
                  f"min={np.min(run_accs):.4f}  max={np.max(run_accs):.4f}")
            print(f"     Macro : mean={np.mean(run_macros):.4f}  "
                  f"std={np.std(run_macros):.4f}  "
                  f"min={np.min(run_macros):.4f}  max={np.max(run_macros):.4f}")
            print(f"     Micro : mean={np.mean(run_micros):.4f}  "
                  f"std={np.std(run_micros):.4f}  "
                  f"min={np.min(run_micros):.4f}  max={np.max(run_micros):.4f}\n")

print("=" * 70)
print(f"All {total} runs complete in {(time.time()-t_global)/60:.1f} min")
print(f"Results saved to: {csv_path}")

# ════════════════════════════════════════════════════════════════════
# SUMMARY TABLE + PLOTS
# ════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("AGGREGATE SUMMARY  (mean +/- std over 10 runs)")
print("=" * 70)
summary = (results_df
           .groupby(['epochs', 'K'])[['test_acc', 'test_macro_f1', 'test_micro_f1']]
           .agg(['mean', 'std'])
           .round(4))
print(summary.to_string())
summary.to_csv('acm_experiment_summary.csv')
print("\nSaved: acm_experiment_results.csv  acm_experiment_summary.csv")

colors  = {2: 'steelblue', 3: 'darkorange', 4: 'green', 5: 'red', 6: 'purple'}
metrics = ['test_acc', 'test_macro_f1', 'test_micro_f1']
titles  = ['Accuracy', 'Macro-F1', 'Micro-F1']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric, title in zip(axes, metrics, titles):
    for K_val in K_LIST:
        subset = results_df[results_df['K'] == K_val]
        means  = subset.groupby('epochs')[metric].mean()
        stds   = subset.groupby('epochs')[metric].std()
        ax.plot(means.index, means.values,
                marker='o', label=f'K={K_val}', color=colors[K_val])
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.15, color=colors[K_val])
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_xticks(EPOCH_LIST)
    ax.legend()
    ax.grid(alpha=0.3)

fig.suptitle(f'ACM (DB / WC / DM) — RAHGH + SimpleGCN  ({N_RUNS} runs mean +/- std)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('acm_experiment_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: acm_experiment_plots.png")
