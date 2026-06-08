# RAHGH — Full Experiment Implementation Guide (v4)
## ACM · DBLP · IMDB | Node Classification · Link Prediction · Graph Clustering · Recommendation

> **v4 Changelog — extends v3 without breaking any existing code**
>
> | # | File | Change |
> |---|------|--------|
> | 1 | `tasks/graph_clustering.py` | **NEW** — unsupervised clustering on Z_final; metrics NMI, ARI, ACC |
> | 2 | `tasks/recommendation.py` | **NEW** — top-K recommendation decoder; metrics Recall@K, NDCG@K, Hit@K, Precision@K, MRR |
> | 3 | `tasks/hparam_search.py` | **Extended** — CV wrappers for clustering and recommendation added |
> | 4 | `train.py` | **Extended** — handles all 4 tasks; final test repeated **10 seeds** (was 5) |
> | 5 | `results/` | **Restructured** — one sub-folder per task; separate CSVs for CV scores, per-run results, epoch-level loss/metrics, summary |
> | 6 | `configs/` | New YAML files for clustering and recommendation tasks |
> | 7 | Hyperparameter grid | Updated with clustering-specific and recommendation-specific params |
>
> v3 fixes (LP GCN output, 80/20 split, α-weighted A_hat, bipartite correction) all retained unchanged.

---

## Dimension flow — must match paper at every stage

The paper states (§4, §5.2): `d = 64`, `Z_final ∈ ℝ^{N×d}`, `K = 3`.
Every tensor below must agree with these constraints or the pipeline collapses silently.

```
Input          X_i  ∈  ℝ^{d_{φ(i)}}   (type-specific, varies per node type)

Stage 1        H^(0) ∈  ℝ^{N × d}      W_{φ(i)} : d_{φ(i)} → d
                                         d = 64  (paper §5.2)

Stage 2        P_r   ∈  ℝ^{N × N}      D^{-½}(A_r+I)D^{-½}

Stage 3        P̃_r  ∈  ℝ^{N × N}      P_r @ P_r^T (bipartite) or P_r (homogeneous)

Stage 4        Z_r   ∈  ℝ^{N × d}      Σ_k β_{r,k} · (P̃_r^k @ H^(0))
               check: (N×N) @ (N×d) = (N×d)  ✓

Stage 5        Z     ∈  ℝ^{N × d}      Σ_r α_r · Z_r

Stage 6        [H^(0)‖Z]  ∈  ℝ^{N × 2d}
               MLP: Linear(2d→d) → ReLU → Dropout → Linear(d→d)
               Z_final     ∈  ℝ^{N × d}    d_prime = d = 64  (paper §4.6)

GCN Layer 1    Â_h @ Z_final  → (N×d) → W₁ → (N × gcn_hidden)
GCN Layer 2    Â_h @ H₁       → (N×gcn_hidden) → W₂ → (N × out_dim)

NC output      out_dim = n_classes          logits ∈ ℝ^{N × n_classes}
LP output      out_dim = d  (= 64)          emb    ∈ ℝ^{N × d}
               LP decoder: [emb[src]‖emb[dst]] ∈ ℝ^{2d} → MLP → score ∈ ℝ
```

**Dataset sizes for sanity checking**

| | DBLP | ACM | IMDB |
|---|---|---|---|
| N (total nodes) | ~26 128 | ~18 274 | ~11 616 |
| Target nodes Nt | ~4 057 (authors) | ~3 025 (papers) | ~4 278 (movies) |
| Relations R | 6 | 7 | 6 |
| n_classes | 4 | 3 | 3 |
| bipartite_flags | `[True]*6` | `[True]*6+[False]` | `[True]*6` |
| X_paper / X_movie feat dim | ~d_paper (≈ vocab) | 1 533 | 3 000 |
| X_author feat dim | d_paper (agg BoW) | 64 (SVD) | Nd (identity) |

---

## 80/20 split + 5-fold CV — how it works for a GNN

> **GNNs are transductive by default.**  This changes how you must apply CV.

In standard ML you can split samples independently. In a GNN, every node's embedding
depends on its neighbours via message passing. The full graph (all N nodes, all edges)
is always used for propagation — what changes per split is only **which node labels
contribute to the loss** (training mask) or **which are measured** (val/test mask).

```
ALL LABELED NODES  (Nt)
│
├── 20 %  HELD-OUT TEST  (never seen during any training or selection)
│
└── 80 %  TRAIN+VAL POOL
    │
    ├─ Random hyperparameter combinations (n_iter = 50 by default)
    │   For each combo:
    │   ┌── Fold 1: val = slice 0   train = slices 1-4
    │   ├── Fold 2: val = slice 1   train = slices 0,2-4
    │   ├── Fold 3: val = slice 2   train = slices 0-1,3-4
    │   ├── Fold 4: val = slice 3   train = slices 0-2,4
    │   └── Fold 5: val = slice 4   train = slices 0-3
    │   mean_val_score = average Macro-F1 / AUC over 5 folds
    │
    ├─ Best combo = argmax(mean_val_score)
    │
    └─ FINAL TRAINING on full 80 % with best combo
           ↓
       Evaluate on held-out 20 %  →  reported test score

For LP (edges):
- Test edges (20%) are REMOVED from the graph during all training.
- Val edges in each fold are ALSO removed from that fold's training graph.
- This prevents any leakage of test structure into message passing.
```

---

## File structure

```
RAHGH/
├── data/raw/
│   ├── DBLP/    ← author_label.txt, paper_author.txt, paper.txt,
│   │               paper_term.txt, paper_conf.txt, term.txt, conf.txt
│   ├── ACM/     ← ACM.mat
│   └── IMDB/    ← movie_metadata.csv
│
├── src/
│   ├── data/
│   │   ├── dblp_loader.py
│   │   ├── acm_loader.py
│   │   └── imdb_loader.py
│   ├── model/
│   │   ├── projector.py
│   │   ├── diffusion.py
│   │   ├── fusion.py        (kept for reference — logic lives in rahgh.py)
│   │   └── rahgh.py
│   ├── tasks/
│   │   ├── hparam_search.py          ← CV wrappers for all 4 tasks
│   │   ├── node_classification.py    ← NC task
│   │   ├── link_prediction.py        ← LP task
│   │   ├── graph_clustering.py       ← NEW: clustering task
│   │   └── recommendation.py         ← NEW: recommendation task
│   └── train.py                      ← orchestrates all 4 tasks
│
├── configs/
│   ├── dblp_nc.yaml    acm_nc.yaml    imdb_nc.yaml
│   ├── dblp_lp.yaml    acm_lp.yaml    imdb_lp.yaml
│   ├── dblp_cl.yaml    acm_cl.yaml    imdb_cl.yaml    ← NEW: clustering
│   └── dblp_rec.yaml   acm_rec.yaml   imdb_rec.yaml   ← NEW: recommendation
│
├── scripts/
│   ├── run_nc.sh
│   ├── run_lp.sh
│   ├── run_cl.sh    ← NEW
│   └── run_rec.sh   ← NEW
│
└── results/
    ├── nc/
    │   ├── cv_fold_scores.csv     ← one row per (dataset, combo_id, fold)
    │   ├── best_params.json       ← best hyperparams per dataset
    │   ├── per_run_results.csv    ← one row per (dataset, seed) — all metrics
    │   ├── epoch_logs/            ← one CSV per (dataset, seed): epoch, train_loss, val_macro
    │   └── summary.csv            ← mean ± std over 10 seeds, per dataset
    ├── lp/
    │   ├── cv_fold_scores.csv
    │   ├── best_params.json
    │   ├── per_run_results.csv
    │   ├── epoch_logs/
    │   └── summary.csv
    ├── clustering/
    │   ├── cv_fold_scores.csv
    │   ├── best_params.json
    │   ├── per_run_results.csv    ← NMI, ARI, ACC per seed
    │   └── summary.csv
    └── recommendation/
        ├── cv_fold_scores.csv
        ├── best_params.json
        ├── per_run_results.csv    ← Recall@K, NDCG@K, Hit@K, Precision@K, MRR per seed
        ├── epoch_logs/
        └── summary.csv
```

---

## Step 1 — `src/data/dblp_loader.py`

```python
# src/data/dblp_loader.py

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS as sk_sw
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_sw
import pandas as pd


class LemmaTokenizer:
    def __init__(self): self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def load_dblp(root="data/raw/DBLP"):
    """
    Node layout : A[0:Na)  P[Na:Na+Np)  T[Na+Np:Na+Np+Nt)  C[Na+Np+Nt:N)
    X_author    : (Na, d_paper)   BoW agg from papers, L2-normed
    X_paper     : (Np, d_paper)   CountVectorizer on titles
    X_term      : (Nt, Nt)        identity
    X_conf      : (Nc, Nc)        identity

    Dimension check (Stage 1):
      in_dims = [d_paper, d_paper, Nt, Nc]
      All project → d = 64   →   H^(0) ∈ ℝ^{N × 64}
    """
    author_label = pd.read_csv(f"{root}/author_label.txt", sep='\t', header=None,
                                names=['author_id','label','author_name'],
                                keep_default_na=False, encoding='utf-8')
    paper_author = pd.read_csv(f"{root}/paper_author.txt", sep='\t', header=None,
                                names=['paper_id','author_id'],
                                keep_default_na=False, encoding='utf-8')
    paper_conf   = pd.read_csv(f"{root}/paper_conf.txt",   sep='\t', header=None,
                                names=['paper_id','conf_id'],
                                keep_default_na=False, encoding='utf-8')
    paper_term   = pd.read_csv(f"{root}/paper_term.txt",   sep='\t', header=None,
                                names=['paper_id','term_id'],
                                keep_default_na=False, encoding='utf-8')
    papers       = pd.read_csv(f"{root}/paper.txt",        sep='\t', header=None,
                                names=['paper_id','paper_title'],
                                keep_default_na=False, encoding='cp1252')
    terms        = pd.read_csv(f"{root}/term.txt",         sep='\t', header=None,
                                names=['term_id','term'],
                                keep_default_na=False, encoding='utf-8')
    confs        = pd.read_csv(f"{root}/conf.txt",         sep='\t', header=None,
                                names=['conf_id','conf'],
                                keep_default_na=False, encoding='utf-8')

    labeled      = author_label['author_id'].tolist()
    paper_author = paper_author[paper_author['author_id'].isin(labeled)].reset_index(drop=True)
    valid_papers = paper_author['paper_id'].unique()
    papers       = papers[papers['paper_id'].isin(valid_papers)].reset_index(drop=True)
    paper_conf   = paper_conf[paper_conf['paper_id'].isin(valid_papers)].reset_index(drop=True)
    paper_term   = paper_term[paper_term['paper_id'].isin(valid_papers)].reset_index(drop=True)
    terms        = terms[terms['term_id'].isin(paper_term['term_id'].unique())].reset_index(drop=True)

    for df in [author_label, papers, terms, confs]:
        df.sort_values(df.columns[0], inplace=True); df.reset_index(drop=True, inplace=True)

    Na, Np, Nt, Nc = len(author_label), len(papers), len(terms), len(confs)
    N = Na + Np + Nt + Nc

    a_map   = {row['author_id']: i             for i, row in author_label.iterrows()}
    p_map   = {row['paper_id']:  i + Na        for i, row in papers.iterrows()}
    t_map   = {row['term_id']:   i + Na+Np     for i, row in terms.iterrows()}
    c_map   = {row['conf_id']:   i + Na+Np+Nt  for i, row in confs.iterrows()}
    p_local = {row['paper_id']:  i             for i, row in papers.iterrows()}

    def build_coo(rows, cols):
        return sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                              shape=(N, N)).tocsr()

    pa_r, pa_c, pt_r, pt_c, pc_r, pc_c = [], [], [], [], [], []
    for _, row in paper_author.iterrows():
        p = p_map.get(row['paper_id']); a = a_map.get(row['author_id'])
        if p and a is not None: pa_r.append(p); pa_c.append(a)
    for _, row in paper_term.iterrows():
        p = p_map.get(row['paper_id']); t = t_map.get(row['term_id'])
        if p and t: pt_r.append(p); pt_c.append(t)
    for _, row in paper_conf.iterrows():
        p = p_map.get(row['paper_id']); c = c_map.get(row['conf_id'])
        if p and c: pc_r.append(p); pc_c.append(c)

    PA = build_coo(pa_r, pa_c); AP = PA.T.tocsr()
    PT = build_coo(pt_r, pt_c); TP = PT.T.tocsr()
    PC = build_coo(pc_r, pc_c); CP = PC.T.tocsr()

    stopwords   = list(sk_sw.union(set(nltk_sw.words('english'))))
    vec         = CountVectorizer(min_df=2, stop_words=stopwords,
                                  tokenizer=LemmaTokenizer())
    X_paper_np  = vec.fit_transform(papers['paper_title'].values).toarray().astype(np.float32)
    d_paper     = X_paper_np.shape[1]

    X_author_np = np.zeros((Na, d_paper), dtype=np.float32)
    for _, row in paper_author.iterrows():
        a = a_map.get(row['author_id']); p = p_local.get(row['paper_id'])
        if a is not None and p is not None: X_author_np[a] += X_paper_np[p]
    norms = np.linalg.norm(X_author_np, axis=1, keepdims=True)
    X_author_np /= np.where(norms == 0, 1.0, norms)

    return dict(
        A_list_sp=[PA, AP, PT, TP, PC, CP],
        bipartite_flags=[True, True, True, True, True, True],
        relation_names=['paper→author','author→paper',
                        'paper→term',  'term→paper',
                        'paper→conf',  'conf→paper'],
        X_dict={'author': torch.tensor(X_author_np),
                'paper' : torch.tensor(X_paper_np),
                'term'  : torch.eye(Nt),
                'conf'  : torch.eye(Nc)},
        labels=torch.tensor(author_label['label'].to_numpy(), dtype=torch.long),
        Na=Na, Np=Np, Nt=Nt, Nc=Nc, N=N,
        target_type='author', target_size=Na,
        n_classes=int(author_label['label'].nunique()),
        # in_dims order must match X_dict.values() order
        # [d_paper, d_paper, Nt, Nc]  → all project to d=64
    )
```

---

## Step 2 — `src/data/acm_loader.py`

```python
# src/data/acm_loader.py

import numpy as np, scipy.io, scipy.sparse as sp
import scipy.sparse.linalg as spla, torch

AREA_CONFS = {
    'Database'     : ['SIGMOD','VLDB','ICDE','PODS'],
    'Wireless_Comm': ['MobiCOMM','SIGCOMM','INFOCOM','ICNP'],
    'Data_Mining'  : ['KDD','WWW','ICDM','SIGIR','CIKM','WSDM'],
}

def load_acm(root="data/raw/ACM"):
    """
    Node layout : P[0:Np)  A[Np:Np+Na)  T[Np+Na:Np+Na+Nt)  V[Np+Na+Nt:N)
    X_paper  : (Np, 1533)   nPvsT  — noun-phrase term features
    X_author : (Na, 64)     SVD-64 of PvsA.T
    X_term   : (Nt, Nt)     identity  (Nt=1533)
    X_venue  : (Nv, Nv)     identity  (Nv=196)

    bipartite_flags: PA,AP,PT,TP,PV,VP → True  |  PP → False
    in_dims = [1533, 64, Nt, Nv]  →  all project to d=64  →  H^(0) ∈ ℝ^{N×64}
    """
    mat   = scipy.io.loadmat(f"{root}/ACM.mat")
    PvsA  = mat['PvsA'].astype(np.float32).tocsr()
    PvsC  = mat['PvsC'].astype(np.float32).tocsr()
    nPvsT = mat['nPvsT'].astype(np.float32).tocsr()
    PvsV  = mat['PvsV'].astype(np.float32).tocsr()
    PvsP  = mat['PvsP'].astype(np.float32).tocsr()

    C_names       = [str(mat['C'][i,0].flat[0]) for i in range(mat['C'].shape[0])]
    conf_to_class = {c: lbl for lbl,(area,confs) in enumerate(AREA_CONFS.items())
                     for c in confs if c in C_names}

    PvsC_d        = np.asarray(PvsC.todense())
    pid_arr, lbl_arr = [], []
    for pid in range(PvsC_d.shape[0]):
        cname = C_names[PvsC_d[pid].argmax()]
        if cname in conf_to_class:
            pid_arr.append(pid); lbl_arr.append(conf_to_class[cname])
    paper_ids = np.array(pid_arr); Np = len(paper_ids)

    PvsA_s  = PvsA[paper_ids]
    nPvsT_s = nPvsT[paper_ids]
    PvsV_s  = PvsV[paper_ids]
    PvsP_s  = PvsP[paper_ids][:, paper_ids]

    Na = PvsA_s.shape[1]; Nt = nPvsT_s.shape[1]
    Nv = PvsV_s.shape[1]; N  = Np + Na + Nt + Nv
    p_off, a_off, t_off, v_off = 0, Np, Np+Na, Np+Na+Nt

    def bip(A, r, c):
        A = A.tocoo().astype(np.float32)
        return sp.coo_matrix((A.data,(A.row+r,A.col+c)),shape=(N,N)).tocsr()
    def hom(A, off):
        A = A.tocoo().astype(np.float32)
        return sp.coo_matrix((A.data,(A.row+off,A.col+off)),shape=(N,N)).tocsr()

    PA = bip(PvsA_s,p_off,a_off);  AP = PA.T.tocsr()
    PT = bip(nPvsT_s,p_off,t_off); TP = PT.T.tocsr()
    PV = bip(PvsV_s,p_off,v_off);  VP = PV.T.tocsr()
    PP = hom(PvsP_s,p_off)

    U, S, _ = spla.svds(PvsA_s.T.astype(np.float64), k=64)

    return dict(
        A_list_sp=[PA,AP,PT,TP,PV,VP,PP],
        bipartite_flags=[True,True,True,True,True,True,False],
        relation_names=['paper→author','author→paper',
                        'paper→term', 'term→paper',
                        'paper→venue','venue→paper','paper→paper'],
        X_dict={'paper' : torch.tensor(np.array(nPvsT_s.todense(),dtype=np.float32)),
                'author': torch.tensor((U*S).astype(np.float32)),
                'term'  : torch.eye(Nt,dtype=torch.float32),
                'venue' : torch.eye(Nv,dtype=torch.float32)},
        labels=torch.tensor(np.array(lbl_arr,dtype=np.int64)),
        Np=Np, Na=Na, Nt=Nt, Nv=Nv, N=N,
        target_type='paper', target_size=Np, n_classes=3,
    )
```

---

## Step 3 — `src/data/imdb_loader.py`

```python
# src/data/imdb_loader.py

import numpy as np, scipy.sparse as sp, torch
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def load_imdb(root="data/raw/IMDB"):
    """
    Node layout : M[0:Nm)  D[Nm:Nm+Nd)  A[Nm+Nd:N)
    X_movie  : (Nm, 3000)   plot keyword BoW
    X_dir    : (Nd, Nd)     identity
    X_act    : (Na, Na)     identity
    All 6 relations are bipartite (cross-type edges only)
    in_dims = [3000, Nd, Na]  →  project to d=64  →  H^(0) ∈ ℝ^{N×64}
    """
    df = pd.read_csv(f"{root}/movie_metadata.csv", encoding='latin-1')
    df['movie_title']   = df['movie_title'].str.strip().str.replace('Â','',regex=False)
    df['genre_primary'] = df['genres'].fillna('').apply(lambda x: x.split('|')[0])
    genre_map = {'Action':0,'Comedy':1,'Drama':2}
    df  = df[df['genre_primary'].isin(genre_map)].reset_index(drop=True); Nm = len(df)

    df['director_name'] = df['director_name'].fillna('Unknown_Director')
    dir_list = sorted(df['director_name'].unique()); dir2idx = {d:i for i,d in enumerate(dir_list)}
    Nd = len(dir_list)

    actor_cols = ['actor_1_name','actor_2_name','actor_3_name']
    for c in actor_cols: df[c] = df[c].fillna('Unknown_Actor')
    act_list = sorted(pd.concat([df[c] for c in actor_cols]).unique())
    act2idx = {a:i for i,a in enumerate(act_list)}; Na = len(act_list); N = Nm+Nd+Na

    def build_coo(rows,cols):
        return sp.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(N,N)).tocsr()

    md_r,md_c,ma_r,ma_c = [],[],[],[]
    for i,row in df.iterrows():
        d = dir2idx.get(row['director_name'])
        if d is not None: md_r.append(i); md_c.append(Nm+d)
        for col in actor_cols:
            a = act2idx.get(row[col])
            if a is not None: ma_r.append(i); ma_c.append(Nm+Nd+a)

    MD = build_coo(md_r,md_c); DM = MD.T.tocsr()
    MA = build_coo(ma_r,ma_c); AM = MA.T.tocsr()
    DA = (DM@MA).tocsr(); DA.data[:]=1.0; DA.eliminate_zeros(); AD = DA.T.tocsr()

    kw  = df['plot_keywords'].fillna('').str.replace('|',' ',regex=False)
    vec = CountVectorizer(max_features=3000)
    X_m = torch.tensor(vec.fit_transform(kw).toarray(), dtype=torch.float32)

    return dict(
        A_list_sp=[MD,DM,MA,AM,DA,AD],
        bipartite_flags=[True,True,True,True,True,True],
        relation_names=['movie→dir','dir→movie','movie→act',
                        'act→movie','dir→act','act→dir'],
        X_dict={'movie':X_m,'director':torch.eye(Nd),'actor':torch.eye(Na)},
        labels=torch.tensor([genre_map[g] for g in df['genre_primary']],dtype=torch.long),
        Nm=Nm, Nd=Nd, Na=Na, N=N,
        target_type='movie', target_size=Nm, n_classes=3,
    )
```

---

## Step 4 — `src/model/projector.py`

```python
# src/model/projector.py
# Stage 1 — H_i^(0) = ReLU(W_{φ(i)} X_i + b_{φ(i)})
# Input  : X_list[t]  ∈  ℝ^{N_t × d_{φ(t)}}
# Output : H^(0)      ∈  ℝ^{N × d}    (all types concatenated)

import torch, torch.nn as nn, torch.nn.functional as F

class TypeSpecificProjector(nn.Module):
    def __init__(self, in_dims: list, d: int):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(i, d) for i in in_dims])

    def forward(self, X_list: list) -> torch.Tensor:
        # Each proj maps (N_t, d_t) → (N_t, d); cat along node dim → (N, d)
        return torch.cat([F.relu(p(X)) for p, X in zip(self.projections, X_list)], dim=0)
```

---

## Step 5 — `src/model/diffusion.py`

```python
# src/model/diffusion.py
# Stages 2, 3, 4, 5 + utility for downstream GCN adjacency

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, scipy.sparse as sp


def normalize_with_selfloops(A_sp, device):
    """
    Stage 2: P_r = D^{-½}(A_r+I)D^{-½}
    Input  : scipy CSR  (N, N)
    Output : sparse COO tensor  (N, N)  on device
    """
    N_  = A_sp.shape[0]
    At  = (A_sp + sp.eye(N_, format='csr', dtype=np.float32)).tocoo()
    deg = np.array(At.sum(axis=1)).flatten()
    di  = np.where(deg > 0, deg**-0.5, 0.0)
    dat = At.data * di[At.row] * di[At.col]
    idx = torch.tensor(np.vstack([At.row, At.col]), dtype=torch.long)
    val = torch.tensor(dat, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (N_, N_)).coalesce().to(device)


def normalize_plain(A_sp, device):
    """D^{-½} A D^{-½} — for the downstream GCN (self-loops already in A)."""
    A   = A_sp.tocoo().astype(np.float32)
    deg = np.array(A_sp.sum(axis=1)).flatten()
    di  = np.where(deg > 0, deg**-0.5, 0.0)
    dat = A.data * di[A.row] * di[A.col]
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(dat, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, A_sp.shape).coalesce().to(device)


def build_operators(A_list_sp, bipartite_flags, device):
    """
    Stage 3 — bipartite correction (paper's central contribution).

    Bipartite: P̃_r = P_r @ P_r^T
      (P̃_r)_{ij} = Σ_k P_r(i,k)·P_r(j,k)
      → same-type similarity graph: nodes i,j linked iff they share target neighbours
      → equivalent to length-2 meta-path  r ∘ r⁻¹  in differentiable matrix form

    Homogeneous: P̃_r = P_r  (unchanged)

    Returns : list of DENSE (N, N) float32 tensors on device.
              Dense because P_r @ P_r^T fills the matrix.
              For N > 50k use: torch.sparse.mm(P_r, P_r.t()) to save memory.

    Dimension: (N,N) @ (N,d) = (N,d)  ✓  used in Stage 4 diffusion loop
    """
    P_list = []
    for A_sp, is_bip in zip(A_list_sp, bipartite_flags):
        P_r = normalize_with_selfloops(A_sp, device)     # sparse (N,N)
        P_d = P_r.to_dense()                              # dense  (N,N)
        P_list.append(P_d @ P_d.t() if is_bip else P_d)
    return P_list                                         # list of dense (N,N)


def build_A_struct(A_list_sp, alpha_np, N, device):
    """
    GCN downstream adjacency — α-weighted relation sum.

        A_struct = Σ_r α_r · A_r
        Â        = D^{-½}(A_struct + I)D^{-½}

    alpha_np : (R,) numpy array — DETACHED from graph before calling
    Returns  : sparse COO tensor (N,N) on device  →  used in torch.sparse.mm
    """
    A = sp.csr_matrix((N, N), dtype=np.float32)
    for A_sp, a in zip(A_list_sp, alpha_np):
        A = A + float(a) * A_sp.astype(np.float32)
    return normalize_plain(A + sp.eye(N, format='csr', dtype=np.float32), device)


class RelationSpecificDiffusion(nn.Module):
    """
    Stage 4 — polynomial diffusion per relation:
        Z_r = Σ_{k=0}^{K} β_{r,k} · P̃_r^k H^(0)
        β_{r,k} = softmax(ψ_r)[k]          learnable hop weights
        check: (N×N) @ (N×d) = (N×d)  ✓

    Stage 5 — relation fusion:
        Z = Σ_r α_r · Z_r
        α_r = softmax(θ)[r]                 learnable relation weights
        Z ∈ ℝ^{N×d}

    P_list must be dense (N,N) tensors from build_operators().
    """
    def __init__(self, R: int, K: int):
        super().__init__()
        self.R, self.K = R, K
        self.phi   = nn.Parameter(torch.zeros(R, K+1))   # hop weights   (R, K+1)
        self.theta = nn.Parameter(torch.ones(R))          # relation importance (R,)

    def forward(self, H0, P_list):
        """
        H0     : (N, d)
        P_list : list of R dense (N, N)
        Returns: Z (N,d)  alpha (R,)  beta (R,K+1)
        """
        beta  = F.softmax(self.phi,   dim=1)   # (R, K+1)
        alpha = F.softmax(self.theta, dim=0)   # (R,)
        Z_list = []
        for r in range(self.R):
            Zr = torch.zeros_like(H0)          # (N, d)
            Hk = H0
            for k in range(self.K + 1):
                Zr = Zr + beta[r, k] * Hk
                if k < self.K:
                    Hk = P_list[r] @ Hk        # (N,N)@(N,d) = (N,d)  ← dense mm
            Z_list.append(Zr)
        Z = sum(alpha[r] * Z_list[r] for r in range(self.R))   # (N, d)
        return Z, alpha, beta
```

---

## Step 6 — `src/model/fusion.py`

```python
# src/model/fusion.py
# Stage 6 — Residual Fusion MLP
# Input  : [H^(0) ‖ Z]  ∈  ℝ^{N × 2d}
# Output : Z_final       ∈  ℝ^{N × d}   (paper §4.6: Z_final ∈ ℝ^{N×d})
# IMPORTANT: d_prime = d (not a free parameter) to match the paper

import torch, torch.nn as nn

class ResidualFusion(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*d, d), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),   # output dim = d  →  Z_final ∈ ℝ^{N×d}
        )
    def forward(self, H0, Z):
        return self.mlp(torch.cat([H0, Z], dim=1))   # (N, d)
```

---

## Step 7 — `src/model/rahgh.py`

**v3 fix for LP:** `forward()` returns `(logits, alpha, beta, logits)` — the 4th value is
**the GCN output** (not Z_final). LP decoders unpack position [0] as embeddings; NC uses
position [0] as logits. Both now correctly use the GCN output.

```python
# src/model/rahgh.py

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from .projector import TypeSpecificProjector
from .diffusion  import RelationSpecificDiffusion, build_operators, build_A_struct


class GCNLayer(nn.Module):
    """Â X W  —  sparse mm then linear projection."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, A, X):
        return self.W(torch.sparse.mm(A, X))   # (N,N)_sparse @ (N,in) → (N,out)


class GCN(nn.Module):
    """
    Two-layer GCN downstream head.
    Input  : Z_final ∈ ℝ^{N×d}          (d = 64, paper-matched)
    Layer1 : Â @ Z_final → (N,gcn_hid)
    Layer2 : Â @ H1      → (N,out_dim)
      NC: out_dim = n_classes
      LP: out_dim = d  (produces embeddings, same dim as Z_final)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.l1, self.l2 = GCNLayer(in_dim, hidden_dim), GCNLayer(hidden_dim, out_dim)
        self.drop = dropout
    def forward(self, A, X):
        H = F.relu(self.l1(A, X))
        H = F.dropout(H, p=self.drop, training=self.training)
        return self.l2(A, H)


class RAHGH(nn.Module):
    """
    Full RAHGH pipeline — 6 stages + GCN downstream head.

    Dimension trace (all must agree with paper §4, §5.2):
      Stage 1  H^(0)    ∈  ℝ^{N × d}       d = 64
      Stage 3  P̃_r     ∈  ℝ^{N × N}       dense after bipartite correction
      Stage 4  Z_r      ∈  ℝ^{N × d}       (N,N)@(N,d)=(N,d) ✓
      Stage 5  Z        ∈  ℝ^{N × d}
      Stage 6  Z_final  ∈  ℝ^{N × d}       MLP(2d→d→d) — paper §4.6
      A_hat    ∈  ℝ^{N × N}               α-weighted, rebuilt each forward
      GCN out  ∈  ℝ^{N × out_dim}

    Constructor
    -----------
    in_dims   : list[int]  input dim per node type (ordered as X_dict.values())
    d         : int        shared projection dim  (= 64, paper)
    R, K      : int        #relations, max hops
    gcn_hidden: int
    out_dim   : int        n_classes (NC)  or  d (LP)
    dropout   : float
    A_list_sp : list[sp.csr_matrix]  stored for dynamic A_hat construction
    N         : int        total nodes
    device    : torch.device
    """
    def __init__(self, in_dims, d, R, K, gcn_hidden, out_dim, dropout,
                 A_list_sp, N, device):
        super().__init__()
        self.projector = TypeSpecificProjector(in_dims, d)
        self.diffusion = RelationSpecificDiffusion(R, K)
        # Stage 6 MLP: [H^(0)||Z] ∈ ℝ^{N×2d} → Z_final ∈ ℝ^{N×d}
        self.fusion    = nn.Sequential(
            nn.Linear(2*d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d)
        )
        self.gcn       = GCN(d, gcn_hidden, out_dim, dropout)
        self.A_list_sp, self.N_nodes, self.device = A_list_sp, N, device

    def forward(self, X_list, P_list):
        """
        X_list : [X_type0, X_type1, ...]  each on device
        P_list : list of R dense (N,N) bipartite-corrected tensors on device

        Returns
        -------
        gcn_out : (N, out_dim)  — GCN output (class logits for NC, embeddings for LP)
        alpha   : (R,)          — learned relation weights
        beta    : (R, K+1)      — learned hop weights
        gcn_out : (N, out_dim)  — same as first; 4th slot used by LP decoder
        """
        H0                = self.projector(X_list)                      # (N, d)
        Z, alpha, beta    = self.diffusion(H0, P_list)                  # (N, d)
        Z_final           = self.fusion(torch.cat([H0, Z], dim=1))      # (N, d)

        # Build α-weighted A_hat from CURRENT learned α (detached — not differentiable)
        alpha_np = alpha.detach().cpu().numpy()                         # (R,)
        A_hat    = build_A_struct(self.A_list_sp, alpha_np,
                                   self.N_nodes, self.device)           # sparse (N,N)

        gcn_out = self.gcn(A_hat, Z_final)                              # (N, out_dim)
        # ── v3 fix ── return gcn_out as 4th value (not Z_final)
        # LP decoder uses position [0] == gcn_out; Z_final no longer leaked out
        return gcn_out, alpha, beta, gcn_out
```

---

## Step 8a — `src/tasks/hparam_search.py` *(updated v4)*

```python
# src/tasks/hparam_search.py
#
# Implements:  80/20 split  →  5-fold CV on 80%  →  best hyperparams
#              →  final train on 80%  →  test on held-out 20%  (10 seeds)
#
# GNN-specific note:
#   The full graph (all N nodes, all edges) is always used for message passing.
#   Only the LABEL MASKS change per fold.  This is correct transductive behaviour.
#   P_list (bipartite-corrected operators) is pre-built ONCE and reused across all
#   folds — it does not depend on which labels are visible.

import itertools, random, time, json, os
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from ..model.rahgh    import RAHGH
from ..model.diffusion import build_operators
import torch.nn.functional as F
from torch.optim import Adam


# ── Hyperparameter search space ──────────────────────────────────────────────
# Shared across NC, LP, Clustering, Recommendation
PARAM_GRID_BASE = {
    'd'         : [64, 128],          # projection dim; 64 is paper default
    'K'         : [2, 3, 4, 5, 6],   # diffusion hops
    'dropout'   : [0.3, 0.5],
    'lr'        : [0.001, 0.005],
    'wd'        : [1e-4, 1e-3],
    'gcn_hidden': [64, 128],
    'epochs'    : [300, 500],
    # d_prime = d always (paper §4.6: Z_final ∈ ℝ^{N×d})
}

# Task-specific additions merged at search time
PARAM_GRID_CLUSTERING = {
    **PARAM_GRID_BASE,
    'cl_loss'       : ['reconstruction', 'contrastive'],  # unsupervised objective
    'cl_temp'       : [0.1, 0.5],                         # contrastive temperature
}

PARAM_GRID_REC = {
    **PARAM_GRID_BASE,
    'K_rec'         : [10, 20, 50],   # top-K for Recall@K, NDCG@K etc.
    'neg_ratio'     : [5, 10],        # negatives per positive
    'bpr_reg'       : [1e-4, 1e-3],   # BPR regularization weight
}

# Full grid size base: 2×5×2×2×2×2×2 = 640; we random-sample n_iter combos
N_ITER    = 50    # number of random combinations to try
N_FOLDS   = 5
N_SEEDS   = 10    # final test repetitions (v4: was 5 in v3)
TEST_FRAC = 0.20  # held-out fraction


def _random_combos(grid, seed=0, n=N_ITER):
    keys = list(grid.keys())
    all_c = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    random.seed(seed); random.shuffle(all_c)
    return all_c[:n]


def _save_best_params(best_params, dataset, task, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'best_params.json')
    try:
        with open(path) as f: existing = json.load(f)
    except FileNotFoundError:
        existing = {}
    existing[f"{dataset}_{task}"] = best_params
    with open(path, 'w') as f: json.dump(existing, f, indent=2)
    print(f"[hparam] Best params saved → {path}")


# ── Single fold: node classification ─────────────────────────────────────────
def _run_fold_nc(data, P_list, X_list, params, tr_idx, va_idx, device):
    """Train one fold for node classification, return val Macro-F1."""
    d       = params['d']
    R       = len(data['A_list_sp'])
    Nt      = data['target_size']
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    labels  = data['labels'].to(device)

    model = RAHGH(
        in_dims=in_dims, d=d, R=R, K=params['K'],
        gcn_hidden=params['gcn_hidden'],
        out_dim=data['n_classes'],
        dropout=params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    opt = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    tr_t = torch.tensor(tr_idx, dtype=torch.long, device=device)
    va_t = torch.tensor(va_idx, dtype=torch.long, device=device)

    best_vm, best_sd = 0.0, None
    for ep in range(1, params['epochs'] + 1):
        model.train(); opt.zero_grad()
        logits, *_ = model(X_list, P_list)
        F.cross_entropy(logits[:Nt][tr_t], labels[tr_t]).backward()
        opt.step()

        if ep % 50 == 0 or ep == params['epochs']:
            model.eval()
            with torch.no_grad():
                logits, *_ = model(X_list, P_list)
                p  = logits[:Nt][va_t].argmax(1).cpu().numpy()
                y  = data['labels'][va_idx].numpy()
                vm = f1_score(y, p, average='macro', zero_division=0)
            if vm > best_vm:
                best_vm = vm
                best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    del model; torch.cuda.empty_cache()
    return best_vm


# ── Single fold: link prediction ──────────────────────────────────────────────
# (same as v3 — imported from link_prediction.py)


# ── Single fold: clustering ───────────────────────────────────────────────────
def _run_fold_cl(data, P_list, X_list, params, tr_idx, va_idx, device):
    """
    Clustering fold: train unsupervised embeddings, cluster va_idx nodes,
    return NMI against ground-truth labels.

    Unsupervised objective: feature reconstruction
        loss = MSE( decoder(Z_final[:Nt]), H0[:Nt] )
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics  import normalized_mutual_info_score

    d       = params['d']
    R       = len(data['A_list_sp'])
    Nt      = data['target_size']
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    n_cl    = data['n_classes']

    model = RAHGH(
        in_dims=in_dims, d=d, R=R, K=params['K'],
        gcn_hidden=params['gcn_hidden'],
        out_dim=d,            # embedding dim for clustering
        dropout=params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)

    # Reconstruction decoder: d → d (maps GCN out back to input space)
    decoder = torch.nn.Linear(d, d).to(device)
    opt = Adam(list(model.parameters()) + list(decoder.parameters()),
               lr=params['lr'], weight_decay=params['wd'])

    tr_t = torch.tensor(tr_idx, dtype=torch.long, device=device)

    best_nmi, best_sd = 0.0, None
    for ep in range(1, params['epochs'] + 1):
        model.train(); decoder.train(); opt.zero_grad()
        emb, *_ = model(X_list, P_list)           # (N, d) GCN output
        # Reconstruction loss on training nodes only
        recon = decoder(emb[:Nt][tr_t])
        target = torch.cat([x for x in X_list], dim=0)[:Nt][tr_t]
        # Project target to d if dimensions differ
        if target.shape[1] != d:
            target = target[:, :d] if target.shape[1] > d \
                     else F.pad(target, (0, d - target.shape[1]))
        loss = F.mse_loss(recon, target.to(device))
        loss.backward(); opt.step()

        if ep % 50 == 0 or ep == params['epochs']:
            model.eval()
            with torch.no_grad():
                emb_v, *_ = model(X_list, P_list)
                emb_np = emb_v[:Nt][va_idx].cpu().numpy()
                km     = KMeans(n_clusters=n_cl, n_init=10, random_state=0)
                pred   = km.fit_predict(emb_np)
                y      = data['labels'][va_idx].numpy()
                nmi    = normalized_mutual_info_score(y, pred)
            if nmi > best_nmi:
                best_nmi = nmi
                best_sd  = {k: v.clone() for k, v in model.state_dict().items()}

    del model, decoder; torch.cuda.empty_cache()
    return best_nmi


# ── Single fold: recommendation ───────────────────────────────────────────────
def _run_fold_rec(data, tr_edges, va_edges, params, device, K_rec=20):
    """
    Recommendation fold using BPR (Bayesian Personalised Ranking) loss.
    Evaluates Recall@K on val edges.

    Treats the task as user-item link prediction with ranking.
    For heterogeneous graphs: 'user' = one node type, 'item' = another.
    For DBLP/ACM/IMDB without explicit users, use the first bipartite relation.
    """
    from .recommendation import bpr_loss, recall_at_k

    d       = params['d']
    R       = len(data['A_list_sp'])
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    X_list  = [x.to(device) for x in data['X_dict'].values()]
    N       = data['N']

    model = RAHGH(
        in_dims=in_dims, d=d, R=R, K=params['K'],
        gcn_hidden=params['gcn_hidden'],
        out_dim=d, dropout=params['dropout'],
        A_list_sp=data['A_list_sp'], N=N, device=device,
    ).to(device)
    opt = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    all_items = np.unique(tr_edges[:, 1])
    user_pos  = {}   # user → set of positive item ids
    for u, i in tr_edges: user_pos.setdefault(u, set()).add(i)

    best_rec, best_sd = 0.0, None
    rng = np.random.default_rng(0)

    for ep in range(1, params['epochs'] + 1):
        model.train(); opt.zero_grad()
        emb, *_ = model(X_list, P_list if 'P_list' in dir() else
                        build_operators(data['A_list_sp'], data['bipartite_flags'], device))
        # BPR: sample triplets (user, pos_item, neg_item)
        users = tr_edges[:, 0]; pos_items = tr_edges[:, 1]
        neg_items = rng.choice(all_items, size=len(users))
        loss = bpr_loss(emb, users, pos_items, neg_items,
                        device, reg=params.get('bpr_reg', 1e-4))
        loss.backward(); opt.step()

        if ep % 50 == 0 or ep == params['epochs']:
            model.eval()
            with torch.no_grad():
                emb_v, *_ = model(X_list, P_list if 'P_list' in dir() else
                                   build_operators(data['A_list_sp'],
                                                   data['bipartite_flags'], device))
                rec = recall_at_k(emb_v, va_edges, user_pos, all_items, K_rec, device)
            if rec > best_rec: best_rec = rec

    del model; torch.cuda.empty_cache()
    return best_rec


# ── 80/20 + 5-fold CV: node classification ───────────────────────────────────
def hparam_search_nc(data, seed=42, out_dir='results/nc'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Nt     = data['target_size']
    lbl_np = data['labels'].numpy()

    tr80, te20 = train_test_split(np.arange(Nt), test_size=TEST_FRAC,
                                   random_state=seed, stratify=lbl_np)
    P_list = build_operators(data['A_list_sp'], data['bipartite_flags'], device)
    X_list = [x.to(device) for x in data['X_dict'].values()]

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_BASE, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows  = []
    best_params, best_mean = None, 0.0

    for ci, params in enumerate(combos):
        fold_scores = []
        for fold, (tr_fold, va_fold) in enumerate(skf.split(tr80, lbl_np[tr80])):
            vm = _run_fold_nc(data, P_list, X_list, params,
                              tr80[tr_fold], tr80[va_fold], device)
            fold_scores.append(vm)
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_macro': round(vm, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
            print(f"  [NC] combo {ci+1}/{len(combos)}  fold {fold+1}  val_macro={vm:.4f}")

        mean_vm = float(np.mean(fold_scores))
        if mean_vm > best_mean: best_mean, best_params = mean_vm, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name',''), 'nc', out_dir)
    print(f"[NC hparam] best_val_macro={best_mean:.4f}  params={best_params}")
    return best_params, tr80, te20


# ── 80/20 + 5-fold CV: clustering ────────────────────────────────────────────
def hparam_search_cl(data, seed=42, out_dir='results/clustering'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Nt     = data['target_size']
    lbl_np = data['labels'].numpy()

    tr80, te20 = train_test_split(np.arange(Nt), test_size=TEST_FRAC,
                                   random_state=seed, stratify=lbl_np)
    P_list = build_operators(data['A_list_sp'], data['bipartite_flags'], device)
    X_list = [x.to(device) for x in data['X_dict'].values()]

    # Clustering uses StratifiedKFold to keep class distribution per fold
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_CLUSTERING, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows  = []
    best_params, best_mean = None, 0.0

    for ci, params in enumerate(combos):
        fold_nmis = []
        for fold, (tr_fold, va_fold) in enumerate(skf.split(tr80, lbl_np[tr80])):
            nmi = _run_fold_cl(data, P_list, X_list, params,
                               tr80[tr_fold], tr80[va_fold], device)
            fold_nmis.append(nmi)
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_nmi': round(nmi, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
            print(f"  [CL] combo {ci+1}/{len(combos)}  fold {fold+1}  val_nmi={nmi:.4f}")

        mean_nmi = float(np.mean(fold_nmis))
        if mean_nmi > best_mean: best_mean, best_params = mean_nmi, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name',''), 'cl', out_dir)
    print(f"[CL hparam] best_val_nmi={best_mean:.4f}  params={best_params}")
    return best_params, tr80, te20


# ── 80/20 + 5-fold CV: recommendation ────────────────────────────────────────
def hparam_search_rec(data, target_edges, seed=42, out_dir='results/recommendation'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_idx    = np.arange(len(target_edges))
    tr80_idx, te20_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=seed)
    tr80_edges = target_edges[tr80_idx]
    te20_edges = target_edges[te20_idx]

    kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos  = _random_combos(PARAM_GRID_REC, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows  = []
    best_params, best_mean = None, 0.0

    for ci, params in enumerate(combos):
        fold_recs = []
        for fold, (tr_fold, va_fold) in enumerate(kf.split(tr80_edges)):
            rec = _run_fold_rec(data, tr80_edges[tr_fold], tr80_edges[va_fold],
                                params, device, K_rec=params.get('K_rec', 20))
            fold_recs.append(rec)
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_recall': round(rec, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
            print(f"  [REC] combo {ci+1}/{len(combos)}  fold {fold+1}  val_recall@K={rec:.4f}")

        mean_rec = float(np.mean(fold_recs))
        if mean_rec > best_mean: best_mean, best_params = mean_rec, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name',''), 'rec', out_dir)
    print(f"[REC hparam] best_val_recall@K={best_mean:.4f}  params={best_params}")
    return best_params, tr80_edges, te20_edges


# ── LP search (v3 unchanged) ──────────────────────────────────────────────────
def hparam_search_lp(data, target_edges, seed=42, out_dir='results/lp'):
    from .link_prediction import _run_fold_lp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_idx = np.arange(len(target_edges))
    tr80_idx, te20_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=seed)
    tr80_edges = target_edges[tr80_idx]
    te20_edges = target_edges[te20_idx]

    kf     = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    combos = _random_combos(PARAM_GRID_BASE, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    cv_rows  = []
    best_params, best_mean = None, 0.0

    for ci, params in enumerate(combos):
        fold_aucs = []
        for fold, (tr_fold, va_fold) in enumerate(kf.split(tr80_edges)):
            auc = _run_fold_lp(data, tr80_edges[tr_fold], tr80_edges[va_fold],
                               te20_edges, params, device)
            fold_aucs.append(auc)
            cv_rows.append({'combo_id': ci, 'fold': fold, 'val_auc': round(auc, 4),
                            **{f'hp_{k}': v for k, v in params.items()}})
            print(f"  [LP] combo {ci+1}/{len(combos)}  fold {fold+1}  val_auc={auc:.4f}")

        mean_auc = float(np.mean(fold_aucs))
        if mean_auc > best_mean: best_mean, best_params = mean_auc, params

    _write_csv(cv_rows, os.path.join(out_dir, 'cv_fold_scores.csv'))
    _save_best_params(best_params, data.get('name',''), 'lp', out_dir)
    print(f"[LP hparam] best_val_auc={best_mean:.4f}  params={best_params}")
    return best_params, tr80_edges, te20_edges


# ── CSV helper ────────────────────────────────────────────────────────────────
def _write_csv(rows, path):
    import csv
    if not rows: return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  Saved → {path}")
```

---

## Step 8 — `src/tasks/node_classification.py`

```python
# src/tasks/node_classification.py
#
# Two entry points:
#   run_single_nc(data, K, epochs, seed, cfg)        ← original single-run (kept for ablations)
#   run_final_nc(data, best_params, tr80, te20, seed) ← final train on 80%, test on 20%

import numpy as np, torch, torch.nn.functional as F, time
from torch.optim import Adam
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from ..model.rahgh    import RAHGH
from ..model.diffusion import build_operators


def _evaluate(logits, target_size, idx, labels_full):
    p = logits[:target_size][idx].argmax(1).cpu().numpy()
    y = labels_full[idx].numpy()
    return ((p==y).mean(),
            f1_score(y, p, average='macro',  zero_division=0),
            f1_score(y, p, average='micro',  zero_division=0))


def run_single_nc(data, K, epochs, seed, cfg):
    """Single (K, epochs, seed) run — used for ablations and the original sweep."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P_list  = build_operators(data['A_list_sp'], data['bipartite_flags'], device)
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
    opt = Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])

    lbl_np  = data['labels'].numpy()
    tr, te  = train_test_split(np.arange(Nt), test_size=0.20,
                                random_state=seed, stratify=lbl_np)
    tr, va  = train_test_split(tr, test_size=0.10/(0.80),
                                random_state=seed, stratify=lbl_np[tr])
    tr_t    = torch.tensor(tr, dtype=torch.long, device=device)
    va_t    = torch.tensor(va, dtype=torch.long, device=device)
    te_t    = torch.tensor(te, dtype=torch.long, device=device)

    best_val, best_alpha, best_beta, best_sd = 0.0, None, None, None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        logits, a, b, _ = model(X_list, P_list)   # A_hat built inside from α
        F.cross_entropy(logits[:Nt][tr_t], labels[tr_t]).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits, a, b, _ = model(X_list, P_list)
            _, vm, _ = _evaluate(logits, Nt, va_t.cpu().numpy(), data['labels'])
        if vm > best_val:
            best_val   = vm
            best_alpha = a.detach().cpu().numpy().copy()
            best_beta  = b.detach().cpu().numpy().copy()
            best_sd    = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd); model.eval()
    with torch.no_grad():
        logits, *_ = model(X_list, P_list)
        acc, macro, micro = _evaluate(logits, Nt, te_t.cpu().numpy(), data['labels'])

    return dict(test_acc=acc, test_macro=macro, test_micro=micro,
                best_val_macro=best_val, alpha=best_alpha, beta=best_beta,
                time_sec=time.time()-t0)


def run_final_nc(data, best_params, tr80_idx, te20_idx, seed=42):
    """
    Final training on FULL 80 % with best hyperparams, test on held-out 20 %.
    Called after hparam_search_nc() selects best_params.

    GNN note: message passing uses the FULL graph throughout.
              Only the loss mask (tr80_idx) changes vs the ablation run.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P_list  = build_operators(data['A_list_sp'], data['bipartite_flags'], device)
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
    opt = Adam(model.parameters(),
               lr=best_params['lr'], weight_decay=best_params['wd'])

    tr_t = torch.tensor(tr80_idx, dtype=torch.long, device=device)
    te_t = torch.tensor(te20_idx, dtype=torch.long, device=device)
    t0   = time.time()

    print(f"[final_nc] Training on {len(tr80_idx)} nodes for {best_params['epochs']} epochs...")
    for ep in range(1, best_params['epochs'] + 1):
        model.train(); opt.zero_grad()
        logits, *_ = model(X_list, P_list)
        F.cross_entropy(logits[:Nt][tr_t], labels[tr_t]).backward()
        opt.step()
        if ep % 100 == 0:
            print(f"  epoch {ep}/{best_params['epochs']}")

    model.eval()
    with torch.no_grad():
        logits, alpha, beta, _ = model(X_list, P_list)
        acc, macro, micro = _evaluate(logits, Nt, te20_idx, data['labels'])

    print(f"[final_nc] test_macro={macro:.4f}  test_micro={micro:.4f}  "
          f"time={time.time()-t0:.1f}s")
    return dict(test_acc=acc, test_macro=macro, test_micro=micro,
                alpha=alpha.detach().cpu().numpy(),
                beta=beta.detach().cpu().numpy(),
                time_sec=time.time()-t0)
```

---

## Step 9 — `src/tasks/link_prediction.py`

**v3 LP fix:** `emb, _, _, _ = backbone(...)` — uses GCN output (position 0), not Z_final (position 3).

```python
# src/tasks/link_prediction.py

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp

from ..model.rahgh    import RAHGH
from ..model.diffusion import build_operators, normalize_plain


def split_edges(edges, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(edges)); n = len(edges)
    n_tr = int(0.8*n); n_va = int(0.1*n)
    return edges[idx[:n_tr]], edges[idx[n_tr:n_tr+n_va]], edges[idx[n_tr+n_va:]]


def sample_negatives(pos, n_neg, all_src, all_dst, seed=0):
    rng = np.random.default_rng(seed); pos_set = set(map(tuple,pos)); negs = []
    while len(negs) < n_neg:
        s = rng.choice(all_src,size=n_neg*2); d = rng.choice(all_dst,size=n_neg*2)
        for si,di in zip(s,d):
            if (si,di) not in pos_set: negs.append((si,di))
            if len(negs)>=n_neg: break
    return np.array(negs[:n_neg])


class MLPDecoder(nn.Module):
    """
    Input  : [emb[src] ‖ emb[dst]]  ∈  ℝ^{2×out_dim}  where out_dim = d = 64
    Output : score  ∈  ℝ  (raw logit for BCE)
    """
    def __init__(self, emb_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
    def forward(self, emb, src, dst):
        return self.mlp(torch.cat([emb[src], emb[dst]], dim=1)).squeeze(-1)


def _build_masked_operators(data, train_edges, device):
    """
    For LP, build operators with ONLY training edges visible.
    This prevents message-passing leakage from val/test edges.
    Returns P_list and A_hat built from train_edges only.
    """
    N = data['N']
    # Remove val/test edges from adjacency for this fold
    tr_r, tr_c = train_edges[:,0], train_edges[:,1]
    A_train = sp.coo_matrix((np.ones(len(tr_r)),(tr_r,tr_c)),shape=(N,N)).tocsr()
    # Rebuild A_list_sp with training edges only for target relation (index 0)
    A_list_masked = list(data['A_list_sp'])   # shallow copy
    A_list_masked[0] = A_train                # replace with train-only edges
    A_list_masked[1] = A_train.T.tocsr()      # reverse direction
    return build_operators(A_list_masked, data['bipartite_flags'], device)


def _run_fold_lp(data, tr_edges, va_edges, te_edges, params, device, neg_ratio=5):
    """Single LP fold for CV. Returns val AUC."""
    torch.manual_seed(0); np.random.seed(0)
    d       = params['d']
    R       = len(data['A_list_sp'])
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    X_list  = [x.to(device) for x in data['X_dict'].values()]

    # Build operators without val+test edges (leakage prevention)
    P_list  = _build_masked_operators(data, tr_edges, device)

    backbone = RAHGH(
        in_dims=in_dims, d=d, R=R, K=params['K'],
        gcn_hidden=params['gcn_hidden'],
        out_dim=d,           # LP: out_dim = d so emb dim matches Z_final
        dropout=params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    decoder = MLPDecoder(d).to(device)
    opt = Adam(list(backbone.parameters())+list(decoder.parameters()),
               lr=params['lr'], weight_decay=params['wd'])

    all_src = np.unique(tr_edges[:,0]); all_dst = np.unique(tr_edges[:,1])
    tr_neg  = sample_negatives(tr_edges, len(tr_edges)*neg_ratio, all_src, all_dst, 0)
    va_neg  = sample_negatives(va_edges, len(va_edges)*neg_ratio, all_src, all_dst, 1)

    def tensors(pos, neg):
        e = np.concatenate([pos,neg],0)
        l = np.concatenate([np.ones(len(pos)),np.zeros(len(neg))]).astype(np.float32)
        return (torch.tensor(e[:,0],dtype=torch.long,device=device),
                torch.tensor(e[:,1],dtype=torch.long,device=device),
                torch.tensor(l,device=device))

    tr_s,tr_d,tr_l = tensors(tr_edges, tr_neg)
    va_s,va_d,va_l = tensors(va_edges, va_neg)

    best_auc = 0.0
    for ep in range(1, params['epochs']+1):
        backbone.train(); decoder.train(); opt.zero_grad()
        # ── v3 fix: use position [0] = GCN output, not position [3] = Z_final
        emb, *_ = backbone(X_list, P_list)
        F.binary_cross_entropy_with_logits(decoder(emb,tr_s,tr_d), tr_l).backward()
        opt.step()

        if ep % 50 == 0 or ep == params['epochs']:
            backbone.eval(); decoder.eval()
            with torch.no_grad():
                emb_v, *_ = backbone(X_list, P_list)    # position [0] = GCN output
                p = torch.sigmoid(decoder(emb_v,va_s,va_d)).cpu().numpy()
                auc = roc_auc_score(va_l.cpu().numpy(), p)
            if auc > best_auc: best_auc = auc

    del backbone, decoder; torch.cuda.empty_cache()
    return best_auc


def run_final_lp(data, best_params, tr80_edges, te20_edges, seed=42, neg_ratio=5):
    """
    Final LP training on full 80 % edges with best hyperparams.
    Tests on held-out 20 % edges.
    Test edges are excluded from graph (no leakage).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d       = best_params['d']
    R       = len(data['A_list_sp'])
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    X_list  = [x.to(device) for x in data['X_dict'].values()]

    # Build operators excluding test edges
    P_list = _build_masked_operators(data, tr80_edges, device)

    backbone = RAHGH(
        in_dims=in_dims, d=d, R=R, K=best_params['K'],
        gcn_hidden=best_params['gcn_hidden'],
        out_dim=d,
        dropout=best_params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    decoder = MLPDecoder(d).to(device)
    opt = Adam(list(backbone.parameters())+list(decoder.parameters()),
               lr=best_params['lr'], weight_decay=best_params['wd'])

    all_src = np.unique(tr80_edges[:,0]); all_dst = np.unique(tr80_edges[:,1])
    tr_neg  = sample_negatives(tr80_edges, len(tr80_edges)*neg_ratio, all_src,all_dst,0)
    te_neg  = sample_negatives(te20_edges, len(te20_edges)*neg_ratio, all_src,all_dst,2)

    def tensors(pos,neg):
        e = np.concatenate([pos,neg],0)
        l = np.concatenate([np.ones(len(pos)),np.zeros(len(neg))]).astype(np.float32)
        return (torch.tensor(e[:,0],dtype=torch.long,device=device),
                torch.tensor(e[:,1],dtype=torch.long,device=device),
                torch.tensor(l,device=device))

    tr_s,tr_d,tr_l = tensors(tr80_edges, tr_neg)
    te_s,te_d,te_l = tensors(te20_edges, te_neg)

    best_sd_b, best_sd_d, best_auc = None, None, 0.0
    t0 = time.time()
    print(f"[final_lp] Training on {len(tr80_edges)} edges for {best_params['epochs']} epochs...")

    for ep in range(1, best_params['epochs']+1):
        backbone.train(); decoder.train(); opt.zero_grad()
        emb, *_ = backbone(X_list, P_list)              # GCN output ∈ ℝ^{N×d}
        F.binary_cross_entropy_with_logits(decoder(emb,tr_s,tr_d), tr_l).backward()
        opt.step()

        if ep % 50 == 0 or ep == best_params['epochs']:
            backbone.eval(); decoder.eval()
            with torch.no_grad():
                emb_v, a, b, _ = backbone(X_list, P_list)
                pv = torch.sigmoid(decoder(emb_v,tr_s,tr_d)).cpu().numpy()
                av = roc_auc_score(tr_l.cpu().numpy(), pv)
            if av > best_auc:
                best_auc = av
                best_sd_b = {k:v.clone() for k,v in backbone.state_dict().items()}
                best_sd_d = {k:v.clone() for k,v in decoder.state_dict().items()}
            if ep % 100 == 0: print(f"  epoch {ep}  train_auc={av:.4f}")

    backbone.load_state_dict(best_sd_b)
    decoder.load_state_dict(best_sd_d)
    backbone.eval(); decoder.eval()
    with torch.no_grad():
        emb_te, alpha, beta, _ = backbone(X_list, P_list)   # GCN output
        p = torch.sigmoid(decoder(emb_te,te_s,te_d)).cpu().numpy()
        auc = roc_auc_score(te_l.cpu().numpy(), p)
        ap  = __import__('sklearn.metrics',fromlist=['average_precision_score']
                         ).average_precision_score(te_l.cpu().numpy(), p)

    print(f"[final_lp] test_auc={auc:.4f}  test_ap={ap:.4f}  time={time.time()-t0:.1f}s")
    return dict(auc=auc, ap=ap,
                alpha=alpha.detach().cpu().numpy(),
                beta=beta.detach().cpu().numpy(),
                time_sec=time.time()-t0)
```

---

## Step 10 — `src/tasks/graph_clustering.py` *(NEW)*

```python
# src/tasks/graph_clustering.py
#
# Graph clustering on Z_final (GCN embeddings).
# Unsupervised training → K-Means evaluation.
#
# Metrics:
#   NMI  — Normalized Mutual Information
#   ARI  — Adjusted Rand Index
#   ACC  — Clustering accuracy via Hungarian algorithm
#
# Protocol:
#   Same 80/20 split as NC.
#   Training uses UNSUPERVISED objective (reconstruction loss).
#   Evaluation clusters ALL target nodes (Nt), measures against ground truth.

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, os, csv
from torch.optim import Adam
from sklearn.cluster  import KMeans
from sklearn.metrics  import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize   import linear_sum_assignment

from ..model.rahgh    import RAHGH
from ..model.diffusion import build_operators


# ── Hungarian-based clustering accuracy ──────────────────────────────────────
def clustering_accuracy(y_true, y_pred):
    """
    Compute clustering accuracy via Hungarian algorithm.
    Allows any permutation of cluster labels.
    """
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    n_classes = max(y_true.max(), y_pred.max()) + 1
    # Confusion matrix
    D = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        D[t, p] += 1
    # Hungarian: maximise diagonal → negate for minimisation
    row_ind, col_ind = linear_sum_assignment(-D)
    return D[row_ind, col_ind].sum() / len(y_true)


# ── Feature reconstruction loss ───────────────────────────────────────────────
class ReconDecoder(nn.Module):
    """Maps GCN embedding back toward H0 space for reconstruction loss."""
    def __init__(self, d: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, out_dim))
    def forward(self, emb):
        return self.mlp(emb)


def _get_target_features(X_list, target_size, d, device):
    """
    Project concatenated X features for target nodes to dim d for reconstruction target.
    Uses first X_list entry (target node type features).
    """
    X_target = X_list[0][:target_size]     # (Nt, d_feat)
    if X_target.shape[1] != d:
        proj = nn.Linear(X_target.shape[1], d, bias=False).to(device)
        with torch.no_grad():
            X_target = proj(X_target)
    return X_target.detach()


# ── Final clustering run (after CV selects best params) ──────────────────────
def run_final_clustering(data, best_params, tr80_idx, te20_idx,
                         seed=42, out_dir='results/clustering'):
    """
    Train with best_params on all 80 % labelled nodes (unsupervised).
    Cluster ALL Nt target nodes. Report NMI, ARI, ACC.

    Note: clustering is evaluated on all Nt nodes (not just test set)
    because K-Means uses the full embedding space. The 80/20 split only
    controls which labels were used during CV — not during inference.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d       = best_params['d']
    R       = len(data['A_list_sp'])
    Nt      = data['target_size']
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    n_cl    = data['n_classes']

    P_list  = build_operators(data['A_list_sp'], data['bipartite_flags'], device)
    X_list  = [x.to(device) for x in data['X_dict'].values()]

    model   = RAHGH(
        in_dims=in_dims, d=d, R=R, K=best_params['K'],
        gcn_hidden=best_params['gcn_hidden'],
        out_dim=d, dropout=best_params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)

    # Reconstruction decoder: d → d_feat of target node type
    d_feat  = data['X_dict'][list(data['X_dict'].keys())[0]].shape[1]
    decoder = ReconDecoder(d, min(d_feat, d)).to(device)

    opt  = Adam(list(model.parameters()) + list(decoder.parameters()),
                lr=best_params['lr'], weight_decay=best_params['wd'])
    tr_t = torch.tensor(tr80_idx, dtype=torch.long, device=device)

    # Epoch-level logging
    epoch_rows = []
    best_nmi, best_sd = 0.0, None
    t0 = time.time()

    for ep in range(1, best_params['epochs'] + 1):
        model.train(); decoder.train(); opt.zero_grad()
        emb, *_ = model(X_list, P_list)             # (N, d)

        # Reconstruction: predict X features for training target nodes
        recon   = decoder(emb[:Nt][tr_t])           # (|tr80|, d_out)
        X_tgt   = X_list[0][:Nt][tr_t]             # (|tr80|, d_feat)
        if X_tgt.shape[1] != recon.shape[1]:
            X_tgt = X_tgt[:, :recon.shape[1]]       # trim if needed
        loss    = F.mse_loss(recon, X_tgt.detach())
        loss.backward(); opt.step()

        epoch_rows.append({'epoch': ep, 'recon_loss': round(loss.item(), 6)})

        if ep % 100 == 0 or ep == best_params['epochs']:
            model.eval()
            with torch.no_grad():
                emb_v, *_ = model(X_list, P_list)
                emb_np    = emb_v[:Nt].cpu().numpy()
                km        = KMeans(n_clusters=n_cl, n_init=10, random_state=0)
                pred      = km.fit_predict(emb_np)
                y         = data['labels'].numpy()
                nmi       = normalized_mutual_info_score(y, pred)
                ari       = adjusted_rand_score(y, pred)
                acc       = clustering_accuracy(y, pred)
            print(f"  ep={ep}  loss={loss.item():.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  ACC={acc:.4f}")
            if nmi > best_nmi:
                best_nmi = nmi
                best_sd  = {k: v.clone() for k, v in model.state_dict().items()}

    # Final evaluation with best checkpoint
    model.load_state_dict(best_sd); model.eval()
    with torch.no_grad():
        emb_f, alpha, beta, _ = model(X_list, P_list)
        emb_np = emb_f[:Nt].cpu().numpy()
        km     = KMeans(n_clusters=n_cl, n_init=20, random_state=seed)
        pred   = km.fit_predict(emb_np)
        y      = data['labels'].numpy()
        nmi    = normalized_mutual_info_score(y, pred)
        ari    = adjusted_rand_score(y, pred)
        acc    = clustering_accuracy(y, pred)

    # Save epoch log
    os.makedirs(os.path.join(out_dir, 'epoch_logs'), exist_ok=True)
    _write_csv_cl(epoch_rows,
                  os.path.join(out_dir, 'epoch_logs', f'seed{seed}_epochs.csv'))

    print(f"[CL seed={seed}] NMI={nmi:.4f}  ARI={ari:.4f}  ACC={acc:.4f}  "
          f"time={time.time()-t0:.1f}s")
    return dict(nmi=nmi, ari=ari, acc=acc,
                alpha=alpha.detach().cpu().numpy(),
                beta=beta.detach().cpu().numpy(),
                time_sec=time.time()-t0)


def _write_csv_cl(rows, path):
    if not rows: return
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
```

---

## Step 11 — `src/tasks/recommendation.py` *(NEW)*

```python
# src/tasks/recommendation.py
#
# Recommendation task using BPR (Bayesian Personalised Ranking) loss.
# Treats heterogeneous graph link prediction as user-item ranking.
#
# For DBLP/ACM/IMDB: the first bipartite relation defines user-item edges.
#   DBLP  : author ↔ paper   (paper→author relation index 0)
#   ACM   : paper  ↔ venue   (paper→venue  relation index 4)
#   IMDB  : movie  ↔ actor   (movie→actor  relation index 2)
#
# For dedicated recommendation datasets (Amazon, LastFM):
#   user ↔ item is the primary target relation.
#
# Metrics:
#   Recall@K, NDCG@K, Hit Rate@K, Precision@K, MRR
#   K ∈ {10, 20, 50}
#
# Protocol:
#   Same 80/20 split as LP.
#   Training: BPR on 80 % interactions.
#   Evaluation: ranking metrics on held-out 20 % per user.

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import time, os, csv
from torch.optim import Adam

from ..model.rahgh    import RAHGH
from ..model.diffusion import build_operators


# ── BPR loss ─────────────────────────────────────────────────────────────────
def bpr_loss(emb, users, pos_items, neg_items, device, reg=1e-4):
    """
    Bayesian Personalised Ranking loss.
    emb       : (N, d) node embeddings
    users     : (B,)   user node indices
    pos_items : (B,)   positive item node indices
    neg_items : (B,)   negative item node indices (sampled uniformly)
    """
    u   = emb[torch.tensor(users,     dtype=torch.long, device=device)]
    pos = emb[torch.tensor(pos_items, dtype=torch.long, device=device)]
    neg = emb[torch.tensor(neg_items, dtype=torch.long, device=device)]

    pos_score = (u * pos).sum(dim=1)
    neg_score = (u * neg).sum(dim=1)
    loss      = -F.logsigmoid(pos_score - neg_score).mean()

    # L2 regularization on embeddings
    reg_loss = reg * (u.norm(2).pow(2) + pos.norm(2).pow(2) + neg.norm(2).pow(2)) / len(users)
    return loss + reg_loss


# ── Top-K recommendation metrics ─────────────────────────────────────────────
def compute_rec_metrics(emb, test_edges, user_train_pos, all_items, K_list, device):
    """
    Compute Recall@K, NDCG@K, Hit@K, Precision@K, MRR for each K in K_list.

    emb           : (N, d) embeddings
    test_edges    : (E, 2) array of (user, item) positive test pairs
    user_train_pos: dict {user: set(train_items)} — exclude from ranking
    all_items     : array of all item node ids
    K_list        : list of K values e.g. [10, 20, 50]
    """
    emb_np    = emb.cpu().numpy()
    user_test = {}
    for u, i in test_edges:
        user_test.setdefault(u, []).append(i)

    results = {K: {'recall':[], 'ndcg':[], 'hit':[], 'precision':[], 'mrr':[]}
               for K in K_list}

    for user, pos_list in user_test.items():
        u_emb    = emb_np[user]                           # (d,)
        item_emb = emb_np[all_items]                      # (|items|, d)
        scores   = item_emb @ u_emb                       # (|items|,)

        # Mask out training positives
        train_pos = user_train_pos.get(user, set())
        for idx, item in enumerate(all_items):
            if item in train_pos:
                scores[idx] = -1e9

        top_K_max = max(K_list)
        top_idx   = np.argpartition(scores, -top_K_max)[-top_K_max:]
        top_idx   = top_idx[np.argsort(scores[top_idx])[::-1]]
        top_items = all_items[top_idx]   # ranked top-K items

        pos_set = set(pos_list)

        for K in K_list:
            recs     = top_items[:K]
            hits     = [1 if i in pos_set else 0 for i in recs]
            n_hits   = sum(hits)

            # Recall@K
            results[K]['recall'].append(n_hits / len(pos_set))

            # NDCG@K
            dcg  = sum(h / np.log2(r+2) for r, h in enumerate(hits))
            idcg = sum(1.0 / np.log2(r+2) for r in range(min(len(pos_set), K)))
            results[K]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

            # Hit Rate@K (1 if at least one hit)
            results[K]['hit'].append(float(n_hits > 0))

            # Precision@K
            results[K]['precision'].append(n_hits / K)

            # MRR (reciprocal rank of first hit)
            rr = 0.0
            for r, i in enumerate(recs):
                if i in pos_set: rr = 1.0 / (r+1); break
            results[K]['mrr'].append(rr)

    # Aggregate
    agg = {}
    for K in K_list:
        for metric, vals in results[K].items():
            agg[f'{metric}@{K}'] = float(np.mean(vals))
    return agg


# ── Negative sampler ─────────────────────────────────────────────────────────
def sample_bpr_negatives(users, all_items, user_pos, rng, n=None):
    """Sample one negative item per user, not in their positive set."""
    n       = n or len(users)
    neg     = rng.choice(all_items, size=n*3)
    out     = []
    ni      = 0
    for u in users:
        pos_set = user_pos.get(u, set())
        while neg[ni] in pos_set:
            ni += 1
            if ni >= len(neg): neg = rng.choice(all_items, size=n*3); ni = 0
        out.append(neg[ni]); ni += 1
    return np.array(out)


# ── Final recommendation run ──────────────────────────────────────────────────
def run_final_recommendation(data, best_params, tr80_edges, te20_edges,
                              target_relation_idx=0,
                              K_list=(10, 20, 50),
                              seed=42,
                              out_dir='results/recommendation'):
    """
    Train BPR on 80 % edges, evaluate ranking metrics on 20 % held-out.

    target_relation_idx : which A_list_sp entry defines user-item edges
                          DBLP=0 (author-paper), ACM=4 (paper-venue), IMDB=2 (movie-actor)
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d       = best_params['d']
    R       = len(data['A_list_sp'])
    in_dims = [x.shape[1] for x in data['X_dict'].values()]
    X_list  = [x.to(device) for x in data['X_dict'].values()]
    K_rec   = best_params.get('K_rec', 20)
    K_list  = list(K_list)

    P_list  = build_operators(data['A_list_sp'], data['bipartite_flags'], device)

    backbone = RAHGH(
        in_dims=in_dims, d=d, R=R, K=best_params['K'],
        gcn_hidden=best_params['gcn_hidden'],
        out_dim=d, dropout=best_params['dropout'],
        A_list_sp=data['A_list_sp'], N=data['N'], device=device,
    ).to(device)
    opt = Adam(backbone.parameters(), lr=best_params['lr'],
               weight_decay=best_params['wd'])

    all_items  = np.unique(tr80_edges[:, 1])
    user_pos   = {}
    for u, i in tr80_edges: user_pos.setdefault(u, set()).add(i)

    rng        = np.random.default_rng(seed)
    epoch_rows = []
    best_agg   = None
    best_rec   = 0.0
    best_sd    = None
    t0         = time.time()

    for ep in range(1, best_params['epochs'] + 1):
        backbone.train(); opt.zero_grad()
        emb, *_ = backbone(X_list, P_list)           # (N, d)

        users     = tr80_edges[:, 0]
        pos_items = tr80_edges[:, 1]
        neg_items = sample_bpr_negatives(users, all_items, user_pos, rng)

        loss = bpr_loss(emb, users, pos_items, neg_items, device,
                        reg=best_params.get('bpr_reg', 1e-4))
        loss.backward(); opt.step()

        epoch_rows.append({'epoch': ep, 'bpr_loss': round(loss.item(), 6)})

        if ep % 50 == 0 or ep == best_params['epochs']:
            backbone.eval()
            with torch.no_grad():
                emb_v, *_ = backbone(X_list, P_list)
                # Quick Recall@K_rec on held-out edges
                agg = compute_rec_metrics(emb_v, te20_edges, user_pos, all_items,
                                          [K_rec], device)
                rec = agg[f'recall@{K_rec}']
            print(f"  ep={ep}  loss={loss.item():.4f}  Recall@{K_rec}={rec:.4f}")
            if rec > best_rec:
                best_rec = rec
                best_sd  = {k: v.clone() for k, v in backbone.state_dict().items()}

    # Final evaluation with all K values
    backbone.load_state_dict(best_sd); backbone.eval()
    with torch.no_grad():
        emb_f, alpha, beta, _ = backbone(X_list, P_list)
        final_agg = compute_rec_metrics(emb_f, te20_edges, user_pos,
                                         all_items, K_list, device)

    # Save epoch log
    os.makedirs(os.path.join(out_dir, 'epoch_logs'), exist_ok=True)
    _write_csv_rec(epoch_rows,
                   os.path.join(out_dir, 'epoch_logs', f'seed{seed}_epochs.csv'))

    print(f"[REC seed={seed}]  " +
          "  ".join([f"R@{K}={final_agg[f'recall@{K}']:.4f}" for K in K_list]))
    return dict(**final_agg,
                alpha=alpha.detach().cpu().numpy(),
                beta=beta.detach().cpu().numpy(),
                time_sec=time.time()-t0)


def _write_csv_rec(rows, path):
    if not rows: return
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)


def recall_at_k(emb, test_edges, user_train_pos, all_items, K, device):
    """Lightweight Recall@K for CV fold evaluation."""
    agg = compute_rec_metrics(emb, test_edges, user_train_pos, all_items, [K], device)
    return agg[f'recall@{K}']
```

---

## Step 12 — `src/train.py` *(updated v4 — all 4 tasks, 10 seeds, full CSV logging)*

```python
# src/train.py
# v4: handles NC, LP, Clustering, Recommendation
#     Final test repeated N_SEEDS=10 times with different random seeds.
#     ALL metrics, losses, and hyperparameter info written to CSV.

import csv, os, json, argparse, time
import numpy as np

from data.dblp_loader import load_dblp
from data.acm_loader  import load_acm
from data.imdb_loader import load_imdb

from tasks.hparam_search      import (hparam_search_nc, hparam_search_lp,
                                       hparam_search_cl, hparam_search_rec)
from tasks.node_classification import run_final_nc
from tasks.link_prediction     import run_final_lp
from tasks.graph_clustering    import run_final_clustering
from tasks.recommendation      import run_final_recommendation

LOADERS  = {'dblp': load_dblp, 'acm': load_acm, 'imdb': load_imdb}
N_SEEDS  = 10    # final test repetitions (paper reports mean ± SD over 10 seeds)

# Target relation index per dataset for recommendation and LP
TARGET_REL_IDX = {'dblp': 0, 'acm': 4, 'imdb': 2}

# Output directories per task
RESULT_DIRS = {
    'nc'  : 'results/nc',
    'lp'  : 'results/lp',
    'cl'  : 'results/clustering',
    'rec' : 'results/recommendation',
}


# ── CSV helpers ───────────────────────────────────────────────────────────────
def write_per_run_csv(rows, path):
    """Append per-run rows to CSV (creates file if missing, adds header once)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists: w.writeheader()
        w.writerows(rows)
    print(f"  Per-run results appended → {path}")


def write_summary_csv(summary_row, path):
    """Write or append one summary row (mean ± SD) to summary CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_row.keys())
        if not file_exists: w.writeheader()
        w.writerow(summary_row)
    print(f"  Summary appended → {path}")


def _flatten_result(r):
    """Convert numpy arrays in result dict to serialisable floats."""
    out = {}
    for k, v in r.items():
        if isinstance(v, np.ndarray):
            # Store α and β as flat scalars with indexed keys
            for i, val in enumerate(v.flatten()):
                out[f"{k}_{i}"] = round(float(val), 6)
        elif isinstance(v, (float, int, np.floating, np.integer)):
            out[k] = round(float(v), 6)
        else:
            out[k] = v
    return out


# ── Node Classification ───────────────────────────────────────────────────────
def run_nc(dataset_name, out_dir):
    data = LOADERS[dataset_name]()
    print(f"\n{'='*60}\n  {dataset_name.upper()} — Node Classification\n{'='*60}")

    # Step 1: 80/20 + 5-fold CV
    best_params, tr80, te20 = hparam_search_nc(data, seed=42, out_dir=out_dir)

    # Step 2: 10-seed final training
    per_run_rows, macros, micros, accs, aucs = [], [], [], [], []

    for seed in range(N_SEEDS):
        r = run_final_nc(data, best_params, tr80, te20, seed=seed)

        macros.append(r['test_macro'])
        micros.append(r['test_micro'])
        accs.append(r['test_acc'])

        # AUC: compute from logits if available, else skip
        aucs.append(r.get('test_auc', float('nan')))

        row = {'dataset': dataset_name, 'task': 'nc', 'seed': seed,
               'test_macro_f1'  : round(r['test_macro'], 4),
               'test_micro_f1'  : round(r['test_micro'], 4),
               'test_accuracy'  : round(r['test_acc'],   4),
               'time_sec'       : round(r['time_sec'],   2),
               **{f'hp_{k}': v for k, v in best_params.items()}}
        per_run_rows.append(_flatten_result({**row}))

    write_per_run_csv(per_run_rows, os.path.join(out_dir, 'per_run_results.csv'))

    # Summary row — mean ± SD over 10 seeds
    summary = {
        'dataset'          : dataset_name,
        'task'             : 'nc',
        'macro_f1_mean'    : round(float(np.mean(macros)), 4),
        'macro_f1_sd'      : round(float(np.std(macros)),  4),
        'micro_f1_mean'    : round(float(np.mean(micros)), 4),
        'micro_f1_sd'      : round(float(np.std(micros)),  4),
        'accuracy_mean'    : round(float(np.mean(accs)),   4),
        'accuracy_sd'      : round(float(np.std(accs)),    4),
        'n_seeds'          : N_SEEDS,
        **{f'best_hp_{k}': v for k, v in best_params.items()},
    }
    write_summary_csv(summary, os.path.join(out_dir, 'summary.csv'))

    print(f"\n{'─'*60}")
    print(f"  {dataset_name} NC  (n={N_SEEDS} seeds)")
    print(f"  Macro-F1 : {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_sd']:.4f}")
    print(f"  Micro-F1 : {summary['micro_f1_mean']:.4f} ± {summary['micro_f1_sd']:.4f}")
    print(f"  Accuracy : {summary['accuracy_mean']:.4f} ± {summary['accuracy_sd']:.4f}")


# ── Link Prediction ───────────────────────────────────────────────────────────
def run_lp(dataset_name, out_dir):
    import scipy.sparse as sp
    data   = LOADERS[dataset_name]()
    A      = data['A_list_sp'][TARGET_REL_IDX[dataset_name]].tocoo()
    target_edges = np.column_stack([A.row, A.col])

    print(f"\n{'='*60}\n  {dataset_name.upper()} — Link Prediction\n{'='*60}")
    best_params, tr80_edges, te20_edges = hparam_search_lp(
        data, target_edges, seed=42, out_dir=out_dir)

    per_run_rows, aucs, aps = [], [], []
    for seed in range(N_SEEDS):
        r = run_final_lp(data, best_params, tr80_edges, te20_edges, seed=seed)
        aucs.append(r['auc']); aps.append(r['ap'])
        row = {'dataset': dataset_name, 'task': 'lp', 'seed': seed,
               'test_auc': round(r['auc'], 4),
               'test_ap' : round(r['ap'],  4),
               'time_sec': round(r['time_sec'], 2),
               **{f'hp_{k}': v for k, v in best_params.items()}}
        per_run_rows.append(_flatten_result(row))

    write_per_run_csv(per_run_rows, os.path.join(out_dir, 'per_run_results.csv'))

    summary = {
        'dataset'    : dataset_name, 'task': 'lp',
        'auc_mean'   : round(float(np.mean(aucs)), 4),
        'auc_sd'     : round(float(np.std(aucs)),  4),
        'ap_mean'    : round(float(np.mean(aps)),  4),
        'ap_sd'      : round(float(np.std(aps)),   4),
        'n_seeds'    : N_SEEDS,
        **{f'best_hp_{k}': v for k, v in best_params.items()},
    }
    write_summary_csv(summary, os.path.join(out_dir, 'summary.csv'))

    print(f"\n  {dataset_name} LP  (n={N_SEEDS} seeds)")
    print(f"  AUC : {summary['auc_mean']:.4f} ± {summary['auc_sd']:.4f}")
    print(f"  AP  : {summary['ap_mean']:.4f} ± {summary['ap_sd']:.4f}")


# ── Graph Clustering ──────────────────────────────────────────────────────────
def run_cl(dataset_name, out_dir):
    data = LOADERS[dataset_name]()
    print(f"\n{'='*60}\n  {dataset_name.upper()} — Graph Clustering\n{'='*60}")

    best_params, tr80, te20 = hparam_search_cl(data, seed=42, out_dir=out_dir)

    per_run_rows, nmis, aris, accs = [], [], [], []
    for seed in range(N_SEEDS):
        r = run_final_clustering(data, best_params, tr80, te20,
                                  seed=seed, out_dir=out_dir)
        nmis.append(r['nmi']); aris.append(r['ari']); accs.append(r['acc'])
        row = {'dataset': dataset_name, 'task': 'cl', 'seed': seed,
               'nmi'     : round(r['nmi'], 4),
               'ari'     : round(r['ari'], 4),
               'acc'     : round(r['acc'], 4),
               'time_sec': round(r['time_sec'], 2),
               **{f'hp_{k}': v for k, v in best_params.items()}}
        per_run_rows.append(_flatten_result(row))

    write_per_run_csv(per_run_rows, os.path.join(out_dir, 'per_run_results.csv'))

    summary = {
        'dataset'  : dataset_name, 'task': 'cl',
        'nmi_mean' : round(float(np.mean(nmis)), 4),
        'nmi_sd'   : round(float(np.std(nmis)),  4),
        'ari_mean' : round(float(np.mean(aris)), 4),
        'ari_sd'   : round(float(np.std(aris)),  4),
        'acc_mean' : round(float(np.mean(accs)), 4),
        'acc_sd'   : round(float(np.std(accs)),  4),
        'n_seeds'  : N_SEEDS,
        **{f'best_hp_{k}': v for k, v in best_params.items()},
    }
    write_summary_csv(summary, os.path.join(out_dir, 'summary.csv'))

    print(f"\n  {dataset_name} CL  (n={N_SEEDS} seeds)")
    print(f"  NMI : {summary['nmi_mean']:.4f} ± {summary['nmi_sd']:.4f}")
    print(f"  ARI : {summary['ari_mean']:.4f} ± {summary['ari_sd']:.4f}")
    print(f"  ACC : {summary['acc_mean']:.4f} ± {summary['acc_sd']:.4f}")


# ── Recommendation ────────────────────────────────────────────────────────────
def run_rec(dataset_name, out_dir, K_list=(10, 20, 50)):
    import scipy.sparse as sp
    data   = LOADERS[dataset_name]()
    A      = data['A_list_sp'][TARGET_REL_IDX[dataset_name]].tocoo()
    target_edges = np.column_stack([A.row, A.col])

    print(f"\n{'='*60}\n  {dataset_name.upper()} — Recommendation\n{'='*60}")
    best_params, tr80_edges, te20_edges = hparam_search_rec(
        data, target_edges, seed=42, out_dir=out_dir)

    K_list       = list(K_list)
    per_run_rows = []
    metric_vals  = {f'{m}@{K}': [] for K in K_list
                    for m in ['recall','ndcg','hit','precision','mrr']}

    for seed in range(N_SEEDS):
        r = run_final_recommendation(
            data, best_params, tr80_edges, te20_edges,
            target_relation_idx=TARGET_REL_IDX[dataset_name],
            K_list=K_list, seed=seed, out_dir=out_dir)

        for key in metric_vals: metric_vals[key].append(r.get(key, float('nan')))

        row = {'dataset': dataset_name, 'task': 'rec', 'seed': seed,
               **{k: round(r[k], 4) for k in metric_vals if k in r},
               'time_sec': round(r['time_sec'], 2),
               **{f'hp_{k}': v for k, v in best_params.items()}}
        per_run_rows.append(_flatten_result(row))

    write_per_run_csv(per_run_rows, os.path.join(out_dir, 'per_run_results.csv'))

    summary = {'dataset': dataset_name, 'task': 'rec', 'n_seeds': N_SEEDS}
    for key, vals in metric_vals.items():
        summary[f'{key}_mean'] = round(float(np.nanmean(vals)), 4)
        summary[f'{key}_sd']   = round(float(np.nanstd(vals)),  4)
    summary.update({f'best_hp_{k}': v for k, v in best_params.items()})
    write_summary_csv(summary, os.path.join(out_dir, 'summary.csv'))

    print(f"\n  {dataset_name} REC  (n={N_SEEDS} seeds)")
    for K in K_list:
        print(f"  Recall@{K:<3}: {summary[f'recall@{K}_mean']:.4f} ± {summary[f'recall@{K}_sd']:.4f}"
              f"   NDCG@{K}: {summary[f'ndcg@{K}_mean']:.4f} ± {summary[f'ndcg@{K}_sd']:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────
TASK_FNS = {'nc': run_nc, 'lp': run_lp, 'cl': run_cl, 'rec': run_rec}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['dblp','acm','imdb'], required=True)
    parser.add_argument('--task',    choices=['nc','lp','cl','rec'], required=True)
    parser.add_argument('--seeds',   type=int, default=N_SEEDS)
    args = parser.parse_args()

    N_SEEDS = args.seeds
    out_dir = RESULT_DIRS[args.task]
    TASK_FNS[args.task](args.dataset, out_dir)
```

---

## Step 13 — Updated configs *(clustering and recommendation)*

```yaml
# configs/dblp_cl.yaml
dataset: dblp
task: cl
d: 64
gcn_hidden: 64
dropout: 0.5
lr: 0.001
wd: 0.0001
K: 3
epochs: 300
cl_loss: reconstruction
cl_temp: 0.1
cv_folds: 5
test_frac: 0.20
n_iter: 50
n_seeds: 10
```

```yaml
# configs/dblp_rec.yaml
dataset: dblp
task: rec
d: 64
gcn_hidden: 64
dropout: 0.5
lr: 0.001
wd: 0.0001
K: 3
epochs: 300
K_rec: 20        # top-K for Recall@K during CV selection
neg_ratio: 5     # BPR negatives per positive
bpr_reg: 0.0001
cv_folds: 5
test_frac: 0.20
n_iter: 50
n_seeds: 10
eval_K: [10, 20, 50]   # K values reported in final table
```

*(Repeat analogously for acm and imdb.)*

---

## Step 14 — Run scripts

```bash
# scripts/run_cl.sh
#!/usr/bin/env bash
set -e
for DATASET in dblp acm imdb; do
    echo "=============================="
    echo "Graph Clustering: $DATASET"
    echo "=============================="
    python -m src.train --dataset $DATASET --task cl --seeds 10
done
echo "Clustering complete."
```

```bash
# scripts/run_rec.sh
#!/usr/bin/env bash
set -e
for DATASET in dblp acm imdb; do
    echo "=============================="
    echo "Recommendation: $DATASET"
    echo "=============================="
    python -m src.train --dataset $DATASET --task rec --seeds 10
done
echo "Recommendation complete."
```

---

## Evaluation protocol — what goes where

```
80 % TRAIN+VAL POOL
│
└── 5-fold CV × 50 random hyperparameter combos
    │
    Saved to:  results/{task}/cv_fold_scores.csv
    Columns:   combo_id, fold, val_metric, hp_d, hp_K, hp_dropout, hp_lr,
               hp_wd, hp_gcn_hidden, hp_epochs, [task-specific hp cols]
    │
    Best params saved to:  results/{task}/best_params.json

20 % HELD-OUT TEST
│
└── Final training on 80% × 10 seeds
    │
    Per-run CSV:   results/{task}/per_run_results.csv
    Columns:
      NC  → dataset, task, seed, test_macro_f1, test_micro_f1,
             test_accuracy, time_sec, hp_*
      LP  → dataset, task, seed, test_auc, test_ap, time_sec, hp_*
      CL  → dataset, task, seed, nmi, ari, acc, time_sec, hp_*
      REC → dataset, task, seed, recall@10, recall@20, recall@50,
             ndcg@10, ndcg@20, ndcg@50, hit@10, hit@20, hit@50,
             precision@10, precision@20, precision@50,
             mrr@10, mrr@20, mrr@50, time_sec, hp_*
    │
    Epoch log (per run):  results/{task}/epoch_logs/seed{N}_epochs.csv
    Columns:   epoch, train_loss, [val_metric if checked at that epoch]
    │
    Summary:   results/{task}/summary.csv
    Columns:
      NC  → macro_f1_mean, macro_f1_sd, micro_f1_mean, micro_f1_sd,
             accuracy_mean, accuracy_sd, n_seeds, best_hp_*
      LP  → auc_mean, auc_sd, ap_mean, ap_sd, n_seeds, best_hp_*
      CL  → nmi_mean, nmi_sd, ari_mean, ari_sd, acc_mean, acc_sd,
             n_seeds, best_hp_*
      REC → recall@K_mean, recall@K_sd, ndcg@K_mean, ndcg@K_sd,
             hit@K_mean, hit@K_sd, precision@K_mean, precision@K_sd,
             mrr@K_mean, mrr@K_sd (for K ∈ {10,20,50}), n_seeds, best_hp_*
```

---

## Metrics — full list by task

| Task | Metric | Description |
|------|--------|-------------|
| NC | Macro-F1 ± SD | Primary; reported for each class equally weighted |
| NC | Micro-F1 ± SD | Instance-weighted F1 |
| NC | Accuracy ± SD | Overall fraction correct |
| LP | AUC-ROC ± SD | Area under ROC curve |
| LP | AP ± SD | Average Precision (area under PR curve) |
| CL | NMI ± SD | Normalized Mutual Information |
| CL | ARI ± SD | Adjusted Rand Index (chance-corrected) |
| CL | ACC ± SD | Clustering accuracy via Hungarian matching |
| REC | Recall@K ± SD | Fraction of true positives in top-K |
| REC | NDCG@K ± SD | Normalized Discounted Cumulative Gain |
| REC | Hit Rate@K ± SD | 1 if any true positive in top-K |
| REC | Precision@K ± SD | True positives / K |
| REC | MRR ± SD | Mean Reciprocal Rank of first hit |

All reported as **mean ± SD over 10 random seeds** on the 20 % held-out test set.

---

## Common errors — v4 additions

**`KMeans: n_samples=X < n_clusters=Y`**
→ Val fold in clustering is too small. Either increase `TEST_FRAC` or ensure `Nt` is large enough. For IMDB (4278 movies), 20 % = 855 nodes — fine. ACM (3025) is also fine.

**`linear_sum_assignment` dimension mismatch**
→ K-Means predicted label range ≠ n_classes. Ensure `KMeans(n_clusters=n_cl)` and `clustering_accuracy(y_true, y_pred)` both reference `data['n_classes']`.

**BPR loss = NaN after epoch 1**
→ Embeddings collapsed. Reduce `lr` to `0.001` or add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.

**Recall@K = 0.0 throughout training**
→ `user_pos` is not excluding training items correctly during ranking. Check `sample_bpr_negatives` uses the same `user_pos` dict as `compute_rec_metrics`.

**`IndexError: index N out of bounds for axis 0 with size N`**
→ Item node indices in `target_edges` include global node IDs (e.g. for DBLP, paper IDs start at Na=4057, not 0). Pass `emb_np[all_items]` using the actual global indices, not local.

**Recommendation on DBLP: no meaningful user-item structure**
→ DBLP/ACM/IMDB are not native recommendation datasets. Results will be lower than on Amazon/LastFM. Use these only as structural benchmarks — add Amazon or LastFM for fair recommendation comparison.

