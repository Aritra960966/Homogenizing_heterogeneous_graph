"""
ACM heterogeneous graph loader.
Follows the EXACT preprocessing pipeline from Data_Preprocessing.ipynb step by step.

Paper selection (PvsC — paper-conference matrix):
    conf [1, 13]  SIGMOD, VLDB            -> label 0  Database          (sampled to 994)
    conf [9, 10]  SIGCOMM, MobiCOMM       -> label 1  Wireless Comm     (all papers)
    conf [0]      KDD                     -> label 2  Data Mining        (all papers)

Node types   :  paper | author | subject
Global layout:  [0 ... N_p-1] papers  |  [N_p ... N_p+N_a-1] authors  |  [N_p+N_a ... N-1] subjects

Relations (4, all bipartite):
    pa  paper  -> author   A_pa
    ap  author -> paper    A_ap  (= A_pa.T)
    ps  paper  -> subject  A_ps
    sp  subject -> paper   A_sp  (= A_ps.T)

Features:
    Built from PvsT / TvsP (paper-term BOW), same vocabulary for all node types.
    Terms are NOT graph nodes — they are used only to construct features.
    paper_feat   = paper-term incidence                  shape (N_p, N_t)
    author_feat  = union of terms across author's papers shape (N_a, N_t)
    subject_feat = union of terms across subject's papers shape (N_s, N_t)
    All three are binary float32 and share the same feature dim N_t.
"""

import numpy as np
import torch
from scipy import io
from scipy.sparse import csr_matrix


# Conference-to-label mapping (from notebook)
_DB_CONFS  = [1, 13]   # SIGMOD=1, VLDB=13     -> label 0  (Database)
_WC_CONFS  = [9, 10]   # SIGCOMM=9, MobiCOMM=10 -> label 1  (Wireless Comm)
_DM_CONFS  = [0]       # KDD=0                  -> label 2  (Data Mining)
_DB_SAMPLE = 994


def load_acm(mat_path: str = "data/raw/ACM/ACM.mat", seed: int = 42) -> dict:
    """
    Load and preprocess the ACM heterogeneous graph from ACM.mat.

    Returns standard RAHGH data dict:
        X_dict             {node_type: (N_t, feat_dim) float32 Tensor}
        A_list_sp          list of 4 scipy CSR matrices, shape (N, N)
                           order: [A_pa, A_ap, A_ps, A_sp]
        bipartite_flags    [True, True, True, True]
        relation_names     ['pa', 'ap', 'ps', 'sp']
        relation_info      {rel_name: (src_type, dst_type)}
        labels             (N_p,) long Tensor  0=DB  1=WC  2=DM
        target_size        N_p  (int)
        target_type        'paper'
        n_classes          3
        N                  total nodes  (int)
        node_type_dims     {type_name: feat_dim}
    """
    rng = np.random.default_rng(seed)
    mat = io.loadmat(mat_path)

    # Step 1 — Select papers by conference
    paper_conf = mat['PvsC'].nonzero()[1]

    db_candidates = np.where(np.isin(paper_conf, _DB_CONFS))[0]
    wc_candidates = np.where(np.isin(paper_conf, _WC_CONFS))[0]
    dm_candidates = np.where(np.isin(paper_conf, _DM_CONFS))[0]

    db_idx = np.sort(rng.choice(db_candidates, _DB_SAMPLE, replace=False))
    wc_idx = wc_candidates
    dm_idx = dm_candidates

    paper_idx = np.sort(np.concatenate([db_idx, wc_idx, dm_idx]))
    N_p = len(paper_idx)

    # Step 2 — Build label array  (0=DB  1=WC  2=DM)
    db_set = set(db_idx.tolist())
    wc_set = set(wc_idx.tolist())
    labels_np = np.array(
        [0 if idx in db_set else (1 if idx in wc_set else 2) for idx in paper_idx],
        dtype=np.int64,
    )

    # Step 3 — Author re-indexing  (global offset = N_p)
    pa_local_rows, pa_orig_cols = mat['PvsA'][paper_idx].nonzero()

    author_map = {}
    re_authors = []
    for a in pa_orig_cols:
        if a not in author_map:
            author_map[a] = len(author_map) + N_p
        re_authors.append(author_map[a])
    re_authors = np.array(re_authors, dtype=np.int64)
    N_a = len(author_map)

    # Step 4 — Subject re-indexing  (global offset = N_p + N_a)
    ps_local_rows, ps_orig_cols = mat['PvsL'][paper_idx].nonzero()

    subject_map = {}
    re_subjects = []
    for s in ps_orig_cols:
        if s not in subject_map:
            subject_map[s] = len(subject_map) + N_p + N_a
        re_subjects.append(subject_map[s])
    re_subjects = np.array(re_subjects, dtype=np.int64)
    N_s = len(subject_map)

    N = N_p + N_a + N_s

    # Step 5 — Adjacency matrices  (all shape NxN, scipy CSR)
    A_pa = csr_matrix(
        (np.ones(len(pa_local_rows), dtype=np.float32),
         (pa_local_rows, re_authors)),
        shape=(N, N),
    )
    A_ap = A_pa.T.tocsr()

    A_ps = csr_matrix(
        (np.ones(len(ps_local_rows), dtype=np.float32),
         (ps_local_rows, re_subjects)),
        shape=(N, N),
    )
    A_sp = A_ps.T.tocsr()

    # Step 6 — Term dictionary for features
    if 'TvsP' in mat:
        PvsT = mat['TvsP'].T.tocsr()
    elif 'PvsT' in mat:
        PvsT = mat['PvsT'].tocsr()
    else:
        raise KeyError("ACM.mat must contain 'TvsP' or 'PvsT'")

    pt_local_rows, pt_orig_cols = PvsT[paper_idx].nonzero()

    term_map = {}
    re_terms = []
    for t in pt_orig_cols:
        if t not in term_map:
            term_map[t] = len(term_map) + N
        re_terms.append(term_map[t])
    re_terms = np.array(re_terms, dtype=np.int64)
    N_t = len(term_map)
    N_tmp = N + N_t

    # Step 7 — Build node features
    A_pt_tmp = csr_matrix(
        (np.ones(len(pt_local_rows), dtype=np.float32),
         (pt_local_rows, re_terms)),
        shape=(N_tmp, N_tmp),
    )

    A_pa_tmp = csr_matrix(
        (np.ones(len(pa_local_rows), dtype=np.float32),
         (pa_local_rows, re_authors)),
        shape=(N_tmp, N_tmp),
    )

    A_ps_tmp = csr_matrix(
        (np.ones(len(ps_local_rows), dtype=np.float32),
         (ps_local_rows, re_subjects)),
        shape=(N_tmp, N_tmp),
    )

    paper_feat   = (A_pt_tmp[:N_p, N:].toarray()                              > 0).astype(np.float32)
    author_feat  = (A_pa_tmp.T.dot(A_pt_tmp)[N_p:N_p + N_a, N:].toarray()    > 0).astype(np.float32)
    subject_feat = (A_ps_tmp.T.dot(A_pt_tmp)[N_p + N_a:N,   N:].toarray()    > 0).astype(np.float32)

    # Step 8 — Package into standard RAHGH data dict
    return {
        'X_dict': {
            'paper':   torch.from_numpy(paper_feat),
            'author':  torch.from_numpy(author_feat),
            'subject': torch.from_numpy(subject_feat),
        },
        'node_type_dims': {
            'paper':   N_t,
            'author':  N_t,
            'subject': N_t,
        },
        'A_list_sp':       [A_pa, A_ap, A_ps, A_sp],
        'bipartite_flags': [True, True, True, True],
        'relation_names':  ['pa', 'ap', 'ps', 'sp'],
        'relation_info': {
            'pa': ('paper', 'author'),
            'ap': ('author', 'paper'),
            'ps': ('paper', 'subject'),
            'sp': ('subject', 'paper'),
        },
        'labels':      torch.from_numpy(labels_np),
        'target_size': N_p,
        'target_type': 'paper',
        'n_classes':   3,
        'N':           N,
    }


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/ACM/ACM.mat'
    print(f'Loading {path} ...')
    data = load_acm(path)

    print(f"\nN total     : {data['N']}")
    print(f"Feature dim : {list(data['X_dict'].values())[0].shape[1]}")
    print(f"n_classes   : {data['n_classes']}")
    print(f"Relations   : {data['relation_names']}")
    print(f"Bipartite   : {data['bipartite_flags']}")

    print(f"\nFeature shapes:")
    for k, v in data['X_dict'].items():
        print(f"  {k:10s}  {tuple(v.shape)}  dtype={v.dtype}")

    print(f"\nAdjacency nnz:")
    for name, A in zip(data['relation_names'], data['A_list_sp']):
        print(f"  {name}  nnz={A.nnz:7d}  shape={A.shape}")

    lbl = data['labels'].numpy()
    counts = np.bincount(lbl, minlength=3)
    print(f"\nLabel distribution:")
    print(f"  0 Database          : {counts[0]}")
    print(f"  1 Wireless Comm     : {counts[1]}")
    print(f"  2 Data Mining       : {counts[2]}")
    print(f"  Total papers (N_p)  : {data['target_size']}")
