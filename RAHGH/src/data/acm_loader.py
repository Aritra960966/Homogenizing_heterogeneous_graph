import numpy as np
import scipy.io
import scipy.sparse as sp
import torch
from pathlib import Path


def load_acm(root: str = "data/raw/ACM") -> dict:
    """
    Load ACM dataset from ACM.mat following Data_Preprocessing.ipynb.

    3 node types: paper, author, subject (56)
    4 link types: paper↔author, paper↔subject
    Features: 1902-dim boolean paper-term (aggregated for author/subject)
    Labels: 3 classes (0=database, 1=wireless, 2=datamining)

    Predefined split from label.dat.train / label.dat.test.
    """
    folder = Path(root)
    mat_path = folder / 'ACM .mat'
    if not mat_path.exists():
        mat_path = folder / 'ACM.mat'
    mat = scipy.io.loadmat(str(mat_path))

    # ── 1. Select papers across 3 research areas ─────────────────────────────
    # PvsC: 12499 papers × 14 conferences; nonzero()[1] gives conf id per paper
    # Database: SIGMOD(1), VLDB(13); Wireless: SIGCOMM(9), MobiCOMM(10); Data Mining: KDD(0)
    paper_conf = mat['PvsC'].nonzero()[1]
    db_mask = np.isin(paper_conf, [1, 13])
    dm_mask = paper_conf == 0
    wc_mask = np.isin(paper_conf, [9, 10])

    db_idx = np.where(db_mask)[0]
    dm_idx = np.where(dm_mask)[0]
    wc_idx = np.where(wc_mask)[0]

    rng = np.random.RandomState(42)
    db_sel = np.sort(rng.choice(db_idx, min(994, len(db_idx)), replace=False))
    paper_idx = np.sort(np.concatenate([db_sel, dm_idx, wc_idx]))
    nP = len(paper_idx)

    paper_global = np.arange(nP)

    # Labels: 0=database, 1=wireless, 2=datamining
    paper_target = np.zeros(nP, dtype=np.int64)
    paper_target[np.isin(paper_idx, wc_idx)] = 1
    paper_target[np.isin(paper_idx, dm_idx)] = 2

    # ── 2. Build node layout ─────────────────────────────────────────────────
    # Authors connected to selected papers
    authors = mat['PvsA'][paper_idx].nonzero()[1]
    unique_authors, author_inv = np.unique(authors, return_inverse=True)
    nA = len(unique_authors)

    # Subjects from PvsL (ACL classification, 56 topics)
    subjects = mat['PvsL'][paper_idx].nonzero()[1]
    unique_subjects, subject_inv = np.unique(subjects, return_inverse=True)
    nS = len(unique_subjects)

    N = nP + nA + nS

    # ── 3. Build adjacency matrices ──────────────────────────────────────────
    def _coo(rows, cols):
        return sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(N, N)
        ).tocsr()

    # Paper→Author (from PvsA)
    pa_rows = mat['PvsA'][paper_idx].nonzero()[0]
    pa_cols = author_inv + nP
    PA = _coo(pa_rows, pa_cols)
    AP = PA.T.tocsr()

    # Paper→Subject (from PvsL)
    ps_rows = mat['PvsL'][paper_idx].nonzero()[0]
    ps_cols = subject_inv + nP + nA
    PS = _coo(ps_rows, ps_cols)
    SP = PS.T.tocsr()

    A_list_sp = [PA, AP, PS, SP]
    bipartite_flags = [True, True, True, True]
    relation_names = ['paper-author', 'author-paper', 'paper-subject', 'subject-paper']
    relation_info = {
        'paper-author':    ('paper', 'author'),
        'author-paper':    ('author', 'paper'),
        'paper-subject':   ('paper', 'subject'),
        'subject-paper':   ('subject', 'paper'),
    }

    edge_index_dict = {}
    for rname, A in zip(relation_names, A_list_sp):
        A_coo = A.tocoo()
        ei = np.vstack([A_coo.row, A_coo.col])
        edge_index_dict[rname] = torch.tensor(ei, dtype=torch.long)

    # ── 4. Build features ────────────────────────────────────────────────────
    # Paper: boolean(paper × term > 0) from PvsT (12499×1903 → 3025×1902 after term filtering)
    # Author: aggregated from paper features via paper-author adjacency
    # Subject: aggregated from paper features via paper-subject adjacency

    # Build paper-feature adjacency: paper → term
    PvsT = mat['PvsT'][paper_idx]  # 3025 × 1903
    paper_terms = PvsT.nonzero()[1]
    unique_terms = np.unique(paper_terms)
    term_map = {t: i for i, t in enumerate(unique_terms)}
    nF = len(unique_terms)  # 1902

    # Paper features: boolean matrix
    pt_rows = PvsT.nonzero()[0]
    pt_cols = np.array([term_map[t] for t in paper_terms])
    P_bool = sp.coo_matrix(
        (np.ones(len(pt_rows), dtype=np.float32), (pt_rows, pt_cols)),
        shape=(nP, nF)
    ).tocsr()

    # Author features: aggregated from paper features via paper-author
    # PA_local is (nP × nA) adjacency
    PA_local = sp.coo_matrix(
        (np.ones(len(pa_rows), dtype=np.float32), (pa_rows, pa_cols - nP)),
        shape=(nP, nA)
    ).tocsr()
    A_bool = (PA_local.T.dot(P_bool) > 0).astype(np.float32)

    # Subject features: aggregated from paper features via paper-subject
    PS_local = sp.coo_matrix(
        (np.ones(len(ps_rows), dtype=np.float32), (ps_rows, ps_cols - nP - nA)),
        shape=(nP, nS)
    ).tocsr()
    S_bool = (PS_local.T.dot(P_bool) > 0).astype(np.float32)

    X_paper = torch.tensor(P_bool.toarray())
    X_author = torch.tensor(A_bool.toarray())
    X_subject = torch.tensor(S_bool.toarray())

    X_dict = {'paper': X_paper, 'author': X_author, 'subject': X_subject}
    node_type_dims = {'paper': nF, 'author': nF, 'subject': nF}

    # Global ID tensors (paper=0..nP-1, author=nP..nP+nA-1, subject=nP+nA..N-1)
    node_type_indices = {
        'paper':   torch.arange(nP),
        'author':  torch.arange(nP, nP + nA),
        'subject': torch.arange(nP + nA, N),
    }

    # ── 5. Parse labels ──────────────────────────────────────────────────────
    # Format: paper_id \t "" \t type_id \t label  (4 fields, label at index 3)
    # label.dat.train: all 3025 paper labels
    # label.dat.test : 907 predefined test indices
    # Train set = all papers minus test set

    label_map = {}
    with open(folder / 'label.dat.train') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                label_map[int(parts[0])] = int(parts[3])

    test_ids = []
    with open(folder / 'label.dat.test') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                test_ids.append(int(parts[0]))

    test_set = set(test_ids)
    train_ids = sorted([pid for pid in label_map if pid not in test_set])
    test_ids.sort()

    unique_labels = sorted(set(label_map.values()))
    lbl_map = {v: i for i, v in enumerate(unique_labels)}

    labels_full = torch.full((nP,), -1, dtype=torch.long)
    for pid in train_ids:
        labels_full[pid] = lbl_map[label_map[pid]]
    for pid in test_ids:
        labels_full[pid] = lbl_map[label_map[pid]]

    train_indices = torch.tensor(train_ids, dtype=torch.long)
    test_indices  = torch.tensor(test_ids,  dtype=torch.long)

    return dict(
        A_list_sp=A_list_sp,
        bipartite_flags=bipartite_flags,
        relation_names=relation_names,
        relation_info=relation_info,
        edge_index_dict=edge_index_dict,
        X_dict=X_dict,
        node_type_indices=node_type_indices,
        node_type_dims=node_type_dims,
        labels=labels_full,
        labels_full=labels_full,
        train_indices=train_indices,
        test_indices=test_indices,
        N=N,
        target_type='paper',
        target_size=nP,
        n_classes=len(unique_labels),
    )
