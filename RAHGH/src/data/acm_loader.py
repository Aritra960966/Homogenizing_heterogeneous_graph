import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


AREA_CONFS = {
    'Database'      : ['SIGMOD','VLDB','ICDE','PODS'],
    'Wireless_Comm' : ['MobiCOMM','SIGCOMM','INFOCOM','ICNP'],
    'Data_Mining'   : ['KDD','WWW','ICDM','SIGIR','CIKM','WSDM'],
}


def load_acm(root="data/raw/ACM"):
    mat = scipy.io.loadmat(f"{root}/ACM.mat")

    PvsA  = mat['PvsA'].astype(np.float32).tocsr()
    PvsC  = mat['PvsC'].astype(np.float32).tocsr()
    nPvsT = mat['nPvsT'].astype(np.float32).tocsr()
    PvsP  = mat['PvsP'].astype(np.float32).tocsr()

    # ── labels from conference areas ──
    C_names = [str(mat['C'][i,0].flat[0]) for i in range(mat['C'].shape[0])]
    conf_to_class = {}
    for label, (area, confs) in enumerate(AREA_CONFS.items()):
        for c in confs:
            if c in C_names:
                conf_to_class[c] = label

    PvsC_dense    = np.asarray(PvsC.todense())
    paper_conf_id = PvsC_dense.argmax(axis=1).flatten()
    paper_ids, labels_np = [], []
    for pid in range(len(paper_conf_id)):
        cname = C_names[paper_conf_id[pid]]
        if cname in conf_to_class:
            paper_ids.append(pid)
            labels_np.append(conf_to_class[cname])

    paper_ids = np.array(paper_ids)
    labels_np = np.array(labels_np)
    Np = len(paper_ids)

    # ── filter matrices to labeled papers ──
    PvsA_s = PvsA[paper_ids]
    PvsP_s = PvsP[paper_ids][:, paper_ids]

    Na = PvsA_s.shape[1]
    N  = Np + Na

    # ── adjacency: only paper↔author + paper↔paper ──
    def embed_bip(A_sp, r_off, c_off):
        A = A_sp.tocoo().astype(np.float32)
        return sp.coo_matrix(
            (A.data, (A.row + r_off, A.col + c_off)),
            shape=(N, N)).tocsr()

    def embed_hom(A_sp, off):
        A = A_sp.tocoo().astype(np.float32)
        return sp.coo_matrix(
            (A.data, (A.row + off, A.col + off)),
            shape=(N, N)).tocsr()

    PA = embed_bip(PvsA_s, 0, Np)
    AP = PA.T.tocsr()
    PP = embed_hom(PvsP_s, 0)

    A_list_sp     = [PA, AP, PP]
    relation_names = ['paper→author', 'author→paper', 'paper→paper']
    bipartite_flags = [True, True, False]

    # ── features: paper uses term TF-IDF, author uses SVD of PvsA ──
    X_paper  = torch.tensor(np.array(nPvsT[paper_ids].todense(), dtype=np.float32))
    U, S, _  = spla.svds(PvsA_s.T.astype(np.float64), k=64)
    X_author = torch.tensor((U * S).astype(np.float32))

    X_dict = {'paper': X_paper, 'author': X_author}

    return dict(
        A_list_sp=A_list_sp,
        relation_names=relation_names,
        bipartite_flags=bipartite_flags,
        X_dict=X_dict,
        labels=torch.tensor(labels_np, dtype=torch.long),
        Np=Np, Na=Na, N=N,
        target_type='paper', target_size=Np,
        n_classes=3,
    )
