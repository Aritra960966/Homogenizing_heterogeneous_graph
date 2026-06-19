"""
src/analysis/homogenization_analysis.py

Homogenization preservation analysis for RAHGH.

Measures whether the unified homogeneous representation Z_final retains:
  1. Node semantic content  (CKA + cosine similarity: features → embeddings)
  2. Relation information   (connectivity score + per-relation CKA)
  3. Structural properties  (spectral similarity + Laplacian preservation)
  4. Neighbourhood topology  (k-NN overlap across diffusion depths)

Usage:
    from src.analysis.homogenization_analysis import run_analysis, print_report, plot_report, latex_table
    report = run_analysis(model, data, device, n_sample=2000)
    print_report(report)
    plot_report(report, save_path='results/preservation.pdf')
    print(latex_table(report))
"""

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import eigsh, norm as spnorm
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

import torch


# ─────────────────────────────────────────────────────────────────────────────
#  1. Representation Extractor
# ─────────────────────────────────────────────────────────────────────────────

class RepresentationExtractor:
    """
    Captures RAHGH's intermediate representations via non-invasive forward hooks.

    Captured tensors (all on CPU, numpy):
        H0       : (N, d)           TypeSpecificProjection output — features in
                                    shared latent space, before any graph propagation
        Z_dict   : {rel: (N, d)}    RelationPolynomialDiffusion output — one
                                    embedding per relation, before fusion
        Z_fused  : (N, d)           AdaptiveRelationFusion output — weighted sum
                                    of relation embeddings, before residual MLP
        Z_final  : (N, d)           ResidualMLP output — the final homogeneous
                                    embedding
        alpha    : (R,)             Relation fusion weights from softmax(theta)
    """

    def __init__(self):
        self._hooks: List = []
        self._captured: Dict = {}

    def attach(self, model: torch.nn.Module) -> "RepresentationExtractor":
        """
        Attach hooks to a RAHGH or RAHGHClassifier instance.

        Handles both:
            model = RAHGH(...)                  → hooks on model directly
            model = RAHGHClassifier(...)        → hooks on model.homogenizer
        """
        from src.model.rahgh import RAHGH, RAHGHClassifier

        if isinstance(model, RAHGHClassifier):
            rahgh = model.homogenizer
        elif isinstance(model, RAHGH):
            rahgh = model
        else:
            for name, mod in model.named_modules():
                if isinstance(mod, RAHGH):
                    rahgh = mod
                    break
            else:
                raise TypeError(f"Cannot find RAHGH backbone in {type(model)}")

        def _hook(name):
            def fn(module, inp, out):
                self._captured[name] = out
            return fn

        self._hooks = [
            rahgh.projector.register_forward_hook(_hook('H0')),
            rahgh.diffuser.register_forward_hook(_hook('Z_dict')),
            rahgh.fusioner.register_forward_hook(_hook('Z_alpha')),
            rahgh.residual.register_forward_hook(_hook('Z_final')),
        ]
        return self

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def extract(
        self,
        model: torch.nn.Module,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        node_type_indices: Dict[str, torch.Tensor],
    ) -> Dict:
        """
        Run one forward pass and return all captured representations.

        Returns dict with keys:
            H0, Z_dict, Z_fused, Z_final (np.ndarray)
            alpha                         (np.ndarray)
        """
        model.eval()
        self._captured.clear()
        _ = model(x_dict, edge_index_dict, node_type_indices)

        H0_t     = self._captured.get('H0')
        Z_dict_t = self._captured.get('Z_dict')
        Z_a_t    = self._captured.get('Z_alpha')
        Z_fin_t  = self._captured.get('Z_final')

        if isinstance(Z_a_t, tuple):
            Z_fused_t, alpha_t = Z_a_t
        else:
            Z_fused_t, alpha_t = Z_a_t, None

        if isinstance(H0_t, tuple):
            H0_t = H0_t[0]

        def _np(t):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().float().numpy()
            return t

        Z_dict_np = (
            {r: _np(v) for r, v in Z_dict_t.items()}
            if isinstance(Z_dict_t, dict) else {}
        )

        return {
            'H0'     : _np(H0_t),
            'Z_dict' : Z_dict_np,
            'Z_fused': _np(Z_fused_t),
            'Z_final': _np(Z_fin_t),
            'alpha'  : _np(alpha_t),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  2. Structural Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _sym_normalized_laplacian(A: csr_matrix) -> csr_matrix:
    """Symmetric normalised Laplacian: L = I - D^{-1/2} A D^{-1/2}."""
    n = A.shape[0]
    d = np.asarray(A.sum(axis=1)).flatten().astype(float)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = csr_matrix(
        (d_inv_sqrt, (range(n), range(n))), shape=(n, n), dtype=np.float32,
    )
    return speye(n, format='csr') - D_inv_sqrt @ A @ D_inv_sqrt


def _safe_eigsh(
    L: csr_matrix,
    k: int,
    rng_seed: int = 0,
) -> Optional[np.ndarray]:
    """
    Robust eigenvalue computation for normalised Laplacians.
    Tries multiple strategies to handle pathological spectra:
      1. which='SM' with seeded v0
      2. shift-invert mode (sigma=0.1) for graphs with many zero eigenvalues
      3. Dense eigh fallback for small matrices
    """
    from scipy.sparse.linalg import eigsh

    n = L.shape[0]
    k = min(k, n - 2)
    if k < 1:
        return None

    rng = np.random.default_rng(rng_seed)
    v0 = rng.uniform(-1, 1, n).astype(np.float32)

    # Strategy 1: standard SM with v0
    try:
        evals = eigsh(L, k=k, which='SM', return_eigenvectors=False, v0=v0)
        return np.sort(np.abs(evals))
    except Exception:
        pass

    # Strategy 2: shift-invert mode (finds eigenvalues near sigma)
    try:
        evals = eigsh(L, k=k, sigma=0.1, which='LM', return_eigenvectors=False, v0=v0)
        evals = np.sort(np.abs(evals))
        # Shift-invert gives eigenvalues near 0.1; shift them back
        evals = np.abs(0.1 - 1.0 / evals) if evals[0] > 0 else evals
        return evals
    except Exception:
        pass

    # Strategy 3: dense eigh (only for small matrices)
    if n <= 5000:
        try:
            import scipy.linalg as la
            L_dense = L.toarray()
            evals = la.eigh(L_dense, check_finite=False)[0][:k]
            return np.sort(np.abs(evals))
        except Exception:
            pass

    return None


def spectral_similarity(
    A1: csr_matrix,
    A2: csr_matrix,
    k: int = 30,
) -> float:
    """
    Spectral similarity between two graphs.

    Computes the k smallest eigenvalues of each graph's normalised Laplacian
    and returns 1 - ||λ1 - λ2|| / ||λ1||.

    Range: ideally [0, 1], 1.0 = identical spectra.
    When comparing graphs with very different edge sets (e.g. individual
    sparse relation vs dense aggregated adjacency), values can dip below 0
    or be numerically unstable. The result is clamped to [0, 1] for
    interpretability; negative values indicate fundamentally different
    spectral profiles.
    """
    n = A1.shape[0]
    k = min(k, n - 2)
    if k < 1:
        return float('nan')
    try:
        L1 = _sym_normalized_laplacian(A1)
        L2 = _sym_normalized_laplacian(A2)
        λ1 = _safe_eigsh(L1, k)
        λ2 = _safe_eigsh(L2, k)
        if λ1 is None or λ2 is None:
            return float('nan')
        norm1 = np.linalg.norm(λ1)
        if norm1 < 1e-10:
            return 1.0
        result = 1.0 - np.linalg.norm(λ1 - λ2) / norm1
        return float(np.clip(result, 0.0, 1.0))
    except Exception as e:
        warnings.warn(f"spectral_similarity failed: {e}")
        return float('nan')


def laplacian_preservation(
    A_homo: csr_matrix,
    A_list: List[csr_matrix],
    alpha: np.ndarray,
) -> float:
    """
    Laplacian preservation score.

    Measures how well L(A_homo) approximates the alpha-weighted sum of
    individual relation Laplacians.
    """
    L_homo = _sym_normalized_laplacian(A_homo)
    L_target = sum(float(alpha[i]) * _sym_normalized_laplacian(A)
                   for i, A in enumerate(A_list))
    diff  = L_homo - L_target
    score = 1.0 - float(spnorm(diff, 'fro')) / (float(spnorm(L_target, 'fro')) + 1e-10)
    return score


def degree_distribution_similarity(
    A1: csr_matrix,
    A2: csr_matrix,
) -> float:
    """
    Jensen-Shannon divergence between degree distributions.
    score = 1 - JSD / log(2).  [0, 1], 1.0 = identical.
    """
    def _deg_dist(A):
        deg = np.asarray(A.sum(1)).flatten().astype(float)
        max_d = max(int(deg.max()), 1)
        hist, _ = np.histogram(deg, bins=max_d + 1, range=(0, max_d + 1), density=True)
        hist += 1e-10
        return hist / hist.sum()

    p = _deg_dist(A1)
    q = _deg_dist(A2)
    max_len = max(len(p), len(q))
    p = np.pad(p, (0, max_len - len(p))) + 1e-10
    q = np.pad(q, (0, max_len - len(q))) + 1e-10
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    return float(1.0 - jsd / np.log(2))


def build_combined_adjacency(A_list: List[csr_matrix], alpha: np.ndarray) -> csr_matrix:
    """
    Build the weighted sum of relation adjacencies: Σ_r α_r A_r.
    """
    A_ref = sum(float(alpha[i]) * A.astype(np.float32) for i, A in enumerate(A_list))
    A_ref = csr_matrix(A_ref)
    A_ref.eliminate_zeros()
    return A_ref


# ─────────────────────────────────────────────────────────────────────────────
#  3. Embedding Metrics
# ─────────────────────────────────────────────────────────────────────────────

def cosine_preservation(
    X: np.ndarray,
    Z: np.ndarray,
    node_idx: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Per-node cosine similarity between source (X) and target (Z) representations.

    Returns dict with mean, median, std, pct_positive.
    """
    if node_idx is not None:
        X, Z = X[node_idx], Z[node_idx]

    if X.shape[1] != Z.shape[1]:
        svd = TruncatedSVD(n_components=min(Z.shape[1], X.shape[1]), random_state=0)
        X   = svd.fit_transform(X)

    X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Z_n = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-10)
    cos = (X_n * Z_n).sum(axis=1)

    return {
        'mean'        : float(np.mean(cos)),
        'median'      : float(np.median(cos)),
        'std'         : float(np.std(cos)),
        'pct_positive': float((cos > 0).mean()),
    }


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment (CKA).

    Range: [0, 1].  1.0 = same linear subspace.  0.0 = orthogonal subspaces.

    Reference: Kornblith et al. (2019) ICML.
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XTY  = X.T @ Y
    XTX  = X.T @ X
    YTY  = Y.T @ Y
    num  = float((XTY * XTY).sum())
    den  = float(np.sqrt((XTX * XTX).sum() * (YTY * YTY).sum()))
    return num / (den + 1e-10)


def neighborhood_preservation(
    Z1: np.ndarray,
    Z2: np.ndarray,
    k_list: Tuple[int, ...] = (5, 10, 20),
    n_sample: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    k-Nearest Neighbour overlap between two embedding spaces.

    Returns {f'k{k}': score}.
    """
    rng  = np.random.default_rng(seed)
    n    = min(n_sample, len(Z1))
    if n < max(k_list) + 1:
        return {f'k{k}': float('nan') for k in k_list}
    idx  = rng.choice(len(Z1), size=n, replace=False)
    Z1s  = Z1[idx].astype(np.float32)
    Z2s  = Z2[idx].astype(np.float32)

    results = {}
    for k in k_list:
        knn1 = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(Z1s)
        knn2 = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(Z2s)
        nn1  = knn1.kneighbors(Z1s)[1][:, 1:]
        nn2  = knn2.kneighbors(Z2s)[1][:, 1:]
        overlaps = [
            len(set(nn1[i].tolist()) & set(nn2[i].tolist())) / k
            for i in range(n)
        ]
        results[f'k{k}'] = float(np.mean(overlaps))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  4. Relation Information Metrics
# ─────────────────────────────────────────────────────────────────────────────

def relation_connectivity_score(
    Z_final: np.ndarray,
    A_list: List[csr_matrix],
    relation_names: List[str],
    n_sample: int = 5000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Relation connectivity preservation score.

    Returns {rel: {'connected_sim': float, 'random_sim': float, 'lift': float}}.
    """
    rng = np.random.default_rng(seed)
    N   = Z_final.shape[0]
    Z_n = Z_final / (np.linalg.norm(Z_final, axis=1, keepdims=True) + 1e-10)

    results = {}
    for rname, A in zip(relation_names, A_list):
        coo   = A.tocoo()
        src   = coo.row.astype(np.int64)
        dst   = coo.col.astype(np.int64)
        if len(src) == 0:
            results[rname] = {'connected_sim': float('nan'),
                              'random_sim': float('nan'), 'lift': float('nan')}
            continue

        m = min(n_sample, len(src))
        sel  = rng.choice(len(src), size=m, replace=False)
        s_e, d_e = src[sel], dst[sel]
        conn_sim = float(np.mean((Z_n[s_e] * Z_n[d_e]).sum(axis=1)))

        r_s = rng.integers(0, N, size=m)
        r_d = rng.integers(0, N, size=m)
        rand_sim = float(np.mean((Z_n[r_s] * Z_n[r_d]).sum(axis=1)))

        results[rname] = {
            'connected_sim': conn_sim,
            'random_sim'   : rand_sim,
            'lift'         : conn_sim - rand_sim,
        }
    return results


def relation_cka_matrix(
    Z_dict: Dict[str, np.ndarray],
    Z_final: np.ndarray,
    n_sample: int = 3000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    CKA between each per-relation embedding Z_r and Z_final.
    Also computes pairwise CKA between relation views.

    Returns {'cka_to_final': {rel: float}, 'pairwise_cka': {rel_i vs rel_j: float}}.
    """
    rng   = np.random.default_rng(seed)
    n     = min(n_sample, Z_final.shape[0])
    idx   = rng.choice(Z_final.shape[0], n, replace=False)

    rel_keys = list(Z_dict.keys())
    Zf       = Z_final[idx]

    cka_to_final = {}
    for r, Zr in Z_dict.items():
        cka_to_final[r] = linear_cka(Zr[idx], Zf)

    pairwise_cka = {}
    for i, ri in enumerate(rel_keys):
        for j, rj in enumerate(rel_keys):
            if j <= i:
                continue
            key = f'{ri}_vs_{rj}'
            pairwise_cka[key] = linear_cka(Z_dict[ri][idx], Z_dict[rj][idx])

    return {'cka_to_final': cka_to_final, 'pairwise_cka': pairwise_cka}


# ─────────────────────────────────────────────────────────────────────────────
#  4.5 Feature Reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def feature_reconstruction_score(
    X: np.ndarray,
    Z: np.ndarray,
) -> Dict[str, float]:
    """
    Recover original features from Z_final using Ridge regression.

    Returns MSE, MAE, R2, and correlation between original and reconstructed.
    """
    reg = Ridge(alpha=1.0)
    reg.fit(Z, X)
    X_hat = reg.predict(Z)

    mse = float(np.mean((X - X_hat) ** 2))
    mae = float(np.mean(np.abs(X - X_hat)))
    r2 = float(r2_score(X, X_hat))
    corr = float(np.corrcoef(X.flatten(), X_hat.flatten())[0, 1])

    return {"mse": mse, "mae": mae, "r2": r2, "corr": corr}


# ─────────────────────────────────────────────────────────────────────────────
#  4.6 Embedding Statistics (collapse detection)
# ─────────────────────────────────────────────────────────────────────────────

def embedding_statistics(Z: np.ndarray) -> Dict[str, float]:
    """
    Compute embedding quality metrics useful for detecting collapse.

    Returns rank, effective rank, participation ratio, mean_norm, std_norm.
    """
    s = np.linalg.svd(Z, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    effective_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    participation_ratio = float((s.sum() ** 2) / ((s ** 2).sum() + 1e-12))

    norms = np.linalg.norm(Z, axis=1)

    return {
        "rank": int(np.linalg.matrix_rank(Z)),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  4.7 Relation Recoverability
# ─────────────────────────────────────────────────────────────────────────────

def relation_recoverability(
    Z: np.ndarray,
    A_list: List[csr_matrix],
    relation_names: List[str],
    n_pairs: int = 5000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Train a logistic regression classifier on [z_i || z_j] to predict relation type.

    Returns accuracy, macro F1.
    """
    rng = np.random.default_rng(seed)
    X_pairs, y = [], []

    for rel_id, (rname, A) in enumerate(zip(relation_names, A_list)):
        coo = A.tocoo()
        m = min(n_pairs, len(coo.row))
        if m == 0:
            continue
        idx = rng.choice(len(coo.row), m, replace=False)
        src = coo.row[idx]
        dst = coo.col[idx]
        feats = np.concatenate([Z[src], Z[dst]], axis=1)
        X_pairs.append(feats)
        y.extend([rel_id] * len(feats))

    if len(X_pairs) < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}

    X_pairs = np.vstack(X_pairs)
    y = np.array(y)

    Xtr, Xte, ytr, yte = train_test_split(
        X_pairs, y, test_size=0.3, random_state=seed, stratify=y,
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  4.8 Compression Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compression_metrics(
    A_list: List[csr_matrix],
    A_homo: csr_matrix,
) -> Dict[str, float]:
    """
    Measure how much the heterogeneous graph is compressed into A_homo.
    """
    hetero_edges = sum(A.nnz for A in A_list)
    homo_edges = A_homo.nnz
    return {
        "hetero_edges": int(hetero_edges),
        "homo_edges": int(homo_edges),
        "compression_ratio": hetero_edges / max(homo_edges, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  4.9 Downstream Preservation
# ─────────────────────────────────────────────────────────────────────────────

def downstream_preservation(
    X_orig: np.ndarray,
    H0: np.ndarray,
    Z_final: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Train logistic regression classifiers on each representation and compare F1.

    Returns {repr_name: macro_f1} and DPS = F1_Zfinal / F1_Xorig.
    """
    labeled_idx = np.where(labels >= 0)[0]
    if len(labeled_idx) == 0:
        return {"X_orig": float("nan"), "H0": float("nan"),
                "Z_final": float("nan"), "DPS": float("nan")}

    y = labels[labeled_idx]
    results = {}
    for name, Z in [("X_orig", X_orig), ("H0", H0), ("Z_final", Z_final)]:
        Z_lbl = Z[labeled_idx]
        Xtr, Xte, ytr, yte = train_test_split(
            Z_lbl, y, test_size=0.3, random_state=seed, stratify=y,
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        results[name] = float(f1_score(yte, pred, average="macro"))

    results["DPS"] = results["Z_final"] / max(results["X_orig"], 1e-10)
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  4.10 Spectral Embedding Preservation (eigenvector similarity)
# ─────────────────────────────────────────────────────────────────────────────

def eigenvector_similarity(
    A1: csr_matrix,
    A2: csr_matrix,
    k: int = 30,
) -> Dict[str, float]:
    """
    Compare top-k eigenvectors of two adjacency matrices.

    Returns CKA and cosine similarity between eigenvector matrices.
    """
    n = A1.shape[0]
    k = min(k, n - 2, A2.shape[0] - 2)
    if k < 1:
        return {"eigenvector_cka": float("nan"), "eigenvector_cosine": float("nan")}

    try:
        L1 = _sym_normalized_laplacian(A1)
        L2 = _sym_normalized_laplacian(A2)
        _, V1 = eigsh(L1, k=k, which="SM")
        _, V2 = eigsh(L2, k=k, which="SM")

        cka = linear_cka(V1, V2)

        V1_f = V1 / (np.linalg.norm(V1, axis=0, keepdims=True) + 1e-10)
        V2_f = V2 / (np.linalg.norm(V2, axis=0, keepdims=True) + 1e-10)
        cos_sim = float(np.mean(np.abs((V1_f * V2_f).sum(axis=0))))

        return {"eigenvector_cka": float(cka), "eigenvector_cosine": cos_sim}
    except Exception as e:
        warnings.warn(f"eigenvector_similarity failed: {e}")
        return {"eigenvector_cka": float("nan"), "eigenvector_cosine": float("nan")}


# ─────────────────────────────────────────────────────────────────────────────
#  4.11 Robustness Study
# ─────────────────────────────────────────────────────────────────────────────

def robustness_study(
    model: torch.nn.Module,
    data: dict,
    device: torch.device,
    noise_levels: List[float] = None,
    n_sample: int = 1000,
    n_spectral: int = 30,
    seed: int = 0,
) -> List[Dict]:
    """
    Inject Gaussian noise into node features and measure preservation metrics.

    Returns list of dicts: [{noise, cka, knn, spectral, laplacian}, ...].
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    from src.model.rahgh import build_edge_index_dict, build_node_type_indices

    ttype = data.get("target_type", list(data["X_dict"].keys())[0])
    Nt = data["target_size"]
    rng = np.random.default_rng(seed)
    sample = rng.choice(Nt, min(n_sample, Nt), replace=False)

    # Base reference (noiseless)
    x_dict_clean = {k: v.to(device) for k, v in data["X_dict"].items()}
    edge_index_dict = build_edge_index_dict(data, device)
    node_type_indices = {k: v.to(device) for k, v in build_node_type_indices(data).items()}
    extractor = RepresentationExtractor().attach(model)
    reps_clean = extractor.extract(model, x_dict_clean, edge_index_dict, node_type_indices)
    extractor.detach()
    Z_clean = reps_clean["Z_final"][:Nt]

    A_list = data["A_list_sp"]
    alpha = reps_clean["alpha"]
    if alpha is None:
        alpha = np.ones(len(A_list)) / len(A_list)
    A_ref = build_combined_adjacency(A_list, alpha)

    results = []
    for noise_lvl in noise_levels:
        x_dict_noisy = {}
        for k, v in data["X_dict"].items():
            noise = torch.randn_like(v) * noise_lvl * v.std()
            x_dict_noisy[k] = (v + noise).to(device)

        extractor = RepresentationExtractor().attach(model)
        reps_noisy = extractor.extract(model, x_dict_noisy, edge_index_dict, node_type_indices)
        extractor.detach()

        Z_noisy = reps_noisy["Z_final"][:Nt]

        cka_val = linear_cka(Z_clean[sample], Z_noisy[sample])
        knn_val = neighborhood_preservation(
            Z_clean, Z_noisy, k_list=(10,), n_sample=n_sample, seed=seed,
        ).get("k10", float("nan"))

        alpha_n = reps_noisy["alpha"]
        if alpha_n is None:
            alpha_n = alpha
        A_homo_n = _extract_A_homo(alpha_n, data)
        spec_val = spectral_similarity(A_homo_n, A_ref, k=n_spectral)
        lap_val = laplacian_preservation(A_homo_n, A_list, alpha_n)

        results.append({
            "noise": noise_lvl,
            "cka": cka_val,
            "knn": knn_val,
            "spectral": spec_val,
            "laplacian": lap_val,
        })

    return results



# ─────────────────────────────────────────────────────────────────────────────
#  4.13 Meta-path Preservation
# ─────────────────────────────────────────────────────────────────────────────

METAPATH_DEFS = {
    "acm": [
        ("PAP", [("paper", "author"), ("author", "paper")]),
        ("PSP", [("paper", "subject"), ("subject", "paper")]),
    ],
    "dblp": [
        ("APA", [("author", "paper"), ("paper", "author")]),
        ("APCPA", [("author", "paper"), ("paper", "conference"),
                   ("conference", "paper"), ("paper", "author")]),
        ("APTPA", [("author", "paper"), ("paper", "term"),
                   ("term", "paper"), ("paper", "author")]),
    ],
    "imdb": [
        ("MAM", [("movie", "actor"), ("actor", "movie")]),
        ("MDM", [("movie", "director"), ("director", "movie")]),
    ],
    "pubmed": [
        ("PAP", [("paper", "paper")]),
    ],
    "freebase": [],
    "lastfm": [
        ("UAU", [("user", "artist"), ("artist", "user")]),
    ],
    "amazon": [
        ("UIU", [("user", "item"), ("item", "user")]),
    ],
}


def _build_metapath_adjacency(
    mp_steps: List[Tuple[str, str]],
    A_list: List[csr_matrix],
    relation_info: Dict[str, Tuple[str, str]],
    relation_names: List[str],
    N: int,
) -> csr_matrix:
    """
    Build meta-path adjacency matrix by chaining relation matrices.

    mp_steps : list like [('paper','author'), ('author','paper')]
    Returns sparse (N, N) matrix for the target node type.
    """
    def _find_rel(src_type: str, dst_type: str):
        for i, rname in enumerate(relation_names):
            info = relation_info.get(rname, ("?", "?"))
            if info[0] == src_type and info[1] == dst_type:
                return A_list[i]
        return None

    M = None
    for step_idx, (src, dst) in enumerate(mp_steps):
        A_step = _find_rel(src, dst)
        if A_step is None:
            return None
        A_f = A_step.astype(np.float32)
        if M is None:
            M = A_f
        else:
            M = M @ A_f

    if M is None:
        return None

    M = csr_matrix(M)
    M.eliminate_zeros()
    return M


def discover_metapaths(
    dataset_name: str,
    relation_info: Dict[str, Tuple[str, str]],
    relation_names: List[str],
    A_list: List[csr_matrix],
    N: int,
    target_type: str,
) -> Dict[str, csr_matrix]:
    """
    Discover and build meta-path adjacency matrices for a dataset.

    Returns {metapath_name: csr_matrix}.
    """
    name_lower = dataset_name.lower().replace("-", "_")
    defs = METAPATH_DEFS.get(name_lower, [])

    if not defs:
        # Auto-discover 2-step meta-paths through target_type
        auto_defs = []
        for rname, (src, dst) in relation_info.items():
            if src == target_type:
                # Find reverse relation
                for rname2, (src2, dst2) in relation_info.items():
                    if src2 == dst and dst2 == target_type:
                        auto_defs.append((f"{target_type[:3]}{dst[:3]}{target_type[:3]}",
                                          [(src, dst), (src2, dst2)]))
        defs = auto_defs

    result = {}
    for mp_name, steps in defs:
        M = _build_metapath_adjacency(steps, A_list, relation_info, relation_names, N)
        if M is not None and M.nnz > 0:
            result[mp_name] = M

    return result


def metapath_spectral_preservation(
    metapath_adjs: Dict[str, csr_matrix],
    A_homo: csr_matrix,
    k: int = 30,
) -> Dict[str, float]:
    """
    Spectral similarity between each meta-path adjacency and A_homo.
    Handles disconnected/empty matrices gracefully.
    """
    results = {}
    for mp_name, M in metapath_adjs.items():
        if M.nnz == 0:
            results[mp_name] = 0.0
        else:
            try:
                results[mp_name] = spectral_similarity(M, A_homo, k=k)
            except Exception:
                results[mp_name] = float("nan")
    return results


def metapath_neighborhood_preservation(
    metapath_adjs: Dict[str, csr_matrix],
    Z_final: np.ndarray,
    target_size: int,
    n_sample: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Meta-path similarity correlation: Spearman correlation between
    flattened meta-path similarity (normalized) and embedding cosine similarity.

    This is much more stable than comparing k-NN sets which have
    fundamentally different density scales.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics.pairwise import cosine_similarity

    rng = np.random.default_rng(seed)
    n = min(n_sample, target_size)
    if n < 10:
        return {mp: float("nan") for mp in metapath_adjs}

    idx = rng.choice(target_size, n, replace=False)
    Z_sub = Z_final[:target_size][idx]
    S_embed = cosine_similarity(Z_sub)

    # Mask upper triangle (excluding diagonal)
    triu = np.triu_indices(n, k=1)

    results = {}
    for mp_name, M in metapath_adjs.items():
        M_self = M[:target_size, :target_size]
        M_sub = M_self[idx][:, idx].astype(np.float32)

        # Normalize meta-path similarity to [0, 1]
        data = M_sub.data
        if M_sub.nnz == 0:
            results[mp_name] = 0.0
            continue

        # Row-normalize to get similarity scores
        row_sums = np.array(M_sub.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        M_norm = M_sub.multiply(1.0 / row_sums[:, np.newaxis])

        S_meta = M_norm.toarray()
        # Symmetrize
        S_meta = (S_meta + S_meta.T) / 2

        # Flatten upper triangle and correlate
        meta_vals = S_meta[triu]
        embed_vals = S_embed[triu]

        if np.std(meta_vals) < 1e-10 or np.std(embed_vals) < 1e-10:
            results[mp_name] = 0.0
        else:
            corr, _ = spearmanr(meta_vals, embed_vals)
            results[mp_name] = float(max(0.0, corr))

    return results


def metapath_classification_preservation(
    metapath_adjs: Dict[str, csr_matrix],
    Z_final: np.ndarray,
    target_size: int,
    n_pairs: int = 5000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    For each meta-path, train a classifier on [z_i || z_j] to predict
    whether nodes are connected by the meta-path.

    Returns {mp_name: {auc, f1, accuracy}}.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    Z_t = Z_final[:target_size]

    results = {}
    for mp_name, M in metapath_adjs.items():
        M_self = M[:target_size, :target_size]
        coo = M_self.tocoo()

        pos_count = min(n_pairs // 2, len(coo.row))
        if pos_count < 10:
            results[mp_name] = {"auc": float("nan"), "f1": float("nan"), "accuracy": float("nan")}
            continue

        pos_idx = rng.choice(len(coo.row), pos_count, replace=False)
        pos_src = coo.row[pos_idx]
        pos_dst = coo.col[pos_idx]

        X_pos = np.concatenate([Z_t[pos_src], Z_t[pos_dst]], axis=1)

        # Build a set of existing edges for fast negative sampling
        edge_set = set(zip(coo.row.tolist(), coo.col.tolist()))
        # Also add self-loops as invalid negatives
        for i in range(target_size):
            edge_set.add((i, i))

        neg_src, neg_dst = [], []
        max_attempts = pos_count * 20
        attempts = 0
        while len(neg_src) < pos_count and attempts < max_attempts:
            s = rng.integers(0, target_size)
            d = rng.integers(0, target_size)
            attempts += 1
            if (s, d) not in edge_set:
                neg_src.append(s)
                neg_dst.append(d)

        if len(neg_src) < 10:
            results[mp_name] = {"auc": float("nan"), "f1": float("nan"), "accuracy": float("nan")}
            continue

        neg_src = np.array(neg_src[:pos_count])
        neg_dst = np.array(neg_dst[:pos_count])
        X_neg = np.concatenate([Z_t[neg_src], Z_t[neg_dst]], axis=1)

        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(X_pos) + [0] * len(X_neg))

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y,
        )

        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, 1]

        results[mp_name] = {
            "accuracy": float(accuracy_score(yte, pred)),
            "f1": float(f1_score(yte, pred)),
            "auc": float(roc_auc_score(yte, proba)),
        }

    return results


def metapath_diversity_preservation(
    metapath_adjs: Dict[str, csr_matrix],
    Z_final: np.ndarray,
    target_size: int,
    n_sample: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Compute CKA between each meta-path embedding (from spectral diffusion on M)
    and Z_final. Also compute a diversity score: 1 - mean(CKA between meta-paths).

    Returns {mp_cka: {mp_name: float}, diversity: float}.
    """
    rng = np.random.default_rng(seed)
    n = min(n_sample, target_size)
    idx = rng.choice(target_size, n, replace=False)
    Z_sub = Z_final[:target_size][idx]

    mp_names = list(metapath_adjs.keys())
    mp_embeddings = {}
    for mp_name, M in metapath_adjs.items():
        M_self = M[:target_size, :target_size]
        try:
            from scipy.sparse.linalg import svds
            U, S, _ = svds(M_self[idx][:, idx], k=min(50, min(M_self.shape) - 1))
            order = np.argsort(-S)
            mp_emb = U[:, order] * S[order]
            mp_embeddings[mp_name] = mp_emb[:, :min(50, mp_emb.shape[1])]
        except Exception:
            from sklearn.decomposition import TruncatedSVD
            try:
                svd = TruncatedSVD(n_components=min(50, min(M_self[idx][:, idx].shape) - 1),
                                   random_state=seed)
                mp_emb = svd.fit_transform(M_self[idx][:, idx])
                mp_embeddings[mp_name] = mp_emb
            except Exception:
                mp_embeddings[mp_name] = None

    mp_cka = {}
    for mp_name, emb in mp_embeddings.items():
        if emb is not None and len(emb) == n:
            mp_cka[mp_name] = linear_cka(emb, Z_sub)
        else:
            mp_cka[mp_name] = float("nan")

    # Pairwise CKA between meta-paths
    pairwise = []
    mp_list = [m for m in mp_names if mp_embeddings.get(m) is not None]
    for i, mi in enumerate(mp_list):
        for j, mj in enumerate(mp_list):
            if j <= i:
                continue
            ei = mp_embeddings[mi]
            ej = mp_embeddings[mj]
            if ei is not None and ej is not None and len(ei) == len(ej):
                pairwise.append(linear_cka(ei, ej))

    diversity = 1.0 - float(np.mean(pairwise)) if pairwise else float("nan")

    return {"mp_cka": mp_cka, "diversity": diversity}


def metapath_weight_alignment(
    metapath_adjs: Dict[str, csr_matrix],
    alpha: np.ndarray,
    A_list: List[csr_matrix],
    relation_info: Dict[str, Tuple[str, str]],
    relation_names: List[str],
    dataset_name: str = "unknown",
) -> Dict[str, float]:
    """
    Estimate meta-path importance from learned alpha weights.

    For a meta-path like (A→B→A), importance = α(A→B) × α(B→A).
    """
    def _find_alpha(src_type: str, dst_type: str) -> float:
        for i, rname in enumerate(relation_names):
            info = relation_info.get(rname, ("?", "?"))
            if info[0] == src_type and info[1] == dst_type:
                return float(alpha[i])
        return 0.0

    # Build a lookup from meta-path name to steps
    name_lower = dataset_name.lower().replace("-", "_")
    defs = METAPATH_DEFS.get(name_lower, [])
    mp_steps = {dmp_name: dsteps for dmp_name, dsteps in defs}

    results = {}
    for mp_name in metapath_adjs:
        steps = mp_steps.get(mp_name)
        if not steps:
            results[mp_name] = float("nan")
            continue

        importance = 1.0
        for src, dst in steps:
            importance *= _find_alpha(src, dst)
        results[mp_name] = importance

    return results


def metapath_cka_preservation(
    metapath_adjs: Dict[str, csr_matrix],
    Z_final: np.ndarray,
    target_size: int,
    n_sample: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    CKA between TruncatedSVD of each meta-path adjacency and Z_final embeddings.

    This measures how much of the meta-path structure is linearly
    recoverable from the learned embeddings (Method 2).
    """
    rng = np.random.default_rng(seed)
    n = min(n_sample, target_size)
    idx = rng.choice(target_size, n, replace=False)
    Z_sub = Z_final[:target_size][idx]
    d = Z_sub.shape[1]

    results = {}
    for mp_name, M in metapath_adjs.items():
        M_self = M[:target_size, :target_size]
        M_sub = M_self[idx][:, idx]
        try:
            k = min(d, min(M_sub.shape) - 1)
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=k, random_state=seed)
            emb = svd.fit_transform(M_sub)
            results[mp_name] = linear_cka(emb, Z_sub)
        except Exception:
            results[mp_name] = float("nan")
    return results


def metapath_retrieval_preservation(
    metapath_adjs: Dict[str, csr_matrix],
    Z_final: np.ndarray,
    target_size: int,
    ks: List[int] = None,
    n_sample: int = 1000,
    seed: int = 0,
) -> Dict[str, Dict]:
    """
    Retrieval-style metrics for meta-path neighbor prediction from embeddings.

    For each node, rank all others by embedding cosine similarity.
    Measure how many true meta-path neighbors appear in top-K.

    Returns {mp_name: {r@{k}: float, ndcg@{k}: float, mrr: float}}.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if ks is None:
        ks = [10, 20, 50]

    rng = np.random.default_rng(seed)
    n = min(n_sample, target_size)
    idx = rng.choice(target_size, n, replace=False)
    Z_sub = Z_final[:target_size][idx]

    S = cosine_similarity(Z_sub)

    results = {}
    for mp_name, M in metapath_adjs.items():
        M_self = M[:target_size, :target_size]
        M_sub = M_self[idx][:, idx].astype(np.float32)

        # Build ground-truth neighbor sets (excluding self)
        gt_neighbors = []
        for i in range(n):
            row = M_sub[i].tocoo()
            neighbors = set(row.col)
            neighbors.discard(i)
            gt_neighbors.append(neighbors)

        # Rank by cosine similarity
        ranks = np.argsort(-S, axis=1)  # descending similarity

        per_k = {}
        for k_val in ks:
            recalls = []
            ndcgs = []
            for i in range(n):
                gt = gt_neighbors[i]
                if len(gt) == 0:
                    continue
                topk = set(ranks[i, 1:k_val+1])  # exclude self (rank[0]=i)
                hits = len(gt & topk)
                recalls.append(hits / min(k_val, len(gt)))

                # NDCG
                dcg = 0.0
                for j, cand in enumerate(ranks[i, 1:k_val+1]):
                    if cand in gt:
                        dcg += 1.0 / np.log2(j + 2)
                idcg = sum(1.0 / np.log2(j + 2) for j in range(min(k_val, len(gt))))
                ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

            per_k[f"r@{k_val}"] = float(np.mean(recalls)) if recalls else float("nan")
            per_k[f"ndcg@{k_val}"] = float(np.mean(ndcgs)) if ndcgs else float("nan")

        # MRR
        mrrs = []
        for i in range(n):
            gt = gt_neighbors[i]
            if len(gt) == 0:
                continue
            for j, cand in enumerate(ranks[i, 1:], start=1):
                if cand in gt:
                    mrrs.append(1.0 / j)
                    break
        per_k["mrr"] = float(np.mean(mrrs)) if mrrs else float("nan")

        results[mp_name] = per_k

    return results


def run_metapath_analysis(
    data: dict,
    Z_final: np.ndarray,
    alpha: np.ndarray,
    A_homo: csr_matrix,
    n_spectral: int = 30,
    n_sample: int = 2000,
    seed: int = 0,
) -> Dict:
    """
    Run all meta-path preservation metrics.

    Returns dict with keys:
        metapath_preservation: {mp_name: spectral_sim}
        metapath_neighborhood: {mp_name: knn_overlap}
        metapath_classification: {mp_name: {auc, f1, accuracy}}
        metapath_diversity: {mp_cka: {mp_name: cka}, diversity: float}
        metapath_weight_alignment: {mp_name: estimated_importance}
    """
    dataset = data.get("name", "unknown")
    relation_info = data.get("relation_info", {})
    relation_names = data.get("relation_names",
                              [f"rel_{i}" for i in range(len(data["A_list_sp"]))])
    A_list = data["A_list_sp"]
    N = data["N"]
    target_type = data.get("target_type", list(data["X_dict"].keys())[0])
    target_size = data["target_size"]

    print("[analysis] Discovering meta-paths ...")
    metapath_adjs = discover_metapaths(
        dataset, relation_info, relation_names, A_list, N, target_type,
    )

    if not metapath_adjs:
        print("[analysis] No meta-paths found for this dataset.")
        return {}

    print(f"[analysis] Found meta-paths: {list(metapath_adjs.keys())}")

    # A. Spectral preservation
    mp_spectral = metapath_spectral_preservation(metapath_adjs, A_homo, k=n_spectral)

    # B. Neighborhood preservation
    mp_neighbor = metapath_neighborhood_preservation(
        metapath_adjs, Z_final, target_size, n_sample=n_sample, seed=seed,
    )

    # C. Classification preservation
    mp_class = metapath_classification_preservation(
        metapath_adjs, Z_final, target_size, n_pairs=5000, seed=seed,
    )

    # D. Diversity preservation
    mp_div = metapath_diversity_preservation(
        metapath_adjs, Z_final, target_size, n_sample=n_sample, seed=seed,
    )

    # E. Weight alignment
    mp_weight = metapath_weight_alignment(
        metapath_adjs, alpha, A_list, relation_info, relation_names,
        dataset_name=dataset,
    )

    # F. CKA preservation (Method 2)
    mp_cka = metapath_cka_preservation(
        metapath_adjs, Z_final, target_size, n_sample=n_sample, seed=seed,
    )

    # G. Retrieval preservation (Method 3)
    mp_retrieval = metapath_retrieval_preservation(
        metapath_adjs, Z_final, target_size, n_sample=n_sample, seed=seed,
    )

    mean_spectral = float(np.mean([v for v in mp_spectral.values() if not np.isnan(v)])) if mp_spectral else float("nan")

    # H. Meta-path Preservation Score (MPS)
    # MPS = 0.30*AUC + 0.25*SimCorr + 0.25*CKA + 0.20*Spectral
    mp_mps = {}
    for mp_name in metapath_adjs:
        auc = mp_class.get(mp_name, {}).get("auc", float("nan"))
        sc = mp_neighbor.get(mp_name, float("nan"))
        cka = mp_cka.get(mp_name, float("nan"))
        spec = mp_spectral.get(mp_name, float("nan"))
        vals = [v for v in [auc, sc, cka, spec] if not np.isnan(v)]
        if len(vals) == 4:
            mp_mps[mp_name] = 0.30 * auc + 0.25 * sc + 0.25 * cka + 0.20 * spec
        else:
            mp_mps[mp_name] = float("nan")

    mps_vals = [v for v in mp_mps.values() if not np.isnan(v)]
    avg_mps = float(np.mean(mps_vals)) if mps_vals else float("nan")

    return {
        "metapath_preservation": mp_spectral,
        "mean_spectral": mean_spectral,
        "metapath_neighborhood": mp_neighbor,
        "metapath_classification": mp_class,
        "metapath_diversity": mp_div,
        "metapath_weight_alignment": mp_weight,
        "metapath_cka": mp_cka,
        "metapath_retrieval": mp_retrieval,
        "metapath_mps": mp_mps,
        "avg_mps": avg_mps,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  5. A_homo builder
# ─────────────────────────────────────────────────────────────────────────────

def _extract_A_homo(alpha: np.ndarray, data: dict) -> csr_matrix:
    A_list = data['A_list_sp']
    A_homo_sp = sum(float(alpha[i]) * A.astype(np.float32) for i, A in enumerate(A_list))
    A_homo_sp = csr_matrix(A_homo_sp); A_homo_sp.eliminate_zeros()
    A_homo_sp = (A_homo_sp + A_homo_sp.T) / 2
    return A_homo_sp


# ─────────────────────────────────────────────────────────────────────────────
#  6. Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    model      : torch.nn.Module,
    data       : dict,
    device     : torch.device,
    n_sample   : int = 2000,
    k_list     : Tuple[int, ...] = (5, 10, 20),
    n_spectral : int = 30,
    target_type: Optional[str] = None,
    seed       : int = 0,
) -> Dict:
    """
    Run the full homogenization preservation analysis.

    Args:
        model       : RAHGH or RAHGHClassifier (trained)
        data        : standard RAHGH data dict
        device      : torch device
        n_sample    : nodes to subsample for embedding metrics
        k_list      : k values for neighbourhood preservation
        n_spectral  : eigenvalues to compare for spectral similarity
        target_type : node type to focus on (default = data['target_type'])
        seed        : random seed

    Returns:
        report dict
    """
    from src.model.rahgh import build_edge_index_dict, build_node_type_indices

    t0 = time.time()
    print('[analysis] Extracting representations ...')

    x_dict            = {k: v.to(device) for k, v in data['X_dict'].items()}
    edge_index_dict   = build_edge_index_dict(data, device)
    node_type_indices = {k: v.to(device) for k, v in build_node_type_indices(data).items()}

    extractor = RepresentationExtractor().attach(model)
    reps      = extractor.extract(model, x_dict, edge_index_dict, node_type_indices)
    extractor.detach()

    H0      = reps['H0']
    Z_dict  = reps['Z_dict']
    Z_final = reps['Z_final']
    alpha   = reps['alpha']

    if Z_final is None:
        Z_final = reps['Z_fused']

    if alpha is None:
        print('[analysis] WARNING: alpha not captured from hooks. '
              'Using uniform weights as fallback.')
        alpha = np.ones(len(data['A_list_sp'])) / len(data['A_list_sp'])

    ttype   = target_type or data.get('target_type', list(data['X_dict'].keys())[0])
    Nt      = data['target_size']
    rng     = np.random.default_rng(seed)
    sample  = rng.choice(Nt, min(n_sample, Nt), replace=False)

    X_orig = data['X_dict'][ttype].float().numpy()

    A_homo = _extract_A_homo(alpha, data)
    A_list = data['A_list_sp']
    rel_names = data.get('relation_names', [f'rel_{i}' for i in range(len(A_list))])

    # ── Embedding metrics ──
    print('[analysis] Computing embedding metrics ...')

    H0_ttype = H0[:Nt]
    Zf_ttype = Z_final[:Nt]

    cos_orig_to_H0 = cosine_preservation(X_orig, H0_ttype, sample)
    cos_H0_to_Zf   = cosine_preservation(H0_ttype, Zf_ttype, sample)
    cos_orig_to_Zf = cosine_preservation(X_orig, Zf_ttype, sample)

    cka_orig_H0    = linear_cka(X_orig[sample], H0_ttype[sample])
    cka_H0_Zf      = linear_cka(H0_ttype[sample], Zf_ttype[sample])
    cka_orig_Zf    = linear_cka(X_orig[sample], Zf_ttype[sample])

    nbr_H0_Zf      = neighborhood_preservation(
        H0_ttype, Zf_ttype, k_list=k_list, n_sample=n_sample, seed=seed,
    )

    # ── Structural metrics ──
    print('[analysis] Computing structural metrics ...')

    A_ref  = build_combined_adjacency(A_list, alpha)

    spec_homo_vs_ref = spectral_similarity(A_homo, A_ref, k=n_spectral)
    spec_per_rel     = {
        r: spectral_similarity(A_homo, A, k=n_spectral)
        for r, A in zip(rel_names, A_list)
    }
    lap_pres  = laplacian_preservation(A_homo, A_list, alpha)
    deg_sim   = degree_distribution_similarity(A_homo, A_ref)

    # ── Relation metrics ──
    print('[analysis] Computing relation metrics ...')

    conn_scores = relation_connectivity_score(
        Z_final, A_list, rel_names, n_sample=5000, seed=seed,
    )
    cka_rel = {}
    if Z_dict:
        cka_rel = relation_cka_matrix(Z_dict, Z_final, n_sample=n_sample, seed=seed)

    # ── New preservation metrics ──
    print('[analysis] Computing reconstruction, embedding stats ...')

    recon = feature_reconstruction_score(X_orig[sample], Zf_ttype[sample])
    embed_stats = embedding_statistics(Zf_ttype)

    rel_recovery = relation_recoverability(
        Z_final, A_list, rel_names, n_pairs=5000, seed=seed,
    )
    compression = compression_metrics(A_list, A_homo)

    eigen_sim = eigenvector_similarity(A_homo, A_ref, k=n_spectral)

    # ── Downstream preservation ──
    print('[analysis] Computing downstream preservation ...')
    labels_full = data.get('labels_full', data.get('labels', None))
    if isinstance(labels_full, torch.Tensor):
        labels_np = labels_full.cpu().numpy()
    elif isinstance(labels_full, np.ndarray):
        labels_np = labels_full
    else:
        labels_np = None

    if labels_np is not None and len(np.unique(labels_np[labels_np >= 0])) > 1:
        downstream = downstream_preservation(
            X_orig, H0_ttype, Zf_ttype, labels_np, seed=seed,
        )
    else:
        downstream = {"X_orig": float("nan"), "H0": float("nan"),
                      "Z_final": float("nan"), "DPS": float("nan")}

    # ── Meta-path preservation ──
    print("[analysis] Computing meta-path preservation ...")
    mp_results = run_metapath_analysis(
        data, Z_final, alpha, A_homo,
        n_spectral=n_spectral, n_sample=n_sample, seed=seed,
    )

    elapsed = time.time() - t0
    print(f"[analysis] Done in {elapsed:.1f}s")

    report = {
        "dataset"   : data.get("name", "unknown"),
        "N"         : data["N"],
        "Nt"        : Nt,
        "n_sample"  : len(sample),
        "alpha"     : alpha.tolist() if alpha is not None else [],
        "rel_names" : rel_names,
        "elapsed_s" : elapsed,

        "cosine": {
            "orig_to_H0": cos_orig_to_H0,
            "H0_to_Zf"  : cos_H0_to_Zf,
            "orig_to_Zf": cos_orig_to_Zf,
        },
        "cka": {
            "orig_to_H0": cka_orig_H0,
            "H0_to_Zf"  : cka_H0_Zf,
            "orig_to_Zf": cka_orig_Zf,
        },
        "neighbourhood": nbr_H0_Zf,

        "spectral": {
            "homo_vs_ref"  : spec_homo_vs_ref,
            "homo_per_rel" : spec_per_rel,
        },
        "laplacian_preservation": lap_pres,
        "degree_sim"            : deg_sim,

        "relation_connectivity": conn_scores,
        "relation_cka"         : cka_rel,

        # New metric groups
        "reconstruction"          : recon,
        "embedding_stats"         : embed_stats,
        "relation_recoverability" : rel_recovery,
        "compression"             : compression,
        "eigenvector_similarity"  : eigen_sim,
        "downstream_preservation" : downstream,
    }

    # Merge meta-path results into report
    for k, v in mp_results.items():
        report[k] = v

    return report


# ─────────────────────────────────────────────────────────────────────────────
#  7. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_report(report: Dict, save_path: Optional[str] = None):
    """
    Multi-panel figure summarising preservation analysis.
    Auto-detects available metrics and arranges accordingly.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.size']   = 11
    except ImportError:
        print('[plot] matplotlib not available — skipping plot')
        return

    has_metapath = bool(report.get("metapath_preservation"))
    nrows, ncols = (3, 2) if has_metapath else (2, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows))
    dataset   = report.get("dataset", "").upper()
    fig.suptitle(f"Homogenization Preservation Analysis — {dataset}",
                 fontsize=14, fontweight="bold", y=1.01)

    BLUE   = "#2563EB"; GREEN  = "#16A34A"; ORANGE = "#EA580C"; GREY = "#6B7280"
    ax_idx = 0

    def _next_ax():
        nonlocal ax_idx
        r, c = divmod(ax_idx, ncols)
        ax = axes[r, c] if nrows > 1 else axes[c]
        ax_idx += 1
        return ax

    # Panel 1: CKA across stages
    ax = _next_ax()
    stages  = ["Feat→H₀", "H₀→Z_final", "Feat→Z_final"]
    cka_v   = [report["cka"]["orig_to_H0"],
               report["cka"]["H0_to_Zf"],
               report["cka"]["orig_to_Zf"]]
    colors  = [BLUE, GREEN, ORANGE]
    bars    = ax.bar(stages, cka_v, color=colors, width=0.5, zorder=3)
    for bar, v in zip(bars, cka_v):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Linear CKA"); ax.set_title("Semantic Preservation (CKA)")
    ax.axhline(0.5, color=GREY, ls="--", lw=0.8, label="random baseline")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4, zorder=0)

    # Panel 2: k-NN neighbourhood preservation
    ax    = _next_ax()
    ks    = sorted(report["neighbourhood"].keys())
    nbr_v = [report["neighbourhood"][k] for k in ks]
    bars  = ax.bar([k.replace("k", "k=") for k in ks], nbr_v,
                   color=BLUE, width=0.4, zorder=3)
    for bar, v in zip(bars, nbr_v):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("k-NN Overlap (H0 vs Z_final)")
    ax.set_title("Neighbourhood Preservation"); ax.grid(axis="y", alpha=0.4, zorder=0)

    # Panel 3: Structural metrics
    ax = _next_ax()
    struct_labels = [
        "Spectral Sim\n(A_homo vs ref)",
        "Laplacian\nPreservation",
        "Degree Dist\nSimilarity",
    ]
    struct_vals = [
        report["spectral"]["homo_vs_ref"],
        report["laplacian_preservation"],
        report["degree_sim"],
    ]
    colors_s = [GREEN, ORANGE, BLUE]
    bars = ax.barh(struct_labels, struct_vals, color=colors_s, height=0.4, zorder=3)
    for bar, v in zip(bars, struct_vals):
        ax.text(max(v + 0.01, 0.05), bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.15); ax.set_xlabel("Score")
    ax.set_title("Structural Preservation"); ax.grid(axis="x", alpha=0.4, zorder=0)

    # Panel 4: Relation connectivity lift
    ax    = _next_ax()
    rels  = list(report["relation_connectivity"].keys())
    lifts = [report["relation_connectivity"][r]["lift"] for r in rels]
    conn  = [report["relation_connectivity"][r]["connected_sim"] for r in rels]
    rand  = [report["relation_connectivity"][r]["random_sim"] for r in rels]

    x    = np.arange(len(rels))
    w    = 0.3
    ax.bar(x - w/2, conn,  width=w, label="Connected",  color=BLUE,  zorder=3)
    ax.bar(x + w/2, rand,  width=w, label="Random",     color=GREY,  alpha=0.7, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(rels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Cosine Similarity"); ax.set_title("Relation Connectivity Preservation")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.axhline(0, color="k", lw=0.5)

    # Panel 5: Meta-path spectral preservation
    if has_metapath:
        ax = _next_ax()
        mp_names = list(report["metapath_preservation"].keys())
        mp_spec  = [report["metapath_preservation"][m] for m in mp_names]
        bars = ax.bar(mp_names, mp_spec, color=ORANGE, width=0.4, zorder=3)
        for bar, v in zip(bars, mp_spec):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, 1.1); ax.set_ylabel("Spectral Similarity")
        ax.set_title("Meta-path Spectral Preservation")
        ax.grid(axis="y", alpha=0.4, zorder=0)

        # Panel 6: Meta-path classification AUC
        ax = _next_ax()
        if report.get("metapath_classification"):
            mp_auc = {m: report["metapath_classification"][m].get("auc", 0)
                      for m in report["metapath_classification"]}
            mp_f1  = {m: report["metapath_classification"][m].get("f1", 0)
                      for m in report["metapath_classification"]}
            mnames = list(mp_auc.keys())
            x = np.arange(len(mnames))
            w = 0.35
            ax.bar(x - w/2, [mp_auc[m] for m in mnames], w, label="AUC", color=BLUE, zorder=3)
            ax.bar(x + w/2, [mp_f1[m] for m in mnames],  w, label="F1",  color=GREEN, zorder=3)
            ax.set_xticks(x); ax.set_xticklabels(mnames, fontsize=9)
            ax.set_ylabel("Score"); ax.set_title("Meta-path Classification")
            ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4, zorder=0)
            ax.set_ylim(0, 1.1)
        else:
            ax.text(0.5, 0.5, "No meta-path data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color=GREY)
            ax.set_title("Meta-path Classification")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[plot] Saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  8. LaTeX table generator
# ─────────────────────────────────────────────────────────────────────────────

def latex_table(report: Dict) -> str:
    """
    Generate a LaTeX table for the paper (booktabs).
    """
    dataset = report.get('dataset', 'Dataset').upper()
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        rf'\caption{{Homogenization Preservation Analysis — {dataset}}}',
        rf'\label{{tab:preservation_{dataset.lower()}}}',
        r'\begin{tabular}{lc}',
        r'\toprule',
        r'\textbf{Metric} & \textbf{Score} \\',
        r'\midrule',
        r'\multicolumn{2}{l}{\textit{Semantic Preservation}} \\',
        rf'CKA (features $\rightarrow$ H$^{{(0)}}$) & {report["cka"]["orig_to_H0"]:.4f} \\',
        rf'CKA (H$^{{(0)}} \rightarrow$ Z$_{{\mathrm{{final}}}}$) & {report["cka"]["H0_to_Zf"]:.4f} \\',
        rf'CKA (features $\rightarrow$ Z$_{{\mathrm{{final}}}}$) & {report["cka"]["orig_to_Zf"]:.4f} \\',
        rf'Cosine similarity (features $\rightarrow$ Z$_{{\mathrm{{final}}}}$) & '
        rf'{report["cosine"]["orig_to_Zf"]["mean"]:.4f} $\pm$ {report["cosine"]["orig_to_Zf"]["std"]:.4f} \\',
        r'\midrule',
        r'\multicolumn{2}{l}{\textit{Neighbourhood Preservation}} \\',
    ]
    for k, v in sorted(report['neighbourhood'].items()):
        lines.append(rf'k-NN overlap ({k}) & {v:.4f} \\')

    lines += [
        r'\midrule',
        r'\multicolumn{2}{l}{\textit{Structural Preservation}} \\',
        rf'Spectral similarity (A$_{{\mathrm{{homo}}}}$ vs ref) & {report["spectral"]["homo_vs_ref"]:.4f} \\',
        rf'Laplacian preservation & {report["laplacian_preservation"]:.4f} \\',
        rf'Degree distribution similarity & {report["degree_sim"]:.4f} \\',
        r'\midrule',
        r'\multicolumn{2}{l}{\textit{Relation Connectivity Preservation}} \\',
    ]
    for r, scores in report['relation_connectivity'].items():
        lift = scores.get('lift', float('nan'))
        conn = scores.get('connected_sim', float('nan'))
        r_safe = r.replace('\u2192', '->')
        lines.append(
            rf'{r_safe} (lift = connected $-$ random) & {conn:.4f} / lift={lift:+.4f} \\'
        )

    if report.get('relation_cka') and report['relation_cka'].get('cka_to_final'):
        lines += [r'\midrule',
                  r'\multicolumn{2}{l}{\textit{Relation CKA to Z$_{\mathrm{final}}$}} \\']
        for r, v in report['relation_cka']['cka_to_final'].items():
            r_safe = r.replace('\u2192', '->')
            lines.append(rf'{r_safe} & {v:.4f} \\')

    # ── New metric sections ──
    if 'reconstruction' in report:
        lines += [r'\midrule',
                  r'\multicolumn{2}{l}{\textit{Feature Reconstruction}} \\']
        rc = report['reconstruction']
        lines.append(rf'MSE & {rc["mse"]:.6f} \\')
        lines.append(rf'MAE & {rc["mae"]:.6f} \\')
        lines.append(rf'R$^2$ & {rc["r2"]:.6f} \\')
        lines.append(rf'Correlation & {rc["corr"]:.6f} \\')

    if 'embedding_stats' in report:
        lines += [r'\midrule',
                  r'\multicolumn{2}{l}{\textit{Embedding Statistics}} \\']
        es = report['embedding_stats']
        lines.append(rf'Rank & {es["rank"]} \\')
        lines.append(rf'Effective Rank & {es["effective_rank"]:.2f} \\')
        lines.append(rf'Participation Ratio & {es["participation_ratio"]:.2f} \\')
        lines.append(rf'Mean Norm & {es["mean_norm"]:.4f} \\')
        lines.append(rf'Std Norm & {es["std_norm"]:.4f} \\')

    if 'relation_recoverability' in report:
        lines += [r'\midrule',
                  r'\multicolumn{2}{l}{\textit{Relation Recoverability}} \\']
        rr = report['relation_recoverability']
        lines.append(rf'Accuracy & {rr["accuracy"]:.4f} \\')
        lines.append(rf'Macro F1 & {rr["macro_f1"]:.4f} \\')

    if 'compression' in report:
        lines += [r'\midrule',
                  r'\multicolumn{2}{l}{\textit{Compression}} \\']
        cm = report['compression']
        lines.append(rf'Hetero Edges & {cm["hetero_edges"]:,} \\')
        lines.append(rf'Homo Edges & {cm["homo_edges"]:,} \\')
        lines.append(rf'Compression Ratio & {cm["compression_ratio"]:.2f}$\times$ \\')

    if 'downstream_preservation' in report:
        lines += [r'\midrule',
                  r'\multicolumn{2}{l}{\textit{Downstream Preservation}} \\']
        dp = report['downstream_preservation']
        lines.append(rf'X$_{{\mathrm{{orig}}}}$ Macro F1 & {dp["X_orig"]:.4f} \\')
        lines.append(rf'H$^{{(0)}}$ Macro F1 & {dp["H0"]:.4f} \\')
        lines.append(rf'Z$_{{\mathrm{{final}}}}$ Macro F1 & {dp["Z_final"]:.4f} \\')
        lines.append(rf'DPS & {dp["DPS"]:.4f} \\')

    if 'eigenvector_similarity' in report:
        lines += [r'\midrule',
                  r'\multicolumn{2}{l}{\textit{Spectral Embedding}} \\']
        ev = report['eigenvector_similarity']
        lines.append(rf'Eigenvector CKA & {ev["eigenvector_cka"]:.4f} \\')
        lines.append(rf'Eigenvector Cosine & {ev["eigenvector_cosine"]:.4f} \\')

    if 'metapath_preservation' in report and report['metapath_preservation']:
        lines += [r'\midrule',
                  r'\multicolumn{2}{l}{\textit{Meta-path Spectral Preservation}} \\']
        for mp, v in report['metapath_preservation'].items():
            lines.append(rf'{mp} & {v:.4f} \\')

    if 'metapath_neighborhood' in report and report['metapath_neighborhood']:
        lines += [r'\midrule',
                  r'\multicolumn{2}{l}{\textit{Meta-path Similarity Correlation}} \\']
        for mp, v in report['metapath_neighborhood'].items():
            lines.append(rf'{mp} & {v:.4f} \\')

    if 'metapath_classification' in report and report['metapath_classification']:
        lines += [r'\midrule',
                  r'\multicolumn{2}{l}{\textit{Meta-path Classification}} \\']
        for mp, sc in report['metapath_classification'].items():
            lines.append(rf'{mp} AUC & {sc.get("auc", 0):.4f} \\')
            lines.append(rf'{mp} F1 & {sc.get("f1", 0):.4f} \\')

    if 'metapath_diversity' in report and report['metapath_diversity']:
        lines += [r'\midrule',
                  r'\multicolumn{2}{l}{\textit{Meta-path Diversity}} \\']
        div = report['metapath_diversity']
        lines.append(rf'Diversity Score & {div.get("diversity", 0):.4f} \\')
        for mp, cka in div.get('mp_cka', {}).items():
            lines.append(rf'{mp} CKA & {cka:.4f} \\')

    if 'metapath_mps' in report and report['metapath_mps']:
        lines += [r'\midrule',
                  r'\multicolumn{2}{l}{\textit{Meta-path Preservation Score (MPS)}} \\']
        for mp, mps in report['metapath_mps'].items():
            lines.append(rf'{mp} MPS & {mps:.4f} \\')
        avg_mps = report.get('avg_mps', float('nan'))
        if not np.isnan(avg_mps):
            lines.append(rf'Average MPS & {avg_mps:.4f} \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  9. Pretty console print
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: Dict):
    """Print a human-readable summary of the analysis report."""
    w = 62
    D = report.get('dataset', 'unknown').upper()
    print(f"\n{'=' * w}")
    print(f"  HOMOGENIZATION PRESERVATION -- {D}")
    print(f"  N={report['N']:,}  N_target={report['Nt']:,}  sampled={report['n_sample']:,}")
    print(f"{'=' * w}")

    print("\n  -- Semantic Preservation (CKA) ------------------------")
    print(f"  Feat -> H0         : {report['cka']['orig_to_H0']:.4f}  (projection stage)")
    print(f"  H0  -> Z_final     : {report['cka']['H0_to_Zf']:.4f}  (diffusion+fusion stage)")
    print(f"  Feat -> Z_final    : {report['cka']['orig_to_Zf']:.4f}  (end-to-end)")
    cos = report['cosine']['orig_to_Zf']
    print(f"  Cosine (orig->Zf)  : {cos['mean']:.4f} +- {cos['std']:.4f}  "
          f"({cos['pct_positive'] * 100:.1f}% positive)")

    print("\n  -- Neighbourhood Preservation (k-NN overlap) ----------")
    for k, v in sorted(report['neighbourhood'].items()):
        print(f"  {k:<6}: {v:.4f}")

    print("\n  -- Structural Preservation ----------------------------")
    print(f"  Spectral sim (A_homo vs ref): {report['spectral']['homo_vs_ref']:.4f}")
    print(f"  Laplacian preservation      : {report['laplacian_preservation']:.4f}")
    print(f"  Degree distribution sim     : {report['degree_sim']:.4f}")
    print(f"\n  Per-relation spectral sim vs A_homo:")
    for r, v in report['spectral']['homo_per_rel'].items():
        r_str = r.replace('\u2192', '->')
        print(f"    {r_str:<20}: {v:.4f}")

    print("\n  -- Relation Connectivity Preservation -----------------")
    print(f"  {'Relation':<20} {'Connected':>10} {'Random':>10} {'Lift':>8}")
    print(f"  {'-' * 50}")
    for r, s in report['relation_connectivity'].items():
        r_str = r.replace('\u2192', '->')
        print(f"  {r_str:<20} {s['connected_sim']:>10.4f} {s['random_sim']:>10.4f} "
              f"{s['lift']:>+8.4f}")

    if report.get('relation_cka') and report['relation_cka'].get('cka_to_final'):
        print("\n  -- Relation CKA to Z_final ----------------------------")
        for r, v in report['relation_cka']['cka_to_final'].items():
            r_str = r.replace('\u2192', '->')
            print(f"  {r_str:<20}: {v:.4f}")

    # ── New metric sections ──
    if 'reconstruction' in report:
        print("\n  -- Feature Reconstruction (Z_final -> X_orig) ----------")
        rc = report['reconstruction']
        print(f"  MSE  : {rc['mse']:.6f}")
        print(f"  MAE  : {rc['mae']:.6f}")
        print(f"  R2   : {rc['r2']:.6f}")
        print(f"  Corr : {rc['corr']:.6f}")

    if 'embedding_stats' in report:
        print("\n  -- Embedding Statistics --------------------------------")
        es = report['embedding_stats']
        print(f"  Rank                : {es['rank']}")
        print(f"  Effective Rank      : {es['effective_rank']:.2f}")
        print(f"  Participation Ratio : {es['participation_ratio']:.2f}")
        print(f"  Mean Norm           : {es['mean_norm']:.4f}")
        print(f"  Std Norm            : {es['std_norm']:.4f}")

    if 'relation_recoverability' in report:
        print("\n  -- Relation Recoverability -----------------------------")
        rr = report['relation_recoverability']
        print(f"  Accuracy : {rr['accuracy']:.4f}")
        print(f"  Macro F1 : {rr['macro_f1']:.4f}")

    if 'compression' in report:
        print("\n  -- Compression Metrics ---------------------------------")
        cm = report['compression']
        print(f"  Hetero Edges       : {cm['hetero_edges']:,}")
        print(f"  Homo Edges         : {cm['homo_edges']:,}")
        print(f"  Compression Ratio  : {cm['compression_ratio']:.2f}x")

    if 'downstream_preservation' in report:
        print("\n  -- Downstream Preservation -----------------------------")
        dp = report['downstream_preservation']
        print(f"  X_orig F1  : {dp['X_orig']:.4f}")
        print(f"  H0 F1      : {dp['H0']:.4f}")
        print(f"  Z_final F1 : {dp['Z_final']:.4f}")
        print(f"  DPS        : {dp['DPS']:.4f}")

    if 'eigenvector_similarity' in report:
        print("\n  -- Spectral Embedding Preservation ---------------------")
        ev = report['eigenvector_similarity']
        print(f"  Eigenvector CKA    : {ev['eigenvector_cka']:.4f}")
        print(f"  Eigenvector Cosine : {ev['eigenvector_cosine']:.4f}")

    if 'metapath_preservation' in report and report['metapath_preservation']:
        print("\n  -- Meta-path Spectral Preservation --------------------")
        for mp, v in report['metapath_preservation'].items():
            print(f"  {mp:<10}: {v:.4f}")

    if 'metapath_neighborhood' in report and report['metapath_neighborhood']:
        print("\n  -- Meta-path Similarity Correlation ------------------")
        for mp, v in report['metapath_neighborhood'].items():
            print(f"  {mp:<10} Spearman corr: {v:.4f}")

    if 'metapath_classification' in report and report['metapath_classification']:
        print("\n  -- Meta-path Classification Preservation --------------")
        print(f"  {'Meta-path':<10} {'AUC':>8} {'F1':>8}")
        print(f"  {'-' * 30}")
        for mp, scores in report['metapath_classification'].items():
            print(f"  {mp:<10} {scores.get('auc', 0):>8.4f} {scores.get('f1', 0):>8.4f}")

    if 'metapath_diversity' in report and report['metapath_diversity']:
        div = report['metapath_diversity']
        print(f"\n  -- Meta-path Diversity Preservation -------------------")
        print(f"  Diversity Score : {div.get('diversity', 0):.4f}")
        for mp, cka in div.get('mp_cka', {}).items():
            print(f"  {mp} CKA with Z_final: {cka:.4f}")

    if 'metapath_cka' in report and report['metapath_cka']:
        print("\n  -- Meta-path CKA Preservation (Method 2) --------------")
        for mp, cka in report['metapath_cka'].items():
            print(f"  {mp:<10} CKA with Z_final: {cka:.4f}")

    if 'metapath_retrieval' in report and report['metapath_retrieval']:
        print("\n  -- Meta-path Retrieval (Method 3) ---------------------")
        print(f"  {'Meta-path':<10} {'MRR':>8} {'R@10':>8} {'R@20':>8} {'R@50':>8}")
        print(f"  {'-' * 55}")
        for mp, sc in report['metapath_retrieval'].items():
            mrr = sc.get('mrr', float('nan'))
            r10 = sc.get('r@10', float('nan'))
            r20 = sc.get('r@20', float('nan'))
            r50 = sc.get('r@50', float('nan'))
            print(f"  {mp:<10} {mrr:>8.4f} {r10:>8.4f} {r20:>8.4f} {r50:>8.4f}")

    if 'metapath_weight_alignment' in report and report['metapath_weight_alignment']:
        print("\n  -- Meta-path Weight Alignment -------------------------")
        for mp, wt in report['metapath_weight_alignment'].items():
            print(f"  {mp:<10} estimated importance: {wt:.4f}")

    if 'metapath_mps' in report and report['metapath_mps']:
        print("\n  -- Meta-path Preservation Score (MPS) -----------------")
        print(f"  {'Meta-path':<10} {'AUC':>8} {'SimCorr':>8} {'CKA':>8} {'Spectral':>9} {'MPS':>8}")
        print(f"  {'-' * 60}")
        for mp in report['metapath_mps']:
            auc = report['metapath_classification'].get(mp, {}).get('auc', 0)
            sc = report['metapath_neighborhood'].get(mp, 0)
            cka = report['metapath_cka'].get(mp, 0)
            spec = report['metapath_preservation'].get(mp, 0)
            mps = report['metapath_mps'].get(mp, 0)
            print(f"  {mp:<10} {auc:>8.4f} {sc:>8.4f} {cka:>8.4f} {spec:>9.4f} {mps:>8.4f}")
        avg_mps = report.get('avg_mps', float('nan'))
        print(f"  {'-' * 50}")
        print(f"  {'Average':<10} {'':>8} {'':>8} {'':>9} {avg_mps:>8.4f}")

    print(f"\n  Elapsed: {report['elapsed_s']:.1f}s")
    print(f"{'=' * w}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  10. CSV / TXT / LaTeX Export Suite
# ─────────────────────────────────────────────────────────────────────────────

def save_report_csv(report: Dict, out_dir: str = "results"):
    """
    Save all preservation metrics as individual CSV files.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── preservation_metrics.csv (summary) ──
    summary_rows = [
        ["cka_orig_H0", report["cka"]["orig_to_H0"]],
        ["cka_H0_Zf", report["cka"]["H0_to_Zf"]],
        ["cka_orig_Zf", report["cka"]["orig_to_Zf"]],
        ["laplacian_preservation", report["laplacian_preservation"]],
        ["degree_similarity", report["degree_sim"]],
    ]
    pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(        out / "preservation_metrics.csv", index=False,
    )

    # ── relation_connectivity.csv ──
    conn_rows = []
    for rname, scores in report.get("relation_connectivity", {}).items():
        conn_rows.append({
            "relation": rname,
            "connected_sim": scores.get("connected_sim"),
            "random_sim": scores.get("random_sim"),
            "lift": scores.get("lift"),
        })
    if conn_rows:
        pd.DataFrame(conn_rows).to_csv(out / "relation_connectivity.csv", index=False, encoding="utf-8")

    # ── relation_cka.csv ──
    cka_rows = []
    for rname, val in report.get("relation_cka", {}).get("cka_to_final", {}).items():
        cka_rows.append({"relation": rname, "cka_to_final": val})
    if cka_rows:
        pd.DataFrame(cka_rows).to_csv(out / "relation_cka.csv", index=False, encoding="utf-8")

    # ── reconstruction.csv ──
    if "reconstruction" in report:
        pd.DataFrame([report["reconstruction"]]).to_csv(            out / "reconstruction.csv", index=False,
        )

    # ── embedding_stats.csv ──
    if "embedding_stats" in report:
        pd.DataFrame([report["embedding_stats"]]).to_csv(            out / "embedding_stats.csv", index=False,
        )

    # ── alpha_weights.csv ──
    alpha = report.get("alpha", [])
    rels = report.get("rel_names", [f"rel_{i}" for i in range(len(alpha))])
    if alpha:
        pd.DataFrame({"relation": rels, "weight": alpha}).to_csv(            out / "alpha_weights.csv", index=False,
        )

    # ── robustness.csv (placeholder, populated by separate call) ──
    if "robustness" in report:
        pd.DataFrame(report["robustness"]).to_csv(            out / "robustness.csv", index=False,
        )

    # ── downstream_preservation.csv ──
    if "downstream_preservation" in report:
        pd.DataFrame([report["downstream_preservation"]]).to_csv(            out / "downstream_preservation.csv", index=False,
        )

    # ── spectral_metrics.csv ──
    spectral_rows = [["homo_vs_ref", report.get("spectral", {}).get("homo_vs_ref", float("nan"))]]
    for r, v in report.get("spectral", {}).get("homo_per_rel", {}).items():
        spectral_rows.append([f"homo_vs_{r}", v])
    if "eigenvector_similarity" in report:
        ev = report["eigenvector_similarity"]
        spectral_rows.append(["eigenvector_cka", ev.get("eigenvector_cka")])
        spectral_rows.append(["eigenvector_cosine", ev.get("eigenvector_cosine")])
    pd.DataFrame(spectral_rows, columns=["metric", "value"]).to_csv(        out / "spectral_metrics.csv", index=False,
    )

    # ── metapath_preservation.csv ──
    mp_rows = []
    for mp, v in report.get("metapath_preservation", {}).items():
        mp_rows.append({"metapath": mp, "spectral_similarity": v})
    for mp, v in report.get("metapath_neighborhood", {}).items():
        for r in mp_rows:
            if r["metapath"] == mp:
                r["knn_overlap_k10"] = v
                break
        else:
            mp_rows.append({"metapath": mp, "knn_overlap_k10": v})
    for mp, sc in report.get("metapath_classification", {}).items():
        for r in mp_rows:
            if r["metapath"] == mp:
                r["auc"] = sc.get("auc")
                r["f1"] = sc.get("f1")
                break
    if "metapath_diversity" in report:
        div = report["metapath_diversity"]
        for mp, cka in div.get("mp_cka", {}).items():
            for r in mp_rows:
                if r["metapath"] == mp:
                    r["cka_with_zfinal"] = cka
                    break
        mp_rows.append({"metapath": "diversity", "diversity_score": div.get("diversity")})
    for mp, wt in report.get("metapath_weight_alignment", {}).items():
        for r in mp_rows:
            if r["metapath"] == mp:
                r["estimated_importance"] = wt
                break
    for mp, mps in report.get("metapath_mps", {}).items():
        for r in mp_rows:
            if r["metapath"] == mp:
                r["mps"] = mps
                break
    if "avg_mps" in report and not np.isnan(report["avg_mps"]):
        mp_rows.append({"metapath": "average_mps", "mps": report["avg_mps"]})
    if mp_rows:
        pd.DataFrame(mp_rows).to_csv(out / "metapath_preservation.csv", index=False, encoding="utf-8")


def save_report_txt(report: Dict, filepath: str = "results/preservation_summary.txt"):
    """
    Save a human-readable plain-text summary.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        D = report.get("dataset", "unknown").upper()
        f.write("=" * 70 + "\n")
        f.write(f"HOMOGENIZATION PRESERVATION REPORT — {D}\n")
        f.write("=" * 70 + "\n\n")

        f.write("CKA Metrics\n")
        f.write("-" * 40 + "\n")
        for k, v in report["cka"].items():
            f.write(f"{k:25s}: {v:.6f}\n")
        f.write("\n")

        f.write("Cosine Similarity\n")
        f.write("-" * 40 + "\n")
        for stage in ["orig_to_H0", "H0_to_Zf", "orig_to_Zf"]:
            c = report["cosine"][stage]
            f.write(f"{stage:25s}: mean={c['mean']:.4f} "
                    f"std={c['std']:.4f} pct_pos={c['pct_positive']:.2%}\n")
        f.write("\n")

        f.write("Structural Metrics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Laplacian Preservation : {report['laplacian_preservation']:.6f}\n")
        f.write(f"Degree Similarity      : {report['degree_sim']:.6f}\n")
        f.write(f"Spectral Sim (vs ref)  : {report['spectral']['homo_vs_ref']:.6f}\n")
        f.write("\n")

        f.write("Neighbourhood Preservation\n")
        f.write("-" * 40 + "\n")
        for k, v in sorted(report["neighbourhood"].items()):
            f.write(f"{k:6s}: {v:.6f}\n")
        f.write("\n")

        if "reconstruction" in report:
            f.write("Feature Reconstruction\n")
            f.write("-" * 40 + "\n")
            rc = report["reconstruction"]
            for k, v in rc.items():
                f.write(f"{k:10s}: {v:.6f}\n")
            f.write("\n")

        if "embedding_stats" in report:
            f.write("Embedding Statistics\n")
            f.write("-" * 40 + "\n")
            es = report["embedding_stats"]
            for k, v in es.items():
                f.write(f"{k:22s}: {v}\n")
            f.write("\n")

        if "relation_recoverability" in report:
            f.write("Relation Recoverability\n")
            f.write("-" * 40 + "\n")
            rr = report["relation_recoverability"]
            f.write(f"Accuracy : {rr['accuracy']:.4f}\n")
            f.write(f"Macro F1 : {rr['macro_f1']:.4f}\n")
            f.write("\n")

        if "compression" in report:
            f.write("Compression Metrics\n")
            f.write("-" * 40 + "\n")
            cm = report["compression"]
            f.write(f"Hetero Edges      : {cm['hetero_edges']:,}\n")
            f.write(f"Homo Edges        : {cm['homo_edges']:,}\n")
            f.write(f"Compression Ratio : {cm['compression_ratio']:.2f}x\n")
            f.write("\n")

        if "downstream_preservation" in report:
            f.write("Downstream Preservation\n")
            f.write("-" * 40 + "\n")
            dp = report["downstream_preservation"]
            f.write(f"X_orig Macro F1  : {dp['X_orig']:.4f}\n")
            f.write(f"H0 Macro F1      : {dp['H0']:.4f}\n")
            f.write(f"Z_final Macro F1 : {dp['Z_final']:.4f}\n")
            f.write(f"DPS              : {dp['DPS']:.4f}\n")
            f.write("\n")

        if "eigenvector_similarity" in report:
            f.write("Spectral Embedding Preservation\n")
            f.write("-" * 40 + "\n")
            ev = report["eigenvector_similarity"]
            f.write(f"Eigenvector CKA    : {ev['eigenvector_cka']:.4f}\n")
            f.write(f"Eigenvector Cosine : {ev['eigenvector_cosine']:.4f}\n")
            f.write("\n")

        if "metapath_preservation" in report and report["metapath_preservation"]:
            f.write("Meta-path Spectral Preservation\n")
            f.write("-" * 40 + "\n")
            for mp, v in report["metapath_preservation"].items():
                f.write(f"{mp:10s}: {v:.6f}\n")
            f.write("\n")

        if "metapath_neighborhood" in report and report["metapath_neighborhood"]:
            f.write("Meta-path Similarity Correlation (Spearman)\n")
            f.write("-" * 40 + "\n")
            for mp, v in report["metapath_neighborhood"].items():
                f.write(f"{mp:10s}: {v:.6f}\n")
            f.write("\n")

        if "metapath_classification" in report and report["metapath_classification"]:
            f.write("Meta-path Classification Preservation\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Meta-path':<10} {'AUC':>8} {'F1':>8}\n")
            f.write("-" * 30 + "\n")
            for mp, sc in report["metapath_classification"].items():
                f.write(f"{mp:<10} {sc.get('auc', 0):>8.4f} {sc.get('f1', 0):>8.4f}\n")
            f.write("\n")

        if "metapath_diversity" in report and report["metapath_diversity"]:
            div = report["metapath_diversity"]
            f.write("Meta-path Diversity Preservation\n")
            f.write("-" * 40 + "\n")
            f.write(f"Diversity Score : {div.get('diversity', 0):.4f}\n")
            for mp, cka in div.get("mp_cka", {}).items():
                f.write(f"{mp} CKA with Z_final: {cka:.4f}\n")
            f.write("\n")

        if "metapath_cka" in report and report["metapath_cka"]:
            f.write("Meta-path CKA Preservation (Method 2)\n")
            f.write("-" * 40 + "\n")
            for mp, cv in report["metapath_cka"].items():
                f.write(f"{mp:10s}: {cv:.6f}\n")
            f.write("\n")

        if "metapath_retrieval" in report and report["metapath_retrieval"]:
            f.write("Meta-path Retrieval (Method 3)\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Meta-path':<10} {'MRR':>8} {'R@10':>8} {'R@20':>8} {'R@50':>8}\n")
            f.write("-" * 55 + "\n")
            for mp, sc in report["metapath_retrieval"].items():
                mrr = sc.get("mrr", float("nan"))
                r10 = sc.get("r@10", float("nan"))
                r20 = sc.get("r@20", float("nan"))
                r50 = sc.get("r@50", float("nan"))
                f.write(f"{mp:<10} {mrr:>8.4f} {r10:>8.4f} {r20:>8.4f} {r50:>8.4f}\n")
            f.write("\n")

        if "metapath_weight_alignment" in report and report["metapath_weight_alignment"]:
            f.write("Meta-path Weight Alignment\n")
            f.write("-" * 40 + "\n")
            for mp, wt in report["metapath_weight_alignment"].items():
                f.write(f"{mp:10s}: {wt:.6f}\n")
            f.write("\n")

        if "metapath_mps" in report and report["metapath_mps"]:
            f.write("Meta-path Preservation Score (MPS)\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Meta-path':<10} {'AUC':>8} {'SimCorr':>8} {'CKA':>8} {'Spectral':>9} {'MPS':>8}\n")
            f.write("-" * 60 + "\n")
            for mp in report["metapath_mps"]:
                auc = report["metapath_classification"].get(mp, {}).get("auc", 0)
                sc = report["metapath_neighborhood"].get(mp, 0)
                cka = report["metapath_cka"].get(mp, 0)
                spec = report["metapath_preservation"].get(mp, 0)
                mps = report["metapath_mps"].get(mp, 0)
                f.write(f"{mp:<10} {auc:>8.4f} {sc:>8.4f} {cka:>8.4f} {spec:>9.4f} {mps:>8.4f}\n")
            avg_mps = report.get("avg_mps", float("nan"))
            if not np.isnan(avg_mps):
                f.write("-" * 60 + "\n")
                f.write(f"{'Average':<10} {'':>8} {'':>8} {'':>8} {'':>9} {avg_mps:>8.4f}\n")
            f.write("\n")

        # Relation connectivity summary
        f.write("Relation Connectivity Preservation\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Relation':<20} {'Connected':>10} {'Random':>10} {'Lift':>8}\n")
        f.write("-" * 50 + "\n")
        for r, s in report["relation_connectivity"].items():
            f.write(f"{r:<20} {s['connected_sim']:>10.4f} {s['random_sim']:>10.4f} "
                    f"{s['lift']:>+8.4f}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")


def export_report(
    report: Dict,
    output_dir: str = "results/homogenization_analysis",
    latex: bool = True,
):
    """
    Master export: saves CSVs, TXT summary, and LaTeX tables in a structured folder.

    Generates:
        homogenization_analysis/
        ├── report.txt
        ├── summary.csv
        ├── relation_metrics.csv
        ├── relation_cka.csv
        ├── reconstruction.csv
        ├── embedding_stats.csv
        ├── alpha.csv
        ├── spectral_metrics.csv
        ├── downstream_preservation.csv
        ├── latex/
        │   ├── preservation_table.tex
        │   └── supplementary_table.tex
        └── (plots/ — generated separately by plot_report)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "latex").mkdir(parents=True, exist_ok=True)

    # TXT summary
    save_report_txt(report, str(out / "report.txt"))

    # CSV suite
    save_report_csv(report, str(out))

    # summary.csv (flat key-value)
    summary_rows = [
        ("dataset", report.get("dataset", "unknown")),
        ("N", report.get("N")),
        ("N_target", report.get("Nt")),
        ("n_sample", report.get("n_sample")),
        ("cka_orig_H0", report["cka"]["orig_to_H0"]),
        ("cka_H0_Zf", report["cka"]["H0_to_Zf"]),
        ("cka_orig_Zf", report["cka"]["orig_to_Zf"]),
        ("laplacian_preservation", report["laplacian_preservation"]),
        ("degree_sim", report["degree_sim"]),
        ("spectral_homo_vs_ref", report["spectral"]["homo_vs_ref"]),
    ]
    for k, v in report.get("reconstruction", {}).items():
        summary_rows.append((f"recon_{k}", v))
    for k, v in report.get("embedding_stats", {}).items():
        summary_rows.append((f"embed_{k}", v))
    for k, v in report.get("relation_recoverability", {}).items():
        summary_rows.append((f"rel_rec_{k}", v))
    for k, v in report.get("downstream_preservation", {}).items():
        summary_rows.append((f"downstream_{k}", v))
    for k, v in report.get("eigenvector_similarity", {}).items():
        summary_rows.append((f"eigen_{k}", v))
    for k, v in report.get("compression", {}).items():
        summary_rows.append((f"compression_{k}", v))
    for mp, v in report.get("metapath_preservation", {}).items():
        summary_rows.append((f"mp_spectral_{mp}", v))
    for mp, v in report.get("metapath_neighborhood", {}).items():
        summary_rows.append((f"mp_knn_{mp}", v))
    for mp, sc in report.get("metapath_classification", {}).items():
        summary_rows.append((f"mp_auc_{mp}", sc.get("auc")))
        summary_rows.append((f"mp_f1_{mp}", sc.get("f1")))
    div = report.get("metapath_diversity", {})
    if div:
        summary_rows.append(("mp_diversity", div.get("diversity")))
        for mp, cka in div.get("mp_cka", {}).items():
            summary_rows.append((f"mp_cka_{mp}", cka))
    for mp, wt in report.get("metapath_weight_alignment", {}).items():
        summary_rows.append((f"mp_weight_{mp}", wt))
    for mp, mps in report.get("metapath_mps", {}).items():
        summary_rows.append((f"mp_mps_{mp}", mps))
    avg_mps = report.get("avg_mps", float("nan"))
    if not np.isnan(avg_mps):
        summary_rows.append(("avg_mps", avg_mps))

    pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(        out / "summary.csv", index=False,
    )

    # ── relation_metrics.csv ──
    conn_rows = []
    for rname, sc in report.get("relation_connectivity", {}).items():
        conn_rows.append({
            "relation": rname,
            "connected_sim": sc.get("connected_sim"),
            "random_sim": sc.get("random_sim"),
            "lift": sc.get("lift"),
        })
    cka_vals = report.get("relation_cka", {}).get("cka_to_final", {})
    for row in conn_rows:
        row["cka_to_final"] = cka_vals.get(row["relation"], float("nan"))
    if conn_rows:
        pd.DataFrame(conn_rows).to_csv(out / "relation_metrics.csv", index=False, encoding="utf-8")

    # ── LaTeX tables ──
    if latex:
        tex_main = latex_table(report)
        with open(out / "latex" / "preservation_table.tex", "w", encoding="utf-8") as f:
            f.write(tex_main)

        # Supplementary table with all metrics
        supp = _supplementary_latex(report)
        with open(out / "latex" / "supplementary_table.tex", "w", encoding="utf-8") as f:
            f.write(supp)

    print(f"[export] All files written to {out}/")
    return str(out)


def _supplementary_latex(report: Dict) -> str:
    """
    Generate a detailed supplementary LaTeX table with every metric.
    """
    D = report.get("dataset", "Dataset").upper()
    lines = [
        r'\begin{landscape}',
        r'\begin{table}[h]',
        r'\centering',
        rf'\caption{{Full Preservation Metrics — {D} (Supplementary)}}',
        rf'\label{{tab:supp_{D.lower()}}}',
        r'\begin{tabular}{lcc}',
        r'\toprule',
        r'\textbf{Metric} & \textbf{Value} & \textbf{Interpretation} \\',
        r'\midrule',
    ]

    def add(k, v, note=""):
        if isinstance(v, float):
            lines.append(rf'{k} & {v:.6f} & {note} \\')
        else:
            lines.append(rf'{k} & {v} & {note} \\')

    add("CKA (Feat → H0)", report["cka"]["orig_to_H0"], "Projection stage")
    add("CKA (H0 → Z_final)", report["cka"]["H0_to_Zf"], "Diffusion+fUSION")
    add("CKA (Feat → Z_final)", report["cka"]["orig_to_Zf"], "End-to-end")
    add("Laplacian Preservation", report["laplacian_preservation"], "Higher = better")
    add("Degree Similarity", report["degree_sim"], "JSD-based")
    add("Spectral Sim (A_homo vs ref)", report["spectral"]["homo_vs_ref"], "Top-k eigenvalues")

    if "reconstruction" in report:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{Feature Reconstruction}} \\']
        rc = report["reconstruction"]
        add("MSE", rc["mse"], "Lower = better")
        add("MAE", rc["mae"], "Lower = better")
        add("R²", rc["r2"], "Higher = better")
        add("Correlation", rc["corr"], "Higher = better")

    if "embedding_stats" in report:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{Embedding Statistics}} \\']
        es = report["embedding_stats"]
        add("Rank", es["rank"])
        add("Effective Rank", es["effective_rank"])
        add("Participation Ratio", es["participation_ratio"], "Higher = more isotropic")
        add("Mean Norm", es["mean_norm"])
        add("Std Norm", es["std_norm"])

    if "relation_recoverability" in report:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{Relation Recoverability}} \\']
        rr = report["relation_recoverability"]
        add("Accuracy", rr["accuracy"], "Higher = better preserved")
        add("Macro F1", rr["macro_f1"], "Higher = better preserved")

    if "compression" in report:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{Compression}} \\']
        cm = report["compression"]
        add("Hetero Edges", cm["hetero_edges"])
        add("Homo Edges", cm["homo_edges"])
        add("Compression Ratio", cm["compression_ratio"], "Higher = more compression")

    if "downstream_preservation" in report:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{Downstream Preservation}} \\']
        dp = report["downstream_preservation"]
        add("X_orig Macro F1", dp["X_orig"])
        add("H0 Macro F1", dp["H0"])
        add("Z_final Macro F1", dp["Z_final"])
        add("DPS", dp["DPS"], "Z_final F1 / X_orig F1")

    if "eigenvector_similarity" in report:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{Spectral Embedding}} \\']
        ev = report["eigenvector_similarity"]
        add("Eigenvector CKA", ev["eigenvector_cka"])
        add("Eigenvector Cosine", ev["eigenvector_cosine"])

    if "metapath_preservation" in report and report["metapath_preservation"]:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{Meta-path Spectral}} \\']
        for mp, v in report["metapath_preservation"].items():
            add(f"{mp} spectral", v)

    if "metapath_neighborhood" in report and report["metapath_neighborhood"]:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{Meta-path Similarity Correlation}} \\']
        for mp, v in report["metapath_neighborhood"].items():
            add(f"{mp} Spearman corr", v)

    if "metapath_classification" in report and report["metapath_classification"]:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{Meta-path Classification}} \\']
        for mp, sc in report["metapath_classification"].items():
            add(f"{mp} AUC", sc.get("auc", 0))
            add(f"{mp} F1", sc.get("f1", 0))

    if "metapath_diversity" in report and report["metapath_diversity"]:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{Meta-path Diversity}} \\']
        div = report["metapath_diversity"]
        add("Diversity", div.get("diversity", 0), "Higher = more diverse")
        for mp, cka in div.get("mp_cka", {}).items():
            add(f"{mp} CKA", cka)

    if "metapath_cka" in report and report["metapath_cka"]:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{Meta-path CKA (Method 2)}} \\']
        for mp, v in report["metapath_cka"].items():
            add(f"{mp} CKA", v, "SVD(M_meta) vs Z_final")

    if "metapath_retrieval" in report and report["metapath_retrieval"]:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{Meta-path Retrieval (Method 3)}} \\']
        for mp, sc in report["metapath_retrieval"].items():
            add(f"{mp} MRR", sc.get("mrr", float("nan")))
            add(f"{mp} R@10", sc.get("r@10", float("nan")))
            add(f"{mp} R@20", sc.get("r@20", float("nan")))
            add(f"{mp} R@50", sc.get("r@50", float("nan")))

    if "metapath_mps" in report and report["metapath_mps"]:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{MPS (Meta-path Preservation Score)}} \\']
        for mp, mps in report["metapath_mps"].items():
            add(f"{mp} MPS", mps, "0.30*AUC+0.25*SimCorr+0.25*CKA+0.20*Spectral")
        avg_mps = report.get("avg_mps", float("nan"))
        if not np.isnan(avg_mps):
            add("Average MPS", avg_mps)

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
        r'\end{landscape}',
    ]
    return '\n'.join(lines)
