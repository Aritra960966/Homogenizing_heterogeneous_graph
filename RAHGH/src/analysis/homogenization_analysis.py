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

import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import eigsh, norm as spnorm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

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
        λ1 = np.sort(np.abs(eigsh(L1, k=k, which='SM', return_eigenvectors=False)))
        λ2 = np.sort(np.abs(eigsh(L2, k=k, which='SM', return_eigenvectors=False)))
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

    elapsed = time.time() - t0
    print(f'[analysis] Done in {elapsed:.1f}s')

    return {
        'dataset'   : data.get('name', 'unknown'),
        'N'         : data['N'],
        'Nt'        : Nt,
        'n_sample'  : len(sample),
        'alpha'     : alpha.tolist() if alpha is not None else [],
        'rel_names' : rel_names,
        'elapsed_s' : elapsed,

        'cosine': {
            'orig_to_H0': cos_orig_to_H0,
            'H0_to_Zf'  : cos_H0_to_Zf,
            'orig_to_Zf': cos_orig_to_Zf,
        },
        'cka': {
            'orig_to_H0': cka_orig_H0,
            'H0_to_Zf'  : cka_H0_Zf,
            'orig_to_Zf': cka_orig_Zf,
        },
        'neighbourhood': nbr_H0_Zf,

        'spectral': {
            'homo_vs_ref'  : spec_homo_vs_ref,
            'homo_per_rel' : spec_per_rel,
        },
        'laplacian_preservation': lap_pres,
        'degree_sim'            : deg_sim,

        'relation_connectivity': conn_scores,
        'relation_cka'         : cka_rel,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  7. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_report(report: Dict, save_path: Optional[str] = None):
    """
    4-panel figure summarising preservation analysis.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.size']   = 11
    except ImportError:
        print('[plot] matplotlib not available — skipping plot')
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    dataset   = report.get('dataset', '').upper()
    fig.suptitle(f'Homogenization Preservation Analysis — {dataset}',
                 fontsize=14, fontweight='bold', y=1.01)

    BLUE   = '#2563EB'; GREEN  = '#16A34A'; ORANGE = '#EA580C'; GREY = '#6B7280'

    # Panel 1: CKA across stages
    ax = axes[0, 0]
    stages  = ['Feat→H₀', 'H₀→Z_final', 'Feat→Z_final']
    cka_v   = [report['cka']['orig_to_H0'],
               report['cka']['H0_to_Zf'],
               report['cka']['orig_to_Zf']]
    colors  = [BLUE, GREEN, ORANGE]
    bars    = ax.bar(stages, cka_v, color=colors, width=0.5, zorder=3)
    for bar, v in zip(bars, cka_v):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Linear CKA'); ax.set_title('Semantic Preservation (CKA)')
    ax.axhline(0.5, color=GREY, ls='--', lw=0.8, label='random baseline')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.4, zorder=0)

    # Panel 2: k-NN neighbourhood preservation
    ax    = axes[0, 1]
    ks    = sorted(report['neighbourhood'].keys())
    nbr_v = [report['neighbourhood'][k] for k in ks]
    bars  = ax.bar([k.replace('k', 'k=') for k in ks], nbr_v,
                   color=BLUE, width=0.4, zorder=3)
    for bar, v in zip(bars, nbr_v):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel('k-NN Overlap (H0 vs Z_final)')
    ax.set_title('Neighbourhood Preservation'); ax.grid(axis='y', alpha=0.4, zorder=0)

    # Panel 3: Structural metrics
    ax = axes[1, 0]
    struct_labels = [
        'Spectral Sim\n(A_homo vs ref)',
        'Laplacian\nPreservation',
        'Degree Dist\nSimilarity',
    ]
    struct_vals = [
        report['spectral']['homo_vs_ref'],
        report['laplacian_preservation'],
        report['degree_sim'],
    ]
    colors_s = [GREEN, ORANGE, BLUE]
    bars = ax.barh(struct_labels, struct_vals, color=colors_s, height=0.4, zorder=3)
    for bar, v in zip(bars, struct_vals):
        ax.text(max(v + 0.01, 0.05), bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=10)
    ax.set_xlim(0, 1.15); ax.set_xlabel('Score')
    ax.set_title('Structural Preservation'); ax.grid(axis='x', alpha=0.4, zorder=0)

    # Panel 4: Relation connectivity lift
    ax    = axes[1, 1]
    rels  = list(report['relation_connectivity'].keys())
    lifts = [report['relation_connectivity'][r]['lift'] for r in rels]
    conn  = [report['relation_connectivity'][r]['connected_sim'] for r in rels]
    rand  = [report['relation_connectivity'][r]['random_sim'] for r in rels]

    x    = np.arange(len(rels))
    w    = 0.3
    ax.bar(x - w/2, conn,  width=w, label='Connected',  color=BLUE,  zorder=3)
    ax.bar(x + w/2, rand,  width=w, label='Random',     color=GREY,  alpha=0.7, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(rels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean Cosine Similarity'); ax.set_title('Relation Connectivity Preservation')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.4, zorder=0)
    ax.axhline(0, color='k', lw=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'[plot] Saved to {save_path}')
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

    print(f"\n  Elapsed: {report['elapsed_s']:.1f}s")
    print(f"{'=' * w}\n")
