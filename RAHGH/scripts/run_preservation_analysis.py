#!/usr/bin/env python3
"""
run_preservation_analysis.py

Standalone script to run homogenization preservation analysis on trained
RAHGH models (ACM, DBLP, IMDB).

Usage:
    # Single dataset
    python RAHGH/scripts/run_preservation_analysis.py --dataset acm --seed 0

    # All three datasets
    python RAHGH/scripts/run_preservation_analysis.py --all

    # Custom checkpoint
    python RAHGH/scripts/run_preservation_analysis.py --dataset dblp --checkpoint RAHGH/results/nc/dblp/final_model_seed42.pt

Output:
    results/analysis/{dataset}_preservation.json   (JSON report)
    results/analysis/{dataset}_preservation.pdf    (4-panel figure)
    results/analysis/{dataset}_latex_table.txt     (LaTeX table)
    Console JSON summary
"""

import argparse
import json
import os
import sys
import warnings
from typing import Optional

import numpy as np
import torch

warnings.filterwarnings('ignore')

# ── Add RAHGH to path ───────────────────────────────────────────────────
RAHGH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(RAHGH_ROOT)
if RAHGH_ROOT not in sys.path:
    sys.path.insert(0, RAHGH_ROOT)

from src.utils.env import setup_training_env, print_env_summary
from src.model.rahgh import (
    RAHGH, build_rahgh_classifier, compile_model,
    build_edge_index_dict, build_node_type_indices,
)
from src.analysis.homogenization_analysis import (
    run_analysis, print_report, plot_report, latex_table,
)

# ── Dataset loader registry ──────────────────────────────────────────────
DATA_ROOT = os.path.join(RAHGH_ROOT, 'data', 'raw')


def _get_nc_dir(dataset: str) -> str:
    """Return path to NC results for a given dataset."""
    return os.path.join(RAHGH_ROOT, 'results', 'nc', dataset)


def _get_loader(dataset: str):
    """Import the correct loader function."""
    if dataset == 'acm':
        from src.data.acm_loader import load_acm
        return load_acm(os.path.join(DATA_ROOT, 'ACM', 'ACM.mat'))
    elif dataset == 'dblp':
        from src.data.dblp_loader import load_dblp
        return load_dblp(os.path.join(DATA_ROOT, 'DBLP'))
    elif dataset == 'imdb':
        from src.data.imdb_loader import load_imdb
        return load_imdb(os.path.join(DATA_ROOT, 'IMDB'))
    elif dataset == 'pubmed':
        from src.data.hgb_unified_loader import load_hgb_unified
        return load_hgb_unified('PubMed', root=DATA_ROOT)
    elif dataset == 'lastfm':
        from src.data.lastfm_loader import load_lastfm
        return load_lastfm(root=DATA_ROOT)
    elif dataset == 'amazon':
        from src.data.lastfm_loader import load_amazon
        return load_amazon(root=DATA_ROOT)
    elif dataset == 'freebase':
        from src.data.freebase_loader import load_freebase
        return load_freebase(root=DATA_ROOT, named=True)
    elif dataset == 'youtube':
        from src.data.hgb_unified_loader import load_hgb_unified
        return load_hgb_unified('youtube', root=DATA_ROOT)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")


def _build_model(data: dict, params: dict, device: torch.device, dataset: str,
                  checkpoint_path: Optional[str] = None):
    """Build a plain RAHGH encoder from data and hyperparameters.
    
    We use the encoder (without classifier head) because the analysis only
    needs intermediate representations from the homogenizer. This avoids
    PyG dependency (GAT head) and head-type mismatch with checkpoints.
    """
    model = RAHGH(
        node_type_dims={k: v.shape[1] for k, v in data['X_dict'].items()},
        relation_info=data.get('relation_info', {}),
        num_nodes=data['N'],
        hidden_dim=params['d'],
        output_dim=params['d'],
        K=params['K'],
        dropout=params['dropout'],
        directed=False,
    ).to(device)
    print(f"  Building RAHGH encoder (no classifier head), d={params['d']}, K={params['K']}")
    return compile_model(model)


def _load_best_params(nc_dir: str, dataset: str) -> dict:
    """Load hyperparameters, preferring per_run_results.csv over best_params.json.
    
    best_params.json may be stale — the actual trained model may have used
    different HPs (e.g., from cross-validation). per_run_results.csv has
    the exact hp_* values used for each seed.
    """
    per_run_csv = os.path.join(nc_dir, 'per_run_results.csv')
    if os.path.exists(per_run_csv):
        import csv
        with open(per_run_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                hp = {}
                for k, v in row.items():
                    if k.startswith('hp_'):
                        try:
                            hp[k[3:]] = int(float(v)) if '.' in v and v.endswith('.0') else float(v)
                        except ValueError:
                            hp[k[3:]] = v
                if hp:
                    # Cast int params
                    for int_key in ['d', 'K', 'epochs', 'hidden', 'warmup']:
                        if int_key in hp:
                            hp[int_key] = int(hp[int_key])
                    return hp

    # Fallback to best_params.json
    params_path = os.path.join(nc_dir, 'best_params.json')
    with open(params_path) as f:
        raw = json.load(f)
    params = raw.get(f'{dataset}_nc', raw)
    return params


def run_single(
    dataset: str,
    checkpoint_path: Optional[str] = None,
    seed: int = 0,
    n_sample: int = 2000,
    n_spectral: int = 30,
    device: torch.device = torch.device('cpu'),
    out_dir: Optional[str] = None,
    skip_plot: bool = False,
) -> dict:
    if out_dir is None:
        out_dir = os.path.join(RAHGH_ROOT, 'results', 'analysis')
    """
    Load a dataset, build the model, load checkpoint, run analysis.

    Returns the report dict.
    """
    print(f"\n{'=' * 60}")
    print(f"  Loading {dataset.upper()} dataset ...")
    print(f"{'=' * 60}")

    data = _get_loader(dataset)
    data['name'] = dataset
    Nt = data['target_size']
    nc = data['n_classes']
    print(f"  N={data['N']:,}  target_type={data['target_type']}  "
          f"N_target={Nt:,}  n_classes={nc}")

    # Resolve checkpoint path
    nc_dir = _get_nc_dir(dataset)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(nc_dir, f'final_model_seed{seed}.pt')
    if not os.path.exists(checkpoint_path):
        alt_seeds = [s for s in [0, 42, 84, 126, 168, 210, 252, 294, 336, 378, 420]
                     if os.path.exists(os.path.join(nc_dir, f'final_model_seed{s}.pt'))]
        if alt_seeds:
            alt_path = os.path.join(nc_dir, f'final_model_seed{alt_seeds[0]}.pt')
            print(f"  [warn] {checkpoint_path} not found. Using {alt_path}")
            checkpoint_path = alt_path
        else:
            print(f"  [warn] No checkpoint found at {nc_dir} — using random weights")

    # Infer hyperparameters from checkpoint (dimensions are authoritative)
    if os.path.exists(checkpoint_path):
        full_state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        homo_state = {k.replace('homogenizer.', ''): v
                      for k, v in full_state.items() if k.startswith('homogenizer.')}
        # Infer d from projector output dim
        sample_proj_key = [k for k in homo_state if 'projector.projections.' in k and 'weight' in k][0]
        d = homo_state[sample_proj_key].shape[0]  # output dim of Linear
        # Infer K from any diffuser.psi entry
        psi_keys = [k for k in homo_state if k.startswith('diffuser.psi.')]
        K = homo_state[psi_keys[0]].shape[0] - 1 if psi_keys else 3
        # Load remaining params from best_params.json (for dropout, lr, etc.)
        params = _load_best_params(nc_dir, dataset)
        params['d'] = d
        params['K'] = K
        print(f"  Inferred from checkpoint: d={d}, K={K}")
    else:
        params = _load_best_params(nc_dir, dataset)
    print(f"  Params: d={params['d']}, K={params['K']}, "
          f"dropout={params['dropout']}, lr={params['lr']}")

    # Build model with the correct dimensions
    model = _build_model(data, params, device, dataset, checkpoint_path)
    model.eval()

    if os.path.exists(checkpoint_path):
        full_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        homo_state = {k.replace('homogenizer.', ''): v
                      for k, v in full_state.items() if k.startswith('homogenizer.')}

        # Handle projection format mismatch: checkpoint may have plain Linear
        # ('.weight') vs Sequential ('.0.weight', '.1.weight')
        adapted = {}
        for k, v in homo_state.items():
            if k in model.state_dict():
                adapted[k] = v
            elif '.projections.' in k and k.endswith('.weight'):
                # Checkpoint uses 'projections.movie.weight' but model expects
                # 'projections.movie.0.weight' (Sequential wrapping)
                base = k.replace('.weight', '')
                k0 = f'{base}.0.weight'
                k1 = f'{base}.0.bias'
                k2 = f'{base}.1.weight'
                k3 = f'{base}.1.bias'
                if k0 in model.state_dict() and k in homo_state:
                    adapted[k0] = v
                if k1 in model.state_dict() and base + '.bias' in homo_state:
                    adapted[k1] = homo_state[base + '.bias']
                # Add LayerNorm params (default identity if not in checkpoint)
                if k2 in model.state_dict():
                    adapted[k2] = torch.ones_like(model.state_dict()[k2])
                if k3 in model.state_dict():
                    adapted[k3] = torch.zeros_like(model.state_dict()[k3])

        # Fill in any other straight mappings
        for k, v in homo_state.items():
            if k not in adapted and k in model.state_dict():
                adapted[k] = v

        missing, unexpected = model.load_state_dict(adapted, strict=False)
        if missing:
            print(f"  Missing keys in state dict: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        print(f"  Homogenizer weights loaded successfully.")
    else:
        print(f"  [warn] Checkpoint not available — analysis with random weights")

    # Run analysis
    report = run_analysis(
        model=model,
        data=data,
        device=device,
        n_sample=n_sample,
        k_list=(5, 10, 20),
        n_spectral=n_spectral,
        seed=seed,
    )

    # Override dataset name
    report['dataset'] = dataset

    return report


def save_results(report: dict, out_dir: str, dataset: str, skip_plot: bool = False):
    """Save report as JSON, figure PDF, and LaTeX table."""
    os.makedirs(out_dir, exist_ok=True)

    # JSON report (sanitize NaNs)
    def sanitize(v):
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return None
            return round(v, 6)
        if isinstance(v, dict):
            return {k: sanitize(v) for k, v in v.items()}
        if isinstance(v, list):
            return [sanitize(x) for x in v]
        return v

    json_path = os.path.join(out_dir, f'{dataset}_preservation.json')
    with open(json_path, 'w') as f:
        json.dump(sanitize(report), f, indent=2)
    print(f"  Report  -> {json_path}")

    # LaTeX table
    latex = latex_table(report)
    tex_path = os.path.join(out_dir, f'{dataset}_latex_table.txt')
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(f"  LaTeX   -> {tex_path}")

    # Plot
    if not skip_plot:
        try:
            pdf_path = os.path.join(out_dir, f'{dataset}_preservation.pdf')
            plot_report(report, save_path=pdf_path)
        except Exception as e:
            print(f"  [plot] Failed: {e}")


def run_all(
    out_dir: str = None,
    n_sample: int = 2000,
    n_spectral: int = 30,
    skip_plot: bool = False,
    device: torch.device = torch.device('cpu'),
):
    if out_dir is None:
        out_dir = os.path.join(RAHGH_ROOT, 'results', 'analysis')
    """Run preservation analysis for all three NC datasets."""
    all_reports = {}
    for dataset in ['acm', 'dblp', 'imdb']:
        print(f"\n{'#' * 60}")
        print(f"  {dataset.upper()}")
        print(f"{'#' * 60}")
        try:
            report = run_single(
                dataset=dataset,
                checkpoint_path=None,
                seed=0,
                n_sample=n_sample,
                n_spectral=n_spectral,
                device=device,
                out_dir=out_dir,
                skip_plot=skip_plot,
            )
            print_report(report)
            save_results(report, out_dir, dataset, skip_plot)
            all_reports[dataset] = report
        except Exception as e:
            print(f"  [error] {dataset} failed: {e}")
            import traceback
            traceback.print_exc()

    # Print combined LaTeX tables
    print("\n\n% ── Combined LaTeX Tables ──────────────────────────────────")
    for ds, rep in all_reports.items():
        print(f"\n% {ds.upper()}")
        print(latex_table(rep))

    # Combined JSON
    comb_path = os.path.join(out_dir, 'all_preservation.json')
    with open(comb_path, 'w') as f:
        json.dump({ds: sanitize(rep) for ds, rep in all_reports.items()}, f, indent=2)
    print(f"\n  Combined -> {comb_path}")

    return all_reports


def sanitize(v):
    if isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(v, 6)
    if isinstance(v, dict):
        return {k: sanitize(v) for k, v in v.items()}
    if isinstance(v, list):
        return [sanitize(x) for x in v]
    return v


def main():
    default_out = os.path.join(RAHGH_ROOT, 'results', 'analysis')
    parser = argparse.ArgumentParser(
        description='RAHGH Homogenization Preservation Analysis'
    )
    parser.add_argument('--dataset', type=str, default='acm',
                        choices=['acm', 'dblp', 'imdb', 'pubmed', 'lastfm',
                                 'amazon', 'freebase', 'youtube'],
                        help='Dataset to analyze (default: acm)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint .pt file')
    parser.add_argument('--seed', type=int, default=0,
                        help='Checkpoint seed (default: 0)')
    parser.add_argument('--n-sample', type=int, default=2000,
                        help='Nodes to sample for embedding metrics (default: 2000)')
    parser.add_argument('--n-spectral', type=int, default=30,
                        help='Eigenvalues for spectral similarity (default: 30)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help=f'Output directory (default: {default_out})')
    parser.add_argument('--all', action='store_true',
                        help='Run on all datasets (acm, dblp, imdb)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if CUDA is available')
    parser.add_argument('--skip-plot', action='store_true',
                        help='Skip PDF figure generation')
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = default_out

    setup_training_env()
    print_env_summary()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"  Device: {device}\n")

    if args.all:
        run_all(
            out_dir=args.out_dir,
            n_sample=args.n_sample,
            n_spectral=args.n_spectral,
            skip_plot=args.skip_plot,
            device=device,
        )
    else:
        report = run_single(
            dataset=args.dataset,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            n_sample=args.n_sample,
            n_spectral=args.n_spectral,
            device=device,
            out_dir=args.out_dir,
            skip_plot=args.skip_plot,
        )
        print_report(report)
        save_results(report, args.out_dir, args.dataset, args.skip_plot)


if __name__ == '__main__':
    main()
