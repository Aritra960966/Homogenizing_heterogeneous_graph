"""
Run final LP evaluation using cached best_params.json, skipping HP search.
Usage: python scripts/run_final_lp.py --dataset amazon --seeds 10
"""
import argparse, json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seeds', type=int, default=10)
    args = parser.parse_args()

    N_SEEDS = args.seeds
    dataset = args.dataset
    out_dir = os.path.join('results', 'lp', dataset)

    # Load cached best params
    params_path = os.path.join(out_dir, 'best_params.json')
    if not os.path.exists(params_path):
        print(f"No cached params at {params_path}. Run with --task lp first.")
        sys.exit(1)
    with open(params_path) as f:
        cached = json.load(f)
    key = f"{dataset}_lp"
    if key not in cached:
        # Try old format without dataset prefix
        best_params = cached
    else:
        best_params = cached[key]
    print(f"Using cached best params: {best_params}")

    # Load data
    from src.train import _get_loader
    data = _get_loader(dataset)
    data['name'] = dataset
    rel_idx = data.get('target_relation_idx', 0)
    import scipy.sparse as sp
    A = data['A_list_sp'][rel_idx].tocoo()
    target_edges = np.column_stack([A.row, A.col])

    # Use same split as HP search (seed=42)
    from sklearn.model_selection import train_test_split
    all_idx = np.arange(len(target_edges))
    tr80_idx, te20_idx = train_test_split(all_idx, test_size=0.20, random_state=42)
    tr80_edges = target_edges[tr80_idx]
    te20_edges = target_edges[te20_idx]

    # Run final evaluation for each seed
    from src.tasks.link_prediction import run_final_lp
    from src.train import write_per_run_csv, write_summary_csv, _flatten_result

    K_HITS = [1, 3, 10]
    per_run_rows, aucs, aps, f1s = [], [], [], []
    hits_vals = {f'hits@{k}': [] for k in K_HITS}
    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed+1}/{N_SEEDS}")
        r = run_final_lp(data, best_params, tr80_edges, te20_edges, seed=seed,
                         out_dir=out_dir)
        aucs.append(r['auc']); aps.append(r['ap']); f1s.append(r['f1_macro'])
        for k in K_HITS:
            hits_vals[f'hits@{k}'].append(r[f'hits@{k}'])
        row = {'dataset': dataset, 'task': 'lp', 'seed': seed,
               'test_auc': round(r['auc'], 4), 'test_ap': round(r['ap'], 4),
               'test_macro_f1': round(r['f1_macro'], 4),
               **{f'hits@{k}': round(r[f'hits@{k}'], 4) for k in K_HITS},
               'time_sec': round(r['time_sec'], 2),
               **{f'hp_{k}': v for k, v in best_params.items()}}
        per_run_rows.append(_flatten_result(row))

    write_per_run_csv(per_run_rows, os.path.join(out_dir, 'per_run_results.csv'))

    summary = {
        'dataset': dataset, 'task': 'lp',
        'auc_mean': round(float(np.mean(aucs)), 4),
        'auc_sd': round(float(np.std(aucs)), 4),
        'ap_mean': round(float(np.mean(aps)), 4),
        'ap_sd': round(float(np.std(aps)), 4),
        'macro_f1_mean': round(float(np.mean(f1s)), 4),
        'macro_f1_sd': round(float(np.std(f1s)), 4),
        **{f'hits@{k}_mean': round(float(np.mean(hits_vals[f'hits@{k}'])), 4) for k in K_HITS},
        **{f'hits@{k}_sd': round(float(np.std(hits_vals[f'hits@{k}'])), 4) for k in K_HITS},
        'n_seeds': N_SEEDS,
        **{f'best_hp_{k}': v for k, v in best_params.items()},
    }
    write_summary_csv(summary, os.path.join(out_dir, 'summary.csv'))

    print(f"\n  {dataset} LP  (n={N_SEEDS} seeds)")
    print(f"  AUC     : {summary['auc_mean']:.4f} +/- {summary['auc_sd']:.4f}")
    print(f"  AP      : {summary['ap_mean']:.4f} +/- {summary['ap_sd']:.4f}")
    print(f"  Macro-F1: {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_sd']:.4f}")
    for k in K_HITS:
        print(f"  Hits@{k:<2}: {summary[f'hits@{k}_mean']:.4f} +/- {summary[f'hits@{k}_sd']:.4f}")

if __name__ == '__main__':
    main()
