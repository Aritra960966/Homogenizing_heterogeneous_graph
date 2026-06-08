import csv
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .data.dblp_loader import load_dblp
from .data.acm_loader  import load_acm
from .data.imdb_loader import load_imdb
from .tasks.hparam_search      import hparam_search_nc, hparam_search_lp
from .tasks.node_classification import run_final_nc
from .tasks.link_prediction     import run_final_lp


LOADERS = {'dblp': load_dblp, 'acm': load_acm, 'imdb': load_imdb}

CFG = dict(d=64, gcn_hidden=64, dropout=0.5, lr=0.001, wd=0.001,
           K=3, epochs=300)  # default; overridden by hparam_search

SKIP_HPARAM_SEARCH = False  # set True to reuse existing best_params.json


def run_experiment(dataset_name, task, out_root, n_seeds=5):
    print(f"\n{'=' * 60}")
    print(f"  Starting experiment: {dataset_name.upper()}  {task.upper()}")
    print(f"{'=' * 60}")

    out_dir = Path(out_root) / task / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data = LOADERS[dataset_name]()

    # ── Hyperparameter search ─────────────────────────────────────────────
    best_params_path = out_dir / 'best_params.json'

    if SKIP_HPARAM_SEARCH and best_params_path.exists():
        best_params = json.load(open(best_params_path))
        print(f"  Reusing cached best_params from {best_params_path}")
        cv_scores = None
    else:
        if task == 'nc':
            best_params, tr80, te20, cv_scores, combos_tried = \
                hparam_search_nc(data, seed=42, out_dir=str(out_dir))
        else:
            import scipy.sparse as sp
            A = data['A_list_sp'][0].tocoo()
            target_edges = np.column_stack([A.row, A.col])
            best_params, tr80_edges, te20_edges, cv_scores, combos_tried = \
                hparam_search_lp(data, target_edges, seed=42,
                                 out_dir=str(out_dir))

        # Save cv_scores.csv
        if cv_scores is not None:
            cv_path = out_dir / 'cv_scores.csv'
            with open(cv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=cv_scores[0].keys())
                w.writeheader()
                w.writerows(cv_scores)
            print(f"  CV scores → {cv_path}")

        # Save best_params.json
        json.dump({k: (v.item() if hasattr(v, 'item') else v)
                   for k, v in best_params.items()},
                  open(best_params_path, 'w'), indent=2)
        print(f"  Best params → {best_params_path}")

    # ── Final runs (n_seeds) ──────────────────────────────────────────────
    rows = []

    if task == 'nc':
        macros, micros = [], []
        for seed in tqdm(range(n_seeds), desc="Seeds"):
            r = run_final_nc(data, best_params, tr80, te20, seed=seed,
                             out_dir=str(out_dir))
            macros.append(r['test_macro'])
            micros.append(r['test_micro'])
            rows.append({**r, 'seed': seed, 'dataset': dataset_name,
                         'task': task,
                         **{f'hp_{k}': v
                            for k, v in best_params.items()}})

        print(f"\n{'─' * 60}")
        print(f"  FINAL  {dataset_name} NC")
        print(f"  Macro-F1 : {np.mean(macros):.4f} ± {np.std(macros):.4f}")
        print(f"  Micro-F1 : {np.mean(micros):.4f} ± {np.std(micros):.4f}")
        print(f"  Best params: {best_params}")

    else:
        aucs, aps = [], []
        for seed in tqdm(range(n_seeds), desc="Seeds"):
            r = run_final_lp(data, best_params, tr80_edges,
                             te20_edges, seed=seed,
                             out_dir=str(out_dir))
            aucs.append(r['auc'])
            aps.append(r['ap'])
            rows.append({**r, 'seed': seed, 'dataset': dataset_name,
                         'task': task,
                         **{f'hp_{k}': v
                            for k, v in best_params.items()}})

        print(f"\n{'─' * 60}")
        print(f"  FINAL  {dataset_name} LP")
        print(f"  AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  AP  : {np.mean(aps):.4f} ± {np.std(aps):.4f}")
        print(f"  Best params: {best_params}")

    # ── Save final_runs.csv ───────────────────────────────────────────────
    runs_path = out_dir / 'final_runs.csv'
    with open(runs_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Final runs → {runs_path}")

    # ── Save summary.csv ──────────────────────────────────────────────────
    metric_keys = [k for k in rows[0] if k not in (
        'seed', 'dataset', 'task', 'alpha', 'beta') and not k.startswith('hp_')]
    summary_rows = {}
    for k in metric_keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float, np.number))]
        if vals:
            summary_rows[k + '_mean'] = float(np.mean(vals))
            summary_rows[k + '_std']  = float(np.std(vals))
    summary_path = out_dir / 'summary.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_rows.keys())
        w.writeheader()
        w.writerow(summary_rows)
    print(f"  Summary     → {summary_path}")
    print(f"{'=' * 60}\n")

    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['dblp', 'acm', 'imdb'],
                        required=True)
    parser.add_argument('--task',    choices=['nc', 'lp'], required=True)
    parser.add_argument('--out-dir', default='results')
    parser.add_argument('--seeds',   type=int, default=5)
    parser.add_argument('--skip-hparam', action='store_true',
                        help='Skip hparam search and reuse existing best_params.json')
    args = parser.parse_args()
    if args.skip_hparam:
        import src.train as tr
        tr.SKIP_HPARAM_SEARCH = True
    run_experiment(args.dataset, args.task, args.out_dir, n_seeds=args.seeds)
