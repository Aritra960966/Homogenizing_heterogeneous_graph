<<<<<<< HEAD
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
N_SEEDS  = 10

TARGET_REL_IDX = {'dblp': 0, 'acm': 4, 'imdb': 2}

RESULT_DIRS = {
    'nc'  : 'results/nc',
    'lp'  : 'results/lp',
    'cl'  : 'results/clustering',
    'rec' : 'results/recommendation',
}
=======
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
>>>>>>> master


def write_per_run_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists: w.writeheader()
        w.writerows(rows)
    print(f"  Per-run results appended -> {path}")

SKIP_HPARAM_SEARCH = False  # set True to reuse existing best_params.json

<<<<<<< HEAD
def write_summary_csv(summary_row, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_row.keys())
        if not file_exists: w.writeheader()
        w.writerow(summary_row)
    print(f"  Summary appended -> {path}")


def _flatten_result(r):
    out = {}
    for k, v in r.items():
        if isinstance(v, np.ndarray):
            for i, val in enumerate(v.flatten()):
                out[f"{k}_{i}"] = round(float(val), 6)
        elif isinstance(v, (float, int, np.floating, np.integer)):
            out[k] = round(float(v), 6)
        else:
            out[k] = v
    return out


def run_nc(dataset_name, out_dir):
=======

def run_experiment(dataset_name, task, out_root, n_seeds=5):
    print(f"\n{'=' * 60}")
    print(f"  Starting experiment: {dataset_name.upper()}  {task.upper()}")
    print(f"{'=' * 60}")

    out_dir = Path(out_root) / task / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

>>>>>>> master
    data = LOADERS[dataset_name]()
    print(f"\n{'='*60}\n  {dataset_name.upper()} - Node Classification\n{'='*60}")

<<<<<<< HEAD
    best_params, tr80, te20 = hparam_search_nc(data, seed=42, out_dir=out_dir)

    per_run_rows, macros, micros, accs = [], [], [], []

    for seed in range(N_SEEDS):
        r = run_final_nc(data, best_params, tr80, te20, seed=seed)
=======
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
>>>>>>> master

        macros.append(r['test_macro'])
        micros.append(r['test_micro'])
        accs.append(r['test_acc'])

<<<<<<< HEAD
        row = {'dataset': dataset_name, 'task': 'nc', 'seed': seed,
               'test_macro_f1'  : round(r['test_macro'], 4),
               'test_micro_f1'  : round(r['test_micro'], 4),
               'test_accuracy'  : round(r['test_acc'],   4),
               'time_sec'       : round(r['time_sec'],   2),
               **{f'hp_{k}': v for k, v in best_params.items()}}
        per_run_rows.append(_flatten_result({**row}))

    write_per_run_csv(per_run_rows, os.path.join(out_dir, 'per_run_results.csv'))

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
=======
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
>>>>>>> master

    print(f"\n  {dataset_name} NC  (n={N_SEEDS} seeds)")
    print(f"  Macro-F1 : {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_sd']:.4f}")
    print(f"  Micro-F1 : {summary['micro_f1_mean']:.4f} +/- {summary['micro_f1_sd']:.4f}")
    print(f"  Accuracy : {summary['accuracy_mean']:.4f} +/- {summary['accuracy_sd']:.4f}")

<<<<<<< HEAD
=======
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
>>>>>>> master

def run_lp(dataset_name, out_dir):
    import scipy.sparse as sp
    data   = LOADERS[dataset_name]()
    A      = data['A_list_sp'][TARGET_REL_IDX[dataset_name]].tocoo()
    target_edges = np.column_stack([A.row, A.col])

    print(f"\n{'='*60}\n  {dataset_name.upper()} - Link Prediction\n{'='*60}")
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
    print(f"  AUC : {summary['auc_mean']:.4f} +/- {summary['auc_sd']:.4f}")
    print(f"  AP  : {summary['ap_mean']:.4f} +/- {summary['ap_sd']:.4f}")


def run_cl(dataset_name, out_dir):
    data = LOADERS[dataset_name]()
    print(f"\n{'='*60}\n  {dataset_name.upper()} - Graph Clustering\n{'='*60}")

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
    print(f"  NMI : {summary['nmi_mean']:.4f} +/- {summary['nmi_sd']:.4f}")
    print(f"  ARI : {summary['ari_mean']:.4f} +/- {summary['ari_sd']:.4f}")
    print(f"  ACC : {summary['acc_mean']:.4f} +/- {summary['acc_sd']:.4f}")


def run_rec(dataset_name, out_dir, K_list=(10, 20, 50)):
    import scipy.sparse as sp
    data   = LOADERS[dataset_name]()
    A      = data['A_list_sp'][TARGET_REL_IDX[dataset_name]].tocoo()
    target_edges = np.column_stack([A.row, A.col])

    print(f"\n{'='*60}\n  {dataset_name.upper()} - Recommendation\n{'='*60}")
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
        print(f"  Recall@{K:<3}: {summary[f'recall@{K}_mean']:.4f} +/- {summary[f'recall@{K}_sd']:.4f}"
              f"   NDCG@{K}: {summary[f'ndcg@{K}_mean']:.4f} +/- {summary[f'ndcg@{K}_sd']:.4f}")


TASK_FNS = {'nc': run_nc, 'lp': run_lp, 'cl': run_cl, 'rec': run_rec}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument('--dataset', choices=['dblp','acm','imdb'], required=True)
    parser.add_argument('--task',    choices=['nc','lp','cl','rec'], required=True)
    parser.add_argument('--seeds',   type=int, default=N_SEEDS)
    args = parser.parse_args()

    N_SEEDS = args.seeds
    out_dir = RESULT_DIRS[args.task]
    TASK_FNS[args.task](args.dataset, out_dir)
=======
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
>>>>>>> master
