import csv, os, json, argparse
import numpy as np

from .data.dblp_loader import load_dblp
from .data.acm_loader  import load_acm
from .data.imdb_loader import load_imdb

from .tasks.hparam_search       import (hparam_search_nc, hparam_search_lp,
                                         hparam_search_cl, hparam_search_rec)
from .tasks.node_classification import run_final_nc
from .tasks.link_prediction     import run_final_lp
from .tasks.graph_clustering    import run_final_clustering
from .tasks.recommendation      import run_final_recommendation

LOADERS  = {'dblp': load_dblp, 'acm': load_acm, 'imdb': load_imdb}
N_SEEDS  = 10

TARGET_REL_IDX = {'dblp': 0, 'acm': 4, 'imdb': 2}

RESULT_DIRS = {
    'nc'  : 'results/nc',
    'lp'  : 'results/lp',
    'cl'  : 'results/clustering',
    'rec' : 'results/recommendation',
}


def write_per_run_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists: w.writeheader()
        w.writerows(rows)
    print(f"  Per-run results → {path}")


def write_summary_csv(summary_row, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_row.keys())
        if not file_exists: w.writeheader()
        w.writerow(summary_row)
    print(f"  Summary → {path}")


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


def run_nc(dataset_name, out_dir, head='gcn'):
    data = LOADERS[dataset_name]()
    data['name'] = dataset_name
    print(f"\n{'='*60}\n  {dataset_name.upper()} — Node Classification\n{'='*60}")

    best_params, tr80, te20 = hparam_search_nc(data, seed=42, out_dir=out_dir,
                                                head=head)

    per_run_rows, macros, micros, accs = [], [], [], []
    for seed in range(N_SEEDS):
        r = run_final_nc(data, best_params, tr80, te20, seed=seed,
                         out_dir=out_dir, head=head)
        macros.append(r['test_macro'])
        micros.append(r['test_micro'])
        accs.append(r['test_acc'])

        row = {'dataset': dataset_name, 'task': 'nc', 'seed': seed,
               'test_macro_f1': round(r['test_macro'], 4),
               'test_micro_f1': round(r['test_micro'], 4),
               'test_accuracy': round(r['test_acc'],   4),
               'time_sec'     : round(r['time_sec'],   2),
               **{f'hp_{k}': v for k, v in best_params.items()}}
        per_run_rows.append(_flatten_result(row))

    write_per_run_csv(per_run_rows, os.path.join(out_dir, 'per_run_results.csv'))

    summary = {
        'dataset'       : dataset_name, 'task': 'nc',
        'macro_f1_mean' : round(float(np.mean(macros)), 4),
        'macro_f1_sd'   : round(float(np.std(macros)),  4),
        'micro_f1_mean' : round(float(np.mean(micros)), 4),
        'micro_f1_sd'   : round(float(np.std(micros)),  4),
        'accuracy_mean' : round(float(np.mean(accs)),   4),
        'accuracy_sd'   : round(float(np.std(accs)),    4),
        'n_seeds'       : N_SEEDS,
        **{f'best_hp_{k}': v for k, v in best_params.items()},
    }
    write_summary_csv(summary, os.path.join(out_dir, 'summary.csv'))

    print(f"\n  {dataset_name} NC  (n={N_SEEDS} seeds)")
    print(f"  Macro-F1 : {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_sd']:.4f}")
    print(f"  Micro-F1 : {summary['micro_f1_mean']:.4f} ± {summary['micro_f1_sd']:.4f}")
    print(f"  Accuracy : {summary['accuracy_mean']:.4f} ± {summary['accuracy_sd']:.4f}")


def run_lp(dataset_name, out_dir, head='gcn'):
    import scipy.sparse as sp
    data   = LOADERS[dataset_name]()
    data['name'] = dataset_name
    A      = data['A_list_sp'][TARGET_REL_IDX[dataset_name]].tocoo()
    target_edges = np.column_stack([A.row, A.col])

    print(f"\n{'='*60}\n  {dataset_name.upper()} — Link Prediction\n{'='*60}")
    best_params, tr80_edges, te20_edges = hparam_search_lp(
        data, target_edges, seed=42, out_dir=out_dir, head=head)

    per_run_rows, aucs, aps = [], [], []
    for seed in range(N_SEEDS):
        r = run_final_lp(data, best_params, tr80_edges, te20_edges, seed=seed,
                         out_dir=out_dir, head=head)
        aucs.append(r['auc']); aps.append(r['ap'])
        row = {'dataset': dataset_name, 'task': 'lp', 'seed': seed,
               'test_auc': round(r['auc'], 4),
               'test_ap' : round(r['ap'],  4),
               'time_sec': round(r['time_sec'], 2),
               **{f'hp_{k}': v for k, v in best_params.items()}}
        per_run_rows.append(_flatten_result(row))

    write_per_run_csv(per_run_rows, os.path.join(out_dir, 'per_run_results.csv'))

    summary = {
        'dataset'  : dataset_name, 'task': 'lp',
        'auc_mean' : round(float(np.mean(aucs)), 4),
        'auc_sd'   : round(float(np.std(aucs)),  4),
        'ap_mean'  : round(float(np.mean(aps)),  4),
        'ap_sd'    : round(float(np.std(aps)),   4),
        'n_seeds'  : N_SEEDS,
        **{f'best_hp_{k}': v for k, v in best_params.items()},
    }
    write_summary_csv(summary, os.path.join(out_dir, 'summary.csv'))

    print(f"\n  {dataset_name} LP  (n={N_SEEDS} seeds)")
    print(f"  AUC : {summary['auc_mean']:.4f} ± {summary['auc_sd']:.4f}")
    print(f"  AP  : {summary['ap_mean']:.4f} ± {summary['ap_sd']:.4f}")


def run_cl(dataset_name, out_dir, head='gcn'):
    data = LOADERS[dataset_name]()
    data['name'] = dataset_name
    print(f"\n{'='*60}\n  {dataset_name.upper()} — Graph Clustering\n{'='*60}")

    best_params, tr80, te20 = hparam_search_cl(data, seed=42, out_dir=out_dir,
                                                head=head)

    per_run_rows, nmis, aris, accs = [], [], [], []
    for seed in range(N_SEEDS):
        r = run_final_clustering(data, best_params, tr80, te20,
                                  seed=seed, out_dir=out_dir,
                                  head=head)
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


def run_rec(dataset_name, out_dir, head='gcn', K_list=(10, 20, 50)):
    import scipy.sparse as sp
    data   = LOADERS[dataset_name]()
    data['name'] = dataset_name
    A      = data['A_list_sp'][TARGET_REL_IDX[dataset_name]].tocoo()
    target_edges = np.column_stack([A.row, A.col])

    print(f"\n{'='*60}\n  {dataset_name.upper()} — Recommendation\n{'='*60}")
    best_params, tr80_edges, te20_edges = hparam_search_rec(
        data, target_edges, seed=42, out_dir=out_dir, head=head)

    K_list       = list(K_list)
    per_run_rows = []
    metric_vals  = {f'{m}@{K}': [] for K in K_list
                    for m in ['recall','ndcg','hit','precision','mrr']}

    for seed in range(N_SEEDS):
        r = run_final_recommendation(
            data, best_params, tr80_edges, te20_edges,
            target_relation_idx=TARGET_REL_IDX[dataset_name],
            K_list=K_list, seed=seed, out_dir=out_dir,
            head=head)

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
        r_mean = summary.get(f'recall@{K}_mean', 0)
        r_sd   = summary.get(f'recall@{K}_sd', 0)
        n_mean = summary.get(f'ndcg@{K}_mean', 0)
        n_sd   = summary.get(f'ndcg@{K}_sd', 0)
        print(f"  Recall@{K:<3}: {r_mean:.4f} ± {r_sd:.4f}"
              f"   NDCG@{K}: {n_mean:.4f} ± {n_sd:.4f}")


TASK_FNS = {'nc': run_nc, 'lp': run_lp, 'cl': run_cl, 'rec': run_rec}

def run_with_head(fn, dataset_name, out_dir, head):
    """Wrap a runner function with head parameter."""
    return fn(dataset_name, out_dir, head=head)


if __name__ == '__main__':
    from .utils.env import setup_training_env, print_env_summary
    setup_training_env()
    print_env_summary()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['dblp','acm','imdb'], required=True)
    parser.add_argument('--task',    choices=['nc','lp','cl','rec'], required=True)
    parser.add_argument('--seeds',   type=int, default=N_SEEDS)
    parser.add_argument('--head', choices=['gcn','gat'], default='gcn',
                        help="GNN head: gcn (default) or gat")
    args = parser.parse_args()

    N_SEEDS = args.seeds
    out_dir = RESULT_DIRS[args.task]
    TASK_FNS[args.task](args.dataset, out_dir, head=args.head)
