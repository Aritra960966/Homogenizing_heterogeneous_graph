import csv
import argparse
import numpy as np
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


def run_experiment(dataset_name, task, out_csv, n_seeds=5):
    print(f"\n{'=' * 60}")
    print(f"  Starting experiment: {dataset_name.upper()}  {task.upper()}")
    print(f"{'=' * 60}")

    data = LOADERS[dataset_name]()

    rows = []

    if task == 'nc':
        best_params, tr80, te20 = hparam_search_nc(data, seed=42)

        macros, micros = [], []
        for seed in tqdm(range(n_seeds), desc="Seeds"):
            r = run_final_nc(data, best_params, tr80, te20, seed=seed)
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
        import scipy.sparse as sp
        A = data['A_list_sp'][0].tocoo()
        target_edges = np.column_stack([A.row, A.col])

        best_params, tr80_edges, te20_edges = hparam_search_lp(
            data, target_edges, seed=42)

        aucs, aps = [], []
        for seed in tqdm(range(n_seeds), desc="Seeds"):
            r = run_final_lp(data, best_params, tr80_edges,
                             te20_edges, seed=seed)
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

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results saved → {out_csv}")
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['dblp', 'acm', 'imdb'],
                        required=True)
    parser.add_argument('--task',    choices=['nc', 'lp'], required=True)
    parser.add_argument('--out',     default='results/logs/results.csv')
    parser.add_argument('--seeds',   type=int, default=5)
    args = parser.parse_args()
    run_experiment(args.dataset, args.task, args.out, n_seeds=args.seeds)
