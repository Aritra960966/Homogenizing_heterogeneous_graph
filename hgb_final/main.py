"""
hgb_final — RAHGH using official HGB train/val/test splits.

Injects pre-computed HGB split indices into the data dict,
then delegates to RAHGH's task runners (NC, LP, CL, Rec).
"""
import sys, os, json, time, argparse
import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJ        = os.path.dirname(_HERE)
_RAHGH       = os.path.join(_PROJ, 'RAHGH')
_HGB_SPLITS  = os.path.join(_PROJ, 'HGB', 'splits')
_RESULTS     = os.path.join(_HERE, 'results')

sys.path.insert(0, _RAHGH)

from src.utils.env import setup_training_env, print_env_summary
from src.train import run_nc, run_lp, run_cl, run_rec, _get_loader, N_SEEDS

TARGET_REL_IDX = {'dblp': 0, 'acm': 0, 'imdb': 2, 'lastfm': 0, 'amazon': 0,
                  'amazon_ini': 0, 'pubmed': 4, 'pubmed_ini': 4}

# ── Dataset → tasks supported (with HGB splits available) ──────────────────
DATASET_TASKS = {
    'dblp'        : ['nc', 'lp', 'cl', 'rec'],
    'acm'         : ['nc', 'cl'],
    'imdb'        : ['nc', 'lp', 'cl'],
    'pubmed'      : ['nc', 'lp', 'cl'],
    'pubmed_ini'  : ['nc', 'lp', 'cl'],
    'freebase'    : ['lp'],
    'freebase_no_name': ['lp'],
    'amazon'      : ['lp', 'rec'],
    'amazon_ini'  : ['lp', 'rec'],
    'lastfm'      : ['lp', 'rec'],
    'lastfm_ini'  : ['lp', 'rec'],
    'lastfm_magnn': ['lp', 'rec'],
    'youtube'     : ['cl'],
}

TASK_DIR_NAMES = {'nc': 'nc', 'lp': 'lp', 'cl': 'clustering', 'rec': 'recommendation'}
TASK_FNS       = {'nc': run_nc, 'lp': run_lp, 'cl': run_cl, 'rec': run_rec}


def inject_hgb_splits(data, dataset_name, task):
    """Inject HGB split indices into the data dict for the given task."""
    split_dir = os.path.join(_HGB_SPLITS, TASK_DIR_NAMES[task], dataset_name)
    if not os.path.isdir(split_dir):
        print(f"  [WARN] No HGB splits found at {split_dir}")
        return data

    if task == 'nc':
        data['train_indices'] = torch.from_numpy(
            np.load(os.path.join(split_dir, 'train_indices.npy')))
        data['val_indices']   = torch.from_numpy(
            np.load(os.path.join(split_dir, 'val_indices.npy')))
        data['test_indices']  = torch.from_numpy(
            np.load(os.path.join(split_dir, 'test_indices.npy')))
        info_path = os.path.join(split_dir, 'info.json')
        if os.path.exists(info_path):
            with open(info_path) as f:
                data['split_info'] = json.load(f)
        print(f"  [HGB NC split] {dataset_name}: "
              f"{len(data['train_indices'])} tr / {len(data['val_indices'])} va / "
              f"{len(data['test_indices'])} te")

    elif task == 'cl':
        data['train_indices'] = torch.from_numpy(
            np.load(os.path.join(split_dir, 'train_indices.npy')))
        data['val_indices']   = torch.from_numpy(
            np.load(os.path.join(split_dir, 'val_indices.npy')))
        data['test_indices']  = torch.from_numpy(
            np.load(os.path.join(split_dir, 'test_indices.npy')))
        info_path = os.path.join(split_dir, 'info.json')
        if os.path.exists(info_path):
            with open(info_path) as f:
                data['split_info'] = json.load(f)
        print(f"  [HGB CL split] {dataset_name}: "
              f"{len(data['train_indices'])} tr / {len(data['val_indices'])} va / "
              f"{len(data['test_indices'])} te")

    elif task in ('lp', 'rec'):
        data['train_edge_indices'] = np.load(
            os.path.join(split_dir, 'train_edge_indices.npy'))
        data['test_edge_indices']  = np.load(
            os.path.join(split_dir, 'test_edge_indices.npy'))
        data['target_edges']       = np.load(
            os.path.join(split_dir, 'target_edges.npy'))

        tr_edges = data['target_edges'][data['train_edge_indices']]
        te_edges = data['target_edges'][data['test_edge_indices']]

        # HGB-compatible val edges for LP (9% of total)
        if task == 'lp':
            val_path = os.path.join(split_dir, 'val_edge_indices.npy')
            if os.path.exists(val_path):
                data['val_edge_indices'] = np.load(val_path)
                va_edges = data['target_edges'][data['val_edge_indices']]
                data['val_edges'] = va_edges
                print(f"  [HGB LP split] {dataset_name}: "
                      f"{len(tr_edges)} tr / {len(va_edges)} va / {len(te_edges)} te")
            else:
                print(f"  [HGB LP split] {dataset_name}: "
                      f"{len(tr_edges)} tr / {len(te_edges)} te (no val)")
        else:
            print(f"  [HGB REC split] {dataset_name}: "
                  f"{len(tr_edges)} tr / {len(te_edges)} te")

        # Store target relation index (needed by LP/Rec runners)
        data['target_relation_idx'] = TARGET_REL_IDX.get(dataset_name, 0)

        # Store the full edges for the hparam_search edge split
        data['train_edges'] = tr_edges
        data['test_edges']  = te_edges

    return data


def run_task(task, dataset_name, out_dir, head='gcn', seeds=N_SEEDS):
    """Load data, inject HGB splits, and run the task."""
    print(f"\n{'='*70}")
    print(f"  Loading {dataset_name} for {task}...")
    print(f"{'='*70}")

    # RAHGH loaders use relative paths from RAHGH/ directory
    old_cwd = os.getcwd()
    os.chdir(_RAHGH)

    # Load data once and inject HGB splits
    data = _get_loader(dataset_name)
    data['name'] = dataset_name
    data = inject_hgb_splits(data, dataset_name, task)

    # Monkey-patch _get_loader so task functions (run_nc, run_lp, etc.)
    # get our pre-loaded data with HGB splits instead of loading fresh data
    import src.train as train_mod
    import src.tasks.hparam_search as hp
    original_get_loader = train_mod._get_loader
    train_mod._get_loader = lambda name: data

    # Quick mode: reduce hparam search to 2 combos, 2 folds, 50 epochs
    if os.environ.get('HGB_QUICK') == '1':
        hp.N_ITER = 2
        hp.N_FOLDS = 2
        for key in hp.PARAM_GRID_BASE:
            if key == 'epochs':
                hp.PARAM_GRID_BASE[key] = [50]
            elif key == 'K':
                hp.PARAM_GRID_BASE[key] = [2]
            elif key in ('d', 'd_prime', 'hidden'):
                hp.PARAM_GRID_BASE[key] = [64]

    original_seeds = train_mod.N_SEEDS
    train_mod.N_SEEDS = seeds

    try:
        TASK_FNS[task](dataset_name, out_dir, head=head)
    finally:
        train_mod._get_loader = original_get_loader
        train_mod.N_SEEDS = original_seeds
        os.chdir(old_cwd)


def main():
    setup_training_env()
    print_env_summary()

    parser = argparse.ArgumentParser(
        description='RAHGH with HGB splits — run experiments')
    parser.add_argument('--dataset', help='Dataset name (default: all supported)')
    parser.add_argument('--task', choices=['nc','lp','cl','rec'],
                        help='Task (default: all supported for dataset)')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--head', choices=['gcn','gat'], default='gcn')
    args = parser.parse_args()

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted(DATASET_TASKS.keys())

    for ds in datasets:
        tasks = DATASET_TASKS.get(ds, ['nc'])
        if args.task:
            tasks = [t for t in tasks if t == args.task]
        if not tasks:
            print(f"  [SKIP] {ds}: no matching tasks")
            continue

        if ds in ('freebase', 'freebase_no_name', 'youtube'):
            data_dir = os.path.join(_PROJ, 'RAHGH', 'data', 'raw', ds)
            if not os.path.isdir(data_dir):
                print(f"  [SKIP] {ds}: data not found")
                continue

        for task in tasks:
            out_dir = os.path.join(_RESULTS, TASK_DIR_NAMES[task], ds)
            os.makedirs(out_dir, exist_ok=True)

            t0 = time.time()
            try:
                run_task(task, ds, out_dir, head=args.head, seeds=args.seeds)
                elapsed = time.time() - t0
                print(f"\n  DONE: {ds} {task}  [{elapsed/60:.1f} min]")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"\n  FAILED: {ds} {task}  —  {e}")

    print(f"\n{'='*70}")
    print(f"  All done. Results in {_RESULTS}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
