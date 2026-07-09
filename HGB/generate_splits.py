"""
HGB-compatible split generator — official HGB benchmark ratios.

  Task    | Train | Val  | Test
  --------+-------+------+-----
  NC      |  24%  |  6%  |  70%   (stratified by label)
  LP      |  81%  |  9%  |  10%   (edge-level)
  Cluster |  24%  |  6%  |  70%   (stratified by label)
  Rec     |  80%  |  —   |  20%   (edge-level, no val)

Usage:
    python generate_splits.py                           # all
    python generate_splits.py --dataset dblp --task nc  # single
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.join(_HERE, '..')
sys.path.insert(0, _PROJ)

# Data loaders default to data/raw/DATASET relative to cwd; set cwd to RAHGH/
_RAHGH = os.path.join(_PROJ, 'RAHGH')
if os.path.isdir(_RAHGH):
    os.chdir(_RAHGH)

import argparse, json, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

SPLIT_DIR = Path(__file__).parent / 'splits'
SEED      = 42

# Official HGB split ratios
NC_SPLIT  = (0.24, 0.06, 0.70)   # train, val, test
LP_SPLIT  = (0.81, 0.09, 0.10)
REC_SPLIT = (0.80, 0.20)          # train, test  (no val)

TARGET_REL_IDX = {'dblp': 0, 'acm': 0, 'imdb': 2, 'lastfm': 0, 'amazon': 0,
                  'amazon_ini': 0, 'pubmed': 4, 'pubmed_ini': 4}

DATASET_TASKS = {
    'dblp'   : ['nc', 'lp', 'cl', 'rec'],
    'acm'    : ['nc', 'cl'],
    'imdb'   : ['nc', 'lp', 'cl'],
    'pubmed' : ['nc', 'lp', 'cl'],
    'pubmed_ini': ['nc', 'lp', 'cl'],
    'freebase': ['lp'],
    'freebase_no_name': ['lp'],
    'amazon'  : ['lp', 'rec'],
    'amazon_ini': ['lp', 'rec'],
    'lastfm'  : ['lp', 'rec'],
    'lastfm_ini': ['lp', 'rec'],
    'lastfm_magnn': ['lp', 'rec'],
    'youtube' : ['cl'],
}

# Datasets not yet downloaded
_SKIP_IF_MISSING = {'freebase', 'freebase_no_name', 'youtube'}

# Datasets that aren't downloaded yet
_SKIP_IF_MISSING = {'freebase', 'freebase_no_name', 'youtube'}


def _get_loader(dataset_name):
    lazy = {
        'dblp'   : lambda: __import__('RAHGH.src.data.dblp_loader',   fromlist=['']).load_dblp(),
        'acm'    : lambda: __import__('RAHGH.src.data.acm_loader',    fromlist=['']).load_acm(),
        'imdb'   : lambda: __import__('RAHGH.src.data.imdb_loader',   fromlist=['']).load_imdb(),
        'lastfm' : lambda: __import__('RAHGH.src.data.lastfm_loader', fromlist=['']).load_lastfm(),
        'lastfm_ini'  : lambda: __import__('RAHGH.src.data.lastfm_loader', fromlist=['']).load_lastfm(),
        'lastfm_magnn': lambda: __import__('RAHGH.src.data.lastfm_loader', fromlist=['']).load_lastfm(),
        'amazon'      : lambda: __import__('RAHGH.src.data.amazon_loader', fromlist=['']).load_amazon(),
        'amazon_ini'  : lambda: __import__('RAHGH.src.data.amazon_loader', fromlist=['']).load_amazon_ini(),
        'pubmed'      : lambda: __import__('RAHGH.src.data.pubmed_loader', fromlist=['']).load_pubmed(),
        'pubmed_ini'  : lambda: __import__('RAHGH.src.data.pubmed_loader', fromlist=['']).load_pubmed_ini(),
        'youtube'     : lambda: __import__('RAHGH.src.data.hgb_unified_loader',
                            fromlist=['']).load_hgb_unified('youtube', root='RAHGH/data/raw'),
        'freebase'    : lambda: __import__('RAHGH.src.data.freebase_loader',
                            fromlist=['']).load_freebase(root='RAHGH/data/raw', named=True),
        'freebase_no_name': lambda: __import__('RAHGH.src.data.freebase_loader',
                            fromlist=['']).load_freebase(root='RAHGH/data/raw', named=False),
    }
    return lazy.get(dataset_name)()


def _stratified_three_way(idx, labels, tr_frac, va_frac, te_frac, seed):
    """Split idx into train/val/test with label stratification."""
    tr_idx, te_idx = train_test_split(
        idx, test_size=te_frac, random_state=seed, stratify=labels)
    rest_idx = np.setdiff1d(idx, te_idx)
    rest_labels = labels[rest_idx]
    va_size = int(va_frac / (1 - te_frac) * len(rest_idx) + 0.5)
    tr_idx2, va_idx2 = train_test_split(
        rest_idx, test_size=va_size, random_state=seed + 1, stratify=rest_labels)
    return tr_idx2, va_idx2, te_idx


def _edge_three_way(idx, tr_frac, va_frac, te_frac, seed):
    """Split edge indices into train/val/test."""
    n = len(idx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(n * tr_frac)
    n_va = int(n * va_frac)
    return idx[perm[:n_tr]], idx[perm[n_tr:n_tr + n_va]], idx[perm[n_tr + n_va:]]


def generate_nc_splits(dataset_name, data):
    """HGB NC: 24% train, 6% val, 70% test (stratified)."""
    out = SPLIT_DIR / 'nc' / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    labels = data['labels'].numpy()
    Nt = data['target_size']
    idx = np.arange(Nt)

    tr, va, te = _stratified_three_way(idx, labels, *NC_SPLIT, seed=SEED)

    np.save(out / 'train_indices.npy', tr)
    np.save(out / 'val_indices.npy',   va)
    np.save(out / 'test_indices.npy',  te)
    np.save(out / 'labels.npy',        labels)

    info = {
        'dataset': dataset_name, 'task': 'nc',
        'n_total': Nt,
        'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'tr_frac': NC_SPLIT[0], 'va_frac': NC_SPLIT[1], 'te_frac': NC_SPLIT[2],
        'n_classes': int(data['n_classes']),
        'seed': SEED,
    }
    with open(out / 'info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"  [NC] {dataset_name}: {len(tr)} tr / {len(va)} va / {len(te)} te  ({info['n_classes']} classes)")
    return info


def generate_lp_splits(dataset_name, data):
    """HGB LP: 81% train, 9% val, 10% test (edge-level)."""
    out = SPLIT_DIR / 'lp' / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    rel_idx = data.get('target_relation_idx', TARGET_REL_IDX.get(dataset_name, 0))
    A = data['A_list_sp'][rel_idx].tocoo()
    target_edges = np.column_stack([A.row, A.col])

    all_idx = np.arange(len(target_edges))
    tr_idx, va_idx, te_idx = _edge_three_way(all_idx, *LP_SPLIT, seed=SEED)

    np.save(out / 'train_edge_indices.npy', tr_idx)
    np.save(out / 'val_edge_indices.npy',   va_idx)
    np.save(out / 'test_edge_indices.npy',  te_idx)
    np.save(out / 'target_edges.npy',       target_edges)

    info = {
        'dataset': dataset_name, 'task': 'lp',
        'n_edges': len(target_edges),
        'n_train': len(tr_idx), 'n_val': len(va_idx), 'n_test': len(te_idx),
        'tr_frac': LP_SPLIT[0], 'va_frac': LP_SPLIT[1], 'te_frac': LP_SPLIT[2],
        'target_relation_idx': rel_idx,
        'seed': SEED,
    }
    with open(out / 'info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"  [LP] {dataset_name}: {len(tr_idx)} tr / {len(va_idx)} va / {len(te_idx)} te edges")
    return info


def generate_cl_splits(dataset_name, data):
    """HGB Cluster: 24% train, 6% val, 70% test (stratified, same as NC)."""
    if data.get('n_classes', 0) == 0:
        print(f"  [CL] {dataset_name}: no labels, skipping")
        return None

    out = SPLIT_DIR / 'cl' / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    labels = data['labels'].numpy()
    Nt = data['target_size']
    idx = np.arange(Nt)

    tr, va, te = _stratified_three_way(idx, labels, *NC_SPLIT, seed=SEED)

    np.save(out / 'train_indices.npy', tr)
    np.save(out / 'val_indices.npy',   va)
    np.save(out / 'test_indices.npy',  te)
    np.save(out / 'labels.npy',        labels)

    info = {
        'dataset': dataset_name, 'task': 'cl',
        'n_total': Nt,
        'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'tr_frac': NC_SPLIT[0], 'va_frac': NC_SPLIT[1], 'te_frac': NC_SPLIT[2],
        'n_classes': int(data['n_classes']),
        'seed': SEED,
    }
    with open(out / 'info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"  [CL] {dataset_name}: {len(tr)} tr / {len(va)} va / {len(te)} te")
    return info


def generate_rec_splits(dataset_name, data):
    """HGB Rec: 80% train, 20% test (edge-level, no val)."""
    out = SPLIT_DIR / 'rec' / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    rel_idx = data.get('target_relation_idx', TARGET_REL_IDX.get(dataset_name, 0))
    if rel_idx >= len(data['A_list_sp']):
        print(f"  [REC] {dataset_name}: no target relation, skipping")
        return None

    A = data['A_list_sp'][rel_idx].tocoo()
    target_edges = np.column_stack([A.row, A.col])

    all_idx = np.arange(len(target_edges))
    tr_frac, te_frac = REC_SPLIT
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_idx))
    n_tr = int(len(all_idx) * tr_frac)
    tr_idx = all_idx[perm[:n_tr]]
    te_idx = all_idx[perm[n_tr:]]

    np.save(out / 'train_edge_indices.npy', tr_idx)
    np.save(out / 'test_edge_indices.npy',  te_idx)
    np.save(out / 'target_edges.npy',       target_edges)

    info = {
        'dataset': dataset_name, 'task': 'rec',
        'n_edges': len(target_edges),
        'n_train': len(tr_idx), 'n_test': len(te_idx),
        'tr_frac': tr_frac, 'te_frac': te_frac,
        'target_relation_idx': rel_idx,
        'seed': SEED,
    }
    with open(out / 'info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"  [REC] {dataset_name}: {len(tr_idx)} tr / {len(te_idx)} te edges")
    return info


GENERATORS = {
    'nc': generate_nc_splits,
    'lp': generate_lp_splits,
    'cl': generate_cl_splits,
    'rec': generate_rec_splits,
}


def generate_all(dataset_filter=None, task_filter=None):
    all_info = {}
    for dname, tasks in DATASET_TASKS.items():
        if dataset_filter and dname not in dataset_filter:
            continue
        # Skip datasets that aren't downloaded
        _rahgh_data = os.path.join(_RAHGH, 'data', 'raw', dname)
        if dname in _SKIP_IF_MISSING and not os.path.isdir(_rahgh_data):
            print(f"  [SKIP] {dname}: data not found at {_rahgh_data}")
            continue
        print(f"\n{'='*60}")
        print(f"  Loading {dname}...")
        print(f"{'='*60}")
        try:
            data = _get_loader(dname)
        except Exception as e:
            print(f"  [SKIP] {dname}: load failed — {e}")
            continue

        dataset_info = {}
        for task in tasks:
            if task_filter and task != task_filter:
                continue
            try:
                fn = GENERATORS[task]
                info = fn(dname, data)
                if info is not None:
                    dataset_info[task] = info
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [SKIP] {dname}/{task}: {e}")
        all_info[dname] = dataset_info

    summary_path = SPLIT_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_info, f, indent=2)
    print(f"\nSplit summary -> {summary_path}")
    return all_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate HGB-compatible splits')
    parser.add_argument('--dataset', help='Dataset name (default: all)')
    parser.add_argument('--task', choices=['nc', 'lp', 'cl', 'rec'], help='Task (default: all)')
    args = parser.parse_args()

    d_filter = {args.dataset} if args.dataset else None
    t_filter = args.task if args.task else None
    generate_all(dataset_filter=d_filter, task_filter=t_filter)
