import numpy as np
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5


def validate_data_dict(data: dict, name: str = "") -> None:
    tag = f"[{name}] " if name else ""

    required = {
        'X_dict', 'A_list_sp', 'bipartite_flags',
        'labels', 'target_size', 'n_classes', 'N',
    }
    missing = required - set(data.keys())
    assert not missing, f"{tag}data dict missing keys: {missing}"

    Nt = data['target_size']
    N  = data['N']
    assert Nt <= N, f"{tag}target_size={Nt} > N={N}"
    assert len(data['labels']) == Nt, \
        f"{tag}labels length {len(data['labels'])} != target_size {Nt}"

    n_classes = data['n_classes']
    lbl       = data['labels'].numpy()
    unique    = np.unique(lbl)
    assert len(unique) == n_classes, \
        f"{tag}n_classes={n_classes} but labels contain {len(unique)} unique values: {unique}"

    counts    = np.bincount(lbl, minlength=n_classes)
    min_count = counts.min()
    if min_count < N_FOLDS:
        raise AssertionError(
            f"{tag}class {counts.argmin()} has only {min_count} samples — "
            f"StratifiedKFold(n_splits={N_FOLDS}) will fail. "
            f"Either reduce N_FOLDS to {min_count} or merge rare classes.\n"
            f"  Full distribution: { {i: int(c) for i, c in enumerate(counts)} }"
        )

    n_relations = len(data['A_list_sp'])
    flags       = data['bipartite_flags']
    assert len(flags) == n_relations, \
        f"{tag}bipartite_flags has {len(flags)} entries but A_list_sp has {n_relations}"

    for node_type, x in data['X_dict'].items():
        assert x.dtype in (
            __import__('torch').float32, __import__('torch').float64
        ), f"{tag}X_dict['{node_type}'] dtype={x.dtype}, expected float32"
        assert not x.is_cuda, \
            f"{tag}X_dict['{node_type}'] is on CUDA — keep X_dict on CPU, move inside train"

    print(f"{tag}OK")
    print(f"  nodes={N}  target={Nt}  relations={n_relations}  classes={n_classes}")
    print(f"  class dist: { {i: int(c) for i, c in enumerate(counts)} }")
    print(f"  bipartite flags: {flags}")
    print(f"  node types: { {k: tuple(v.shape) for k, v in data['X_dict'].items()} }")
