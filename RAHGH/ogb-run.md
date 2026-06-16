# OGB Experiments on RAHGH

## 1. Supported OGB datasets

| OGB Name | Task | Nodes | Node types | Edge types | Target |
|----------|------|-------|------------|------------|--------|
| `ogbn-mag` | Node classification (NC) | ~1.9M | paper, author, institution, field_of_study | 4 relations | paper (349 classes, 121 task labels) |

`ogbn-mag` is the only heterogeneous OGB dataset. Homogeneous OGB datasets (ogbn-arxiv, ogbn-products, etc.) can only be used AFTER the RAHGH encoder, as standard GNN inputs.

---

## 2. Download

```bash
# Download ogbn-mag only
python scripts/download_datasets.py --only ogbn_mag

# Verify
python scripts/verify_downloads.py --dataset ogbn_mag
```

Data lands in `data/raw/ogbn_mag/`.

---

## 3. Write an OGB loader

OGB returns a PyG `HeteroData` object. Convert it to RAHGH's standard data dict.

Create `src/data/ogbn_mag_loader.py`:

```python
import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path


def load_ogbn_mag(root: str = "data/raw") -> dict:
    """
    Load OGBN-MAG into RAHGH's standard data dict format.

    Node layout (global ID space):
        paper[0:Np)  author[Np:Np+Na)  institution[Np+Na:Np+Na+Ni)
        field_of_study[Np+Na+Ni:N)
    """
    from ogb.nodeproppred import PygNodePropPredDataset

    folder = Path(root) / "ogbn_mag"
    ds = PygNodePropPredDataset(name="ogbn-mag", root=str(folder))
    data = ds[0]
    split_idx = ds.get_idx_split()

    # ── Map node types to global ID ranges ──────────────────────────────────
    # PyG stores each type with 0-based local IDs. We assign contiguous global
    # blocks: paper first (largest), then author, institution, field_of_study.
    type_order = ["paper", "author", "institution", "field_of_study"]
    N_types = {}
    for t in type_order:
        N_types[t] = data[t].num_nodes

    offsets = {}
    offset = 0
    for t in type_order:
        offsets[t] = offset
        offset += N_types[t]
    N_total = offset

    # ── Features ────────────────────────────────────────────────────────────
    # Paper: 128-dim (from ogb). Others: identity (no raw features in OGB).
    X_dict = {}
    X_dict["paper"] = data["paper"].x.float()
    for t in type_order[1:]:
        nt = N_types[t]
        dim = min(nt, 128)
        X_dict[t] = torch.eye(nt, dim, dtype=torch.float32)

    # ── Edge index ──────────────────────────────────────────────────────────
    # OGB stores edges as (src_type, rel_type, dst_type) tuples.
    # Convert local → global IDs.
    edge_index_dict = {}
    relation_info = {}
    relation_names = []
    bipartite_flags = []
    A_list_sp = []

    for src_type, rel_type, dst_type in data.edge_types:
        ei = data[src_type, rel_type, dst_type].edge_index
        src_global = ei[0] + offsets[src_type]
        dst_global = ei[1] + offsets[dst_type]

        rname = f"{src_type}→{dst_type}"
        edge_index_dict[rname] = torch.stack([src_global, dst_global], dim=0)
        relation_info[rname] = (src_type, dst_type)
        relation_names.append(rname)
        bipartite_flags.append(src_type != dst_type)

        # Build CSR for model
        r = src_global.numpy().astype(np.int64)
        c = dst_global.numpy().astype(np.int64)
        A = sp.coo_matrix(
            (np.ones(len(r), dtype=np.float32), (r, c)),
            shape=(N_total, N_total)
        ).tocsr()
        A_list_sp.append(A)

    # ── Labels ──────────────────────────────────────────────────────────────
    # ogbn-mag labels: shape (N_paper, 1), 121 classes (0-120), -1 = unlabeled
    labels_raw = data["paper"].y.squeeze(-1).numpy()       # (N_paper,)
    train_idx = split_idx["train"]["paper"].numpy()
    valid_idx = split_idx["valid"]["paper"].numpy()
    test_idx  = split_idx["test"]["paper"].numpy()

    # Filter to labeled nodes only
    labeled = np.where(labels_raw >= 0)[0]
    n_classes = int(labels_raw[labeled].max()) + 1

    # Map -1 → -1 (unlabeled), rest 0-indexed (already 0-indexed in ogbn-mag)
    labels_full = torch.full((N_types["paper"],), -1, dtype=torch.long)
    labels_full[labeled] = torch.tensor(labels_raw[labeled], dtype=torch.long)

    labeled_mask = labels_full >= 0
    labels_labeled = labels_full[labeled_mask]

    # Train/val/test masks for paper nodes
    train_mask = torch.zeros(N_types["paper"], dtype=torch.bool)
    train_mask[train_idx] = True
    valid_mask = torch.zeros(N_types["paper"], dtype=torch.bool)
    valid_mask[valid_idx] = True
    test_mask  = torch.zeros(N_types["paper"], dtype=torch.bool)
    test_mask[test_idx] = True

    # ── Node type indices ──────────────────────────────────────────────────
    node_type_indices = {}
    for t in type_order:
        off = offsets[t]
        sz = N_types[t]
        node_type_indices[t] = torch.arange(off, off + sz, dtype=torch.long)

    node_type_dims = {t: X_dict[t].shape[1] for t in type_order}

    return dict(
        A_list_sp=A_list_sp,
        bipartite_flags=bipartite_flags,
        relation_names=relation_names,
        X_dict=X_dict,
        node_type_indices=node_type_indices,
        node_type_dims=node_type_dims,
        relation_info=relation_info,
        edge_index_dict=edge_index_dict,
        labels=labels_labeled,
        labels_full=labels_full,
        labeled_mask=labeled_mask,
        N=N_total,
        target_type="paper",
        target_size=N_types["paper"],
        n_classes=n_classes,
        # OGB fixed splits
        train_indices=torch.tensor(train_idx, dtype=torch.long),
        valid_indices=torch.tensor(valid_idx, dtype=torch.long),
        test_indices=torch.tensor(test_idx, dtype=torch.long),
    )
```

---

## 4. Register in `src/train.py`

Add the loader and target relation index:

```python
from .data.ogbn_mag_loader import load_ogbn_mag

LOADERS = {
    'dblp': load_dblp, 'acm': load_acm, 'imdb': load_imdb,
    'ogbn_mag': load_ogbn_mag,
}

# ogbn-mag uses directed relations (paper→author, etc.)
# LP index: use the first relation (paper→author) as target
TARGET_REL_IDX = {'dblp': 0, 'acm': 0, 'imdb': 2, 'ogbn_mag': 0}
```

---

## 5. Run

```bash
# Node classification (uses OGB's fixed train/val/test split)
python -m src.train --dataset ogbn_mag --task nc --seeds 10

# Link prediction (predict paper→author edges)
python -m src.train --dataset ogbn_mag --task lp --seeds 10

# Graph clustering (paper node clusters)
python -m src.train --dataset ogbn_mag --task cluster --seeds 10
```

### What happens with `ogbn_mag` + NC

The hparam search in `hparam_search_nc` detects `train_indices` in the data dict (line 267-271 of `hparam_search.py`) and uses OGB's official fixed split instead of generating a random 80/20 split. This is critical for leaderboard-compatible results.

### Passing the data to the model

The loaders must produce these exact keys for the RAHGH model:

```
A_list_sp         : list[sp.csr_matrix]   N×N adjacency per relation
X_dict            : dict[str, Tensor]     {type: (N_t, d_t)}
node_type_indices : dict[str, Tensor]     {type: (N_t,)} global IDs
node_type_dims    : dict[str, int]        {type: feature_dim}
relation_info     : dict[str, tuple]      {rel: (src_type, dst_type)}
labels            : Tensor(N_labeled,)    class IDs
labels_full       : Tensor(N_target,)     -1 for unlabeled
N                 : int                   total nodes
target_type       : str                   node type to classify
target_size       : int                   nodes of target type
n_classes         : int                   number of classes
```

See `rahgh_model_spec.md` or `CLAUDE.md` section 7 for the full contract.

---

## 6. Handling `ogbn-mag` specifics

### Directed relations

OGBN-MAG has directed semantics (paper→author, paper→field_of_study, etc.). The model's `build_propagation_operator` in `src/model/normalize.py` supports `directed=True`, which uses row normalization `D^{-1}(A+I)` instead of symmetric `D^{-1/2}(A+I)D^{-1/2}`.

To enable directed propagation, pass `directed=True` when building the model. The current `build_rahgh_classifier` and `build_classifier` default to `directed=False`. For OGBN-MAG, either:
- Set `directed=True` in the loader before building, or
- Pass it through the params dict

### Memory

OGBN-MAG has ~1.9M nodes. The 6 adjacency matrices (one per relation) each consume ~60-80MB as sparse COO. Ensure ≥32GB RAM.

The loader above avoids duplicating the graph in memory by reusing edge_index_dict and building CSR matrices lazily. If you hit OOM, use the `edge_index_dict` only (drop `A_list_sp`) and modify the model forward to accept edge_index directly.

### Feature-less node types

Author, institution, and field_of_study have no raw features in OGBN-MAG. The loader above assigns identity encoding. For better performance, consider:
- Aggregating neighbor paper features (e.g., mean-pool paper → author)
- Using SVD on the adjacency matrix
- Using learnable embeddings (TypeSpecificProjection already handles this — each type has its own `nn.Linear`, so identity features work fine)

---

## 7. Adding another OGB dataset

To add a new OGB dataset (e.g., `ogbn-arxiv`):

1. **Check if it's heterogeneous**: RAHGH is designed for heterogeneous graphs. Homogeneous datasets (ogbn-arxiv, ogbn-products) should be used with the GNN head directly (skipping the RAHGH encoder), or wrapped as single-type heterogeneous graphs.

2. **For heterogeneous OGB datasets** (only `ogbn-mag` as of 2026): follow the pattern in `load_ogbn_mag` above.

3. **For homogeneous OGB datasets**: use the `HomogeneousWrapper`:

```python
def load_ogbn_arxiv(root="data/raw") -> dict:
    from ogb.nodeproppred import PygNodePropPredDataset
    ds = PygNodePropPredDataset(name="ogbn-arxiv", root=f"{root}/ogbn_arxiv")
    data = ds[0]
    split_idx = ds.get_idx_split()

    N = data.num_nodes
    edge_index = data.edge_index

    A = sp.coo_matrix(
        (np.ones(edge_index.size(1), dtype=np.float32),
         (edge_index[0].numpy(), edge_index[1].numpy())),
        shape=(N, N)
    ).tocsr()

    labels = data.y.squeeze(-1)
    labeled = labels >= 0
    n_classes = int(labels[labeled].max()) + 1

    return dict(
        A_list_sp=[A],
        bipartite_flags=[False],
        relation_names=["cites"],
        X_dict={"node": data.x.float()},
        node_type_indices={"node": torch.arange(N)},
        node_type_dims={"node": data.x.size(1)},
        relation_info={"cites": ("node", "node")},
        edge_index_dict={"cites": edge_index},
        labels=labels[labeled],
        labels_full=labels,
        labeled_mask=labeled,
        N=N, target_type="node", target_size=N, n_classes=n_classes,
        train_indices=split_idx["train"].numpy(),
        valid_indices=split_idx["valid"].numpy(),
        test_indices=split_idx["test"].numpy(),
    )
```

---

## 8. Reference: ogbn-mag statistics

| Property | Value |
|----------|-------|
| Papers | 736,389 |
| Authors | 1,134,649 |
| Institutions | 8,740 |
| Fields of study | 59,965 |
| Relations | paper→author, paper→field_of_study, author→institution, paper→paper (cites) |
| Classes | 121 (task-specific subset of 349) |
| Train/Val/Test | 629,571 / 64,879 / 41,939 |
| Feature dim | 128 (paper only) |

Source: [OGB official](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag)
