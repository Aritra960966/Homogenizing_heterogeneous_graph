# CLAUDE.md — RAHGH Project Instructions for Claude Code

This file is your complete operating manual. Read it fully before touching any code or data file.

---

## 1. What this project is

Relation-Aware Heterogeneous Graph Homogenization (RAHGH). The model transforms heterogeneous graphs into homogeneous latent embeddings compatible with any standard GNN. Four tasks: node classification (NC), link prediction (LP), graph clustering (Cluster), recommendation (Rec).

The full model is in `rahgh_model_spec.md`. The full experiment protocol is in `EXPERIMENT_GUIDE_v4.md`. This file tells you how to read the HGB datasets and wire them into the pipeline.

---

## 2. Where the data lives — exact paths

```
data/
└── raw/
    ├── IMDB/           ← movie node classification (3-class)
    ├── ACM/            ← paper node classification (3-class)
    ├── DBLP/           ← author node classification (4-class)
    ├── Freebase/       ← knowledge graph (entity classification + LP)
    ├── Freebase_no_name/
    ├── amazon/         ← user-item recommendation + LP
    ├── amazon_ini/     ← alternative amazon format
    ├── LastFM/         ← user-artist recommendation + LP
    ├── LastFM_ini/     ← alternative LastFM format
    ├── LastFM_magnn/   ← MAGNN format of LastFM
    ├── PubMed/         ← biomedical citation (NC)
    ├── PubMed_ini/     ← alternative PubMed format
    └── youtube/        ← social network (community detection)
```

**Before running any experiment, always call `inspect_dataset(dataset_name)` first.** This function (defined below) prints the exact files present, their sizes, and auto-detects the format.

---

## 3. HGB dataset formats — there are two

HGB datasets come in one of two formats. You must detect which one you have before loading.

### Format A — HGB unified format (newer, preferred)

Files present: `node.dat`, `link.dat`, `label.dat`, `info.dat`

```
node.dat   : tab-separated   node_id \t node_type_id \t feature_values...
link.dat   : tab-separated   src_id \t dst_id \t link_type_id \t weight
label.dat  : tab-separated   node_id \t node_type_id \t label
info.dat   : JSON or plain text metadata (type names, num_nodes, num_links)
```

### Format B — legacy per-type files (DBLP, ACM MAGNN-style)

Files present: multiple named `.txt` or `.mat` files specific to each dataset.

```
DBLP:     author_label.txt, paper_author.txt, paper_conf.txt,
          paper_term.txt, paper.txt, term.txt, conf.txt
ACM:      ACM.mat  (scipy loadmat)
IMDB:     IMDB.mat  OR  movie_metadata.csv
Freebase: entity2id.txt, relation2id.txt, train.txt, valid.txt, test.txt
LastFM:   user_artist.dat, user_taggedartists.dat, tags.dat, artists.dat
```

---

## 4. Step 1 — always do this first: inspect the dataset

```python
# src/data/hgb_inspector.py

import os
import json
import numpy as np
from pathlib import Path


def inspect_dataset(dataset_name: str, root: str = "data/raw") -> dict:
    """
    Auto-inspect any HGB dataset folder.
    Prints all files, sizes, detected format, node/edge counts.
    Returns a dict with detected metadata.

    Call this BEFORE writing any loader.

    Usage:
        info = inspect_dataset("DBLP")
        info = inspect_dataset("LastFM")
        info = inspect_dataset("amazon")
    """
    folder = Path(root) / dataset_name
    assert folder.exists(), f"Dataset folder not found: {folder}"

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Path   : {folder}")
    print(f"{'='*60}")

    files = sorted(folder.iterdir())
    print(f"\n  Files ({len(files)} total):")
    for f in files:
        size = f.stat().st_size
        print(f"    {f.name:<35} {_human_size(size)}")

    # Detect format
    names = {f.name for f in files}
    if 'node.dat' in names and 'link.dat' in names:
        fmt = 'hgb_unified'
    elif 'ACM.mat' in names or 'IMDB.mat' in names:
        fmt = 'mat'
    elif 'author_label.txt' in names:
        fmt = 'dblp_legacy'
    elif 'entity2id.txt' in names or 'train.txt' in names:
        fmt = 'kg_triples'
    elif 'user_artist.dat' in names or 'user_artists.dat' in names:
        fmt = 'lastfm'
    else:
        fmt = 'unknown'

    print(f"\n  Detected format: {fmt}")

    meta = {'dataset': dataset_name, 'folder': str(folder), 'format': fmt, 'files': list(names)}

    # Parse metadata if available
    if fmt == 'hgb_unified':
        meta.update(_inspect_hgb_unified(folder))
    elif fmt == 'dblp_legacy':
        meta.update(_inspect_dblp(folder))
    elif fmt == 'lastfm':
        meta.update(_inspect_lastfm(folder))

    print(f"\n  Summary: {json.dumps({k:v for k,v in meta.items() if k not in ['files','folder']}, indent=2)}")
    return meta


def _human_size(n):
    for unit in ['B','KB','MB','GB']:
        if n < 1024: return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _inspect_hgb_unified(folder):
    """Parse HGB unified format metadata."""
    info = {}
    # info.dat
    info_path = folder / 'info.dat'
    if info_path.exists():
        raw = open(info_path).read()
        print(f"\n  info.dat contents:\n{raw[:800]}")

    # node.dat — count node types
    node_path = folder / 'node.dat'
    if node_path.exists():
        type_counts = {}
        with open(node_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    t = parts[1]
                    type_counts[t] = type_counts.get(t, 0) + 1
        info['node_type_counts'] = type_counts
        info['total_nodes'] = sum(type_counts.values())
        print(f"\n  node.dat: {info['total_nodes']} nodes")
        for t, c in type_counts.items():
            print(f"    type {t}: {c} nodes")

    # link.dat — count relation types
    link_path = folder / 'link.dat'
    if link_path.exists():
        rel_counts = {}
        with open(link_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    r = parts[2]
                    rel_counts[r] = rel_counts.get(r, 0) + 1
        info['relation_counts'] = rel_counts
        info['total_edges'] = sum(rel_counts.values())
        print(f"\n  link.dat: {info['total_edges']} edges across {len(rel_counts)} relation types")

    # label.dat — count label types and classes
    label_path = folder / 'label.dat'
    if label_path.exists():
        labels = {}
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    lbl = parts[2]
                    labels[lbl] = labels.get(lbl, 0) + 1
        info['label_counts'] = labels
        info['n_classes'] = len(labels)
        print(f"\n  label.dat: {sum(labels.values())} labeled nodes, {len(labels)} classes")

    return info


def _inspect_dblp(folder):
    """Parse DBLP legacy format."""
    info = {}
    al = folder / 'author_label.txt'
    if al.exists():
        lines = open(al).readlines()
        info['n_authors'] = len(lines)
        labels = set(l.split('\t')[1] for l in lines if '\t' in l)
        info['n_classes'] = len(labels)
        print(f"\n  author_label.txt: {len(lines)} authors, classes={labels}")

    pa = folder / 'paper_author.txt'
    if pa.exists():
        info['n_paper_author_edges'] = sum(1 for _ in open(pa))

    pt = folder / 'paper_term.txt'
    if pt.exists():
        info['n_paper_term_edges'] = sum(1 for _ in open(pt))

    return info


def _inspect_lastfm(folder):
    """Parse LastFM format."""
    info = {}
    ua = folder / 'user_artist.dat'
    if not ua.exists():
        ua = folder / 'user_artists.dat'
    if ua.exists():
        edges = sum(1 for _ in open(ua)) - 1  # minus header
        users = set(); artists = set()
        with open(ua) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    users.add(parts[0]); artists.add(parts[1])
        info['n_users'] = len(users)
        info['n_artists'] = len(artists)
        info['n_user_artist_edges'] = edges
        print(f"\n  user_artist: {len(users)} users, {len(artists)} artists, {edges} interactions")
    return info
```

---

## 5. Step 2 — dataset loaders (one per dataset family)

### 5.1 HGB unified format loader (covers IMDB, ACM new format, PubMed, Freebase)

```python
# src/data/hgb_unified_loader.py

import numpy as np
import torch
from pathlib import Path
from collections import defaultdict


def load_hgb_unified(dataset_name: str, root: str = "data/raw") -> dict:
    """
    Load any HGB unified-format dataset.
    Reads: node.dat, link.dat, label.dat, info.dat

    Returns the standard data dict used by all RAHGH task files:
    {
        'A_list_sp'       : list of scipy CSR matrices,
        'bipartite_flags' : list of bool,
        'relation_names'  : list of str,
        'X_dict'          : {type_name: Tensor(N_t, d_t)},
        'labels'          : LongTensor(N_target,),
        'node_type_indices': {type_name: Tensor(N_t,)},  # global IDs
        'node_type_dims'  : {type_name: int},
        'relation_info'   : {rel_name: (src_type, dst_type)},
        'N'               : int total nodes,
        'target_type'     : str,
        'target_size'     : int,
        'n_classes'       : int,
        'edge_index_dict' : {rel_name: Tensor(2, E)},   # for PyG-style models
    }
    """
    import scipy.sparse as sp
    folder = Path(root) / dataset_name

    # ── 1. Parse info.dat ────────────────────────────────────────────────────
    info = _parse_info(folder / 'info.dat')
    type_id_to_name = info.get('type_id_to_name', {})    # e.g. {0: 'movie', 1: 'actor'}
    rel_id_to_info  = info.get('rel_id_to_info',  {})    # e.g. {0: ('movie','director')}
    target_type     = info.get('target_type', None)

    # ── 2. Parse node.dat ────────────────────────────────────────────────────
    # node_id \t type_id \t feat1 \t feat2 ...
    nodes_by_type = defaultdict(list)   # type_name → list of (global_id, feat_vec)
    max_node_id = 0

    with open(folder / 'node.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            nid = int(parts[0])
            tid = parts[1]
            tname = type_id_to_name.get(tid, f"type_{tid}")
            feats = np.array(parts[2:], dtype=np.float32) if len(parts) > 2 else None
            nodes_by_type[tname].append((nid, feats))
            max_node_id = max(max_node_id, nid)

    N = max_node_id + 1

    # Build global index tensors and feature matrices
    node_type_indices = {}
    X_dict = {}
    node_type_dims = {}

    for tname, node_list in nodes_by_type.items():
        ids   = np.array([n[0] for n in node_list], dtype=np.int64)
        feats = [n[1] for n in node_list]
        node_type_indices[tname] = torch.tensor(ids)

        if feats[0] is not None:
            feat_mat = np.stack(feats, axis=0)
        else:
            # No features → identity encoding (small types) or zeros (large types)
            dim = min(len(node_list), 512)
            feat_mat = np.eye(len(node_list), dim, dtype=np.float32)

        X_dict[tname] = torch.tensor(feat_mat)
        node_type_dims[tname] = feat_mat.shape[1]

    # ── 3. Parse link.dat ────────────────────────────────────────────────────
    # src_id \t dst_id \t rel_type_id \t weight
    edges_by_rel = defaultdict(lambda: ([], []))
    relation_type_info = {}

    with open(folder / 'link.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            src, dst, rid = int(parts[0]), int(parts[1]), parts[2]
            edges_by_rel[rid][0].append(src)
            edges_by_rel[rid][1].append(dst)
            if rid not in relation_type_info and rid in rel_id_to_info:
                relation_type_info[rid] = rel_id_to_info[rid]

    # Build scipy CSR and edge_index tensors
    A_list_sp = []
    bipartite_flags = []
    relation_names  = []
    edge_index_dict = {}
    relation_info   = {}

    for rid, (rows, cols) in sorted(edges_by_rel.items(), key=lambda x: int(x[0])):
        rname = f"rel_{rid}"
        src_type, dst_type = relation_type_info.get(rid, ('?', '?'))
        is_bip = (src_type != dst_type)

        r = np.array(rows, dtype=np.int64)
        c = np.array(cols, dtype=np.int64)
        A = sp.coo_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                           shape=(N, N)).tocsr()

        A_list_sp.append(A)
        bipartite_flags.append(is_bip)
        relation_names.append(rname)
        edge_index_dict[rname] = torch.tensor(np.stack([r, c], axis=0))
        relation_info[rname]   = (src_type, dst_type)

    # ── 4. Parse label.dat ───────────────────────────────────────────────────
    # node_id \t type_id \t label
    labeled_ids, raw_labels = [], []
    with open(folder / 'label.dat') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                labeled_ids.append(int(parts[0]))
                raw_labels.append(int(parts[2]))

    # Map raw label values to 0-indexed classes
    unique_labels = sorted(set(raw_labels))
    lbl_map = {v: i for i, v in enumerate(unique_labels)}
    labels_mapped = [lbl_map[l] for l in raw_labels]

    # Find the target type by checking which type most labeled nodes belong to
    if target_type is None:
        lbl_id_set = set(labeled_ids)
        counts = {t: sum(1 for nid in node_type_indices[t].numpy()
                         if nid in lbl_id_set)
                  for t in node_type_indices}
        target_type = max(counts, key=counts.get)

    # Re-index labels to target-type-local indices
    tgt_global = node_type_indices[target_type].numpy()
    global_to_local = {gid: li for li, gid in enumerate(tgt_global)}
    local_ids    = [global_to_local[gid] for gid in labeled_ids if gid in global_to_local]
    local_labels = [labels_mapped[i] for i, gid in enumerate(labeled_ids)
                    if gid in global_to_local]

    # Build dense label tensor (size = target_size); -1 for unlabeled
    target_size = len(tgt_global)
    labels_full = torch.full((target_size,), -1, dtype=torch.long)
    for li, lbl in zip(local_ids, local_labels):
        labels_full[li] = lbl

    # Labeled mask (for CV splits — only split labeled nodes)
    labeled_mask = labels_full >= 0
    labels_labeled = labels_full[labeled_mask]

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
        N=N,
        target_type=target_type,
        target_size=target_size,
        n_classes=len(unique_labels),
    )


def _parse_info(info_path: Path) -> dict:
    """Parse info.dat into usable metadata dicts. Handles JSON and plain text."""
    if not info_path.exists():
        return {}
    raw = open(info_path).read().strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Plain text fallback — common HGB format
    result = {'type_id_to_name': {}, 'rel_id_to_info': {}}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if 'node type' in line.lower() or 'entity type' in line.lower():
            # e.g. "0 movie" or "Node type 0: movie"
            parts = line.split()
            for i, p in enumerate(parts):
                if p.isdigit() and i + 1 < len(parts):
                    result['type_id_to_name'][p] = parts[i+1]
    return result
```

### 5.2 DBLP legacy loader

```python
# src/data/dblp_loader.py
# (Full version already in EXPERIMENT_GUIDE_v4.md — Step 1)
# Key reminder: run inspect_dataset("DBLP") first to confirm file names
```

### 5.3 ACM .mat loader

```python
# src/data/acm_loader.py
# (Full version already in EXPERIMENT_GUIDE_v4.md — Step 2)
# Key reminder: run inspect_dataset("ACM") first; mat keys vary by version
# If ACM.mat is absent, fall back to load_hgb_unified("ACM")
```

### 5.4 LastFM / Amazon recommendation loader

```python
# src/data/lastfm_loader.py

import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
from collections import defaultdict


def load_lastfm(root: str = "data/raw") -> dict:
    """
    Load LastFM user-artist-tag heterogeneous graph for recommendation.
    Supports both LastFM/ and LastFM_ini/ folder variants.

    Node layout: U[0:Nu)  A[Nu:Nu+Na)  T[Nu+Na:N)
    Target relation for recommendation: User → Artist (index 0 in A_list_sp)

    Returns standard RAHGH data dict.
    """
    folder = Path(root) / 'LastFM'
    if not (folder / 'user_artist.dat').exists():
        folder = Path(root) / 'LastFM_ini'
    if not (folder / 'user_artist.dat').exists():
        folder = Path(root) / 'LastFM_magnn'

    assert folder.exists(), f"LastFM folder not found under {root}"

    # ── User-Artist edges ─────────────────────────────────────────────────────
    ua_file = folder / 'user_artist.dat'
    if not ua_file.exists():
        ua_file = folder / 'user_artists.dat'

    users, artists = set(), set()
    ua_edges = []
    with open(ua_file) as f:
        header = f.readline()   # skip header line
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                u, a = parts[0].strip(), parts[1].strip()
                users.add(u); artists.add(a)
                ua_edges.append((u, a))

    user_list   = sorted(users)
    artist_list = sorted(artists)
    u2i = {u: i for i, u in enumerate(user_list)}
    a2i = {a: i for i, a in enumerate(artist_list)}

    Nu = len(user_list)
    Na = len(artist_list)

    # ── Tag edges (if present) ────────────────────────────────────────────────
    tag_file = folder / 'user_taggedartists.dat'
    tags = set(); at_edges = []
    if tag_file.exists():
        with open(tag_file) as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    a, t = parts[1].strip(), parts[2].strip()
                    artists.add(a); tags.add(t)
                    at_edges.append((a, t))
    tag_list = sorted(tags)
    t2i = {t: i for i, t in enumerate(tag_list)}
    Nt = len(tag_list)
    N = Nu + Na + Nt

    # ── Build adjacency matrices ──────────────────────────────────────────────
    def build_sp(src_idx, dst_idx):
        r = np.array(src_idx, dtype=np.int64)
        c = np.array(dst_idx, dtype=np.int64)
        return sp.coo_matrix((np.ones(len(r), np.float32), (r, c)),
                               shape=(N, N)).tocsr()

    # UA: user→artist (global: user offset=0, artist offset=Nu)
    ua_r = [u2i[u]       for u, a in ua_edges]
    ua_c = [a2i[a] + Nu  for u, a in ua_edges]
    UA   = build_sp(ua_r, ua_c)
    AU   = UA.T.tocsr()

    # AT: artist→tag (if tags exist)
    if at_edges:
        at_r = [a2i[a] + Nu       for a, t in at_edges]
        at_c = [t2i[t] + Nu + Na  for a, t in at_edges]
        AT   = build_sp(at_r, at_c)
        TA   = AT.T.tocsr()
        A_list_sp      = [UA, AU, AT, TA]
        bipartite_flags = [True, True, True, True]
        relation_names  = ['user→artist','artist→user','artist→tag','tag→artist']
    else:
        A_list_sp      = [UA, AU]
        bipartite_flags = [True, True]
        relation_names  = ['user→artist','artist→user']

    # ── Features ─────────────────────────────────────────────────────────────
    X_user   = torch.eye(Nu, min(Nu, 512), dtype=torch.float32)
    X_artist = torch.eye(Na, min(Na, 512), dtype=torch.float32)
    X_tag    = torch.eye(Nt, min(Nt, 256), dtype=torch.float32) if Nt > 0 else None

    X_dict = {'user': X_user, 'artist': X_artist}
    if X_tag is not None:
        X_dict['tag'] = X_tag

    # Edge array for recommendation task
    ua_edges_np = np.array(list(zip(ua_r, ua_c)), dtype=np.int64)

    return dict(
        A_list_sp=A_list_sp,
        bipartite_flags=bipartite_flags,
        relation_names=relation_names,
        X_dict=X_dict,
        labels=torch.zeros(Nu, dtype=torch.long),   # placeholder — no class labels
        N=N, Nu=Nu, Na=Na, Nt=Nt,
        target_type='user', target_size=Nu,
        n_classes=0,    # recommendation task — no class labels
        # For recommendation: use ua_edges_np as target_edges
        target_edges=ua_edges_np,
        user_list=user_list, artist_list=artist_list,
    )


def load_amazon(root: str = "data/raw") -> dict:
    """
    Load Amazon user-item graph for recommendation.
    Tries amazon/ then amazon_ini/ folder.

    Expects either:
      - user_item.dat  (HGB unified format)
      - OR reads from link.dat via load_hgb_unified
    """
    folder = Path(root) / 'amazon'
    if not folder.exists():
        folder = Path(root) / 'amazon_ini'

    # Check if it's HGB unified format
    if (folder / 'node.dat').exists():
        return load_hgb_unified('amazon', root)

    # Otherwise parse raw files
    return _parse_amazon_raw(folder, root)


def _parse_amazon_raw(folder: Path, root: str) -> dict:
    """Parse Amazon raw interaction files."""
    # Amazon typically has user-item interaction as:
    # user_id \t item_id \t rating \t timestamp
    # or just user_id \t item_id
    import scipy.sparse as sp

    edges = []
    users, items = set(), set()

    for fname in ['user_item.dat', 'ratings.dat', 'interactions.txt', 'edges.txt']:
        fp = folder / fname
        if fp.exists():
            with open(fp) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        u, it = parts[0].strip(), parts[1].strip()
                        users.add(u); items.add(it)
                        edges.append((u, it))
            break

    if not edges:
        # Fall back to HGB unified
        return load_hgb_unified('amazon', root)

    user_list = sorted(users); item_list = sorted(items)
    u2i = {u: i for i, u in enumerate(user_list)}
    it2i = {it: i for i, it in enumerate(item_list)}

    Nu = len(user_list); Ni = len(item_list); N = Nu + Ni

    ui_r = [u2i[u]        for u, it in edges]
    ui_c = [it2i[it] + Nu for u, it in edges]
    UI = sp.coo_matrix((np.ones(len(ui_r), np.float32), (ui_r, ui_c)),
                        shape=(N, N)).tocsr()
    IU = UI.T.tocsr()

    X_user = torch.eye(Nu, min(Nu, 512), dtype=torch.float32)
    X_item = torch.eye(Ni, min(Ni, 512), dtype=torch.float32)

    target_edges = np.array(list(zip(ui_r, ui_c)), dtype=np.int64)

    return dict(
        A_list_sp=[UI, IU],
        bipartite_flags=[True, True],
        relation_names=['user→item','item→user'],
        X_dict={'user': X_user, 'item': X_item},
        labels=torch.zeros(Nu, dtype=torch.long),
        N=N, Nu=Nu, Ni=Ni,
        target_type='user', target_size=Nu, n_classes=0,
        target_edges=target_edges,
        user_list=user_list, item_list=item_list,
    )
```

### 5.5 Freebase / KG triples loader

```python
# src/data/freebase_loader.py

import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
from collections import defaultdict


def load_freebase(root: str = "data/raw",
                  named: bool = True) -> dict:
    """
    Load Freebase for link prediction and entity classification.

    named=True  → uses Freebase/ (with entity names)
    named=False → uses Freebase_no_name/

    Files expected:
      entity2id.txt    : entity_name \t entity_id   (or just entity_id per line)
      relation2id.txt  : relation_name \t rel_id
      train.txt        : h \t r \t t
      valid.txt        : h \t r \t t
      test.txt         : h \t r \t t
      (optional) entity_labels.txt  : entity_id \t label
    """
    folder = Path(root) / ('Freebase' if named else 'Freebase_no_name')

    # Check for HGB unified format
    if (folder / 'node.dat').exists():
        return load_hgb_unified('Freebase' if named else 'Freebase_no_name', root)

    # ── Parse entity and relation maps ────────────────────────────────────────
    ent2id, rel2id = {}, {}

    e2id_path = folder / 'entity2id.txt'
    if e2id_path.exists():
        with open(e2id_path) as f:
            first = f.readline().strip()
            if first.isdigit():
                N_ent = int(first)
            else:
                parts = first.split('\t')
                if len(parts) == 2:
                    ent2id[parts[0]] = int(parts[1])
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    ent2id[parts[0]] = int(parts[1])
    else:
        # Auto-build from train/valid/test
        for split in ['train.txt','valid.txt','test.txt']:
            fp = folder / split
            if fp.exists():
                with open(fp) as f:
                    for line in f:
                        h, r, t = line.strip().split('\t')
                        if h not in ent2id: ent2id[h] = len(ent2id)
                        if t not in ent2id: ent2id[t] = len(ent2id)
                        if r not in rel2id: rel2id[r] = len(rel2id)

    N = len(ent2id) if ent2id else 0

    # ── Parse triples ─────────────────────────────────────────────────────────
    edges_by_rel = defaultdict(lambda: ([], []))
    for split in ['train.txt','valid.txt','test.txt']:
        fp = folder / split
        if not fp.exists(): continue
        with open(fp) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    h, r, t = parts
                    hi = ent2id.get(h, 0); ti = ent2id.get(t, 0)
                    ri = rel2id.get(r, len(rel2id))
                    edges_by_rel[ri][0].append(hi)
                    edges_by_rel[ri][1].append(ti)

    A_list_sp = []
    relation_names = []
    bipartite_flags = []

    for rid in sorted(edges_by_rel.keys()):
        rows, cols = edges_by_rel[rid]
        A = sp.coo_matrix((np.ones(len(rows), np.float32),
                            (np.array(rows, np.int64), np.array(cols, np.int64))),
                           shape=(N, N)).tocsr()
        A_list_sp.append(A)
        rname = {v:k for k,v in rel2id.items()}.get(rid, f'rel_{rid}')
        relation_names.append(rname)
        bipartite_flags.append(False)   # KG: all entities in one type space

    # Features: random or identity (entities have no features in raw Freebase)
    X = torch.eye(N, min(N, 512), dtype=torch.float32)

    return dict(
        A_list_sp=A_list_sp,
        bipartite_flags=bipartite_flags,
        relation_names=relation_names,
        X_dict={'entity': X},
        labels=torch.zeros(N, dtype=torch.long),
        N=N, target_type='entity', target_size=N,
        n_classes=0,   # LP task — no class labels unless entity_labels.txt present
        ent2id=ent2id, rel2id=rel2id,
    )
```

---

## 6. Dataset → task mapping

| Dataset | Folder | NC | LP | Cluster | Rec | Target type | n_classes |
|---------|--------|----|----|---------|----|------------|-----------|
| DBLP | `DBLP/` | ✓ | — | ✓ | — | author | 4 |
| ACM | `ACM/` | ✓ | — | ✓ | — | paper | 3 |
| IMDB | `IMDB/` | ✓ | ✓ | ✓ | — | movie | 3 |
| Freebase | `Freebase/` | — | ✓ | — | — | entity | — |
| Freebase_no_name | `Freebase_no_name/` | — | ✓ | — | — | entity | — |
| Amazon | `amazon/` | — | ✓ | — | ✓ | user | — |
| Amazon_ini | `amazon_ini/` | — | ✓ | — | ✓ | user | — |
| LastFM | `LastFM/` | — | ✓ | — | ✓ | user | — |
| LastFM_ini | `LastFM_ini/` | — | ✓ | — | ✓ | user | — |
| LastFM_magnn | `LastFM_magnn/` | — | ✓ | — | ✓ | user | — |
| PubMed | `PubMed/` | ✓ | ✓ | ✓ | — | paper | varies |
| PubMed_ini | `PubMed_ini/` | ✓ | ✓ | ✓ | — | paper | varies |
| YouTube | `youtube/` | — | — | ✓ | — | user | — |

---

## 7. Model interface — what the loaders must produce

The model (`rahgh_model_spec.md`) expects these exact keys in the data dict:

```python
data = {
    # ── Graph structure ───────────────────────────────────────────────────────
    'A_list_sp'        : list[sp.csr_matrix],   # one per relation, shape (N, N)
    'bipartite_flags'  : list[bool],            # True if src_type != dst_type
    'relation_names'   : list[str],             # human-readable relation names
    'edge_index_dict'  : dict[str, Tensor],     # {rel_name: (2, E)} global IDs

    # ── Features ─────────────────────────────────────────────────────────────
    'X_dict'           : dict[str, Tensor],     # {type_name: (N_t, d_t)}
    'node_type_indices': dict[str, Tensor],     # {type_name: (N_t,)} global IDs
    'node_type_dims'   : dict[str, int],        # {type_name: feature_dim}
    'relation_info'    : dict[str, tuple],      # {rel_name: (src_type, dst_type)}

    # ── Labels ───────────────────────────────────────────────────────────────
    'labels'           : LongTensor(N_labeled,),  # class IDs for labeled nodes
    'labels_full'      : LongTensor(N_target,),   # -1 for unlabeled nodes

    # ── Sizes ────────────────────────────────────────────────────────────────
    'N'                : int,   # total nodes across all types
    'target_type'      : str,   # which node type to classify
    'target_size'      : int,   # number of nodes of target type
    'n_classes'        : int,
}
```

---

## 8. Step-by-step experiment sequence

Follow this exact sequence every time you run an experiment.

```bash
# Step 0: Inspect the dataset first — ALWAYS
python -c "from src.data.hgb_inspector import inspect_dataset; inspect_dataset('DBLP')"

# Step 1: Run the loader and verify shapes
python -c "
from src.data.dblp_loader import load_dblp
data = load_dblp()
print('N:', data['N'])
print('target_size:', data['target_size'])
print('n_classes:', data['n_classes'])
print('X_dict:', {k: tuple(v.shape) for k, v in data['X_dict'].items()})
print('A_list_sp:', [A.shape for A in data['A_list_sp']])
print('bipartite_flags:', data['bipartite_flags'])
"

# Step 2: Run one experiment
python -m src.train --dataset dblp --task nc --seeds 10

# Step 3: Collect results
python scripts/collect_results.py
```

---

## 9. How to pick the right loader for each dataset folder

```python
# src/data/loader_registry.py

from .dblp_loader     import load_dblp
from .acm_loader      import load_acm
from .imdb_loader     import load_imdb
from .lastfm_loader   import load_lastfm, load_amazon
from .freebase_loader import load_freebase
from .hgb_unified_loader import load_hgb_unified


def load_dataset(name: str, root: str = "data/raw") -> dict:
    """
    Central loader registry. Always call this — it routes to the right loader.

    Args:
        name : one of the keys below (case-insensitive)
        root : path to data/raw

    If the specific loader fails (missing files, wrong format),
    it automatically falls back to load_hgb_unified().
    """
    name_lower = name.lower().replace('-', '_')

    LOADERS = {
        'dblp'              : lambda: load_dblp(root=f"{root}/DBLP"),
        'acm'               : lambda: load_acm(root=f"{root}/ACM"),
        'imdb'              : lambda: load_imdb(root=f"{root}/IMDB"),
        'lastfm'            : lambda: load_lastfm(root=root),
        'lastfm_ini'        : lambda: load_lastfm(root=root),
        'lastfm_magnn'      : lambda: load_lastfm(root=root),
        'amazon'            : lambda: load_amazon(root=root),
        'amazon_ini'        : lambda: load_amazon(root=root),
        'freebase'          : lambda: load_freebase(root=root, named=True),
        'freebase_no_name'  : lambda: load_freebase(root=root, named=False),
        'pubmed'            : lambda: load_hgb_unified('PubMed', root=root),
        'pubmed_ini'        : lambda: load_hgb_unified('PubMed_ini', root=root),
        'youtube'           : lambda: load_hgb_unified('youtube', root=root),
    }

    loader_fn = LOADERS.get(name_lower)
    if loader_fn is None:
        print(f"[warning] No specific loader for '{name}'. Trying HGB unified format.")
        return load_hgb_unified(name, root=root)

    try:
        return loader_fn()
    except Exception as e:
        print(f"[warning] Loader failed ({e}). Falling back to HGB unified format.")
        return load_hgb_unified(name, root=root)
```

---

## 10. What to do when a file is missing or has unexpected structure

If `inspect_dataset()` reveals unexpected files:

1. Print the first 5 lines of each file:
   ```python
   with open('data/raw/DATASET/file.dat') as f:
       for i, line in enumerate(f):
           if i >= 5: break
           print(repr(line))
   ```

2. Check separators — HGB uses `\t`; some old files use spaces or commas.

3. Check if node IDs are 0-indexed or 1-indexed. Subtract 1 if needed.

4. If `.mat` file: run `scipy.io.loadmat(path).keys()` to see available matrices.

5. If format is completely unrecognized: open a GitHub issue at https://github.com/THUDM/HGB and compare against the documented format.

---

## 11. Sanity checks — run these after loading any new dataset

```python
def sanity_check(data: dict, name: str):
    """Run after any load_* call to verify the data dict is correct."""
    import scipy.sparse as sp

    print(f"\n{'='*50}  Sanity check: {name}")

    # 1. All adjacency matrices are N×N
    N = data['N']
    for i, A in enumerate(data['A_list_sp']):
        assert A.shape == (N, N), f"A[{i}] shape {A.shape} != ({N},{N})"
    print(f"  ✓ All {len(data['A_list_sp'])} adjacency matrices are ({N},{N})")

    # 2. Node type indices are within [0, N)
    for tname, idx in data['node_type_indices'].items():
        assert idx.max() < N, f"type {tname} has global idx {idx.max()} >= N={N}"
        assert idx.min() >= 0
    print(f"  ✓ All node_type_indices in [0, {N})")

    # 3. Feature dims match node_type_dims
    for tname, X in data['X_dict'].items():
        assert X.shape[0] == len(data['node_type_indices'][tname]), \
            f"{tname}: X rows {X.shape[0]} != {len(data['node_type_indices'][tname])} nodes"
    print(f"  ✓ Feature matrix shapes consistent")

    # 4. Labels are within [0, n_classes)
    if data['n_classes'] > 0 and len(data['labels']) > 0:
        assert data['labels'].max() < data['n_classes'], "label value >= n_classes"
        assert data['labels'].min() >= 0, "negative label"
    print(f"  ✓ Labels in [0, {data['n_classes']})")

    # 5. Bipartite flags length matches A_list_sp
    assert len(data['bipartite_flags']) == len(data['A_list_sp'])
    print(f"  ✓ bipartite_flags length = {len(data['bipartite_flags'])}")

    print(f"  All checks passed.\n")
```

---

## 12. Results folder structure (recap)

All CSVs are written here. Create the folder before running.

```bash
mkdir -p results/nc/dblp results/nc/acm results/nc/imdb
mkdir -p results/lp/dblp results/lp/acm results/lp/imdb results/lp/freebase
mkdir -p results/clustering/dblp results/clustering/acm results/clustering/imdb
mkdir -p results/recommendation/amazon results/recommendation/lastfm results/recommendation/dblp
```

Every task writes:
- `cv_scores.csv` — fold × combo scores during hyperparameter search
- `best_params.json` — the winning hyperparameter combination
- `final_runs.csv` — one row per seed (10 seeds total), all metrics
- `summary.csv` — mean ± std across 10 seeds
- `epoch_metrics_seed{n}.csv` — per-epoch loss and eval metric for each final run

---

## 13. Quick reference — run commands

```bash
# ── Inspect ──────────────────────────────────────────────────────────────────
python -c "from src.data.hgb_inspector import inspect_dataset; inspect_dataset('LastFM')"

# ── Node Classification ───────────────────────────────────────────────────────
python -m src.train --dataset dblp    --task nc      --seeds 10
python -m src.train --dataset acm     --task nc      --seeds 10
python -m src.train --dataset imdb    --task nc      --seeds 10
python -m src.train --dataset pubmed  --task nc      --seeds 10

# ── Link Prediction ───────────────────────────────────────────────────────────
python -m src.train --dataset dblp      --task lp    --seeds 10
python -m src.train --dataset freebase  --task lp    --seeds 10
python -m src.train --dataset imdb      --task lp    --seeds 10

# ── Graph Clustering ──────────────────────────────────────────────────────────
python -m src.train --dataset dblp    --task cluster --seeds 10
python -m src.train --dataset acm     --task cluster --seeds 10
python -m src.train --dataset imdb    --task cluster --seeds 10

# ── Recommendation ────────────────────────────────────────────────────────────
python -m src.train --dataset amazon  --task rec     --seeds 10
python -m src.train --dataset lastfm  --task rec     --seeds 10
python -m src.train --dataset dblp    --task rec     --seeds 10

# ── Run everything ────────────────────────────────────────────────────────────
bash scripts/run_all.sh

# ── Collect and print LaTeX table ────────────────────────────────────────────
python scripts/collect_results.py
```
