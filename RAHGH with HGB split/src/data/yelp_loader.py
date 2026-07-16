"""
Yelp dataset loader for recommendation task.

Dataset structure (data/raw/Yelp/):
  entity_list.txt   : org_id remap_id           -> 136499 entities (IDs 0..136498)
  item_list.txt     : org_id col2 [freebase_id] -> 45538 items (IDs 0..45537)
  user_list.txt     : org_id remap_id           -> 45919 users (IDs 0..45918)
  train.txt         : user_id item_id1 item_id2 ... -> 45919 lines (one per user)
  test.txt          : user_id item_id1 item_id2 ...  -> 45919 lines
  kg_final.txt      : head_entity_id rel_id tail_entity_id -> 1,853,704 triples, 43 relations
  relation_list.txt : relation_name remap_id    -> 43 relations (IDs 0..42)

Node layout (contiguous global IDs):
  Users    [0, Nu)
  Entities [Nu, N)   includes both items (IDs 0..Ni-1) and pure KG entities (IDs Ni..N_entities-1)

Target relation for recommendation: user→item (index 0 in A_list_sp)
"""

import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
from collections import defaultdict


def load_yelp(root="data/raw"):
    folder = Path(root) / 'Yelp'
    assert folder.exists(), f"Yelp folder not found at {folder}"

    # -- 1. Parse entity list -------------------------------------------------
    entity_id_to_org = {}
    with open(folder / 'entity_list.txt') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                eid = int(parts[-1])
                org = ' '.join(parts[:-1])
                entity_id_to_org[eid] = org

    N_entities = len(entity_id_to_org)      # 136499
    org_to_entity_id = {org: eid for eid, org in entity_id_to_org.items()}

    # -- 2. Parse user list ---------------------------------------------------
    with open(folder / 'user_list.txt') as f:
        Nu = sum(1 for _ in f) - 1           # 45919 (skip header)

    # -- 3. Parse item list ---------------------------------------------------
    # Format: org_id  <remap_id><org_id>  [freebase_id]
    # col2 = remap_id concatenated with org_id (no delimiter).
    # Items correspond 1:1 with entities for IDs 0..Ni-1.
    item_interaction_to_entity = {}
    with open(folder / 'item_list.txt') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                org_id = parts[0]
                col2 = parts[1]
                if col2.endswith(org_id):
                    item_id = int(col2[:-len(org_id)])
                    entity_id = org_to_entity_id.get(org_id)
                    if entity_id is not None:
                        item_interaction_to_entity[item_id] = entity_id

    Ni = len(item_interaction_to_entity)    # 45538

    # -- 4. Parse train/test interactions ------------------------------------
    def parse_interactions(filepath):
        edges = []
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    u = int(parts[0])
                    for p in parts[1:]:
                        i = int(p)
                        edges.append((u, i))
        return np.array(edges, dtype=np.int64)

    train_edges = parse_interactions(folder / 'train.txt')
    test_edges  = parse_interactions(folder / 'test.txt')

    # -- 5. Node layout -------------------------------------------------------
    # Users [0, Nu), Entities [Nu, Nu+N_entities)
    N = Nu + N_entities
    ENTITY_OFFSET = Nu

    # -- 6. Build adjacency matrices -----------------------------------------
    def build_mat(rows, cols, n):
        return sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n, n)).tocsr()

    # 6a. User-item edges (bipartite, forward + reverse)
    all_ui_edges = np.concatenate([train_edges, test_edges], axis=0)
    u_global = all_ui_edges[:, 0]
    i_global = all_ui_edges[:, 1] + ENTITY_OFFSET

    UI = build_mat(u_global, i_global, N)
    IU = UI.T.tocsr()

    A_list_sp = [UI, IU]
    relation_names = ['user→item', 'item→user']
    bipartite_flags = [True, True]

    # 6b. KG triples -> per-relation entity<->entity adjacency
    kg_edges_by_rel = defaultdict(lambda: ([], []))
    with open(folder / 'kg_final.txt') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                h_global = h + ENTITY_OFFSET
                t_global = t + ENTITY_OFFSET
                kg_edges_by_rel[r][0].append(h_global)
                kg_edges_by_rel[r][1].append(t_global)

    for rid in sorted(kg_edges_by_rel.keys()):
        rows, cols = kg_edges_by_rel[rid]
        A = build_mat(rows, cols, N)
        A_list_sp.append(A)
        relation_names.append(f'kg_rel_{rid}')
        bipartite_flags.append(True)

    # -- 7. Features ---------------------------------------------------------
    # Random features: torch.eye(N, d) produces zero vectors for all
    # nodes with index >= d, killing gradient for >99% of the graph.
    # Random Gaussian features give every node a unique initialisation
    # that the GNN can propagate through the KG and user-item edges.
    user_dim = 128
    entity_dim = 128

    X_user = torch.randn(Nu, user_dim, dtype=torch.float32)
    X_entity = torch.randn(N_entities, entity_dim, dtype=torch.float32)
    # Normalise each row to unit length so initial magnitudes are
    # consistent (prevents the BPR regularisation from dominating).
    X_user = X_user / X_user.norm(dim=1, keepdim=True).clamp(min=1e-8)
    X_entity = X_entity / X_entity.norm(dim=1, keepdim=True).clamp(min=1e-8)

    X_dict = {'user': X_user, 'entity': X_entity}
    node_type_dims = {'user': user_dim, 'entity': entity_dim}

    # -- 8. Node type index mappings -----------------------------------------
    node_type_global_ids = {
        'user':   list(range(Nu)),
        'entity': list(range(ENTITY_OFFSET, N)),
    }
    node_type_indices = {
        'user':   torch.arange(0, Nu, dtype=torch.long),
        'entity': torch.arange(ENTITY_OFFSET, N, dtype=torch.long),
    }

    # -- 9. Relation info (consistent with relation_names using →) -----------
    relation_info = {'user→item': ('user', 'entity'), 'item→user': ('entity', 'user')}
    for rid in sorted(kg_edges_by_rel.keys()):
        relation_info[f'kg_rel_{rid}'] = ('entity', 'entity')

    # -- 10. Build output dict ------------------------------------------------
    return dict(
        # Graph structure
        A_list_sp=A_list_sp,
        relation_names=relation_names,
        bipartite_flags=bipartite_flags,

        # Features
        X_dict=X_dict,
        node_type_dims=node_type_dims,
        node_type_indices=node_type_indices,
        node_type_global_ids=node_type_global_ids,

        # Labels (unused in recommendation, but expected by model interface)
        labels=torch.zeros(max(Nu, 1), dtype=torch.long),
        labels_full=torch.zeros(max(N, 1), dtype=torch.long),
        n_classes=0,

        # Sizes
        N=N, Nu=Nu, Ni=Ni, Ne=N_entities,
        target_type='user',
        target_size=Nu,

        # Task configuration
        target_relation_idx=0,
        relation_info=relation_info,

        # Raw data for downstream feature engineering
        item_interaction_to_entity=item_interaction_to_entity,
        kg_edges_by_rel=dict(kg_edges_by_rel),
    )
