import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.data.amazon_loader import load_amazon
from src.data.pubmed_loader import load_pubmed

for name, loader in [('amazon', load_amazon), ('pubmed', load_pubmed)]:
    data = loader()
    print(f'\n=== {name} ===')
    print(f'  N: {data["N"]}')
    keys = list(data["X_dict"].keys())
    print(f'  X_dict keys: {keys}')
    for k in keys:
        print(f'    {k}: {tuple(data["X_dict"][k].shape)}')
    print(f'  A_list_sp: {[a.shape for a in data["A_list_sp"]]}')
    print(f'  relation_names: {data["relation_names"]}')
    print(f'  bipartite_flags: {data["bipartite_flags"]}')
    print(f'  target_relation_idx: {data.get("target_relation_idx", "?")}')
    print(f'  relation_info: {list(data["relation_info"].keys())}')
