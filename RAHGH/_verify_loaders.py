import sys; sys.path.insert(0, 'RAHGH')
from src.data.acm_loader import load_acm
from src.data.dblp_loader import load_dblp
from src.data.imdb_loader import load_imdb
from src.data.validate import validate_data_dict

def check(name, fn, **kw):
    try:
        data = fn(**kw)
        data['name'] = name
        print(f'\n=== {name} ===')
        print(f'Relation count: {len(data["A_list_sp"])}')
        for i, (n, b) in enumerate(zip(data['relation_names'], data['bipartite_flags'])):
            label = n.replace('\u2192', '->')
            print(f'  {i}: {label:30s} bipartite={b}')
        print(f'N={data["N"]}  target={data["target_type"]}[{data["target_size"]}]  classes={data["n_classes"]}')
        print(f'X_dict keys: {list(data["X_dict"].keys())}')
        validate_data_dict(data, name)
    except Exception as e:
        print(f'{name} load failed: {e}')

check('ACM', load_acm, root='..')
check('IMDB', load_imdb, root='..')

import os
if os.path.isdir("data/raw/DBLP"):
    check('DBLP', load_dblp)
else:
    print('\n=== DBLP ===\nData files not found at data/raw/DBLP')
