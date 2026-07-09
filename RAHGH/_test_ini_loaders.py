import sys; sys.path.insert(0, '.')
from src.data.amazon_loader import load_amazon_ini
from src.data.pubmed_loader import load_pubmed_ini
from src.train import TARGET_REL_IDX

for name, loader in [('amazon_ini', load_amazon_ini), ('pubmed_ini', load_pubmed_ini)]:
    data = loader()
    shapes = [a.shape for a in data['A_list_sp']]
    print(f'{name}: N={data["N"]}, target_size={data["target_size"]}, A_list={shapes}, target_rel_idx={TARGET_REL_IDX[name]}')
