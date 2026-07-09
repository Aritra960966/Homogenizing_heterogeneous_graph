import sys; sys.path.insert(0, '.')
from src.data.pubmed_loader import load_pubmed
data = load_pubmed()
lpt = data.get('lp_test_edges', {})
for k, v in lpt.items():
    print("  %s: %d test edges, src=%s, dst=%s" % (k, len(v['edges']), v['src_type'], v['dst_type']))
print()
print('Total A_list entries:')
for i, (rn, A) in enumerate(zip(data['relation_names'], data['A_list_sp'])):
    print("  %d: %s -> %d edges" % (i, rn, A.nnz))
