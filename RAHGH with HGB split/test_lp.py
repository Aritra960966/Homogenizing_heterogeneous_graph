import sys, os; sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'
from src.data.amazon_loader import load_amazon_ini
from src.data.pubmed_loader import load_pubmed_ini

d1 = load_amazon_ini()
print('amazon_ini:')
print('  N=%d, types=%s' % (d1['N'], list(d1['X_dict'].keys())))
te = d1.get('lp_test_edges')
if te:
    for rn, v in te.items():
        print('  test %s: %d edges, src=%s, dst=%s' % (rn, len(v['edges']), v['src_type'], v['dst_type']))
else:
    print('  NO lp_test_edges')

d2 = load_pubmed_ini()
print('pubmed_ini:')
print('  N=%d, types=%s' % (d2['N'], list(d2['X_dict'].keys())))
te2 = d2.get('lp_test_edges')
if te2:
    for rn, v in te2.items():
        print('  test %s: %d edges, src=%s, dst=%s' % (rn, len(v['edges']), v['src_type'], v['dst_type']))
else:
    print('  NO lp_test_edges')
