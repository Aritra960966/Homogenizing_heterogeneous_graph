import sys, os, numpy as np
sys.path.insert(0, 'D:/Aritra/graph/Homogenizing_heterogeneous_graph/RAHGH/src')
sys.path.insert(0, 'D:/Aritra/graph/Homogenizing_heterogeneous_graph')
import RAHGH.src.train as train_mod
train_mod.N_SEEDS = 10
data = train_mod._get_loader('amazon')
data['name'] = 'amazon'
split_dir = 'D:\Aritra\graph\Homogenizing_heterogeneous_graph\HGB\splits\lp\amazon'
if os.path.exists(split_dir):
    tr = np.load(os.path.join(split_dir, 'train_indices.npy'))
    va = np.load(os.path.join(split_dir, 'val_indices.npy'))
    te = np.load(os.path.join(split_dir, 'test_indices.npy'))
    data['train_indices'] = np.concatenate([tr, va])
    data['test_indices'] = te
    print(f'HGB split: {len(tr)} train + {len(va)} val + {len(te)} test')
fn = train_mod.TASK_FNS['lp']
fn('amazon', 'D:\Aritra\graph\Homogenizing_heterogeneous_graph\HGB\results\lp\amazon')
