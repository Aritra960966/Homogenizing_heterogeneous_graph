import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def load_imdb(root="data/raw/IMDB"):
    df = pd.read_csv(f"{root}/movie_metadata.csv", encoding='latin-1')
    df['movie_title']   = df['movie_title'].str.strip().str.replace('Â','',regex=False)
    df['genre_primary'] = df['genres'].fillna('').apply(lambda x: x.split('|')[0])

    genre_map = {'Action':0, 'Comedy':1, 'Drama':2}
    df = df[df['genre_primary'].isin(genre_map)].reset_index(drop=True)
    Nm = len(df)

    df['director_name'] = df['director_name'].fillna('Unknown_Director')
    dir_list = sorted(df['director_name'].unique())
    dir2idx  = {d: i for i, d in enumerate(dir_list)}
    Nd       = len(dir_list)

    actor_cols = ['actor_1_name','actor_2_name','actor_3_name']
    for c in actor_cols: df[c] = df[c].fillna('Unknown_Actor')
    act_list = sorted(pd.concat([df[c] for c in actor_cols]).unique())
    act2idx  = {a: i for i, a in enumerate(act_list)}
    Na       = len(act_list)
    N        = Nm + Nd + Na

    m_off, d_off, a_off = 0, Nm, Nm + Nd

    def build_coo(rows, cols):
        return sp.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(N,N)).tocsr()

    md_r, md_c, ma_r, ma_c = [], [], [], []
    for i, row in df.iterrows():
        d = dir2idx.get(row['director_name'])
        if d is not None: md_r.append(i); md_c.append(d_off + d)
        for col in actor_cols:
            a = act2idx.get(row[col])
            if a is not None: ma_r.append(i); ma_c.append(a_off + a)

    MD = build_coo(md_r, md_c); DM = MD.T.tocsr()
    MA = build_coo(ma_r, ma_c); AM = MA.T.tocsr()
    A_list_sp = [MD, DM, MA, AM]
    relation_names = ['movie→dir','dir→movie',
                      'movie→act','act→movie']

    keywords = df['plot_keywords'].fillna('').str.replace('|',' ',regex=False)
    vec      = CountVectorizer(max_features=3000)
    X_movie  = torch.tensor(vec.fit_transform(keywords).toarray(), dtype=torch.float32)
    X_dir    = torch.eye(Nd, dtype=torch.float32)
    X_act    = torch.eye(Na, dtype=torch.float32)

    labels   = torch.tensor([genre_map[g] for g in df['genre_primary']], dtype=torch.long)

    return dict(
        A_list_sp=A_list_sp, relation_names=relation_names,
        X_dict={'movie':X_movie, 'director':X_dir, 'actor':X_act},
        labels=labels,
        Nm=Nm, Nd=Nd, Na=Na, N=N,
        target_type='movie', target_size=Nm,
        n_classes=3,
        bipartite_flags=[True,True,True,True],
    )
