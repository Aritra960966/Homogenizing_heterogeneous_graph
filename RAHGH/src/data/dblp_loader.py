import warnings
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sk_sw
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_sw
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning,
                        module='sklearn.feature_extraction.text')


class LemmaTokenizer:
    def __init__(self): self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def load_dblp(root="data/raw/DBLP"):
    author_label = pd.read_csv(f"{root}/author_label.txt", sep='\t',
                                header=None,
                                names=['author_id','label','author_name'],
                                keep_default_na=False, encoding='utf-8')
    paper_author = pd.read_csv(f"{root}/paper_author.txt", sep='\t',
                                header=None, names=['paper_id','author_id'],
                                keep_default_na=False, encoding='utf-8')
    paper_conf   = pd.read_csv(f"{root}/paper_conf.txt",   sep='\t',
                                header=None, names=['paper_id','conf_id'],
                                keep_default_na=False, encoding='utf-8')
    paper_term   = pd.read_csv(f"{root}/paper_term.txt",   sep='\t',
                                header=None, names=['paper_id','term_id'],
                                keep_default_na=False, encoding='utf-8')
    papers       = pd.read_csv(f"{root}/paper.txt",         sep='\t',
                                header=None, names=['paper_id','paper_title'],
                                keep_default_na=False, encoding='cp1252')
    terms        = pd.read_csv(f"{root}/term.txt",          sep='\t',
                                header=None, names=['term_id','term'],
                                keep_default_na=False, encoding='utf-8')
    confs        = pd.read_csv(f"{root}/conf.txt",          sep='\t',
                                header=None, names=['conf_id','conf'],
                                keep_default_na=False, encoding='utf-8')

    labeled = author_label['author_id'].tolist()
    paper_author = paper_author[paper_author['author_id'].isin(labeled)].reset_index(drop=True)
    valid_papers = paper_author['paper_id'].unique()
    papers       = papers[papers['paper_id'].isin(valid_papers)].reset_index(drop=True)
    paper_conf   = paper_conf[paper_conf['paper_id'].isin(valid_papers)].reset_index(drop=True)
    paper_term   = paper_term[paper_term['paper_id'].isin(valid_papers)].reset_index(drop=True)
    valid_terms  = paper_term['term_id'].unique()
    terms        = terms[terms['term_id'].isin(valid_terms)].reset_index(drop=True)

    author_label = author_label.sort_values('author_id').reset_index(drop=True)
    papers       = papers.sort_values('paper_id').reset_index(drop=True)
    terms        = terms.sort_values('term_id').reset_index(drop=True)
    confs        = confs.sort_values('conf_id').reset_index(drop=True)

    Na = len(author_label)
    Np = len(papers)
    Nt = len(terms)
    Nc = len(confs)
    N  = Na + Np + Nt + Nc

    a_map = {row['author_id']: i          for i, row in author_label.iterrows()}
    p_map = {row['paper_id']:  i + Na     for i, row in papers.iterrows()}
    t_map = {row['term_id']:   i + Na+Np  for i, row in terms.iterrows()}
    c_map = {row['conf_id']:   i + Na+Np+Nt for i, row in confs.iterrows()}
    p_local = {row['paper_id']: i for i, row in papers.iterrows()}

    def build_coo(rows, cols, shape):
        return sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                              shape=shape).tocsr()

    pa_r, pa_c = [], []
    for _, row in paper_author.iterrows():
        p = p_map.get(row['paper_id']); a = a_map.get(row['author_id'])
        if p and a is not None: pa_r.append(p); pa_c.append(a)
    PA = build_coo(pa_r, pa_c, (N, N))
    AP = PA.T.tocsr()

    pt_r, pt_c = [], []
    for _, row in paper_term.iterrows():
        p = p_map.get(row['paper_id']); t = t_map.get(row['term_id'])
        if p and t: pt_r.append(p); pt_c.append(t)
    PT = build_coo(pt_r, pt_c, (N, N))
    TP = PT.T.tocsr()

    pc_r, pc_c = [], []
    for _, row in paper_conf.iterrows():
        p = p_map.get(row['paper_id']); c = c_map.get(row['conf_id'])
        if p and c: pc_r.append(p); pc_c.append(c)
    PC = build_coo(pc_r, pc_c, (N, N))
    CP = PC.T.tocsr()

    A_list_sp = [AP, PA, TP, PT, CP, PC]
    relation_names = ['author→paper','paper→author',
                      'term→paper','paper→term',
                      'conf→paper','paper→conf']

    stopwords = list(sk_sw.union(set(nltk_sw.words('english'))))
    vec = CountVectorizer(min_df=2, stop_words=stopwords,
                          tokenizer=LemmaTokenizer())
    X_paper_np = vec.fit_transform(papers['paper_title'].values
                                   ).toarray().astype(np.float32)
    d_paper = X_paper_np.shape[1]

    X_author_np = np.zeros((Na, d_paper), dtype=np.float32)
    for _, row in paper_author.iterrows():
        a = a_map.get(row['author_id'])
        p = p_local.get(row['paper_id'])
        if a is not None and p is not None:
            X_author_np[a] += X_paper_np[p]
    norms = np.linalg.norm(X_author_np, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X_author_np /= norms

    X_dict = {
        'author': torch.tensor(X_author_np),
        'paper' : torch.tensor(X_paper_np),
        'term'  : torch.eye(Nt),
        'conf'  : torch.eye(Nc),
    }

    labels = torch.tensor(author_label['label'].to_numpy(), dtype=torch.long)

    return dict(
        A_list_sp=A_list_sp, relation_names=relation_names,
        X_dict=X_dict, labels=labels,
        Na=Na, Np=Np, Nt=Nt, Nc=Nc, N=N,
        target_type='author', target_size=Na,
        n_classes=len(author_label['label'].unique()),
        bipartite_flags=[True, True, True, True, True, True],
        relation_info={
            'author→paper': ('author', 'paper'),
            'paper→author': ('paper', 'author'),
            'term→paper':   ('term', 'paper'),
            'paper→term':   ('paper', 'term'),
            'conf→paper':   ('conf', 'paper'),
            'paper→conf':   ('paper', 'conf'),
        },
    )
