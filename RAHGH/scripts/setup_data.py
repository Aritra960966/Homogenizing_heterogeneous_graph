import shutil
import os
import sys


def setup_data():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_raw = os.path.join(root, 'data', 'raw')
    src_dir = os.path.dirname(root)

    acm_mat_src = os.path.join(src_dir, 'ACM.mat')
    acm_mat_dst = os.path.join(data_raw, 'ACM', 'ACM.mat')
    if os.path.exists(acm_mat_src) and not os.path.exists(acm_mat_dst):
        shutil.copy2(acm_mat_src, acm_mat_dst)
        print(f"Copied ACM.mat")

    movie_csv_src = os.path.join(src_dir, 'movie_metadata.csv')
    movie_csv_dst = os.path.join(data_raw, 'IMDB', 'movie_metadata.csv')
    if os.path.exists(movie_csv_src) and not os.path.exists(movie_csv_dst):
        shutil.copy2(movie_csv_src, movie_csv_dst)
        print(f"Copied movie_metadata.csv")

    print("Data setup complete.")
    print("Note: DBLP raw text files must be placed in data/raw/DBLP/")
    print("  Required: author_label.txt, paper_author.txt, paper.txt,")
    print("            paper_term.txt, paper_conf.txt, term.txt, conf.txt")


if __name__ == '__main__':
    setup_data()
