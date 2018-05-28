import argparse
import logging
import re
import string

import gensim
import numpy as np
import ujson
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def song_iterator(fpath, max_songs=None, return_type='tokens'):
    logging.info("Loading dataset {}...".format(fpath))
    song_cnt = 0

    for line in open(fpath):
        song = ujson.loads(line)
        if return_type == 'tokens':
            tokens = []
            for verse in song['text']:
                for line in verse:
                    for i, word in enumerate(line):
                        token = word.get('token', word.get('word')).lower()
                        token = ''.join(c for c in token if c.isalpha())
                        if token:
                            tokens.append(token)
            yield ' '.join(tokens)
        elif return_type == 'id':
            yield song['id']

        song_cnt += 1
        if max_songs and song_cnt >= max_songs:
            break


if __name__ == '__main__':
    """
    python3.5 train_topics.py --training_file='../data/lazy_ohhla.json' --output_file='topics' --max_songs=100
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_file", type=str)
    parser.add_argument("--output_file", default='topic_', type=str)

    parser.add_argument("--min_df",  default=5, type=int)
    parser.add_argument("--max_songs",  default=None, type=int)
    parser.add_argument("--max_df",  default=.2, type=int)
    parser.add_argument("--max_features",  default=5000, type=int)

    parser.add_argument("--n_components", default=60, type=int)
    parser.add_argument("--max_iter", default=50, type=int)

    args = parser.parse_args()

    songs = song_iterator(args.training_file, args.max_songs)
    vectorizer = TfidfVectorizer(max_features=args.max_features,
                                 min_df=args.min_df,
                                 max_df=args.max_df)
    X = vectorizer.fit_transform(songs)

    nmf = NMF(n_components=args.n_components,
              random_state=878763,
              verbose=1, max_iter=args.max_iter)
    X_ = normalize(nmf.fit_transform(X), norm='l1')

    df = pd.DataFrame(X_, columns=['t' + str(i) for i in range(args.n_components)])
    df['id'] = list(song_iterator(args.training_file, args.max_songs, return_type='id')) 
    df.to_csv(args.output_file + '_repr.csv')

    for topic_idx, topic in enumerate(nmf.components_):
        top_idxs = np.argsort(topic)[::-1][:25]
        top_words = np.array(vectorizer.get_feature_names())[top_idxs]
        logging.info('topic' + str(topic_idx) + ' -> ' + ' - '.join(top_words))

    joblib.dump(vectorizer, args.output_file + '_vector.pkl')
    joblib.dump(nmf, args.output_file + '_nmf.pkl')

    logging.info("vectorizer and model saved")
