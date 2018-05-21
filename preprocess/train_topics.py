import argparse
import logging
import re
import string

import gensim
import numpy as np
import ujson

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.decomposition import NMF


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def song_iterator(fpath):
    logging.info("Loading dataset... {}".format(fpath))
    with open(fpath) as f:
        data = ujson.load(f)

    lines = []
    for song in data:
        tokens = []
        for verse in song['text']:
            for line in verse:
                for i, word in enumerate(line):
                    token = word.get('token', word.get('word')).lower()
                    token = ''.join(c for c in token if c.isalpha())
                    if token:
                        tokens.append(token)
        yield ' '.join(tokens)


if __name__ == '__main__':
    """
    python3 train_topics.py --training_file='../data/ohhla-enriched.json' --output_file='topics'
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_file", type=str)
    parser.add_argument("--output_file", default='topic_vectorizer', type=str)

    parser.add_argument("--min_df",  default=5, type=int)
    parser.add_argument("--max_df",  default=.2, type=int)
    parser.add_argument("--max_features",  default=5000, type=int)

    parser.add_argument("--n_components", default=60, type=int)
    parser.add_argument("--max_iter", default=50, type=int)

    args = parser.parse_args()

    songs = song_iterator(args.training_file)
    vectorizer = TfidfVectorizer(max_features=args.max_features,
                                 min_df=args.min_df,
                                 max_df=args.max_df)
    X = vectorizer.fit_transform(songs)

    nmf = NMF(n_components=args.n_components,
              random_state=878763,
              verbose=1, max_iter=args.max_iter).fit(X)

    for topic_idx, topic in enumerate(nmf.components_):
        top_idxs = np.argsort(topic)[::-1][:25]
        top_words = np.array(vectorizer.get_feature_names())[top_idxs]
        logging.info('topic' + str(topic_idx) + '->' + ' - '.join(top_words))

    joblib.dump(vectorizer, args.output_file + '_vector.pkl')
    joblib.dump(vectorizer, args.output_file + '_nmf.pkl')

    logging.info("vectorizer and model saved")
