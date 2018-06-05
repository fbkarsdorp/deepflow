from collections import Counter
from itertools import chain
import json
import numpy as np

from sklearn.preprocessing import LabelEncoder

SYMBOLS = ['<PAD>', '<UNK>', '<BOS>']


class SequenceVectorizer(object):
    def __init__(self, min_cnt=0, bptt=15,
                 syll2idx=None, idx2syll=None):
        self.min_cnt = min_cnt
        self.bptt = bptt
        self.syll2idx = syll2idx
        self.idx2syll = {}
        if idx2syll:
            for s in idx2syll:
                self.idx2syll[int(s)] = idx2syll[s]

        self.cnt = Counter()
        self.fitted = False

    def partial_fit(self, batch):
        self.cnt.update(chain(*batch))

    def finalize_fit(self):
        self.syll2idx = {}
        for syll in SYMBOLS + sorted([k for k, v in self.cnt.most_common()
                                      if v >= self.min_cnt]):
            self.syll2idx[syll] = len(self.syll2idx)

        # construct a dict mapping indices to characters:
        self.idx2syll = {i: s for s, i in self.syll2idx.items()}
        self.dim = len(self.syll2idx)

        return self

    def transform(self, lines):
        if self.fitted:
            self.finalize_fit()

        X = []
        for line in lines:
            x = []
            for syll in line:
                try:
                    x.append(self.syll2idx[syll])
                except KeyError:
                    x.append(self.syll2idx['<UNK>'])
                # truncate longer tokens
                if len(x) >= self.bptt:
                    break
            # left-pad shorter BPTT:
            while len(x) < self.bptt:
                x = [self.syll2idx['<PAD>']] + x
            X.append(x)

        return np.array(X, dtype=np.int32)

    def inverse_transform(self, lines):
        strs = []
        for line in lines:
            sylls = [self.idx2syll[int(syll)] for syll in line]
            strs.append(sylls)
        return strs

    def fit_transform(self, tokens):
        return self.fit(tokens).transform(tokens)

    def dump(self, path):
        with open(path, 'w') as f:
            json.dump({'min_cnt': self.min_cnt,
                       'max_len': self.max_len,
                       'idx2syll': self.idx2syll,
                       'syll2idx': self.syll2idx}, f)

    @classmethod
    def load(self, path):
        with open(path, 'r') as f:
            params = json.load(f)
            return Vocabulary(**params)
