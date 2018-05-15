from collections import Counter
from itertools import chain
import json
import numpy as np

SYMBOLS = ['<PAD>', '<UNK>']


class Vocabulary:
    def __init__(self, min_cnt=0, max_len=None,
                 syll2idx=None, idx2syll=None):
        self.min_cnt = min_cnt
        self.max_len = max_len
        self.syll2idx = syll2idx
        self.idx2syll = {}
        if idx2syll:
            for s in idx2syll:
                self.idx2syll[int(s)] = idx2syll[s]

    def fit(self, lines):
        cnt = Counter(chain(*lines))

        # construct a dict mapping characters to an index
        # we add special symbols for unknown characters (UNK),
        # padding (PAD) and the beginning (BOS) and end (EOS)
        self.syll2idx = {}
        for syll in SYMBOLS + sorted([k for k, v in cnt.most_common()
                                      if v >= self.min_cnt]):
            self.syll2idx[syll] = len(self.syll2idx)

        x = sorted(self.syll2idx.items(), key=lambda x: x[1])

        # construct a dict mapping indices to characters:
        self.idx2syll = {i: s for s, i in self.syll2idx.items()}

        # determine max length if required:
        if not self.max_len:
            self.max_len = max([len(t) for t in lines])
        else:
            self.max_len

        return self

    def transform(self, lines):
        if not self.syll2idx or not self.idx2syll:
            raise NotFittedError('Vocab not fitted yet...')

        X = []
        for line in lines:
            x = []
            for syll in line:
                try:
                    x.append(self.syll2idx[syll])
                except KeyError:
                    x.append(self.syll2idx['<UNK>'])
                # truncate longer tokens
                if len(x) >= self.max_len:
                    break
            # pad shorter tokens:
            while len(x) < self.max_len:
                x.append(self.syll2idx['<PAD>'])
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
