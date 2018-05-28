from collections import Counter
from itertools import chain
import json
import numpy as np

from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

SYMBOLS = ['<PAD>', '<UNK>']


class LabelVectorizer(object):
    def __init__(self, min_cnt=0):
        self.classes = Counter()
        self.min_cnt = min_cnt

    def partial_fit(self, batch):
        self.classes.update(batch)

    def finalize_fit(self):
        self.class2idx = {}
        for cl in sorted([k for k, v in self.classes.most_common() if v >= self.min_cnt]):
            self.class2idx[cl] = len(self.class2idx)

        self.idx2class = {i: s for s, i in self.class2idx.items()}
        self.encoder = LabelEncoder().fit(list(self.class2idx.keys()) + ['<UNK>'])
        self.dim = len(self.encoder.classes_)

        self.fitted = True

    def transform(self, labels):
        if not self.fitted:
            self.finalize_fit()

        labels = [s if s in self.encoder.classes_ else '<UNK>' for s in labels]

        return to_categorical(self.encoder.transform(labels),
                              num_classes=self.dim)


class SequenceVectorizer(object):
    def __init__(self, min_cnt=0, max_len=None,
                 syll2idx=None, idx2syll=None):
        self.min_cnt = min_cnt
        self.max_len = max_len
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

        for k in self.syll2idx:
            if 'eminem' in k.lower():
                print(k)

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
                if len(x) >= self.max_len:
                    break
            # left-pad shorter BPTT:
            while len(x) < self.max_len:
                x = ['<PAD>'] + x
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
