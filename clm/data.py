import json
from collections import OrderedDict
import random
random.seed(7676)

import numpy as np

import extraction

def to_categorical(y, num_classes, bptt, batch_size):
    y = np.array(y, dtype='int').ravel()
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical.reshape((batch_size, bptt, num_classes))

class ClmData(object):

    def __init__(self, batch_size, max_songs, json_path,
                 conditions, bptt):
        self.batch_size = batch_size
        self.bptt = bptt
        self.max_songs = max_songs
        self.json_path = json_path
        self.conditions = set(conditions)
        self.num_batches = None
        self.extractor = extraction.Extractor(conditions)

    def get_batches(self, endless=False):

        while True:
            # shuffle lines in json file before parsing in each epoch?

            batch_cnt = 0

            batch = {'syllables': [], 'targets': []}
            for c in self.conditions:
                batch[c] = []

            for song_idx, song in enumerate(open(self.json_path)):
                song = json.loads(song)

                song_data = OrderedDict()
                song_data['syllables'] = []  # always include syllables

                features = {}
                for cond in sorted(self.conditions):
                    song_data[cond] = []
                    if cond != 'rhythms':
                        features[cond] = self.extractor[cond](song)

                for verse in song['text']:
                    for line in verse:
                        rhythm_type = line['rhythm_type']
                        syllables = ['<BR>']
                        for word in line['tokens']:
                            for i, s in enumerate(word['syllables']):
                                if i < len(word['syllables']) - 1:
                                    syllables.append(s + '/')
                                else:
                                    syllables.append(s)

                        song_data['syllables'].extend(syllables)

                        for k in self.conditions:
                            if k != 'rhythms':
                                song_data[k].extend([features[k]] * len(syllables))
                            else:
                                song_data[k].extend([rhythm_type] * len(syllables))

                # shift start position randomly at beginning of song
                shift_ = random.choice(range(self.bptt))

                song_data['syllables'] = (shift_ - 1) * ['<PAD>'] + ['<BOS>'] + song_data['syllables']

                si, ei = 0, self.bptt

                while ei + 1 < len(song_data['syllables']):
                    batch['syllables'].append(song_data['syllables'][si:ei])

                    for c in self.conditions:
                        batch[c].append(song_data[c][si:ei])

                    batch['targets'].append(song_data['syllables'][si + 1: ei + 1])

                    si += self.bptt
                    ei += self.bptt

                    if len(batch['targets']) == self.batch_size:
                        batch_cnt += 1

                        yield batch

                        batch = {'syllables': [], 'targets': []}
                        for c in self.conditions:
                            batch[c] = []

                if self.max_songs and song_idx >= self.max_songs:
                    break

            if not self.num_batches:
                self.num_batches = batch_cnt

            if not endless:
                break

    def get_transformed_batches(self, vectorizers, endless=False):
        for batch in self.get_batches(endless=endless):
            batch_dict = {}
            for k, vectorizer in vectorizers.items():
                batch_dict[k] = vectorizer.transform(batch[k])
            
            X = {k: batch_dict[k] for k in ['syllables'] + list(self.conditions)}
            Y = vectorizers['syllables'].transform(batch['targets'])
            Y = to_categorical(Y, num_classes=vectorizers['syllables'].dim,
                               bptt=self.bptt, batch_size=self.batch_size)
            yield X, Y