import json
from collections import OrderedDict
import random
random.seed(7676)

import extraction

extractors = {'artists': extraction.extract_artist,
              'topics': extraction.extract_topic}

class ClmData(object):

    def __init__(self, batch_size, max_songs, json_path,
                 conditions, bptt, shift):
        self.batch_size = batch_size
        self.bptt = bptt
        self.shift = shift
        self.max_songs = max_songs
        self.json_path = json_path
        self.conditions = set(conditions)
        self.num_batches = None

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
                    features[cond] = extractors[cond](song)

                for verse in song['text']:
                    for line in verse:
                        syllables = ['<BR>']
                        for word in line:
                            #for syllable in word['syllables']:
                            #    syllables.extend(syllable)
                            #syllables.append(' ')

                            for i, s in enumerate(word['syllables']):
                                if i < len(word['syllables']) - 1:
                                    syllables.append(s + '/')
                                else:
                                    syllables.append(s)

                        song_data['syllables'].extend(syllables)

                        for k in self.conditions:
                            song_data[k].extend([features[k]] * len(syllables))

                # shift start position randomly at beginning of song
                shift_ = random.choice(range(self.shift))
                si, ei = 0 + shift_, self.bptt + shift_

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
            yield X, Y