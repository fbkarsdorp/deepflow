import collections
import json

import gensim
import torch


PAD, EOS, BOS, UNK = '<PAD>', '<EOS>', '<BOS>', '<UNK>'


def load_gensim_embeddings(fpath: str):
    model = gensim.models.KeyedVectors.load(fpath)
    # model.init_sims(replace=True)
    return model.index2word, model.syn0

def identity(x): return x

def word_boundaries(syllables):
    if syllables:
        return [1] + [0] * (len(syllables) - 1)
    return []

def normalize_stress(stress):
    return [0] if len(stress) == 1 else [int(s) if s != '.' else 0 for s in stress]

def normalize_beats(stress):
    return [int(s) if s != '.' else 0 for s in stress]

def lowercase(syllables):
    return [syllable.lower() for syllable in syllables]

def clean_syllables(syllables):
    return [syllable.strip('+,') for syllable in syllables]

def format_syllables(syllables):
    if len(syllables) == 1:
        return syllables
    return ['{}{}{}'.format('-' if i > 0 else '',
                            s,
                            '-' if (i < len(syllables) - 1) else '')
            for i, s in enumerate(syllables)]

class Encoder:
    def __init__(self, name, pad_token=PAD, eos_token=EOS, bos_token=BOS, unk_token=UNK,
                 vocab=None, fixed_vocab=False, preprocessor=identity, word_based=False):
        self.index = collections.defaultdict()
        self.index.default_factory = lambda: len(self.index)
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.unk_token = unk_token
        self.reserved_tokens = list(filter(None, (pad_token, eos_token, bos_token, unk_token)))
        self.name = name
        self.preprocessor = preprocessor
        self.word_based = word_based
        self.fixed_vocab = fixed_vocab
        self._unknowns = set()

        for token in self.reserved_tokens:
            self.index[token]
        if vocab is not None:
            for token in vocab:
                self.index[token]

    def __getitem__(self, item):
        if self.fixed_vocab:
            index = self.index.get(item, self.unk_index)
            if index == self.unk_index:
                self._unknowns.add(item)
            return index
        return self.index[item]

    def __len__(self):
        return len(self.index)

    def size(self):
        return len(self)

    @property
    def pad_index(self):
        return self.index[self.pad_token]

    @property
    def bos_index(self):
        return self.index[self.bos_token]

    @property
    def eos_index(self):
        return self.index[self.eos_token]

    @property
    def unk_index(self):
        return self.index[self.unk_token]

    def __repr__(self):
        return '<Encoder({})>'.format(self.name)

    def transform(self, sample):
        eos = [self.eos_index] if self.eos_token is not None else []
        bos = [self.bos_index] if self.bos_token is not None else []
        if not self.word_based:
            sample = bos + [
                self[elt] for item in sample for elt in self.preprocessor(item[self.name])
            ] + eos
        else:
            sample = bos + [self[''.join(map(str, self.preprocessor(item[self.name])))] for item in sample] + eos
        return torch.LongTensor(sample)

    def decode(self, sample):
        if not hasattr(self, 'index2item'):
            self.index2item = sorted(self.index, key=self.index.get)
        return [self.index2item[elt] for elt in sample]

    def unknowns(self):
        return self._unknowns


class Batch:
    def __init__(self, fields, include_length=True, length_field=None):

        self.include_length = include_length
        self.length_field = length_field

        if include_length and length_field is None:
            raise ValueError("`include_length` requires `length_field`")

        self.fields = set(fields)
        if length_field:
            self.fields.add('length')

        self.reset()

    def __len__(self):
        return self.size

    def add(self, data):
        done = set()
        for f, item in data.items():
            if f not in self.fields:
                raise ValueError("Got data for unexisting field: {}".format(f))
            if item is not None:
                self.data[f].append(item)
                done.add(f)

        if self.include_length:
            self.data['length'].append(len(data[self.length_field]))

        # check missing inputs
        missing = done.difference(self.fields)
        if missing:
            print("Missing fields: {}".format(', '.join(missing)))

        # increment
        self.size += 1

    def reset(self):
        self.data = {f: [] for f in self.fields}
        self.size = 0

    def get_batch(self):
        return self.data

    
class DataSet:
    def __init__(self, fpath, batch_size=1, **encoders):
        self.encoders = encoders
        self.fpath = fpath
        self.batch_size = batch_size

    def __iter__(self):
        return self.batches()

    def batches(self):
        batch = Batch(list(self.encoders.keys()) + ['song_id'], length_field='stress')

        for line in open(self.fpath):
            song = json.loads(line.strip())
            for verse in song['text']:
                for line in verse:
                    if not [s for w in line for s in w['syllables']]:
                        continue
                    # yield if needed
                    if len(batch) == self.batch_size:
                        yield batch.get_batch()
                        batch.reset()
                    # accumulate data
                    data = {f: t.transform(line) for f, t in self.encoders.items()}
                    data = {'song_id': song['id'], **data}
                    batch.add(data)

        if len(batch) > 0:
            yield batch.get_batch()


def chunks(it, size):
    buf = []
    for s in it:
        buf.append(s)
        if len(buf) == size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def buffer_groups(groups, batch_size):
    gchunks = chunks(groups, batch_size)

    while True:
        try:
            cgroups = [list(group) for group in next(gchunks)]
            # sort by number of lines per group in descending order
            cgroups = sorted(cgroups, key=len, reverse=True)
            max_lines = len(cgroups[0])

            for _ in range(max_lines):
                batch = []
                for group in cgroups:
                    try:
                        batch.append(group.pop(0))
                    except:
                        pass

                yield batch

        except StopIteration:
            break


class BlockDataSet(DataSet):
    def batches(self):
        batch = Batch(list(self.encoders.keys()) + ['song_id'], length_field='syllables')

        with open(self.fpath) as f:
            groups = (((song['id'], line) for verse in song['text'] for line in verse)
                      for song in ijson.items(f, 'item'))

            for lines in buffer_groups(groups, self.batch_size):
                for song_id, line in lines:
                    data = {f: t.transform(line) for f, t in self.encoders.items()}
                    data = {'song_id': song_id, **data}
                    batch.add(data)

                yield batch.get_batch()
                batch.reset()


if __name__ == '__main__':
    stress_encoder = Encoder('stress')
    beat_encoder = Encoder('beatstress')
    syllable_encoder = Encoder('syllables', preprocessor=format_syllables)
    wb_encoder = Encoder('syllables', preprocessor=word_boundaries)
    data = BlockDataSet('../data/lyrics-corpora/ohhla-beatstress.json', batch_size=5,
                        syllables=syllable_encoder)
    for batch in data.batches():
        pass
    print(syllable_encoder.size())
    print(syllable_encoder.index)

    def reverse(*bs):
        for lines in zip(*[b['syllables'] for b in bs]):
            for idx, line in enumerate( lines):
                print(idx, ' '.join(syllable_encoder.decode(line)))
            print()

    bat = data.batches()

    reverse(*[next(bat) for _ in range(20)])
