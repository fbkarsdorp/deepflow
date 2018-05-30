import collections
import ijson

import gensim
import torch


PAD, EOS, BOS, UNK = '<PAD>', '<EOS>', '<BOS>', '<UNK>'


def load_gensim_embeddings(fpath: str):
    model = gensim.models.KeyedVectors.load(fpath)
    # model.init_sims(replace=True)
    return model.index2word, model.vectors

def identity(x): return x

def word_boundaries(syllables):
    if syllables:
        return [1] + [0] * (len(syllables) - 1)
    return []

def normalize_stress(stress):
    return [int(s) if s != '.' else 0 for s in stress]


def format_syllables(syllables):
    if len(syllables) == 1:
        return syllables
    return ['{}{}{}'.format('-' if i > 0 else '',
                            s,
                            '-' if (i < len(syllables) - 1) else '')
            for i, s in enumerate(syllables)]


class Encoder:
    def __init__(self, name, pad_token=PAD, eos_token=EOS, bos_token=BOS, unk_token=UNK,
                 vocab=None, fixed_vocab=False, preprocessor=identity):
        self.index = collections.defaultdict()
        self.index.default_factory = lambda: len(self.index)
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.unk_token = unk_token
        self.reserved_tokens = list(filter(None, (pad_token, eos_token, bos_token, unk_token)))
        self.name = name
        self.preprocessor = preprocessor
        self.fixed_vocab = fixed_vocab

        for token in self.reserved_tokens:
            self.index[token]
        if vocab is not None:
            for token in vocab:
                self.index[token]

    def __getitem__(self, item):
        if self.fixed_vocab:
            return self.index.get(item, self.unk_index)
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
        sample = bos + [
            self[elt] for item in sample for elt in self.preprocessor(item[self.name])
        ] + eos
        return torch.LongTensor(sample)

    def decode(self, sample):
        if not hasattr(self, 'index2item'):
            self.index2item = sorted(self.index, key=self.index.get)
        return [self.index2item[elt] for elt in sample]


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

        with open(self.fpath) as f:
            for song in ijson.items(f, 'item'):
                for verse in song['text']:
                    for line in verse:
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
    syllable_vocab, syllable_embeddings = load_gensim_embeddings(
        '../data/syllable-embeddings/syllables.200.10.10.syllable.cbow.gensim')
    stress_encoder = Encoder('stress')
    beat_encoder = Encoder('beatstress')
    syllable_encoder = Encoder('syllables', vocab=syllable_vocab)
    wb_encoder = Encoder('syllables', preprocessor=word_boundaries)
    data = BlockDataSet('../data/ohhla-beatstress.json', batch_size=5,
                        syllables=syllable_encoder)

    def reverse(*bs):
        for lines in zip(*[b['syllables'] for b in bs]):
            for idx, line in enumerate( lines):
                print(idx, ' '.join(syllable_encoder.decode(line)))
            print()

    bat = data.batches()

    reverse(*[next(bat) for _ in range(20)])
