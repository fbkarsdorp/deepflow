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
    return [1] + [0] * (len(syllables) - 1)


class Encoder:
    def __init__(self, name, pad_token=PAD, eos_token=EOS, bos_token=BOS, unk_token=UNK,
                 vocab=None, preprocessor=identity):
        self.index = collections.defaultdict()
        self.index.default_factory = lambda: len(self.index)
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.unk_token = unk_token
        self.name = name
        self.preprocessor = preprocessor

        for token in (pad_token, eos_token, bos_token, unk_token):
            self.index[token]
        if vocab is not None:
            for token in vocab:
                self.index[token]

    def __len__(self):
        return len(self.index)

    def transform(self, sample, max_seq_len):
        eos = [self.index[self.eos_token]] if self.eos_token is not None else []
        sample = [self.index[elt] for item in sample for elt in self.preprocessor(item[self.name])] + eos
        sample = sample + [self.index[self.pad_token]] * (max_seq_len - len(sample) - 1)
        return sample

    
class DataSet:
    def __init__(self, fpath, max_seq_len=30, batch_size=1, **encoders):
        self.encoders = encoders
        self.fpath = fpath
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def __iter__(self):
        return self.batches()

    def batches(self):
        batch_size = 0
        sample = {f: [] for f in self.encoders.keys()}
        with open(self.fpath) as f:
            songs = ijson.items(f, 'item')
            for song in songs:
                for verse in song['text']:
                    for line in verse:
                        for f, t in self.encoders.items():
                            sample[f].append(t.transform(line, self.max_seq_len))
                        sample['length'].append(sample[f][-1].index(t.pad_index))
                        batch_size += 1
                        if batch_size == self.batch_size:
                            yield self.make_tensors(sample)
                            batch_size = 0
                            sample = {f: [] for f in self.encoders.keys()}
                            sample['length'] = []                            
            if sample:
                yield self.make_tensors(sample)

    def make_tensors(self, sample):
        return {f: torch.LongTensor(item) for f, item in sample.items()}
                    
                
if __name__ == '__main__':
    syllable_vocab, syllable_embeddings = load_gensim_embeddings(
        '../data/syllable-embeddings/syllables.200.10.10.fasttext.gensim')
    stress_encoder = Encoder('stress')
    beat_encoder = Encoder('beatstress')
    syllable_encoder = Encoder('syllables', vocab=syllable_vocab)
    wb_encoder = Encoder('syllables', preprocessor=word_boundaries)
    data = DataSet('../data/mcflow/mcflow-primary-recip.json',
                   stress=stress_encoder, syllables=syllable_encoder,
                   beat=beat_encoder, wb=wb_encoder)
        

