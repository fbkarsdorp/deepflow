import ujson
import logging

import numpy as np

import gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SPACE = '<SPACE>'

class VerseIterator(object):

    def __init__(self, songsfile, max_songs=None, include_space=True):
        self.songsfile = songsfile
        self.max_songs = max_songs
        self.include_space = include_space

    def __iter__(self):
        cnt = 0

        for song in open(self.songsfile, 'r'):
            song = ujson.loads(song)
            for verse in song['text']:
                for line in verse:
                    syllables = []
                    for word in line:
                        syllables.extend(word['syllables'])
                        if self.include_space:
                            syllables.append(SPACE)
                        # Q: include space symbols?
                    yield syllables

            cnt += 1
            if self.max_songs and cnt >= self.max_songs:
                break

def get_lazy(oldfile, newfile):
    with open(oldfile, 'r') as f:
        songs = ujson.load(f)

    with open(newfile, 'w') as f:
        for song in songs:
            f.write(ujson.dumps(song) + '\n')

def pretrain_embeddings(songsfile, modelfile):
    verses = VerseIterator(songsfile=songsfile,
                           max_songs=None, include_space=True)
    model = gensim.models.Word2Vec(verses, min_count=10,
                           size=300, window=10, workers=4)
    model.save(modelfile)

def load_embeddings(modelfile):
    model = gensim.models.Word2Vec.load(modelfile)
    model.init_sims(replace=True)
    vocab = tuple(sorted(set(model.wv.vocab.keys())))
    weights = np.array([model[w] for w in vocab], dtype=np.float32)

    return vocab, weights


def main():
    #get_lazy('../data/ohhla-enriched.json', '../data/lazy_ohhla.json')
    pretrain_embeddings(songsfile='../data/lazy_ohhla.json',
                        modelfile='syllable_embeddings')
    vocab, weights = load_embeddings('syllable_embeddings')
    print(weights.shape)


if __name__ == '__main__':
    main()
