import argparse
import logging

import gensim
import numpy as np
import ujson


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_data(fpath):
    logging.info("Loading dataset... {}".format(fpath))
    with open(fpath) as f:
        data = ujson.load(f)
    lines = []
    for song in data:
        for verse in song['text']:
            for line in song:
                words = []
                for word in line:
                    words.extend(word['syllables'])
                    words.append("<SPACE>")
                lines.append(words)
    logging.info("Loading done!")
    return lines


def train_model(data, output, min_count, dim, window, worker, model):
    sg = 1 if model == 'skipgram' else 0
    model = gensim.models.Word2Vec(
        data, min_count=min_count, size=dim, window=window, workers=workers, sg=sg)
    model.save(output)


def load_embeddings(fpath):
    model = gensim.models.Word2Vec.load(fpath)
    model.init_sims(replace=True)
    vocab = tuple(sorted(model.wv.vocab.keys()))
    weights = np.array([model[w] for w in vocab], dtype=np.float32)
    return vocab, weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_files", nargs='+')
    parser.add_argument("--output_file")
    parser.add_argument("--dim", default=300, type=int)
    parser.add_argument("--window", default=5, type=int)
    parser.add_argument("--model", default="cbow")
    parser.add_argument("--workers", default=4)
    parser.add_argument("--min_count", default=5, type=int)
    args = parser.parse_args()

    lines = []
    for file in args.training_files:
        lines.extend(load_data(file))
    train_model(lines, args.output_file, args.min_count, args.dim,
                args.window, args.workers, args.model)

