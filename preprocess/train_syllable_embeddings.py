import argparse
import logging
import re
import string

import gensim
import numpy as np
import json


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


PUNCT_RE = re.compile(r'[^\w\s]+$')
def is_punct(string):
    return PUNCT_RE.match(string) is not None

ft_home = '/home/folgert/local/fastText-0.1.0/fasttext'

def load_data(fpath):
    logging.info("Loading dataset... {}".format(fpath))
    lines = []    
    for line in open(fpath):
        song = json.loads(line.strip())
        for verse in song['text']:
            for line in verse:
                words = []
                n_words = len(line)
                for i, word in enumerate(line):
                    token = word.get('token', word.get('word'))
                    if not is_punct(token):
                        syllables = []
                        for j, syllable in enumerate(word['syllables']):
                            if len(word['syllables']) > 1:
                                if j == 0:
                                    syllable = syllable + '-'
                                elif j == (len(word['syllables']) - 1):
                                    syllable = '-' + syllable
                                else:
                                    syllable = '-' + syllable + '-'
                            words.append(syllable.lower())
                lines.append(words)
    logging.info("Loading done!")
    return lines


def train_model(data, output, min_count, dim, window, workers, model):
    sg = 1 if model == 'skipgram' else 0
    if model == 'fasttext':
        model = gensim.models.FastText(size=dim, window=window, workers=workers, min_count=min_count)
        model.build_vocab(data)
        model.train(data, total_examples=model.corpus_count, epochs=10)
    else:
        model = gensim.models.Word2Vec(
            data, min_count=min_count, size=dim, window=window, workers=workers, sg=sg)
    model.wv.save(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_files", nargs='+')
    parser.add_argument("--output_file")
    parser.add_argument("--dim", default=300, type=int)
    parser.add_argument("--window", default=5, type=int)
    parser.add_argument("--model", default="cbow", choices=('skipgram', 'cbow', 'fasttext'))
    parser.add_argument("--workers", default=4)
    parser.add_argument("--min_count", default=5, type=int)
    args = parser.parse_args()

    lines = []
    for file in args.training_files:
        lines.extend(load_data(file))
    train_model(lines, args.output_file, args.min_count, args.dim,
                args.window, args.workers, args.model)

