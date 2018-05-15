import ujson
import logging
from itertools import chain
from copy import deepcopy

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import accuracy_score

from pretrain import load_embeddings
from vectorization import Vocabulary
from utils import pad, make_splits, VerseIterator
from modelling import build_model


def main():
    embeddingsfile = 'syllable_embeddings'
    datafile = '../data/mcflow.json'
    modelpath = 'scanning'
    max_len = 20
    max_songs = None
    include_space = False
    random_state = 1066
    train_prop = 0.9
    num_layers = 2
    recurrent_dim = 512
    learning_rate = 0.001
    batch_size = 32
    patience = 2
    num_epochs = 10
    freeze = False

    vocab, weights = load_embeddings(embeddingsfile)
    
    # preinitialize vocabulary:
    idx2syll = {0: '<PAD>', 1: '<UNK>'}
    for w in vocab:
        idx2syll[len(idx2syll)] = w
    syll2idx = {v: k for k, v in idx2syll.items()}

    weights = np.vstack((np.zeros((2, weights.shape[1])), weights)) # better initialization?

    vocab = Vocabulary(max_len=max_len, idx2syll=idx2syll,
                       syll2idx=syll2idx)

    verses = VerseIterator(songsfile=datafile,
                           max_songs=max_songs, include_space=include_space,
                           return_type='syllables')

    X = vocab.transform(verses)
    Y = pad(VerseIterator(songsfile=datafile,
                           max_songs=max_songs, include_space=include_space,
                           return_type='stresses'), max_syllables=max_len)

    assert X.shape[0] == Y.shape[0]

    train_X, dev_X, test_X, train_Y_int, dev_Y_int, test_Y_int = make_splits(X, Y,
                         random_state=random_state,
                         train_prop=train_prop)

    train_Y = to_categorical(train_Y_int, num_classes=2)
    dev_Y = to_categorical(dev_Y_int, num_classes=2)
    test_Y = to_categorical(test_Y_int, num_classes=2)

    print(train_X.shape)
    print(train_Y.shape)
    print(dev_X.shape)
    print(dev_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)

    model = build_model(seq_len=max_len,
                        vocab_size=len(vocab.syll2idx),
                        num_layers=num_layers,
                        embed_dim=weights.shape[1],
                        recurrent_dim=recurrent_dim,
                        embedding_weights=weights,
                        learning_rate=learning_rate,
                        freeze=freeze)
    model.summary()

    checkpoint = ModelCheckpoint(modelpath, monitor='val_loss',
                                 verbose=1, save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss', min_delta=0,
                            patience=patience, verbose=1)

    try:
        model.fit({'syll': train_X}, {'out': train_Y},
                  validation_data=({'syll': dev_X}, {'out': dev_Y}),
                  batch_size=batch_size, epochs=num_epochs,
                  shuffle=True,
                  callbacks=[checkpoint, stopper])
    except KeyboardInterrupt:
        pass

    del model
    model = load_model(modelpath)

    test_pred = model.predict({'syll': test_X}).argmax(axis=-1).astype('int32')

    print('Verse accuracy: ',
          accuracy_score(test_pred, test_Y_int))

    print('Syllable accuracy: ',
          accuracy_score(list(chain(*test_pred)),
                         list(chain(*test_Y_int))))


if __name__ == '__main__':
    main()
