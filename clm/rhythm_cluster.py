from keras.layers import Input, LSTM, RepeatVector, Embedding, Dropout, TimeDistributed, Dense, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import LambdaCallback, ReduceLROnPlateau

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

import json

import numpy as np

from vectorization import SequenceVectorizer

def to_categorical(y, num_classes, bptt, batch_size):
    y = np.array(y, dtype='int').ravel()
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    for idx, p in enumerate(y):
        categorical[idx, p - 1] = 1  # ???
    return categorical.reshape((batch_size, bptt, num_classes))

class BeatGenerator:

    def __init__(self, json_path, batch_size=32, max_songs=None):
        self.json_path = json_path
        self.batch_size = batch_size
        self.num_batches = None
        self.max_songs = max_songs

    def get_batches(self, endless=False):
        batch = []
        batch_cnt = 0

        while True:
            for song_idx, song in enumerate(open(self.json_path)):
                song = json.loads(song)
                for verse in song['text']:
                    for line in verse:
                        verse = ['<BOS>']
                        for word in line:
                            verse.extend([str(s) for s in word['beatstress']])

                        batch.append(verse + ['<EOS>'])
                        if len(batch) >= self.batch_size:
                            batch_cnt += 1
                            yield batch
                            batch = []

                if self.max_songs and song_idx >= self.max_songs:
                    break

            if not self.num_batches:
                self.num_batches = batch_cnt

            if not endless:
                break

    def get_lines(self, max_lines=None):
        line_cnt = 0
        for song_idx, song in enumerate(open(self.json_path)):
            song = json.loads(song)
            for verse in song['text']:
                for line in verse:
                    beats, l = ['<BOS>'], ''
                    for word in line:
                        beats.extend([str(s) for s in word['beatstress']])
                        for s in word['syllables']:
                            l += s + ' '
                    line_cnt += 1
                    yield (l.replace('/ ', ''), beats + ['<EOS>'])
            if max_lines and line_cnt >= max_lines:
                break


    def get_transformed_batches(self, vectorizer, bptt, endless=False):
        for batch in self.get_batches(endless=endless):
            x = vectorizer.transform(batch)
            yield {'stresses': x}, to_categorical(x, num_classes=vectorizer.dim,
                                                  bptt=bptt, batch_size=self.batch_size)

def main():
    bptt = 30
    max_songs = 2000
    batch_size = 256
    hidden_dim = 10
    embed_dim = 5
    n_neighbors = 25
    n_clusters = 20
    dropout = 0.2
    epochs = 30
    lr = 0.001
    input_file = '../data/lazy_ohhla-beatstress.json'
    output_file = '../data/lazy_ohhla-beatfamilies.json'
    model_path = 'beat_autoencoder'

    cnt = 0
    vectorizer = SequenceVectorizer(min_cnt=0, bptt=bptt)

    generator = BeatGenerator(input_file, batch_size=batch_size,
                              max_songs=max_songs)

    for batch in generator.get_batches():
        vectorizer.partial_fit(batch)

    vectorizer.finalize_fit()

    input_ = Input(shape=(bptt,), dtype='int32', name='stresses')
    input_embed = Embedding(output_dim=embed_dim,
                            input_dim=vectorizer.dim,
                            input_length=bptt)(input_)
    encoded = Bidirectional(LSTM(hidden_dim, return_sequences=False), merge_mode='sum', name='encoder')(input_embed)
    encoded = Dropout(dropout)(encoded)

    decoder = RepeatVector(bptt)(encoded)
    decoder = LSTM(hidden_dim, return_sequences=True)(decoder)
    decoder = TimeDistributed(Dense(vectorizer.dim, activation='softmax'))(decoder)

    autoencoder = Model(input_, decoder)
    optim = Adam(lr)
    autoencoder.compile(optimizer=optim, loss='categorical_crossentropy')

    layer_name = 'encoder'
    repr_fn = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer(layer_name).output)
    repr_fn.compile(loss='categorical_crossentropy', optimizer='Adam')


    checkpoint = ModelCheckpoint(model_path, monitor='loss',
                                  verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3,
                                  patience=1, min_lr=0.000001,
                                  verbose=1, epsilon=0.03)

    try:
        autoencoder.fit_generator(generator=generator.get_transformed_batches(vectorizer, endless=True, bptt=bptt),
                            steps_per_epoch=generator.num_batches,
                            epochs=epochs,
                            callbacks=[checkpoint, reduce_lr])
    except KeyboardInterrupt:
        pass

    # get representations:
    lines, beats, reprs = [], [], []
    batch_lines, batch_beats = [], []
    for line, beat in generator.get_lines(max_lines=10000):
        batch_lines.append(line)
        batch_beats.append(beat)

        if len(batch_lines) >= batch_size:
            x = vectorizer.transform(batch_beats)
            p = repr_fn.predict(x)
            
            lines.extend(batch_lines)
            beats.extend(batch_beats)
            reprs.extend(p)

            batch_lines = []
            batch_beats = []

    # flush:
    if len(batch_lines):
        x = vectorizer.transform(batch_beats)
        p = repr_fn.predict(x)
            
        lines.extend(batch_lines)
        beats.extend(batch_beats)
        reprs.extend(p)

    reprs = np.array(reprs)
    print(len(lines), reprs.shape)

    km = MiniBatchKMeans(n_clusters=n_clusters)
    X_ = km.fit_predict(reprs)

    print(X_)

    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute')
    knn.fit(reprs)

    for idx, centroid in enumerate(km.cluster_centers_):
        print('cluster ', idx + 1)
        nns = knn.kneighbors([centroid], return_distance=False).ravel()
        for idx, nn in enumerate(nns):
            print('    ', lines[nn])
            print('    ', beats[nn])
            print('     ==================')

    with open(output_file, 'w') as f:
        for song_idx, song in enumerate(open(input_file)):
            beats = []

            song = json.loads(song)
            for verse in song['text']:
                for line in verse:
                    verse = ['<BOS>']
                    for word in line:
                        verse.extend([str(s) for s in word['beatstress']])
                    verse += ['<EOS>']
                    beats.append(verse)

            x = vectorizer.transform([beats])
            pred = repr_fn.predict(x)
            cluster_idxs = list(km.predict(pred).ravel())

            for verse in song['text']:
                for line in verse:
                    line['rhythm_type'] = 'r' + str(cluster_idxs.pop(0))
            f.write(json.dumps(song) + '\n')

            if max_songs and song_idx >= max_songs:
                break

    return



if __name__ == '__main__':
    main()




