import numpy as np

from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout, TimeDistributed
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import LambdaCallback, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam

from collections import OrderedDict

def build_model(conditions, bptt, vectorizers, dropout,
                syll_emb_dim, cond_emb_dim, lstm_dim, lr):
    
    # inputs:
    input_dict = OrderedDict()
    input_dict['syllables'] = Input(shape=(bptt,), dtype='int32', name='syllables')

    for c in sorted(conditions):
        input_dict[c] = Input(shape=(bptt,), dtype='int32', name=c)

    # embeddings:
    embed_dict = OrderedDict()
    embed_dict['syllables'] = Embedding(output_dim=syll_emb_dim,
                             input_dim=vectorizers['syllables'].dim,
                             #mask_zero=True,
                             input_length=bptt)(input_dict['syllables'])

    for c in input_dict:
        embed_dict[c] = Embedding(output_dim=cond_emb_dim,
                                    input_dim=vectorizers[c].dim,
                                    input_length=bptt)(input_dict[c])

    concat_emb = Concatenate(axis=-1)([embed_dict[k] for k in embed_dict])
    concat_emb = Dropout(dropout)(concat_emb)

    lstm_out = LSTM(lstm_dim, recurrent_dropout=dropout,
                    return_sequences=True, activation='tanh')(concat_emb)
    syll_pred = TimeDistributed(Dense(vectorizers['syllables'].dim,
                          activation='softmax'))(lstm_out)

    model = Model(inputs=[input_dict[k] for k in input_dict],
                  outputs=[syll_pred])

    optim = Adam(lr=lr)
    model.compile(optimizer=optim,
                  loss='categorical_crossentropy')

    model.summary()

    return model

def fit_model(model, generator, nb_epochs, bptt,
              model_path, vectorizers, max_gen_len,
              gen_conditions):

    checkpoint = ModelCheckpoint(model_path, monitor='loss',
                                  verbose=1, save_best_only=True)

    generation_callback = GenerationCallback(vectorizers=vectorizers,
                                             bptt=bptt,
                                             max_len=max_gen_len,
                                             gen_conditions=gen_conditions)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3,
                                  patience=1, min_lr=0.000001,
                                  verbose=1)

    try:
        model.fit_generator(generator=generator.get_transformed_batches(vectorizers, endless=True),
                            steps_per_epoch=generator.num_batches,
                            epochs=nb_epochs,
                            callbacks=[checkpoint, generation_callback, reduce_lr])
    except KeyboardInterrupt:
        return


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    preds = preds.ravel()
    #probas = np.random.multinomial(1, preds, 1)
    #return np.argmax(probas)
    return np.random.choice(range(len(preds)), p=preds)


class GenerationCallback(Callback):

    def __init__(self, vectorizers, gen_conditions,
                 bptt, max_len=12):
        super(GenerationCallback, self).__init__()
        self.vectorizers = vectorizers
        self.bptt = bptt
        self.gen_conditions = gen_conditions
        self.max_len = max_len

    def on_epoch_end(self, epoch, logs):
        #batch = {'syllables' : [['<PAD>'] * (self.bptt - 1) + ['<BR>']]}
        batch = {'syllables' : [['<BR>', 'beat', 'you', 'to', 'death', 'and', 'teach', 'you', 'a', 'le/']]}
        for c in self.vectorizers:
            if c not in ('syllables', 'targets'):
                batch[c] = [[self.gen_conditions[c]] * self.bptt]

        batch_dict = {}
        for k, vectorizer in self.vectorizers.items():
            if k == 'targets':
                continue
            batch_dict[k] = vectorizer.transform(batch[k])

        for diversity in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            print('\n-> diversity:', diversity)
            preds = []
            for i in range(self.max_len):
                pred_proba = self.model.predict(batch_dict, verbose=0)[0][-1]

                pred_syllab = '<UNK>'
                patience = 20
                while pred_syllab == '<UNK>':
                    pred_idx = sample(pred_proba, diversity)
                    pred_syllab = self.vectorizers['syllables'].idx2syll[pred_idx]
                    patience -= 1
                    if patience <= 0:
                        break

                preds.append(pred_syllab)

                batch_dict['syllables'][0] = batch_dict['syllables'][0][1:].tolist() + [pred_idx]

            print(' '.join(preds).replace('/ ', ''))
            print()
