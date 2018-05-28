import numpy as np

from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Conv1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import LambdaCallback
from keras.models import load_model
from keras.optimizers import Adam

from collections import OrderedDict

def build_model(conditions, bptt, vectorizers,
                syll_emb_dim, cond_emb_dim, lstm_dim):
    
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

    #conv_out = Conv1D(lstm_dim, 3, strides=1)(concat_emb)
    #lstm_out = LSTM(lstm_dim, activation='tanh', return_sequences=True)(conv_out)

    lstm_out = LSTM(lstm_dim, return_sequences=False, activation='tanh')(concat_emb)
    syll_pred = Dense(vectorizers['targets'].dim, activation='softmax',
                      name='targets')(lstm_out)

    model = Model(inputs=[input_dict[k] for k in input_dict],
                  outputs=[syll_pred])

    optim = Adam(lr=0.001)
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

    try:
        model.fit_generator(generator=generator.get_transformed_batches(vectorizers, endless=True),
                            steps_per_epoch=generator.num_batches,
                            epochs=nb_epochs,
                            callbacks=[checkpoint, generation_callback])
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

        for diversity in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.4]:
            print('\n-> diversity:', diversity)
            for i in range(self.max_len):
                pred_proba = self.model.predict(batch_dict, verbose=0)[0]
                pred_idx = sample(pred_proba, diversity)
                try:
                    pred_syllab = self.vectorizers['targets'].idx2class[pred_idx]
                    idx_new_syll = self.vectorizers['syllables'].syll2idx[pred_syllab]
                except KeyError:
                    pred_syllab = '<UNK>'
                    idx_new_syll = self.vectorizers['syllables'].syll2idx[pred_syllab]

                print(pred_syllab, end=' ')
                batch_dict['syllables'][0] = batch_dict['syllables'][0][1:].tolist() + [idx_new_syll]
