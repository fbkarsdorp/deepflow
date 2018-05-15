from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.layers import Dense, Dropout, Bidirectional,\
    Input, Lambda, Embedding, LSTM, Flatten, TimeDistributed,\
    Activation


def build_model(seq_len, vocab_size,
                num_layers=2,
                embed_dim=128,
                recurrent_dim=128,
                embedding_weights=None,
                learning_rate=0.01,
                freeze=True):

    input_ = Input(shape=(seq_len,), dtype='int32', name='syll')
    m = Embedding(input_dim=vocab_size,
                  mask_zero=True,
                  output_dim=embed_dim,
                  input_length=seq_len,
                  weights=[embedding_weights],
                  trainable=not freeze)(input_)

    for i in range(num_layers):
        if i < (num_layers ):
            m = Bidirectional(LSTM(recurrent_dim, return_sequences=True),
                              merge_mode='sum', name='out' + str(i + 1))(m)

    dense = TimeDistributed(Dense(2, activation='softmax'), name='out')(m)
    model = Model(inputs=[input_], outputs=[dense])

    optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim,
                  metrics=['accuracy'],
                  loss={'out': 'categorical_crossentropy'})

    return model
