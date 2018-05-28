"""
CUDA_VISIBLE_DEVICES=1 python3 clm.py
"""

import pandas as pd

from keras.models import load_model

import modelling
import data
from vectorization import SequenceVectorizer, LabelVectorizer


def main():
    bptt = 22
    batch_size = 128
    min_syll_cnt = 50
    max_songs = 10000
    random_shift = 6
    syll_emb_dim = 150
    cond_emb_dim = 10
    lstm_dim = 512
    nb_epochs = 200
    max_gen_len = 20
    min_artist_cnt = 10
    model_path = 'clm_model'
    conditions = {'artists',
                  #'topics',
                  }
    json_path = '../data/lazy_ohhla.json'
    gen_conditions = {'artists': 'eminem',
                      #'topics': 't58',
                      }

    clm_data = data.ClmData(batch_size=batch_size,
                            bptt=bptt,
                            shift=random_shift,
                            max_songs=max_songs,
                            json_path=json_path,
                            conditions=conditions)

    vectorizers = {'syllables': SequenceVectorizer(min_cnt=min_syll_cnt,
                                                   max_len=bptt),
                   'targets': LabelVectorizer(min_cnt=min_syll_cnt)}
    for c in conditions:
        vectorizers[c] = SequenceVectorizer(max_len=bptt, min_cnt=min_artist_cnt)

    for batch in clm_data.get_batches():
        for k, vectorizer in vectorizers.items():
            vectorizer.partial_fit(batch[k])

    for vectorizer in vectorizers.values():
        vectorizer.finalize_fit()

    """
    model = modelling.build_model(conditions=clm_data.conditions,
                                  vectorizers=vectorizers,
                                  bptt=bptt,
                                  syll_emb_dim=syll_emb_dim,
                                  cond_emb_dim=cond_emb_dim,
                                  lstm_dim=lstm_dim)

    modelling.fit_model(model=model,
                        bptt=bptt,
                        vectorizers=vectorizers,
                        gen_conditions=gen_conditions,
                        generator=clm_data,
                        nb_epochs=nb_epochs,
                        model_path=model_path,
                        max_gen_len=max_gen_len)

    model = load_model(model_path)
    """


if __name__ == '__main__':
    main()