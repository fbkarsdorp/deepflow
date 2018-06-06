"""
CUDA_VISIBLE_DEVICES=1 python3 clm.py
"""
import argparse
import shutil
import os

import modelling
import data
from vectorization import SequenceVectorizer


def main():
    parser = argparse.ArgumentParser(description='Conditional Language Model')
    parser.add_argument('--json_path', type=str, default='../data/lazy_ohhla-beatfamilies.json',
                        help='path to data file (lazy json)')
    parser.add_argument('--model_path', type=str, default='clm_model',
                        help='path to store model')
    parser.add_argument('--vectorizer_dir', type=str, default='vectorizers',
                        help='path to dir where to store vectorizers')
    parser.add_argument('--emsize', type=int, default=512,
                        help='size of word embeddings')
    parser.add_argument('--cond_emsize', type=int, default=56,
                        help='size of condition embeddings')
    parser.add_argument('--nhid', type=int, default=2048,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=5.0, # 0.25
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=500,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=20,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--min_syll_cnt', type=int, default=100,
                        help='minimum syllable frequency')
    parser.add_argument('--max_songs', type=int, default=0,
                        help='maximum number of songs to parse')
    parser.add_argument('--max_gen_len', type=int, default=35,
                        help='number of syllables to generate')
    parser.add_argument('--min_artist_cnt', type=int, default=10,
                        help='minimum number of songs for artist embeddings')
    args = parser.parse_args()

    conditions = {'artists',
                  'topics',
                  'rhythms'
                  }
    gen_conditions = {'artists': 'eminem',
                      'topics': 't40',   # topic7: gun - kill - dead - shot - murder - guns - blood - street - son
                      'rhythms': 'r6'}

    clm_data = data.ClmData(batch_size=args.batch_size,
                            bptt=args.bptt,
                            max_songs=args.max_songs,
                            json_path=args.json_path,
                            conditions=conditions)

    vectorizers = {'syllables': SequenceVectorizer(min_cnt=args.min_syll_cnt,
                                                   bptt=args.bptt,
                                                   )}
    for c in conditions:
        vectorizers[c] = SequenceVectorizer(bptt=args.bptt,
                                            min_cnt=args.min_artist_cnt)

    for batch in clm_data.get_batches():
        for k, vectorizer in vectorizers.items():
            vectorizer.partial_fit(batch[k])

    for vectorizer in vectorizers.values():
        vectorizer.finalize_fit()

    # save the vectorizers:
    try:
        shutil.rmtree(args.vectorizer_dir)
    except FileNotFoundError:
        pass
    os.mkdir(args.vectorizer_dir)

    for n in sorted(vectorizers.keys()):
        vectorizers[n].dump(args.vectorizer_dir + '/' + n + '.json')

    #for batch in clm_data.get_batches():
    #    for sylls, trgts, arts in zip(batch['syllables'], batch['targets'], batch['artists']):
    #        print(sylls, trgts)

    #for batch in clm_data.get_transformed_batches(vectorizers=vectorizers):
    #    X, Y = batch
    
    model = modelling.build_model(conditions=clm_data.conditions,
                                  vectorizers=vectorizers,
                                  bptt=args.bptt,
                                  syll_emb_dim=args.emsize,
                                  cond_emb_dim=args.cond_emsize,
                                  lstm_dim=args.nhid,
                                  lr=args.lr,
                                  dropout=args.dropout)

    modelling.fit_model(model=model,
                        bptt=args.bptt,
                        vectorizers=vectorizers,
                        gen_conditions=gen_conditions,
                        generator=clm_data,
                        nb_epochs=args.epochs,
                        model_path=args.model_path,
                        max_gen_len=args.max_gen_len)


if __name__ == '__main__':
    main()