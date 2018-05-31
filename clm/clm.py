"""
CUDA_VISIBLE_DEVICES=1 python3 clm.py
"""
import math
import numpy as np

import torch
import torch.nn as nn
import argparse
import time
#from keras.models import load_model

#import modelling
import data
from model import RNNModel, repackage_hidden
from vectorization import SequenceVectorizer


def main():
    parser = argparse.ArgumentParser(description='Conditional Language Model')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--json_path', type=str, default='../data/lazy_ohhla.json',
                        help='path to data file (lazy json)')
    parser.add_argument('--model_path', type=str, default='clm_model',
                        help='path to store model')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--log_interval', type=int, default=2,
                        help='every n batches')
    parser.add_argument('--cond_emsize', type=int, default=30,
                        help='size of condition embeddings')
    parser.add_argument('--nhid', type=int, default=1024,
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
    parser.add_argument('--bptt', type=int, default=15,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='sampling temperature')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--min_syll_cnt', type=int, default=5,
                        help='minimum syllable frequency')
    parser.add_argument('--max_songs', type=int, default=None,
                        help='maximum number of songs to parse')
    parser.add_argument('--random_shift', type=int, default=3,
                        help='random uniform shift at the beginning of file to increase shingling variety')
    parser.add_argument('--max_gen_len', type=int, default=20,
                        help='number of syllables to generate')
    parser.add_argument('--min_artist_cnt', type=int, default=10,
                        help='minimum number of songs for artist embeddings')
    args = parser.parse_args()
    
    conditions = {'artists',
                  #'topics',
                  }
    gen_conditions = {'artists': 'eminem',
                      #'topics': 't58',
                      }

    clm_data = data.ClmData(batch_size=args.batch_size,
                            bptt=args.bptt,
                            shift=args.random_shift,
                            max_songs=args.max_songs,
                            json_path=args.json_path,
                            conditions=conditions,)

    vectorizers = {'syllables': SequenceVectorizer(min_cnt=args.min_syll_cnt,
                                                   max_len=args.bptt)}
    for c in conditions:
        vectorizers[c] = SequenceVectorizer(max_len=args.bptt, min_cnt=args.min_artist_cnt)

    for batch in clm_data.get_batches():
        for k, vectorizer in vectorizers.items():
            vectorizer.partial_fit(batch[k])

    for vectorizer in vectorizers.values():
        vectorizer.finalize_fit()
    
    #for batch in clm_data.get_batches():
    #    for sylls, trgts, arts in zip(batch['syllables'], batch['targets'], batch['artists']):
    #        print(sylls, trgts)

    torch.manual_seed(args.seed)

    ninp_syllables = vectorizers['syllables'].dim

    model = RNNModel(rnn_type=args.model,
                     ntoken=ninp_syllables,
                     ninp=args.emsize,
                     nhid=args.nhid,
                     cond_emsize=args.cond_emsize,
                     nlayers=args.nlayers,
                     conditions=conditions,
                     vectorizers=vectorizers,
                     dropout=args.dropout,
                     tie_weights=False)
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for batch_idx, (batch_X, batch_Y) in enumerate(clm_data.get_transformed_batches(vectorizers, endless=False)):
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()
            
            sylls = batch_X['syllables']
            conds = {k: torch.LongTensor(batch_X[k]).t().contiguous().to(device) for k in conditions}

            targets = torch.LongTensor(batch_Y).contiguous().view(-1).to(device)
            sylls = torch.LongTensor(sylls).t().contiguous().to(device)

            output, hidden = model(input=sylls, hidden=hidden, conds=conds)

            loss = criterion(output.view(-1, ninp_syllables), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            #for p in model.parameters():
            #    p.data.add_(-args.lr, p.grad.data)

            total_loss += loss.item()

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                total_loss = 0
                start_time = time.time()

                rnd_hidden = model.init_hidden(1)
                rnd_input = torch.LongTensor([[vectorizers['syllables'].syll2idx['<BR>']]]).to(device)
                #rnd_input = torch.randint(ninp_syllables, (1, 1), dtype=torch.long).to(device)

                rnd_conds = {}
                for c in conditions:
                    rnd_conds[c] = torch.LongTensor([[vectorizers['artists'].syll2idx['eminem']]]).to(device)

                with torch.no_grad():
                    for temperature in np.linspace(0.1, 1.0, 10):
                        print('\ntemp @ ' + str(temperature), ':')
                        try:
                            for i in range(args.max_gen_len):
                                output, rnd_hidden = model(rnd_input, rnd_hidden, conds=rnd_conds)
                                word_weights = output.squeeze().div(temperature).exp().cpu()
                                word_idx = torch.multinomial(word_weights, 1)[0]
                                rnd_input.fill_(word_idx)
                                word = vectorizers['syllables'].idx2syll[int(word_idx.data)]

                                print(word, end=' ')
                            print()
                        except:
                            pass
                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, batch_idx, clm_data.num_batches, args.lr,
                            elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))


if __name__ == '__main__':
    main()