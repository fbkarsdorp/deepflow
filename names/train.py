import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from modeling import RNN
from utils import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bptt', type=int, default=25)
    parser.add_argument('--rnd_seed', type=int, default=825424)
    parser.add_argument('--num_batches', type=int, default=10000)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--max_artists', type=int, default=None)
    parser.add_argument('--input_file', type=str, default='../data/ohhla-new.jsonl')
    parser.add_argument('--real_names_file', type=str, default='real_names.txt')
    parser.add_argument('--real_names_unique_file', type=str, default='real_names_unique.txt')
    parser.add_argument('--model_file', type=str, default='name_generator.pt')
    parser.add_argument('--vocab_file', type=str, default='vocabulary.txt')
    parser.add_argument('--unicize', action='store_true', default=False)
    parser.add_argument('--reparse', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=.001)

    args = parser.parse_args()
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    if args.reparse:
        artists = load_artists(args.input_file,
                               max_artists=args.max_artists)

        with open(args.real_names_file, 'w') as f:
            f.write('\n'.join(artists))    

        with open(args.real_names_unique_file, 'w') as f:
            f.write('\n'.join(sorted(set(artists))))

    else:
        artists = [l.strip() for l in open(args.real_names_file)]
        artists = artists[:args.max_artists]

    data = stringify(artists, unicize=args.unicize)
    vocabulary = tuple(sorted(set(data)))
    with open(args.vocab_file, 'w') as f:
            f.write('\n'.join(sorted(vocabulary)))
    
    model = RNN(input_size=len(vocabulary),
                hidden_size=args.hidden_size,
                output_size=len(vocabulary),
                n_layers=args.num_layers,
                embed_dim=args.embed_dim)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    loss_avg = 0

    try:
        for batch_idx in range(1, args.num_batches + 1):
            inp, target = random_batch(data, vocabulary, args.bptt)

            hidden = model.init_hidden()
            model.zero_grad()
            loss = 0

            for c in range(args.bptt):
                output, hidden = model.forward(inp[c], hidden)
                loss += criterion(output, target[c])

            loss.backward()
            optimizer.step()
            loss = loss.data[0] / args.bptt

            if batch_idx % args.print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start),
                    batch_idx, batch_idx / args.num_batches * 100, loss))
                print(' - '.join(generate(model, vocabulary, '#', 100).split('#')[:-1]), '\n')

    except KeyboardInterrupt:
        pass

    torch.save(model, args.model_file)


if __name__ == '__main__':
    main()