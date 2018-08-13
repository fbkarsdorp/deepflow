import argparse

import torch

from utils import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--rnd_seed', type=int, default=127666)
    parser.add_argument('--num_names', type=int, default=1000000)
    parser.add_argument('--real_names_unique_file', type=str, default='real_names_unique.txt')
    parser.add_argument('--output_file', type=str, default='user_names.json')
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--generation_len', type=int, default=1000)
    parser.add_argument('--vocab_file', type=str, default='vocabulary.txt')
    parser.add_argument('--model_file', type=str, default='name_generator.pt')

    args = parser.parse_args()

    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    real_names = {l.strip() for l in open(args.real_names_unique_file)}

    vocabulary = tuple([re.sub(r'\n', '', w)\
                    for w in open(args.vocab_file).readlines()])

    model = torch.load(args.model_file)
    model.eval()

    names = set()
    while len(names) < args.num_names:
        s = generate(model, vocabulary, '#', args.generation_len,
                     temperature=args.temperature)
        s = s.split('#')[:-1]  # cut off last
        s = set([n.strip() for n in s if n.strip() and n not in real_names])
        names.update(s)
        print(len(names))

    names = list(names)
    random.shuffle(names)

    for n in names:
        print(n)

    with open(args.output_file, 'w') as f:
        f.write(json.dumps(tuple(names), indent=4))


if __name__ == '__main__':
    main()