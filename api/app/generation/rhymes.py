
import os
import random
import math
import json
import collections

from . import utils
random.seed(1001)


def sample_rhymes(model, encoder, rhyme,
                  ntries=100, nsamples=1000, length=15, tau=0.85, tau_std=0.075):
    conds = {}
    if 'length' in encoder.conds:
        conds['length'] = encoder.conds['length'].w2i[length]
    conds['rhyme'] = encoder.conds['rhyme'].w2i[rhyme]

    run = True
    tsamples = 0
    ntries_ = ntries
    while run and ntries_ > 0:
        try:
            (samples, _), _, _, _ = model.sample(
                encoder, tau=random.gauss(tau, tau_std), conds=conds, batch=100)

            for sample in samples:
                if tsamples >= nsamples:
                    run = False
                    break
                tsamples += 1
                yield rhyme, utils.join_syllables(sample.split())
        except KeyboardInterrupt:
            return
        except Exception as e:
            print('Error', e)
            ntries_ -= 1
            if ntries_ % 10 == 0:
                print("Trying another {} times".format(ntries_))


def count_missing(total, fpath):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                key, val = line.split('\t')
                if key not in total:
                    continue
                total[key] -= 1
            except:
                pass

    return {k: v for k, v in total.items() if v > 0}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--nsamples', default=1000, type=int)
    parser.add_argument('--length', default=15, type=int)
    parser.add_argument('--tau', default=0.85, type=float)
    parser.add_argument('--tau_std', default=0.075, type=float)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--debug', action='store_true', help="Print sampled examples")
    args = parser.parse_args()

    from generation import model_loader

    model, encoder = model_loader(args.model)
    model.to(args.device)

    path = '{}.rhymes.csv'.format(model.modelname)

    missing = {k: args.nsamples for k in encoder.conds['rhyme'].w2i.keys()}
    if os.path.isfile(path):
        missing = count_missing(missing, path)
    import pprint
    pprint.pprint(missing)

    with open(path, 'a+') as f:
        for rhyme, missing in missing.items():
            print(rhyme, missing)
            for rhyme, sample in sample_rhymes(
                    model, encoder, rhyme,
                    nsamples=args.nsamples, length=args.length,
                    tau=args.tau, tau_std=args.tau_std):
                f.write('\t'.join([rhyme, sample]) + '\n')
