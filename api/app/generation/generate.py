
import random

from .cache import Cache
from . import utils


def generate_stanza(model, encoder,
                    nlines=4, nstanzas=20,
                    rhyme=None, length=15, tau=1.0,
                    verbose=False,
                    cache_size=0, alpha=0.0, theta=0.0):

    hidden = None
    rhyme = rhyme or random.choice(list(encoder.conds['rhyme'].w2i.keys()))
    cache = None
    if cache_size > 0:
        cache = Cache(model.hidden_dim, cache_size, len(encoder.word.w2i), model.device)

    conds = {'rhyme': encoder.conds['rhyme'].w2i[rhyme],
             'length': encoder.conds['length'].w2i[length]}

    stanzas = None
    for _ in range(nlines):
        (hyps, _), _, hidden = model.sample(
            encoder, hidden=hidden, conds=conds, batch=nstanzas, tau=tau,
            cache=cache, alpha=alpha, theta=theta, avoid_unk=True)

        if stanzas is None:
            stanzas = [[hyp.split()] for hyp in hyps]
        else:
            for idx in range(len(hyps)):
                stanzas[idx].append(hyps[idx].split())

    output = []
    for stanza in stanzas:
        valid, prev = True, None
        for line in stanza:
            # check line
            if not utils.is_valid(line, verbose=verbose):
                valid = False
            # check consecutive
            if prev is not None:
                valid = valid and utils.is_valid_pair(line, prev, verbose=verbose)
            prev = line

            if not valid:
                continue

        if valid:
            lines = []
            for line in stanza:
                lines.append(utils.detokenize(utils.join_syllables(line), debug=verbose))
            output.append(lines)

    return output, {'rhyme': rhyme, 'length': length}


def format_stanzas(stanzas, conds):
    print(conds)
    print()
    for idx, stanza in enumerate(stanzas):
        print("Stanza {}:\n{}\n\n".format(idx+1, '\n'.join(stanza)))
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='RNNLanguageModel.2018-08-01+19:33:04.pt')
    parser.add_argument('--tau', default=1.0, type=float)
    parser.add_argument('--length', default=15, type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--nlines', default=4, type=int)
    parser.add_argument('--nstanzas', default=20, type=int)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--theta', type=float, default=0.0)
    parser.add_argument('--cache_size', type=int, default=0)
    args = parser.parse_args()

    from generation import model_loader
    model, encoder = model_loader(args.model)

    rhymes = list(filter(lambda rhyme: len(rhyme.split('-')) < 2, encoder.conds['rhyme'].w2i))
    rhyme = random.choice(rhymes)

    stanzas, conds = generate_stanza(
        model, encoder, nlines=args.nlines, nstanzas=args.nstanzas,
        tau=args.tau, length=args.length, rhyme=rhyme,
        cache_size=args.cache_size, alpha=args.alpha, theta=args.theta,
        verbose=args.verbose)

    format_stanzas(stanzas, conds)
