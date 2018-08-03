
import random
import torch
from models.pytorch.model import RNNLanguageModel


def join(sylls):
    joint = ''

    for syl in sylls:
        if syl.startswith('-'):
            syl = syl[1:]
        if syl.endswith('-'):
            syl = syl[:-1]
        else:
            syl = syl + ' '
        joint += syl

    return joint


def is_valid(sylls, verbose=False):

    if len(sylls) == 0:
        if verbose:
            print("Empty line")
        return False

    if '<unk>' in sylls:
        if verbose:
            print("Found <unk> in line")
        return False

    if sylls[0].startswith('-'):
        if verbose:
            print("Incongruent start syllabification: {}".format(sylls[0]))
        return False

    if sylls[-1].endswith('-'):
        if verbose:
            print("Incongruent ending syllabification: {}".format(sylls[-1]))
        return False

    for idx in range(len(sylls) - 1):
        if sylls[idx].endswith('-') and not sylls[idx+1].startswith('-'):
            if verbose:
                print("Incongruent syllabification: {} {}".format(
                    sylls[idx], sylls[idx+1]))
            return False

    return True


def is_valid_pair(sylls1, sylls2, verbose=False):

    def get_last(line):
        last = []
        for syl in line[::-1]:
            last.append(syl)
            if not syl.startswith('-'):
                break
        return join(last[::-1])

    # avoid same word in the end of consecutive lines
    if get_last(sylls1) == get_last(sylls2):
        if verbose:
            print("Lines end equal:\n\t- {}\n\t- {}\n\t".format(line1, line2))
        return False

    return True


def generate_stanza(model, encoder,
                    nlines=4, nstanzas=20,
                    rhyme=None, length=15, tau=1.0,
                    verbose=False):

    hidden = None
    rhyme = rhyme or random.choice(list(encoder.conds['rhyme'].w2i.keys()))

    conds = {'rhyme': encoder.conds['rhyme'].w2i[rhyme],
             'length': encoder.conds['length'].w2i[length]}

    stanzas = None
    for _ in range(nlines):
        (hyps, _), hidden = model.sample(
            encoder, hidden=hidden, conds=conds, reverse=True,
            batch=nstanzas, tau=tau)

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
            if not is_valid(line, verbose=verbose):
                valid = False
            # check consecutive
            if prev is not None:
                valid = valid and is_valid_pair(line, prev, verbose=verbose)
            prev = line

            if not valid:
                continue

        if valid:
            output.append([join(line) for line in stanza])

    return output, {'rhyme': rhyme, 'length': length}


def format_stanzas(stanzas, conds):
    print(conds)
    print()
    for idx, stanza in enumerate(stanzas):
        print("Stanza {}:\n{}\n\n".format(idx+1, '\n'.join(stanza)))
    print()


def create_rhyme_eval_set(model, encoder, nsamples=25, length=15, tau=0.85):
    conds = {'length': encoder.conds['length'].w2i[length]}
    for cond in encoder.conds['rhyme'].w2i.keys():
        conds['rhyme'] = encoder.conds['rhyme'].w2i[cond]
        (samples, _), _ = model.sample(encoder, tau=tau, conds=conds, batch=nsamples)
        for sample in samples:
            sample = ' '.join(sample.split()[::-1])
            yield cond, sample


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='RNNLanguageModel.2018-08-01+19:33:04.pt')
    parser.add_argument('--tau', default=1.0, type=float)
    parser.add_argument('--length', default=15, type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--nlines', default=4, type=int)
    parser.add_argument('--nstanzas', default=20, type=int)
    args = parser.parse_args()

    # d = torch.load('./RNNLanguageModel.2018-07-31+16:28:30.pt')
    # d = torch.load('./RNNLanguageModel.2018-08-01+19:33:04.pt')
    d = torch.load(args.model)
    model, encoder = d['model'], d['encoder']

    rhymes = list(filter(lambda rhyme: len(rhyme.split('-')) < 2, encoder.conds['rhyme'].w2i))
    rhyme = random.choice(rhymes)

    stanzas, conds = generate_stanza(
        model, encoder, nlines=args.nlines, nstanzas=args.nstanzas,
        tau=args.tau, length=args.length, rhyme=rhyme,
        verbose=args.verbose)

    format_stanzas(stanzas, conds)

    # import collections
    # samples = collections.defaultdict(list)
    # for idx, (cond, sample) in enumerate(create_rhyme_eval_set(model, encoder)):
    #     if idx % 50 == 0:
    #         print(idx)
    #     samples[cond].append(sample)

