
import math
import torch
import json
import collections

from . import utils


def counts_entropy(counts):
    total = sum(counts.values())
    ent = 0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p)
    return ent


def gather_statistics(model, encoder, d, nsamples=100, length=15, tau=0.85):
    """
    Compute rhyming statistics for a given model by sampling and counting rhyming lines
    """
    conds = {'length': encoder.conds['length'].w2i[length]}
    for rhyme in encoder.conds['rhyme'].w2i.keys():
        conds['rhyme'] = encoder.conds['rhyme'].w2i[rhyme]
        (samples, _), _, _ = model.sample(encoder, tau=tau, conds=conds, batch=nsamples)
        counts, fails, examples = collections.Counter(), 0, []
        for sample in samples:
            sample = utils.join_syllables(sample.split())
            examples.append(sample)
            words = sample.split()
            last = words[-1] if words else ''
            try:
                phonology = d[last]
                phon = utils.get_final_phonology(phonology)
                if phon == rhyme.split('-'):
                    counts[last] += 1
            except KeyError:
                fails += 1
                continue

        accuracy = sum(counts.values()) / nsamples
        dispersion = len(counts) / (nsamples - fails)

        yield {"rhyme": rhyme,
               "counts": counts,
               "fails": fails,
               "accuracy": sum(counts.values()) / nsamples,
               "dispersion": len(counts) / (nsamples - fails),
               "entropy": counts_entropy(counts),
               "examples": examples}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--dpath', default='./data/ohhla.vocab.phon.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--debug', action='store_true', help="Print sampled examples")
    args = parser.parse_args()

    from generation import model_loader

    model, encoder = model_loader(args.model)

    model.to(args.device)
    with open(args.dpath) as f:
        d = json.loads(f.read())

    with open('{}.rhyme.stats.json'.format(model.modelname), 'w') as f:
        for idx, stuff in enumerate(gather_statistics(model, encoder, d)):
            if idx % 50 == 0:
                print(idx+1)
            if not args.debug:
                del stuff['examples']
                del stuff['counts']
            f.write(json.dumps(stuff) + '\n')
