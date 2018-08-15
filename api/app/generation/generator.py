
import time
import sys
import os
import json
import random
import uuid
import warnings

from . import utils, model_loader, Cache


def load_models(dirpath):
    models = {}

    for modelname in os.listdir(dirpath):
        if not modelname.endswith('.pt'):
            continue

        modelname = '.'.join(modelname.split('.')[:-1])

        if modelname in models:
            continue

        print("Loading model: {}".format(os.path.join(dirpath, modelname)))
        model, encoder = model_loader(os.path.join(dirpath, modelname))
        models[modelname] = {'model': model, 'encoder': encoder}

    return models


def sample_file(fpath, rate=0.01):
    f = open(fpath)

    while True:
        try:
            r = random.random()
            if r < rate:
                yield json.loads(next(f).strip())
            else:
                next(f)
        except StopIteration:
            f.close()
            f = open(fpath)

        except UnicodeDecodeError:
            # corrupted file
            continue


class TemplateSampler:
    """
    Sample from a file line by line based on a certain rate (n out of all lines)
    """
    def __init__(self, fpath, dpath):
        self.sampler = sample_file(fpath)
        with open(dpath) as f:
            self.d = json.loads(f.read())

    def sample(self, nlines=None, verbose=False):
        """
        Arguments:
        ----------
        nlines : int or None, minimum number of lines in template

        Returns: template, metadata
        --------

        - template : list of dicts specifying conditions
        - metadata : dict of metadata associated with the verse (artist, id, album, etc.)
        """
        tries, sample = 1, None

        while sample is None:
            try:
                sample = self.sample_template(next(self.sampler), nlines)
            except RuntimeError:
                tries += 1
                continue

        if verbose:
            print("Got sample after #{} tries".format(tries))

        return sample

    def sample_template(self, song, nlines):
        """
        Auxiliary method
        """
        verses = list(range(len(song['text'])))
        random.shuffle(verses)

        metadata = {"id": song['id'],
                    "album": song["album"],
                    "artist": song['artist'],
                    "song": song["song"]}

        while len(verses) > 0:
            verse = verses.pop()

            if nlines is None or len(song['text'][verse]) >= nlines:
                metadata["verse"] = verse

                template = []
                for line in song['text'][verse]:
                    _, conds = utils.prepare_line(line, d=self.d)
                    template.append(conds)

                return template, metadata

        raise RuntimeError("Couldn't find template of #{} lines".format(nlines))


def get_weights(encoder):
    weights = {}

    for cond, vocab in encoder.conds.items():
        if cond == 'rhyme':
            # sample more from single syllable rhymes
            weights[cond] = [{1: 10}.get(len(c.split('-')), 1) for c in vocab.w2i]
        elif cond == 'length':
            # uniform for now (maybe change it later)
            weights[cond] = [1 for _ in encoder.conds['length'].w2i]

    return weights


def sample_conditions(encoder, get_weights=get_weights):
    weights = get_weights(encoder)

    conds = {}
    for cond, vocab in encoder.conds.items():
        # choices returns a list
        conds[cond] = random.choices(list(vocab.w2i), weights[cond])[0]

    return conds


class Generator:
    """
    Generate examples

    Arguments:
    ----------
    - modelpath : str, path to dir with models
    - template_sampler : TemplateSampler
    - nlines : tuple of ints, range of lines to sample
    """
    def __init__(self, modelpath, template_sampler=None, nlines=(2, 3, 4), device='cpu'):
        self.models = load_models(modelpath)
        self.template_sampler = template_sampler
        self.nlines = nlines
        self.device = device

    def sample(self, tries=1, sample_template=True, avoid_unk=True,
               cache_size=0, alpha=0.15, theta=0.75):
        """
        Generate a stanza sampling:

            - model (uniformly)
            - nlines (uniformly)
            - tau (gaussian)
            - conds: either using a sampled template as per `TemplateSampler` or
                sampled conds based on multinomial sampling in `sample_conditions`
            - candidates (multinomial), based on the generator own estimates

        Arguments:
        ----------

        - tries : int, number of candidates per line
        - sample_template : bool, whether to use the , ignored if `Generator` wasn't
            instantiated with a `TemplateSampler`
        - avoid_unk : bool, whether to downweight the logits for `<unk>` so as to
            avoid sampling it during generation

        Returns: None if failed or a dict with the sample metadata
        --------
        { "id" : str,
          "model" : str,
          "params": {...},
          "text" : [{"params": {...}, "text" : str}] }
        """
        if tries > 1 and cache_size:
            warnings.warn("Using cache with a batch size bigger than 1 "
                          "might lead to unexpected results")

        mconfig = random.choice(list(self.models.values()))
        model, encoder = mconfig['model'], mconfig['encoder']
        model.to(self.device)

        # best guess for nlines
        nlines = random.choice(self.nlines)
        # best guess for temperature
        tau = random.gauss(0.8, 0.075)

        template, tmeta, conds = None, None, None
        if sample_template and self.template_sampler is not None:
            template, tmeta = self.template_sampler.sample(nlines)
            template = [{c: line[c] for c in encoder.conds} for line in template]
        else:
            conds = sample_conditions(encoder)
            conds = {cond: conds[val] for cond in encoder.conds}

        cache = None
        if cache_size:
            cache = Cache.new(model.hidden_dim, cache_size, device=self.device)

        text = []
        hidden, prev = None, None
        for line in range(nlines):

            if template is not None:
                conds = template[line]

            (hyps, _), scores, hidden, cache = model.sample(
                encoder,
                batch=tries,
                tau=tau,
                conds={k: encoder.conds[k].w2i[v] for k, v in conds.items()},
                avoid_unk=avoid_unk,
                cache=cache, alpha=alpha, theta=theta)

            if not hyps:
                return

            # sort by score to ensure best is last
            scores, hyps = zip(*sorted(list(zip(scores, hyps))))
            scores, hyps = list(scores), list(hyps)

            # downgrade sentences with <unk>
            c = 0
            while utils.UNK in hyps[-1] and c < len(hyps):
                hyps[0], hyps[-1] = hyps[-1], hyps[0]
                scores[0], scores[-1] = scores[-1], scores[0]
                c += 1

            # filter out invalid hyps (but always leave last one at least)
            c = 0
            while c < len(hyps):
                hyp = hyps[c].split()
                if not utils.is_valid(hyp) or \
                   (prev and not utils.is_valid_pair(hyp, prev)):
                    hyps.pop(c)
                    scores.pop(c)
                else:
                    c += 1

            if not hyps or sum(scores) == 0.0:
                return

            # sample from the filtered hyps
            idx = random.choices(list(range(len(hyps))), scores).pop()
            # select sampled
            hyp, score = hyps[idx], scores[idx]
            # update prev
            prev = hyp.split()

            # reformat hidden
            hidden_ = []
            for h in hidden:
                if isinstance(h, tuple):
                    h = h[0][:, idx].repeat(1, tries, 1), \
                        h[1][:, idx].repeat(1, tries, 1)
                else:
                    h = h[:, idx].repeat(1, tries, 1)
                hidden_.append(h)
            hidden = hidden_

            # prepare output
            line = hyp
            if model.modelname.startswith('RNN'):
                line = utils.join_syllables(hyp.split())
            line = utils.detokenize(line)

            text.append({"line": line, "original": hyp, "params": {"score": score}})

        return {"id": str(uuid.uuid1())[:8],
                "params": {
                    # sampled parameters
                    "tau": tau,
                    "nlines": nlines,
                    "template": template[:nlines] if template else None,
                    "template_metadata": tmeta,
                    "conds": None if template is not None else conds,
                    # passed parameters (for reference)
                    "avoid_unk": avoid_unk,
                    "tries": tries,
                    "cache_size": cache_size,
                    "alpha": alpha,
                    "theta": theta
                },
                "model": model.modelname,
                "text": text}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', required=True)
    parser.add_argument('--nsamples', type=int, default=100)
    parser.add_argument('--outputpath', help='/path/to/output file (jsonl)')
    parser.add_argument('--datapath', help='/path/to/data to sample templates')
    parser.add_argument('--dpath', help='/path/to/phonological dict')
    parser.add_argument('--tries', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    tsampler = None
    if args.datapath:
        if not args.dpath:
            print("`datapath` requires path to phonological dictionary `dpath`")
            sys.exit(1)
        tsampler = TemplateSampler(fpath=args.datapath, dpath=args.dpath)

    generator = Generator(args.modelpath, template_sampler=tsampler, device=args.device)

    opts = {'tries': args.tries,
            'avoid_unk': True,
            'cache_size': 0,
            'alpha': 0.15,
            'theta': 0.75}

    c, repfreq = 0, 1000
    if args.outputpath:
        with open(args.outputpath, 'w') as f:
            start = time.time()
            while c < args.nsamples:
                sample = generator.sample(**opts)
                if sample:
                    f.write('{}\n'.format(json.dumps(sample)))
                    c += 1
                if c % repfreq == 0:
                    print("Processed {:>8} items at speed: {:g} items/sec".format(
                        c, repfreq / (time.time() - start)))
                    start = time.time()
    else:
        while c < args.nsamples:
            sample = generator.sample(**opts)
            if sample:
                if args.debug:
                    for idx, line in enumerate(sample['text']):
                        if sample['params']['template']:
                            print(sample['model'],
                                  line['line'], sample['params']['template'][idx])
                        else:
                            print(sample['model'],
                                  line['line'], sample['params']['conds'])
                    print()
                else:
                    print(json.dumps(sample, indent=2))
                c += 1
