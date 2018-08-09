
import json
import os
import uuid
import random

from allennlp.predictors import Predictor

from .generation import RNNLanguageModel, CharLanguageModel, model_loader
from .generation import Cache
from .generation import utils


def load_models(config):
    """
    Load the models into a dictionary. Only called at loading time.
    """
    models = {}
    for modelname in os.listdir(config['MODEL_DIR']):
        if not modelname.endswith('.pt'):  # avoid non-weight files if present
            continue

        modelname = '.'.join(modelname.split('.')[:-1])  # remove extension

        if modelname in models:  # already loaded
            continue

        mconfig = {"path": modelname}
        if config['MODELS']:
            if modelname in config['MODELS']:
                mconfig = config['MODELS'][modelname]
            else:
                continue

        # load model
        print("Loading model: {}".format(modelname))
        model, encoder = model_loader(os.path.join(config['MODEL_DIR'], modelname))

        # load rhyme weights
        rweights = {}
        rpath = os.path.join(config['MODEL_DIR'], modelname + '.rhyme.stats.json')
        if os.path.isfile(rpath):
            with open(rpath) as f:
                rweights = {r: w['entropy'] for r, w in json.loads(f.read()).items()}

        # create cache if necessary
        cache = None
        if mconfig.get("options", {}).get("cache"):
            cache = Cache(
                model.hidden_dim,               # hidden_dim
                config['DEFAULTS']["cache_size"],  # cache_size
                len(encoder.word.w2i))          # vocabulary

        # add mconfig
        models[modelname] = {
            "model": model,
            "encoder": encoder,
            "options": mconfig.get("options", {}),
            "rweights": rweights,
            "cache": cache,
            "hidden": None}

    return models


def resample_conds(encoder, conds, counter):
    """
    Takes care of sampling conditions and planning the song over consecutive generations
    """
    def sample_conds(encoder):
        conds = {}
        for cond, vocab in encoder.conds.items():
            # TODO: add weights to conditions based on rhyme evaluation
            #       e.g. shorter phonological endings are more productive in general
            conds[cond] = random.choice(list(vocab.w2i.keys()))
        return conds

    # TODO: better planning
    if not conds or counter % 4 == 0:
        return sample_conds(encoder)

    return conds


def process_seed(mconfig, seed, seed_conds):
    """
    Process the picked option (seed) and return the new hidden state / cache

    Arguments:
    ----------

    - mconfig : dict corresponding to the model config data
    - seed : list of str, input to transform_batch
    - seed_conds : dict of conditions corresponding to the seed
    """
    (word, nwords), (char, nchars), conds = \
        mconfig["encoder"].transform_batch([seed], [seed_conds])

    _, hidden = mconfig["model"](
        # TODO: add cache
        word, nwords, char, nchars, conds, mconfig.get("hidden"))

    return hidden


def syllabify(syllabifier, words):
    """
    Transform into syllables

    Arguments:
    - syllabifier : Predictor
    - words : list of str
    """
    sent = []
    for word in words:
        syllables = []
        pred = syllabifier.predict(' '.join(word))
        for char, tag in zip(pred['words'], pred['tags']):
            if int(tag) > 0:
                syllables.append('')
            syllables[-1] += char

        sent.extend(utils.format_syllables(syllables))

    return sent


def get_model_generation(mconfig, conds, tries, defaults,
                         # seed params
                         seed=None, seed_conds=None):
    """
    Run a model over a given seed to generate a continuation. This function
    shouldn't have any side-effects. Instead all updates are done by the
    Generator class itself.

    Arguments:
    ----------
    - mconfig : dict, model-specific dictionary with keys:
        "model", "encoder", "hidden", "cache", "options"
    - config : AppConfig, global AppConfig object
    - conds : dict, dictionary of currently active conditions
    - seed : str, picked sentence to use as seed (already syllabified)
    - seed_conds : dict, dictionary of conditions used to generate previous sentence

    Returns:
    --------
    hyp : str, generated sentence
    score : float, score for the generated sentence
    hidden : tuple, torch.Tensor or None, hidden state after reading currently
        picked candidate
    """
    model, encoder = mconfig["model"], mconfig['encoder']

    # preprocess seed
    hidden = mconfig.get("hidden")
    if seed is not None:
        hidden = process_seed(mconfig, seed, seed_conds)

    # expand hidden to batch size
    hidden_ = None
    if hidden is not None:
        hidden_ = []
        for h in hidden:
            if isinstance(h, tuple):
                hidden_.append((h[0].repeat(1, tries, 1), h[1].repeat(1, tries, 1)))
            else:
                hidden_.append(h.repeat(1, tries, 1))

    # transform conditions to actual input
    conds = {key: encoder.conds[key].w2i[val] for key, val in conds.items()}

    (hyps, _), scores, _ = model.sample(
        encoder,
        batch=tries,
        conds=conds,
        hidden=hidden_,
        tau=mconfig.get("options", {}).get("tau", defaults["tau"]),
        # cache: TODO: don't force-update cache until a pick has been done
        cache=mconfig.get("cache"))

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
    while c < len(hyps) - 1:
        hyp = hyps[c].split()
        if not utils.is_valid(hyp) or (seed and not utils.is_valid_pair(hyp, seed)):
            hyps.pop(c)
            scores.pop(c)
        else:
            c += 1

    # sample from the filtered hyps
    idx = random.choices(list(range(len(hyps))), scores).pop()
    # select sampled
    hyp, score = hyps[idx], scores[idx]

    return hyp, score, hidden


class Generator:
    """
    Attributes:
    - config : AppConfig
    - counter : number of accepted candidates so far
    - syllabifier : allennlp.services.Predictor
    - models : dictionary
    - candidates : dictionary,

        { "conds": dict,
          "hyps": {
            modelname: {
                id: {
                    "hyp": str,
                    "score": score
                }
            }
          }
        }
    """
    def __init__(self, config):
        self.config = config
        # counter
        self.counter = 0
        # load syllabifier
        self.syllabifier = Predictor.from_path(
            os.path.join(config['MODEL_DIR'], config['SYLLABIFIER']))
        # load models
        self.models = load_models(config)
        # first sample
        self.candidates = {"conds": {}, "hyps": {}}

    def resample(self):
        """
        Update candidates for the current step without modifying any local state
        (hidden, cache, counter, conds) except, of course, the current candidates
        """
        candidates, payload = {}, []
        conds = self.candidates['conds']
        candidates['conds'] = conds
        candidates['hyps'] = {}

        for modelname, mconfig in self.models.items():
            hyp, score, _ = get_model_generation(
                mconfig, conds, self.config['TRIES'], self.config['DEFAULTS'])
            # update new candidates (always kept as they come from model.sample)
            id = str(uuid.uuid1())[:8]
            candidates["hyps"][modelname] = {id: {"hyp": hyp}}

            if isinstance(self.models[modelname], RNNLanguageModel):
                hyp = utils.join_syllables(hyp.split())
            hyp = utils.detokenize(hyp)
            payload.append({"id": "{}/{}".format(modelname, id), "text": hyp})

        self.candidates = candidates

        return payload

    def sample(self, seed_id=None):
        """
        Create new generations based on a picked seed. Update local state (hidden,
        cache, counter, conds) and the current candidates
        """
        candidates, payload = {}, []

        # add new conditions
        encoder = list(self.models.values())[0]["encoder"]
        conds = resample_conds(encoder, self.candidates["conds"], self.counter)
        candidates["conds"] = conds

        # prepare seed
        seed = None
        if seed_id is not None:
            if not self.candidates["hyps"]:
                raise ValueError("Generator was passed seed {} but ")

            modelname, id = seed_id.split('/')
            try:
                seed = self.candidates["hyps"][modelname][id]["hyp"]
            except KeyError:
                raise ValueError("Couldn't find hyp {}".format(seed_id))

            # list of words for CharLanguageModel, syllables for RNNLanguageModel
            seed = seed.split()
            if isinstance(self.models[modelname], CharLanguageModel):
                # needs syllabification
                seed = syllabify(self.syllabifier, seed)

        # add generations
        candidates["hyps"] = {}

        for modelname, mconfig in self.models.items():
            # TODO: this should run multiprocessing
            # Warning! hidden is the state after reading the seed and NOT
            # the state after generating the new candidates
            hyp, score, hidden = get_model_generation(
                mconfig, conds, self.config['TRIES'], self.config['DEFAULTS'],
                seed=seed, seed_conds=self.candidates["conds"])

            # update new candidates (always kept as they come from model.sample)
            id = str(uuid.uuid1())[:8]
            candidates["hyps"][modelname] = {id: {"hyp": hyp}}

            # payload
            if isinstance(mconfig['model'], RNNLanguageModel):
                hyp = utils.join_syllables(hyp.split())
            hyp = utils.detokenize(hyp)
            payload.append({"id": "{}/{}".format(modelname, id), "text": hyp})

            # side-effects
            self.models[modelname]["hidden"] = hidden
            # TODO: update cache

        # reset candidates
        self.candidates = candidates

        # increment counter
        self.counter += 1

        return payload

    def reset(self):
        self.counter = 0
        # needs to stop any other model-related process first
        for model, mconfig in self.models.items():
            # reset hidden
            mconfig["hidden"] = None
            # reset cache if necessary
            if mconfig.get("cache"):
                mconfig["cache"].reset()

        self.candidates = {"conds": {}, "hyps": {}}
