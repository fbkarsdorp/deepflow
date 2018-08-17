
import json
import os
import uuid
import random

from allennlp.predictors import Predictor

from .generation import RNNLanguageModel, HybridLanguageModel, CharLanguageModel
from .generation import model_loader
from .generation import TemplateSampler, sample_conditions
# from .generation import Cache
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
        # if mconfig.get("options", {}).get("cache"):
        #     cache = Cache.new(
        #         model.hidden_dim,                  # hidden_dim
        #         config['MODEL_DEFAULTS']["cache_size"])  # cache_size

        # add mconfig
        models[modelname] = {
            "model": model,
            "encoder": encoder,
            "options": mconfig.get("options", {}),
            "rweights": rweights,
            "cache": cache,
            "hidden": None}

        print("Model options: ")
        print(json.dumps(models[modelname]['options']))

    return models


def process_seed(model, encoder, hidden, seed, seed_conds):
    """
    Process the picked option (seed) and return the new hidden state

    Arguments:
    ----------

    - mconfig : dict corresponding to the model config data
    - seed : list of str, input to transform_batch
    - seed_conds : dict of conditions corresponding to the seed
    """
    (word, nwords), (char, nchars), conds = encoder.transform_batch([seed], [seed_conds])

    # TODO: add cache
    _, hidden = model(word, nwords, char, nchars, conds, hidden)

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
    - tries : int, number of candidates to sample from based on their probs
    - defaults : dict, app-level defaults for sampling options
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

    # update hidden with given seed
    seed_hidden = mconfig["hidden"]
    if seed is not None:
        seed_hidden = process_seed(model, encoder, seed_hidden, seed, seed_conds)

    # prepare hidden for generation
    hidden = None
    if seed_hidden is not None:
        hidden = []
        for h in seed_hidden:
            if isinstance(h, tuple):
                hidden.append((h[0].repeat(1, tries, 1), h[1].repeat(1, tries, 1)))
            else:
                hidden.append(h.repeat(1, tries, 1))

    # transform conditions to actual input
    conds = {c: vocab.w2i[conds[c]] for c, vocab in encoder.conds.items()}

    print(mconfig["options"].get("tau", defaults["tau"]))

    (hyps, _), scores, _, _ = model.sample(
        encoder,
        batch=tries,
        conds=conds,
        hidden=hidden,
        avoid_unk=defaults["avoid_unk"],
        tau=mconfig["options"].get("tau", defaults["tau"]),
        cache=mconfig["cache"])

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

    return hyp, score, seed_hidden


class Generator:
    """
    Attributes:
    - config : AppConfig
    - counter : number of accepted candidates so far
    - syllabifier : allennlp.services.Predictor
    - tsampler : TemplateSampler or None
    - models : dictionary
    - state : dictionary,

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
        if os.path.isfile(os.path.join(config['MODEL_DIR'], config['SYLLABIFIER'])):
            self.syllabifier = Predictor.from_path(
                os.path.join(config['MODEL_DIR'], config['SYLLABIFIER']))
        else:
            print("Couldn't load Syllabifier, file not found:\n\t{}".format(
                os.path.join(config['MODEL_DIR'], config['SYLLABIFIER'])))
        # load template sample if given
        self.tsampler = None
        if os.path.isfile(config['SONG_PATH']) and os.path.isfile(config['PHON_DICT']):
            self.tsampler = TemplateSampler(config['SONG_PATH'], config['PHON_DICT'])
        else:
            print("Couldn't load TemplateSampler, files not found:\n\t{}\n\t{}".format(
                config['SONG_PATH'], config['PHON_DICT']))
        # load models
        self.models = load_models(config)
        # reset state structures
        self.state = {"hyps": {}, "conds": {}, "template": {}}
        self.reset()

    def get_conditions(self, encoder=None):
        """
        Takes care of sampling conditions and planning the song over consecutive
        generations
        """
        encoder = encoder or list(self.models.values())[0]["encoder"]
        conds = self.state["conds"]

        if self.state['template']:
            template = self.state["template"]["data"]
            # next step
            conds = template[self.counter % len(template)]

        else:
            if not conds or self.counter % 4 == 0:
                # resample every 4
                conds = sample_conditions(encoder)

        return conds

    def resample(self):
        """
        Recreate candidates for the current step without modifying any local state
        (hidden, cache, counter, conds).
        """
        state = {**self.state, "hyps": {}}  # reset hyps
        conds = state["conds"]
        payload = []

        for modelname, mconfig in self.models.items():
            hyp, score, _ = get_model_generation(
                mconfig, conds,
                self.config['MODEL_TRIES'], self.config['MODEL_DEFAULTS'])
            # seed is the same as previous time around (no need to update hidden)
            # update new hyps (always kept as they come from model.sample)
            id = str(uuid.uuid1())[:8]
            state["hyps"][modelname] = {id: {"hyp": hyp}}

            # payload
            if isinstance(mconfig['model'], (RNNLanguageModel, HybridLanguageModel)):
                hyp = utils.join_syllables(hyp.split())
            hyp = utils.detokenize(hyp)

            data = {
                "id": "{}/{}".format(modelname, id),
                "text": hyp,
                "score": score,
                "conds": conds}

            payload.append(data)

        self.state = state

        return payload

    def sample(self, seed_id=None):
        """
        Create new generations based on a picked seed. Update local state (hidden,
        cache, counter, conds) and the current hyps
        """
        payload = []

        # prepare seed
        seed = None
        if seed_id is not None:
            if not self.state["hyps"]:
                raise ValueError("Generator was passed seed {} but ")

            modelname, id = seed_id.split('/')
            try:
                seed = self.state["hyps"][modelname][id]["hyp"]
            except KeyError:
                raise ValueError("Couldn't find hyp {}".format(seed_id))

            # list of words for CharLanguageModel, syllables for RNNLanguageModel
            seed = seed.split()
            if isinstance(self.models[modelname], CharLanguageModel):
                # needs syllabification
                seed = syllabify(self.syllabifier, seed)

        # reset hyps and conditions
        conds = self.get_conditions()
        state = {**self.state, "conds": conds, "hyps": {}}

        for modelname, mconfig in self.models.items():
            # Warning! hidden is the state after reading the seed and NOT
            # the state after generating the new candidates
            hyp, score, hidden = get_model_generation(
                mconfig, conds,
                self.config['MODEL_TRIES'], self.config['MODEL_DEFAULTS'],
                seed=seed, seed_conds=self.state["conds"])

            # update new candidates (always kept as they come from model.sample)
            id = str(uuid.uuid1())[:8]
            state["hyps"][modelname] = {id: {"hyp": hyp}}

            # payload
            if isinstance(mconfig['model'], (RNNLanguageModel, HybridLanguageModel)):
                hyp = utils.join_syllables(hyp.split())
            hyp = utils.detokenize(hyp)

            data = {
                "id": "{}/{}".format(modelname, id),
                "text": hyp,
                "score": score,
                "conds": conds}

            payload.append(data)

            # side-effects
            self.models[modelname]["hidden"] = hidden
            # self.models[modelname]["cache"] = cache

        # reset state
        self.state = state

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
            # if mconfig["cache"]:
            #     mconfig["cache"] = mconfig["cache"].reset()

        self.state["conds"] = {}
        self.state["hyps"] = {}
        self.state["template"] = {}

        if self.tsampler is not None:
            if random.random() <= self.config['TEMPLATE_RATIO']:
                tdata, tmetadata = self.tsampler.sample(
                    nlines=self.config['TEMPLATE_MIN_LEN'])
                self.state["template"] = {"data": tdata, "metadata": tmetadata}
