
from .model import RNNLanguageModel
from .char_model import CharLanguageModel, CharLevelCorpusEncoder
from .hybrid_model import HybridLanguageModel
from . import utils


def model_loader(modelpath):
    """
    General function to load models

    modelpath is a string /path/to/modelname (no extension)
    """
    import os

    # remove path
    modelname = os.path.basename(modelpath)

    if modelname.startswith('RNNLanguageModel'):
        model, encoder = RNNLanguageModel.load(modelpath, utils.CorpusEncoder)
    elif modelname.startswith('HybridLanguageModel'):
        model, encoder = HybridLanguageModel.load(modelpath, utils.CorpusEncoder)
    elif modelname.startswith('CharLanguageModel'):
        model, encoder = CharLanguageModel.load(modelpath, CharLevelCorpusEncoder)
    else:
        raise ValueError("Couldn't identify {} as model".format(modelpath))

    # always load on eval mode
    model.eval()

    return model, encoder


from .cache import Cache
from .generator import TemplateSampler, sample_conditions
