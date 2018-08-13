import os

basedir = os.path.abspath(os.path.dirname(__file__))


class AppConfig:
    # GENERATION
    # - model configuration
    MODEL_TRIES = 1             # parallel tries per sentence
    MODEL_DEFAULTS = {
        "tau": 0.8,
        "avoid_unk": True
    }
    MODEL_DIR = os.path.join(basedir, 'data/models/')
    MODELS = {
        # # add model-specific configuration in the following form
        # "path": "ModelName.pt",
        # "options": {
        #    "tau": 0.95 }
    }
    # - syllabification
    SYLLABIFIER = "syllable-model.tar.gz"     # fpath of syllabifier in MODEL_DIR
    # - condition templates
    SONG_PATH = "data/ohhla-new.jsonl"        # file with songs in jsons format
    PHON_DICT = "data/ohhla.vocab.phon.json"  # file with phonological dictionary
    TEMPLATE_MIN_LEN = 3   # minimum #lines for a template
    # prop of songs created with template (only if template data is available)
    TEMPLATE_RATIO = 1.0

    # TURING
    TURING_FILE = os.path.join(basedir, 'data/turing-pairs.jsonl')

    # DATABASE
    if os.environ.get('DEEPFLOW_DB_URL') is None:
        print('Falling back to SQLite database.')
        SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'turing.db')
    else:
        print('Using MySQL database.')
        SQLALCHEMY_DATABASE_URI = os.environ['DEEPFLOW_DB_URL']
    SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')

    # OTHERS
    SECRET_KEY = "=\x07BoZ\xeb\xb0\x13\x88\xf8mW(\x93}\xe6k\r\xebA\xbf\xff\xb1v"
    LOG_DIR = 'data/logs/'


class CeleryConfig:
    task_serializer = 'pickle'
    result_serializer = 'pickle'
    accept_content = ['pickle']
