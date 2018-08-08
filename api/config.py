import os

basedir = os.path.abspath(os.path.dirname(__file__))


class AppConfig:
    # generation
    TRIES = 10  # parallel tries per sentence
    DEFAULTS = {
        "tau": 0.8,
        "cache_size": 100
    }
    MODEL_DIR = os.path.join(basedir, 'data/models/')
    MODELS = {
        # # add model-specific configuration in the following form
        # "path": "ModelName.pt",
        # "options": {
        #    "cache": {"alpha": 0.01, "theta": 0.17}
        #    "tau": 0.95 }
    }
    SYLLABIFIER = "syllable-model.tar.gz"

    TURING_FILE = os.path.join(basedir, 'data/turing-pairs.json')

    # database
    if os.environ.get('DEEPFLOW_DB_URL') is None:
        print('Falling back to SQLite database.')
        SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'turing.db')
    else:
        print('Using MySQL database.')
        SQLALCHEMY_DATABASE_URI = os.environ['DEEPFLOW_DB_URL']
    SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')

    # others
    SECRET_KEY = "=\x07BoZ\xeb\xb0\x13\x88\xf8mW(\x93}\xe6k\r\xebA\xbf\xff\xb1v"

    # log dir for storing sessions
    LOG_DIR = 'data/logs/'


class CeleryConfig:
    task_serializer = 'pickle'
    result_serializer = 'pickle'
    accept_content = ['pickle']
