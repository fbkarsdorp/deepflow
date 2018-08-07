import os

basedir = os.path.abspath(os.path.dirname(__file__))


class AppConfig:
    MODEL_DIR = os.path.join(basedir, 'data/models/')
    TURING_FILE = os.path.join(basedir, 'data/turing-pairs.json')

    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'turing.db')
    SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SECRET_KEY = "=\x07BoZ\xeb\xb0\x13\x88\xf8mW(\x93}\xe6k\r\xebA\xbf\xff\xb1v"


class CeleryConfig:
    task_serializer = 'pickle'
    result_serializer = 'pickle'
    accept_content = ['pickle']
