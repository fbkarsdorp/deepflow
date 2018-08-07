import os

import flask
import flask_login
import flask_sqlalchemy
import celery

from .turing import ExampleSampler
from .lyrics import Generator


app = flask.Flask(__name__)
app.config.from_object('config.AppConfig')

db = flask_sqlalchemy.SQLAlchemy(app)

celery = celery.Celery(
    __name__,
    broker=os.environ.get('CELERY_BROKER_URL', 'redis://')
    backend=os.environ.get('CELERY_BROKER_URL', 'redis://'))
celery.config_from_object('config.CeleryConfig')

lm = flask_login.LoginManager()
lm.session_protection = 'strong'
lm.init_app(app)
lm.login_view('login')

app.ExampleSampler = ExampleSampler(config.AppConfig.TURING_FILE)
app.Generator = Generator(config.AppConfig.MODEL_DIR)

from app import views, models
