import os

import flask
import flask_login
import flask_sqlalchemy
import celery
import tweepy

from .turing import ExampleSampler
from .lyrics import Generator

import config
import twitterconfig as tw

app = flask.Flask(__name__, static_folder='static', template_folder='templates')
app.config.from_object('config.AppConfig')

db = flask_sqlalchemy.SQLAlchemy(app)

celery = celery.Celery(
    __name__,
    broker=os.environ.get('CELERY_BROKER_URL', 'redis://127.0.0.1:6379'),
    backend=os.environ.get('CELERY_BROKER_URL', 'redis://127.0.0.1:6379'))
celery.config_from_object('config.CeleryConfig')

lm = flask_login.LoginManager()
lm.session_protection = 'strong'
lm.init_app(app)

print(app.config)
app.ExampleSampler = ExampleSampler(app.config['TURING_FILE'])
app.Generator = Generator(app.config)

auth = tweepy.OAuthHandler(tw.consumer_key, tw.consumer_secret)
auth.set_access_token(tw.access_token, tw.access_secret)
app.twitter_api = tweepy.API(auth)


from app import views, models
