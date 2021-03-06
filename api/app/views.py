
import json
import os
import random
import time
import uuid

from typing import Dict

import flask
import flask_login
import tweepy

from celery import states
from app import app, db, celery, lm
from .models import Turn, Machine, Artist
from .forms import LoginForm
from .social import create_image_file
from . import twitterconfig as tw


@lm.user_loader
def load_user(id):
    return Machine.query.get(int(id))


@app.before_request
def before_request():
    flask.g.user = flask_login.current_user


###############################################################################
# Landing 
###############################################################################

@app.route('/', methods=['GET', 'POST'])
def landing():
    # return flask.render_template('landing/index.html')
    return flask.send_from_directory('static/landing', 'index.html')

###############################################################################
# MC Turing views
###############################################################################


@app.route('/turing', methods=['GET', 'POST'])
def turing():
    return flask.send_from_directory('static/turing', 'index.html')


@app.route('/scoreboard', methods=['GET'])
def get_scoreboard() -> flask.Response:
    ranking = Turn.query.order_by(Turn.score.desc(), Turn.timestamp.desc()).limit(10).all()
    ranking = [{'name': row.name, 'score': row.score} for row in ranking]
    return flask.jsonify(status='OK', ranking=ranking)


def get_artist_name():
    artist_name = None
    while artist_name is None:
        name = Artist.query.filter_by(taken=0).first()
        name.taken = Artist.taken + 1
        db.session.commit()
        artist = Artist.query.filter_by(name=name.name).first()
        if artist.taken == 1:
            artist_name = name.name
    return artist_name


@app.route('/saveturing', methods=['POST'])
def save_turn() -> flask.Response:
    data = flask.request.json
    name = get_artist_name()
    turn = Turn(name=name, log=data['log'], score=data['score'])
    db.session.add(turn)
    db.session.commit()
    return flask.jsonify(status='OK', message='turn saved', name=name)


@app.route('/pair', methods=['GET', 'POST'])
def get_pair() -> flask.Response:
    data = flask.request.json
    if data is None:
        iteration, level, seen, artist, album = 0, 0, [], "", ""
    else:
        iteration, level, seen = data['iteration'], data['level'], data['seen']
    id, true, false, artist, album, iteration, level = app.ExampleSampler.next(
        iteration, level, seen)
    return flask.jsonify(
        status='OK', id=id, real=true, fake=false, iteration=iteration, level=level,
        artist=artist, album=album)


###############################################################################
# Lyrics composition views
###############################################################################


@app.route('/lyrics', methods=['GET', 'POST'])
def lyrics():
    return flask.send_from_directory('static/lyrics', 'index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask.g.user is not None and flask.g.user.is_authenticated:
        return flask.redirect('/lyrics')
    form = LoginForm()
    if form.validate_on_submit() and form.validate_fields():
        machine = form.get_machine()
        print(machine)
        flask.session['remember_me'] = form.remember_me.data
        flask_login.login_user(machine, remember=form.remember_me.data)
        return flask.redirect('/lyrics')
    else:
        print(form.validate_fields())
        print(form.validate_on_submit())
        print('NOT VALID')
    return flask.render_template('login.html', title='Sign in', form=form)


@app.route('/generate', methods=['POST', 'GET'])
@flask_login.login_required
def generate() -> flask.Response:
    data = flask.request.json
    seed_id = (data or {}).get('seed_id', None)
    resample = (data or {}).get('resample', False)
    job = generate_task.apply_async(
        args=(seed_id, resample), queue=f'{flask_login.current_user.name}-queue')
    return flask.jsonify({"id": job.id})


@app.route('/status/<id>', methods=['GET'])
def get_status(id) -> flask.Response:
    job = generate_task.AsyncResult(id)
    if job.state in (states.PENDING, states.RECEIVED, states.STARTED):
        return flask.jsonify({"status": "busy"})
    return flask.jsonify(job.info)


@celery.task
def generate_task(seed_id, resample) -> Dict[str, str]:
    with app.app_context():
        try:
            if resample:
                payload = app.Generator.resample()
            else:
                payload = app.Generator.sample(seed_id=seed_id)
                random.shuffle(payload)
            return {'status': 'OK', 'payload': payload}
        except Exception as e:
            if app.debug is True:
                raise e
            return {'status': 'fail', 'message': str(e), 'code': 500}


@celery.task
def tweet_image(lines, username):
    with app.app_context():
        try:
            image_file = create_image_file(lines, app.config['LYRICS_SVG'])
            statuses = (
                f'Check out this new track by {username}',
                f'MC Turing ft {username} present'
            )
            status = random.choice(statuses) + ' #LL18 #LLScience #deepflow'
            auth = tweepy.OAuthHandler(tw.consumer_key, tw.consumer_secret)
            auth.set_access_token(tw.access_token, tw.access_secret)
            twitter_api = tweepy.API(auth)
            twitter_api.update_with_media(image_file, status=status)
            time.sleep(5)
            os.unlink(image_file)
            return {'status': 'OK', 'message': 'image tweeted'}
        except Exception as e:
            if app.debug is True:
                raise e
            return {'status': 'fail', 'message': str(e), 'code': 500}


@app.route('/upload', methods=['POST'])
def save_session() -> flask.Response:
    data = flask.request.json
    with open(f'{app.config["LOG_DIR"]}/{uuid.uuid1()}.txt', 'w') as f:
        json.dump(data, f)
    app.Generator.reset()
    lines = [line['text'].strip() for line in data['lyric']]
    name = get_artist_name()
    if lines and random.random() <= 0.2:
        job = tweet_image.apply_async(
            args=(lines, name), queue='twitter-queue')
    return flask.jsonify(
        {'status': 'OK', 'message': 'session saved', 'username': name})
