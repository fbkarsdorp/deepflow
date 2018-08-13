
import json
import random
import uuid

from typing import Dict

import flask
import flask_login

from celery import states
from app import app, db, celery, lm
from .models import Turn, Machine
from .forms import LoginForm


@lm.user_loader
def load_user(id):
    return Machine.query.get(int(id))


@app.before_request
def before_request():
    flask.g.user = flask_login.current_user

###############################################################################
# MC Turing views
###############################################################################


@app.route('/scoreboard', methods=['GET'])
def get_scoreboard() -> flask.Response:
    ranking = Turn.query.order_by(Turn.score.desc(), Turn.timestamp.desc()).limit(10).all()
    ranking = [{'name': row.name.split('^^^')[0], 'score': row.score} for row in ranking]
    return flask.jsonify(status='OK', ranking=ranking)


@app.route('/saveturing', methods=['POST'])
def save_turn() -> flask.Response:
    data = flask.request.json
    name = f'{uuid.uuid1()}'[:5]
    exists = Turn.query.filter_by(name=name).first()
    while exists is not None:
        name = f'{uuid.uuid1()}'[:5]
        exists = Turn.query.filter_by(name=name).first()
    turn = Turn(name=name, log=data['log'], score=data['score'])
    db.session.add(turn)
    db.session.commit()
    return flask.jsonify(status='OK', message='turn saved', name=name)


@app.route('/pair', methods=['GET', 'POST'])
def get_pair() -> flask.Response:
    data = flask.request.json
    if data is None:
        iteration, level, seen = 0, 0, []
    else:
        iteration, level, seen = data['iteration'], data['level'], data['seen']
    id, true, false, iteration, level = app.ExampleSampler.next(
        iteration, level, seen)
    return flask.jsonify(
        status='OK', id=id, real=true, fake=false, iteration=iteration, level=level)


###############################################################################
# Lyrics composition views
###############################################################################


@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask.g.user is not None and flask.g.user.is_authenticated:
        return flask.redirect('static/lyrics/index.html')
    form = LoginForm()
    if form.validate_on_submit() and form.validate_fields():
        machine = form.get_machine()
        print(machine)
        flask.session['remember_me'] = form.remember_me.data
        flask_login.login_user(machine, remember=form.remember_me.data)
        return flask.redirect('static/lyrics/index.html')
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


@app.route('/upload', methods=['POST'])
def save_session() -> flask.Response:
    data = flask.request.json
    with open(f'{app.config["LOG_DIR"]}/{uuid.uuid1()}.txt', 'w') as f:
        json.dump(data, f)
    app.Generator.reset()
    return flask.jsonify({'status': 'OK', 'message': 'session saved'})
