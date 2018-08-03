import datetime
import json
import random
import uuid

from typing import Dict

import flask
import flask_login

from celery import states
from app import app, db, celery
from .models import Turn


################################################################################
# MC Turing views
################################################################################


@app.route('/scoreboard', methods=['GET'])
def get_scoreboard() -> flask.Response:
    ranking = Turn.query.order_by(Turn.score.desc()).all()
    ranking = [(row.name.split('^^^')[0], row.score) for row in ranking]
    return flask.jsonify(status='OK', ranking=ranking)


@app.route('/saveturn', methods=['POST'])
def save_turn() -> flask.Response:
    data = flask.request.json
    name = f'{name}^^^{uuid.uuid1()}'
    turn = Turn(name=name, log=data['log'], score=data['score'])
    db.session.add(turn)
    db.session.commit()
    return flask.jsonify(status='OK', message='turn saved')


@app.route('/pair', methods=['GET'])
def get_pair() -> flask.Response:
    id, real, fake = app.SentenceSampler.next()
    return flask.jsonify(status='OK', id=id, real=real, fake=fake)


################################################################################
# Lyrics composition views
################################################################################


@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask.g.user is not None and flask.g.user.is_authenticated:
        return flask.redirect(flask.url_for('index'))
    form = LoginForm()
    if form.validate_on_submit() and form.validate_fields():
        machine = form.get_machine()
        flask.session['remember_me'] = form.remember_me.data
        flask_login.login_user(machine, remember=form.remember_me.data)
        return flask.redirect(flask.url_for('index'))
    return flask.render_template('login.html', title='Sign in', form=form)


@app.route('/generate', methods=['POST', 'GET'])
@flask_login.login_required
def generate() -> flask.Response:
    data = flask.request.json
    user_id = int(flask_login.current_user.id)
    job = generate_task.apply_async(
        args=(user_id,), queue=flask_login.current_user.name
    )
    return flask.jsonify({}), 202, {
        'Location': flask.url_for('get_status', id=job.id)}


@app.route('/status/<id>', methods=['GET'])
def get_status(id) -> flask.Response:
    job = generate_task.AsyncResult(id)
    if job.state in (states.PENDING, states.RECEIVED, states.STARTED):
        return flask.jsonify({}), 202, {
            'Location': flask.url_for('get_status', id=id)}
    return flask.jsonify(job.info)


@celery.task
def generate_task(user_id) -> Dict[str, str]:
    with app.app_context():
        try:
            hyps = app.Generator.sample()
            return {'status': 'OK', 'hyps': hyps}
        except Exception as e:
            if app.debug is True:
                raise e
            return {'status': 'fail', 'message': str(e), 'code': 500}


@app.route('/upload', methods='POST')
def save_session() -> flask.Response:
    data = flask.request.json
    with open(f'{app.config.RESULT_DIR}/{uuid.uuid1()}.txt', 'w') as f:
        json.dump(data, f)
    return flask.jsonify({'status': OK, 'message': 'session saved'})