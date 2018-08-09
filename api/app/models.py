import datetime
import json
import os

from app import app, db
from sqlalchemy.dialects.mysql import MEDIUMTEXT


if os.environ.get('DEEPFLOW_DB_URL') is None:
    MEDIUMTEXT = db.VARCHAR


class JSONEncodedDict(db.TypeDecorator):
    "Represents an immutable structure as a json-encoded string."

    impl = MEDIUMTEXT

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


class Machine(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(30), unique=True)
    active = db.Column(db.SmallInteger, default=False)

    def __repr__(self):
        return f'<Machine({self.name})>'

    def is_authenticated(self):
        return True

    def is_active(self):
        return self.active

    def is_anonymous(self):
        return False

    def get_id(self):
        return self.id


class Turn(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(30), unique=True)
    score = db.Column(db.Integer)
    log = db.Column(JSONEncodedDict)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
