import datetime
import json

from app import app, db


class JSONEncodedDict(db.TypeDecorator):
    "Represents an immutable structure as a json-encoded string."

    impl = db.VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


class Machine:
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(10), unique=True)

    def __repr__(self):
        return f'<Machine({self.name})>'


class Turn(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Integer, unique=True)
    score = db.Column(db.Integer)
    log = db.Column(JSONEncodedDict)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
