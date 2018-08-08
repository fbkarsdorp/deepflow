from flask_wtf import Form
from wtforms import StringField, BooleanField
from wtforms.validators import DataRequired


class LoginForm(Form):
    name = StringField('name', validators=[DataRequired()])
    remember_me = BooleanField('remember_me', default=True)

    def validate_fields(self):
        name = self.get_machine()
        if name is None:
            self.name.errors = ('Unknown Machine ID',)
            return False
        return True

    def get_machine(self):
        return Machine.query.filter_by(name=self.name.data).first()
