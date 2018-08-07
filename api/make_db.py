from app import db, models

db.create_all()

MACHINES = 'tupac', 'big', 'kendrick'
for machine in MACHINES:
    machine = models.Machine(name=machine)
    db.session.add(machine)
db.session.commit()
    
