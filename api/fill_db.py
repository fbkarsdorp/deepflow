from app import db, models

MACHINES = 'tupac', 'big', 'kendrick'
for machine in MACHINES:
    print(f'adding {machine} to DB')
    machine = models.Machine(name=machine)
    db.session.add(machine)
db.session.commit()
