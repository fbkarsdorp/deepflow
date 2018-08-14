from app import db, models
import json
import tqdm

MACHINES = 'tupac', 'big', 'kendrick'
for machine in MACHINES:
    print(f'adding {machine} to DB')
    machine = models.Machine(name=machine)
    db.session.add(machine)
db.session.commit()

with open('data/user_names.json') as f:
    names = set(json.load(f))
for name in tqdm.tqdm(names):
    if len(name) <= 15:
        artist = models.Artist(name=name)
        db.session.add(artist)
db.session.commit()
