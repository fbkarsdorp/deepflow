Steps to setup the server:

1. Make and fill the database:

```bash
> python make_db.py 
> python fill_db.py
```

2. start the redis server:

```bash
> redis-server
```

3. start celery in `deepflow/api` (possible nodes are 'big', 'tupac', and 'kendrick'; make sure to also add them in the queues):

```bash
> celery multi start big -A app.celery --loglevel=INFO --time-limit=300 -c 1 -Q:big big-queue --logfile=celery.log
```

or with more nodes:

```bash
> celery multi start big tupac -A app.celery --loglevel=INFO --time-limit=300 -c 1 -Q:big big-queue -Q:tupac tupac-queue --logfile=celery.log
```

4. Start a flask server. In `deepflow/api` run:

```bash
> python wsgi.py
```
