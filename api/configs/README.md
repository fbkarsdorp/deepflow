# Server configuration

This directory contains the config files for the server. It provides configuration for (i)
celery, (ii) redis, (iii) gunicorn and (iv) nginx. All components can and should be run as
systemd services.

## Redis

The message queue is controled by redis which should be run first. Place the file
`redis.service` in `/etc/systemd/system/` and run `sudo systemctl start redis`. Check its
status with `systemctl status redis` to see if it's running.

## Celery

Next in line are the celery workers. For this we first copy the ennvironment config file
`celeryd` to `/etc/default`. Next, we put `celery.service` in `/etc/systemd/system/` and
start it using `sudo systemctl start celery`.

## Gunicorn

To launch the webservers, we first move the `deepflow.target` file and the
`deepflow@.service` file to `/etc/systemd/system/`. Next, start the target file using
`sudo systemctl start deepflow.target`. To launch different servers, we employ the
template mechanism offered by systemd. For example, to start a server at port 5000, we
execute: `sudo systemctl start deepflow@5000`. Do this for all servers listed in
`/etc/nginx/sites-available/deepflow`, i.e. 5000:5009.

## Nginx

Nginx is started and restarted with: `sudo systemctl start nginx`. 
