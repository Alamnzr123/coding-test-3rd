import os
from celery import Celery

redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("backend_tasks", broker=redis_url, backend=redis_url)
celery_app.conf.task_routes = {"backend.app.tasks.parse_tasks.*": {"queue": "parsers"}}