"""
Celery worker configuration for UniDepth service
"""

from modules.common.celery_utils import create_celery_app

celery_app = create_celery_app("unidepth")

# Ensure tasks are registered when Celery loads this app
celery_app.conf.include = ["modules.unidepth.tasks"]

