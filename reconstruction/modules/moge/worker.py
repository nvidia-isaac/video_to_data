"""
Celery worker configuration for MoGe service
"""

from modules.common.celery_utils import create_celery_app

celery_app = create_celery_app("moge")

# Ensure tasks are registered when Celery loads this app
celery_app.conf.include = ["modules.moge.tasks"]

