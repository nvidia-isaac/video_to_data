"""
Celery worker configuration for SAM2 service
"""

from modules.common.celery_utils import create_celery_app

celery_app = create_celery_app("sam2")

# Ensure tasks are registered when Celery loads this app
celery_app.conf.include = ["modules.sam2.tasks"]

