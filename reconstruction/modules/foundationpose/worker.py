"""
Celery worker configuration for FoundationPose service
"""

from modules.common.celery_utils import create_celery_app

celery_app = create_celery_app("foundationpose")

# Ensure tasks are registered when Celery loads this app
celery_app.conf.include = ["modules.foundationpose.tasks"]

