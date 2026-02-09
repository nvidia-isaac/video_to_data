"""
Celery worker configuration for SAM3D service
"""

from modules.common.celery_utils import create_celery_app

celery_app = create_celery_app("sam3d")

# Ensure tasks are registered when Celery loads this app
celery_app.conf.include = ["modules.sam3d.tasks"]

