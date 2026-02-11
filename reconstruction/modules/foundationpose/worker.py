"""
Celery worker configuration for FoundationPose service
"""

import os
import multiprocessing
import sys

# Force spawn start method for multiprocessing to avoid CUDA hangs in Celery workers
# This MUST be done before any other imports that might start multiprocessing
try:
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
        # Use stderr to ensure it shows up in logs even if stdout is buffered
        sys.stderr.write("Successfully set multiprocessing start method to 'spawn'\n")
        sys.stderr.flush()
except RuntimeError:
    sys.stderr.write(f"Could not set multiprocessing start method to 'spawn', already set to {multiprocessing.get_start_method()}\n")
    sys.stderr.flush()

from modules.common.celery_utils import create_celery_app

celery_app = create_celery_app("foundationpose")

# Ensure tasks are registered when Celery loads this app
celery_app.conf.include = ["modules.foundationpose.tasks"]

