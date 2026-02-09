"""
Common utilities for Celery configuration across services
"""
import os
from celery import Celery


def create_celery_app(app_name: str, broker_url: str = None) -> Celery:
    """
    Create a standardized Celery app instance.
    
    Args:
        app_name: Name of the Celery application (e.g., 'moge', 'sam2')
        broker_url: Redis broker URL. If None, reads from REDIS_URL env var.
    
    Returns:
        Configured Celery app instance
    """
    if broker_url is None:
        broker_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    
    celery_app = Celery(
        app_name,
        broker=broker_url,
        backend=broker_url  # Use Redis as both broker and result backend
    )
    
    # Standard configuration
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=3600 * 24,  # 24 hour timeout
        worker_prefetch_multiplier=1,  # Disable prefetching for better load balancing
        # Route all tasks from this app to module-specific queue
        task_default_queue=app_name,
        task_routes={
            f'modules.{app_name}.tasks.*': {'queue': app_name},
        },
    )
    
    return celery_app


