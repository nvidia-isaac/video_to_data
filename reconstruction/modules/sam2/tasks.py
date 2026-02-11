"""
Celery tasks for SAM2 service.
Wraps functions from functions/ module with Celery task decorators.
"""
from modules.sam2.worker import celery_app
from modules.sam2.functions.video_to_masks import video_to_masks as video_to_masks_func

@celery_app.task(name='sam2.video_to_masks', queue='sam2.video_to_masks')
def video_to_masks(video_path: str, prompts_path: str, masks_dir: str):
    """Process a video with SAM2 prompts and save masks to files."""
    return video_to_masks_func(video_path, prompts_path, masks_dir)

