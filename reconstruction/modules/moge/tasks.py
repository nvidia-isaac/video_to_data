"""
Celery tasks for MoGe service.
Wraps functions from functions/ module with Celery task decorators.
"""
from modules.moge.worker import celery_app
from modules.moge.functions.image_to_depth import image_to_depth as image_to_depth_func
from modules.moge.functions.video_to_depth import video_to_depth as video_to_depth_func

@celery_app.task(name='moge.image_to_depth', queue='moge.image_to_depth')
def image_to_depth(image_path: str, depth_path: str, intrinsics_path: str):
    """Process single image to depth."""
    return image_to_depth_func(image_path, depth_path, intrinsics_path)

@celery_app.task(name='moge.video_to_depth', queue='moge.video_to_depth')
def video_to_depth(video_path: str, depth_folder: str, intrinsics_folder: str, batch_size: int = 8):
    """Process video to depth frames."""
    return video_to_depth_func(video_path, depth_folder, intrinsics_folder, batch_size)

