import os
from celery import Celery
from modules.nlf.functions.video_to_smpl import video_to_smpl
from modules.nlf.functions.render_smpl_overlay import render_smpl_overlay
from modules.nlf.datatypes import CameraIntrinsics

# Configure Celery
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("nlf_tasks", broker=REDIS_URL, backend=REDIS_URL)

@celery_app.task(name="nlf.video_to_smpl")
def video_to_smpl_task(
    video_path: str, 
    masks_dir: str, 
    intrinsics_path: str, 
    gender: str, 
    model_type: str = "smplh",
    output_path: str = None,
    render_debug: bool = False,
    debug_dir: str = None
):
    """Celery task for end-to-end NLF processing."""
    result = video_to_smpl(
        video_path=video_path,
        masks_dir=masks_dir,
        intrinsics_path=intrinsics_path,
        gender=gender,
        model_type=model_type,
        output_path=output_path
    )
    
    if render_debug and debug_dir:
        render_smpl_overlay(
            video_path=video_path,
            smpl_params_path=output_path,
            intrinsics_path=intrinsics_path,
            output_dir=debug_dir
        )

    return {
        "status": "completed",
        "output_path": output_path,
        "num_frames": len(result.frames),
        "debug_dir": debug_dir if render_debug else None
    }

