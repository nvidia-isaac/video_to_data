from modules.foundationpose.worker import celery_app
from modules.foundationpose.functions.video_to_poses import video_to_poses as video_to_poses_func
from modules.foundationpose.functions.simplify_mesh import simplify_mesh as simplify_mesh_func
from modules.foundationpose.functions.transform_mesh import transform_mesh as transform_mesh_func
from modules.foundationpose.functions.align_mesh_scale import align_mesh_scale as align_mesh_scale_func
from modules.foundationpose.functions.render_foundationpose_overlay import render_foundationpose_overlay as render_foundationpose_overlay_func

@celery_app.task(name='foundationpose.video_to_poses', queue='foundationpose.video_to_poses')
def video_to_poses(video_path: str, depth_folder: str, masks_folder: str, camera_intrinsics_path: str, mesh_path: str, poses_dir: str, reference_frame: int = 0, target_width: int = None, target_height: int = None, render_debug: bool = False, debug_dir: str = None):
    """Process a video to track object poses."""
    res = video_to_poses_func(
        video_path, depth_folder, masks_folder, camera_intrinsics_path, mesh_path, poses_dir,
        reference_frame=reference_frame,
        target_width=target_width,
        target_height=target_height
    )
    
    if render_debug and debug_dir:
        render_foundationpose_overlay_func(
            video_path=video_path,
            poses_dir=poses_dir,
            mesh_path=mesh_path,
            camera_intrinsics_path=camera_intrinsics_path,
            output_dir=debug_dir
        )
    
    return res

@celery_app.task(name='foundationpose.render_overlay', queue='foundationpose.render_overlay')
def render_overlay(video_path: str, poses_dir: str, mesh_path: str, camera_intrinsics_path: str, output_dir: str):
    """Render FoundationPose overlay on video."""
    return render_foundationpose_overlay_func(video_path, poses_dir, mesh_path, camera_intrinsics_path, output_dir)

@celery_app.task(name='foundationpose.simplify_mesh', queue='foundationpose.simplify_mesh')
def simplify_mesh(input_mesh_path: str, output_mesh_path: str, face_count: int = None, factor: float = None):
    """Simplify a mesh."""
    return simplify_mesh_func(input_mesh_path, output_mesh_path, face_count, factor)

@celery_app.task(name='foundationpose.transform_mesh', queue='foundationpose.transform_mesh')
def transform_mesh(input_mesh_path: str, output_mesh_path: str, transform_path: str):
    """Transform a mesh."""
    return transform_mesh_func(input_mesh_path, output_mesh_path, transform_path)

@celery_app.task(name='foundationpose.align_mesh_scale', queue='foundationpose.align_mesh_scale')
def align_mesh_scale(depth_path: str, mask_path: str, intrinsics_path: str, input_mesh_path: str, output_mesh_path: str, transform_path: str):
    """Align a mesh scale."""
    return align_mesh_scale_func(input_mesh_path, output_mesh_path, transform_path)