import os
import json
import time
import shutil
from celery import Celery
from modules.common.server_utils import zip_directory

# Configure Celery
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("orchestrator_tasks", broker=REDIS_URL, backend=REDIS_URL)

# Import tasks from other modules
# Note: We use send_task with task names to avoid importing dependencies
# from other modules.

@celery_app.task(name="orchestrator.reconstruction", bind=True, queue="orchestrator.reconstruction")
def orchestrate_reconstruction(self, job_id):
    """
    End-to-end orchestration of the reconstruction pipeline.
    """
    base_dir = f"/data/jobs/{job_id}"
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    
    video_path = os.path.join(input_dir, "video.mp4")
    prompts_path = os.path.join(input_dir, "prompts.json")
    
    with open(prompts_path, 'r') as f:
        prompts_data = json.load(f)
    
    # 1. SAM2 Segmentation
    self.update_state(state='PROGRESS', meta={'current_step': 'sam2'})
    masks_dir = os.path.join(output_dir, "masks")
    sam2_task = celery_app.send_task("sam2.video_to_masks", args=[video_path, prompts_path, masks_dir], queue="sam2.video_to_masks")
    sam2_task.get(timeout=3600, disable_sync_subtasks=False)
    
    # Identify object and human IDs from prompts
    object_id = None
    human_id = None
    ref_frame = 0
    for p in prompts_data['prompts']:
        if p['role'] == 'object' and object_id is None:
            object_id = p['object_id']
            ref_frame = p['frame_index']
        if p['role'] == 'human' and human_id is None:
            human_id = p['object_id']
    
    # 2. MoGe Depth Estimation
    self.update_state(state='PROGRESS', meta={'current_step': 'moge'})
    depth_dir = os.path.join(output_dir, "depth")
    intrinsics_dir = os.path.join(output_dir, "intrinsics")
    moge_task = celery_app.send_task("moge.video_to_depth", args=[video_path, depth_dir, intrinsics_dir], queue="moge.video_to_depth")
    moge_task.get(timeout=3600, disable_sync_subtasks=False)
    
    # 3. SAM3D Mesh Generation (using ref_frame)
    self.update_state(state='PROGRESS', meta={'current_step': 'sam3d'})
    # Extract ref frame image
    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
    ret, frame = cap.read()
    ref_image_path = os.path.join(input_dir, f"ref_frame_{ref_frame:06d}.jpg")
    cv2.imwrite(ref_image_path, frame)
    cap.release()
    
    ref_mask_path = os.path.join(masks_dir, str(object_id), f"{ref_frame:06d}.png")
    raw_mesh_path = os.path.join(output_dir, "object_mesh_raw.glb")
    raw_transform_path = os.path.join(output_dir, "object_transform_raw.json")
    raw_intrinsics_path = os.path.join(output_dir, "object_intrinsics_raw.json")
    
    sam3d_task = celery_app.send_task("sam3d.image_to_mesh", args=[ref_image_path, ref_mask_path, raw_mesh_path, raw_transform_path, raw_intrinsics_path], queue="sam3d.image_to_mesh")
    sam3d_task.get(timeout=3600, disable_sync_subtasks=False)
    
    # 4. Alignment & Simplification
    self.update_state(state='PROGRESS', meta={'current_step': 'alignment'})
    
    # Simplify
    simplified_mesh_path = os.path.join(output_dir, "object_mesh_simplified.glb")
    simplify_task = celery_app.send_task("foundationpose.simplify_mesh", args=[raw_mesh_path, simplified_mesh_path], kwargs={'factor': 0.1}, queue="foundationpose.simplify_mesh")
    simplify_task.get(timeout=600, disable_sync_subtasks=False)
    
    # Align
    ref_depth_path = os.path.join(depth_dir, f"{ref_frame:06d}.png")
    ref_intrinsics_path = os.path.join(intrinsics_dir, f"{ref_frame:06d}.json")
    aligned_transform_path = os.path.join(output_dir, "object_transform_aligned.json")
    
    align_task = celery_app.send_task("foundationpose.align_mesh_scale", args=[ref_depth_path, ref_mask_path, ref_intrinsics_path, raw_mesh_path, aligned_transform_path, raw_transform_path], queue="foundationpose.align_mesh_scale")
    align_task.get(timeout=600, disable_sync_subtasks=False)
    
    # Transform simplified mesh
    final_mesh_path = os.path.join(output_dir, "object_mesh_final.glb")
    transform_task = celery_app.send_task("foundationpose.transform_mesh", args=[simplified_mesh_path, final_mesh_path, aligned_transform_path], queue="foundationpose.transform_mesh")
    transform_task.get(timeout=600, disable_sync_subtasks=False)
    
    # 5. FoundationPose Tracking
    self.update_state(state='PROGRESS', meta={'current_step': 'foundationpose'})
    object_poses_dir = os.path.join(output_dir, "object_poses")
    fp_task = celery_app.send_task("foundationpose.video_to_poses", args=[
        video_path, depth_dir, os.path.join(masks_dir, str(object_id)), 
        ref_intrinsics_path, final_mesh_path, object_poses_dir, 
        ref_frame, None, None, True, None
    ], queue="foundationpose.video_to_poses")
    fp_task.get(timeout=3600, disable_sync_subtasks=False)
    
    # 6. NLF SMPL Fitting
    self.update_state(state='PROGRESS', meta={'current_step': 'nlf'})
    smpl_output_path = os.path.join(output_dir, "human_smpl.h5")
    nlf_task = celery_app.send_task("nlf.video_to_smpl", kwargs={
        'video_path': video_path, 
        'masks_dir': os.path.join(masks_dir, str(human_id)), 
        'intrinsics_path': ref_intrinsics_path, 
        'gender': "neutral", 
        'render_debug': True,
        'debug_dir': os.path.join(output_dir, "human_debug_render"),
        'output_path': smpl_output_path
    }, queue="nlf.video_to_smpl")
    nlf_task.get(timeout=3600, disable_sync_subtasks=False)
    
    # 7. Final Visualization & Packaging
    self.update_state(state='PROGRESS', meta={'current_step': 'rendering'})
    
    # Zip everything
    zip_path = os.path.join(output_dir, "results.zip")
    zip_directory(output_dir, zip_path)
    
    return {"status": "completed", "job_id": job_id}

