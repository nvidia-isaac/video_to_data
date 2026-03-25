"""
Pipeline: airplane rayban_airplane_1 — SAM3D mesh + MoGe depth + FP tracking

  1. SAM3D: reference frame + mask → OBJ mesh (UV-textured)
  2. FP mesh scale estimation: SAM3D mesh × frame-0 RGB/depth/mask → scale factor
  3. FP tracking: MoGe depth + rescaled SAM3D mesh → raw poses
  4. EKF smoothing
  5. Render raw + smoothed poses
  6. Encode videos

Assumes frames, masks, and MoGe depth are already computed (run
airplane_rayban_airplane_1.py first through at least step 4 / MoGe depth).

Run from reconstruction/:
    python experiments/airplane_rayban_airplane_1_sam3d.py
"""

import os

from v2d.common.utils import frames_to_video, stitch_videos
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME            = "airplane"
SESSION         = "rayban_airplane_1"
OBJECT_ID       = 1
REFERENCE_FRAME = 0

VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/airplane.mp4"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"

SAM3D_WEIGHTS = "data/weights/sam3d"
FP_WEIGHTS    = "data/weights/foundation_pose"

# Inputs shared with the base pipeline (must already exist)
frames_dir      = f"{OUTPUT_DIR}/frames"
masks_dir       = f"{OUTPUT_DIR}/masks/{OBJECT_ID}"
moge_depth_dir  = f"{OUTPUT_DIR}/depth_moge"
moge_intrinsics = f"{OUTPUT_DIR}/intrinsics_moge_stable.json"

rgb_frame0   = f"{frames_dir}/{REFERENCE_FRAME:06d}.png"
depth_frame0 = f"{moge_depth_dir}/{REFERENCE_FRAME:06d}.png"
mask_frame0  = f"{masks_dir}/{REFERENCE_FRAME:06d}.png"

# SAM3D outputs
sam3d_dir        = f"{OUTPUT_DIR}/sam3d"
sam3d_mesh       = f"{sam3d_dir}/mesh.obj"
sam3d_transform  = f"{sam3d_dir}/transform.json"
sam3d_intrinsics = f"{sam3d_dir}/intrinsics.json"

# Scale estimation outputs
mesh_scale_path   = f"{sam3d_dir}/mesh_scale.json"
mesh_scaled_path  = f"{sam3d_dir}/mesh_scaled.obj"

# FP tracking outputs
poses_dir        = f"{OUTPUT_DIR}/poses_sam3d_moge"
poses_smooth_dir = f"{OUTPUT_DIR}/poses_sam3d_moge_smoothed"
renders_dir      = f"{OUTPUT_DIR}/renders_sam3d_moge"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_sam3d_moge_smoothed"

os.makedirs(sam3d_dir, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


# ---------------------------------------------------------------------------
# Step 1: SAM3D image → OBJ mesh
# ---------------------------------------------------------------------------
if not os.path.exists(sam3d_mesh):
    print("Step 1: SAM3D image → mesh...")
    run_image_to_mesh(
        image_path=rgb_frame0,
        mask_path=mask_frame0,
        mesh_path=sam3d_mesh,
        transform_path=sam3d_transform,
        intrinsics_path=sam3d_intrinsics,
        weights_dir=SAM3D_WEIGHTS,
        with_texture_baking=True,
        with_mesh_postprocess=True,
        use_vertex_color=False,
        dev=True,
    )
    print(f"  SAM3D mesh: {sam3d_mesh}")
else:
    print("Step 1: Skipping (SAM3D mesh cached)")

# ---------------------------------------------------------------------------
# Step 2: FP mesh scale estimation
# ---------------------------------------------------------------------------
if not os.path.exists(mesh_scale_path):
    print("Step 2: FP mesh scale estimation...")
    scale = run_estimate_mesh_scale(
        mesh_path=sam3d_mesh,
        rgb_path=rgb_frame0,
        depth_path=depth_frame0,
        mask_path=mask_frame0,
        intrinsics_path=moge_intrinsics,
        weights_dir=FP_WEIGHTS,
        scale_path=mesh_scale_path,
        rescaled_mesh_path=mesh_scaled_path,
        lo=0.5,
        hi=2.0,
        n_samples=7,
        n_levels=3,
        iou_weight=1.0,
        depth_weight=1.0,
        chamfer_weight=0.0,
        registration_iterations=5,
    )
    print(f"  Best scale factor: {scale:.4f}")
    print(f"  Rescaled mesh: {mesh_scaled_path}")
else:
    print("Step 2: Skipping (mesh scale cached)")

# ---------------------------------------------------------------------------
# Step 3: FP tracking — MoGe depth + rescaled SAM3D mesh
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 3: FP tracking (MoGe depth + rescaled SAM3D mesh)...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=moge_depth_dir,
        masks_folder=masks_dir,
        camera_intrinsics_path=moge_intrinsics,
        mesh_path=mesh_scaled_path,
        poses_dir=poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
        mask_depth=True,
    )
    print(f"  Poses: {poses_dir}")
else:
    print("Step 3: Skipping (poses cached)")

# ---------------------------------------------------------------------------
# Step 4: EKF smoothing
# ---------------------------------------------------------------------------
if not _has_files(poses_smooth_dir):
    print("Step 4: EKF smoothing...")
    run_ekf_smoothing(
        poses_dir=poses_dir,
        mesh_path=mesh_scaled_path,
        intrinsics_path=moge_intrinsics,
        weights_dir=FP_WEIGHTS,
        output_dir=poses_smooth_dir,
        masks_folder=masks_dir,
        process_noise_xy=0.01,
        process_noise_z=0.01,
        process_noise_r=0.02,
        measurement_noise_xy=0.01,
        measurement_noise_z=0.04,
        measurement_noise_r=0.02,
    )
    print(f"  Smoothed poses: {poses_smooth_dir}")
else:
    print("Step 4: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 5: Render raw poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 5: Rendering raw poses...")
    run_render_poses(
        mesh_path=mesh_scaled_path,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=moge_intrinsics,
        output_dir=renders_dir,
    )
    print(f"  Renders: {renders_dir}")
else:
    print("Step 5: Skipping (raw renders cached)")

# ---------------------------------------------------------------------------
# Step 6: Render smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_smooth_dir):
    print("Step 6: Rendering smoothed poses...")
    run_render_poses(
        mesh_path=mesh_scaled_path,
        poses_dir=poses_smooth_dir,
        frames_dir=frames_dir,
        intrinsics_path=moge_intrinsics,
        output_dir=renders_smooth_dir,
    )
    print(f"  Smoothed renders: {renders_smooth_dir}")
else:
    print("Step 6: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 7: Encode videos + stitch comparison
# ---------------------------------------------------------------------------
print("Step 7: Encoding videos...")
renders_mp4        = f"{OUTPUT_DIR}/renders_sam3d_moge.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_sam3d_moge_smoothed.mp4"
comparison_mp4     = f"{OUTPUT_DIR}/comparison_sam3d_moge.mp4"

if not os.path.exists(renders_mp4):
    frames_to_video(renders_dir, renders_mp4)
if not os.path.exists(renders_smooth_mp4):
    frames_to_video(renders_smooth_dir, renders_smooth_mp4)
if not os.path.exists(comparison_mp4):
    stitch_videos([renders_mp4, renders_smooth_mp4], comparison_mp4)

print(f"\nDone.")
print(f"  SAM3D mesh:             {sam3d_mesh}")
print(f"  Mesh scale:             {mesh_scale_path}")
print(f"  Rescaled mesh:          {mesh_scaled_path}")
print(f"  Renders (raw):          {renders_mp4}")
print(f"  Renders (smoothed):     {renders_smooth_mp4}")
print(f"  Comparison:             {comparison_mp4}")
