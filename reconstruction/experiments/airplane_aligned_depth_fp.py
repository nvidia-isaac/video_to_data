"""
Pipeline: airplane Session_20260310_130642_f50 — aligned depth FP tracking

Uses depth_moge_aligned (affine-corrected MoGe depth from depth_alignment experiment) to:
  1. Render aligned depth as a video
  2. Run FoundationPose tracking with aligned depth
  3. EKF smoothing of aligned-depth poses
  4. Render raw + smoothed FP poses
  5. Comparison video: original MoGe smoothed (left) vs aligned MoGe smoothed (right)

Run from reconstruction/:
    python experiments/airplane_aligned_depth_fp.py
"""

import os

from v2d.common.utils import frames_to_video
from v2d.common.utils import stitch_videos
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_render_poses import run_render_poses

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME      = "airplane"
SESSION   = "Session_20260310_130642_f50"
OBJECT_ID = 1

MESH_PATH  = f"data/objects/{NAME}/mesh/textured_mesh.obj"
VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/{SESSION}_color.mp4"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"
FP_WEIGHTS = "data/weights/foundation_pose"

frames_dir        = f"{OUTPUT_DIR}/frames"
masks_dir         = f"{OUTPUT_DIR}/masks/{OBJECT_ID}"
moge_intrinsics   = f"{OUTPUT_DIR}/intrinsics_moge_stable.json"
aligned_depth_dir = f"{OUTPUT_DIR}/depth_moge_aligned"

poses_dir          = f"{OUTPUT_DIR}/poses_moge_aligned"
poses_smooth_dir   = f"{OUTPUT_DIR}/poses_moge_aligned_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_moge_aligned"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_moge_aligned_smoothed"

# Reference for comparison (already computed by airplane_f50.py)
moge_smooth_renders_dir = f"{OUTPUT_DIR}/renders_moge_smoothed"

aligned_depth_mp4  = f"{OUTPUT_DIR}/depth_moge_aligned.mp4"
renders_mp4        = f"{OUTPUT_DIR}/renders_moge_aligned.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_moge_aligned_smoothed.mp4"
comparison_mp4     = f"{OUTPUT_DIR}/comparison_moge_vs_aligned.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


# ---------------------------------------------------------------------------
# Step 1: Render aligned depth as video
# ---------------------------------------------------------------------------
if not os.path.exists(aligned_depth_mp4):
    print("Step 1: Rendering aligned depth as video...")
    frames_to_video(aligned_depth_dir, aligned_depth_mp4)
    print(f"Aligned depth video: {aligned_depth_mp4}")
else:
    print("Step 1: Skipping (aligned depth video cached)")

# ---------------------------------------------------------------------------
# Step 2: FP tracking with aligned depth
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 2: FoundationPose tracking (aligned depth)...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=aligned_depth_dir,
        masks_folder=masks_dir,
        camera_intrinsics_path=moge_intrinsics,
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=0,
    )
    print(f"Poses written to {poses_dir}")
else:
    print("Step 2: Skipping (poses cached)")

# ---------------------------------------------------------------------------
# Step 3: Render raw FP poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 3: Rendering FP poses...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=moge_intrinsics,
        output_dir=renders_dir,
    )
    print(f"Renders written to {renders_dir}")
else:
    print("Step 3: Skipping (renders cached)")

# ---------------------------------------------------------------------------
# Step 4: Encode raw renders video
# ---------------------------------------------------------------------------
if not os.path.exists(renders_mp4):
    print("Step 4: Encoding renders as video...")
    frames_to_video(renders_dir, renders_mp4)
    print(f"Renders video: {renders_mp4}")
else:
    print("Step 4: Skipping (renders video cached)")

# ---------------------------------------------------------------------------
# Step 5: EKF smoothing — aligned depth
# ---------------------------------------------------------------------------
if not _has_files(poses_smooth_dir):
    print("Step 5: EKF smoothing (aligned depth)...")
    run_ekf_smoothing(
        poses_dir=poses_dir,
        mesh_path=MESH_PATH,
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
    print(f"Smoothed poses written to {poses_smooth_dir}")
else:
    print("Step 5: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 6: Render smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_smooth_dir):
    print("Step 6: Rendering smoothed poses...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_smooth_dir,
        frames_dir=frames_dir,
        intrinsics_path=moge_intrinsics,
        output_dir=renders_smooth_dir,
    )
    print(f"Smoothed renders written to {renders_smooth_dir}")
else:
    print("Step 6: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 7: Encode smoothed renders + comparison
# ---------------------------------------------------------------------------
if not os.path.exists(renders_smooth_mp4):
    print("Step 7: Encoding smoothed renders video...")
    frames_to_video(renders_smooth_dir, renders_smooth_mp4)

if not os.path.exists(comparison_mp4):
    print("Step 7: Creating comparison video (moge_smoothed vs aligned_smoothed)...")
    moge_smooth_mp4 = f"{OUTPUT_DIR}/renders_moge_smoothed.mp4"
    if not os.path.exists(moge_smooth_mp4):
        frames_to_video(moge_smooth_renders_dir, moge_smooth_mp4)
    stitch_videos([moge_smooth_mp4, renders_smooth_mp4], comparison_mp4)
    print(f"Comparison video: {comparison_mp4}")
else:
    print("Step 7: Skipping (comparison video cached)")

print(f"\nDone.")
print(f"  Aligned depth video         : {aligned_depth_mp4}")
print(f"  FP renders video            : {renders_mp4}")
print(f"  FP smoothed renders video   : {renders_smooth_mp4}")
print(f"  Comparison video            : {comparison_mp4}")
