"""
Pipeline: airplane rayban_airplane_1 — DA3 metric depth + alignment + FP tracking

  DA3 metric depth (already computed)
  -> align_depth_to_object (frame 0 reference)
  -> align_depth_to_reference_depth (all frames)
  -> FP tracking (mask_depth=True)
  -> EKF smoothing -> render

Assumes DA3 metric depth is already computed (run airplane_rayban_airplane_1_da3metric.py first).

Run from reconstruction/:
    python experiments/airplane_rayban_airplane_1_da3metric_aligned.py
"""

import os

from v2d.common.utils import frames_to_video, stitch_videos
from v2d.foundation_pose.docker.run_align_depth_to_object import run_align_depth_to_object
from v2d.foundation_pose.docker.run_align_depth_to_reference_depth import run_align_depth_to_reference_depth
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME            = "airplane"
SESSION         = "rayban_airplane_1"
OBJECT_ID       = 1
REFERENCE_FRAME = 0

MESH_PATH  = f"data/objects/{NAME}/mesh/textured_mesh.obj"
VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/airplane.mp4"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"
FP_WEIGHTS = "data/weights/foundation_pose"

frames_dir        = f"{OUTPUT_DIR}/frames"
masks_dir         = f"{OUTPUT_DIR}/masks/{OBJECT_ID}"
da3_depth_dir     = f"{OUTPUT_DIR}/depth_da3metric"
intrinsics_stable = f"{OUTPUT_DIR}/intrinsics_da3metric_stable.json"

aligned_depth_dir   = f"{OUTPUT_DIR}/depth_da3metric_aligned"
rgb_frame0          = f"{frames_dir}/{REFERENCE_FRAME:06d}.png"
depth_frame0        = f"{da3_depth_dir}/{REFERENCE_FRAME:06d}.png"
mask_frame0         = f"{masks_dir}/{REFERENCE_FRAME:06d}.png"
depth_reference_out = f"{aligned_depth_dir}/depth_reference.png"

poses_dir          = f"{OUTPUT_DIR}/poses_da3metric_aligned"
poses_smooth_dir   = f"{OUTPUT_DIR}/poses_da3metric_aligned_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_da3metric_aligned"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_da3metric_aligned_smoothed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


# ---------------------------------------------------------------------------
# Step 1: Align reference frame depth to object mesh
# ---------------------------------------------------------------------------
if not os.path.exists(depth_reference_out):
    print("Step 1: Aligning reference frame DA3 metric depth to object mesh...")
    run_align_depth_to_object(
        mesh_path=MESH_PATH,
        rgb_path=rgb_frame0,
        depth_path=depth_frame0,
        mask_path=mask_frame0,
        intrinsics_path=intrinsics_stable,
        weights_dir=FP_WEIGHTS,
        output_depth_path=depth_reference_out,
        scale_lo=0.5,
        scale_hi=2.0,
        shift_lo=-0.5,
        shift_hi=0.5,
        n_scale_samples=7,
        n_shift_samples=5,
        n_levels=3,
        iou_weight=1.0,
        depth_weight=1.0,
        registration_iterations=5,
    )
    print(f"Reference depth saved to {depth_reference_out}")
else:
    print("Step 1: Skipping (reference depth cached)")

# ---------------------------------------------------------------------------
# Step 2: Align all frames to reference depth via ICP + affine
# ---------------------------------------------------------------------------
if not os.path.exists(f"{aligned_depth_dir}/{REFERENCE_FRAME:06d}.png"):
    print("Step 2: Aligning all DA3 metric frames to reference depth via ICP...")
    run_align_depth_to_reference_depth(
        depth_folder=da3_depth_dir,
        depth_reference_path=depth_reference_out,
        intrinsics_path=intrinsics_stable,
        output_folder=aligned_depth_dir,
        masks_folder=masks_dir,
        reference_mask_path=mask_frame0,
        n_iterations=3,
        outlier_trim_ratio=0.2,
        max_points=20000,
    )
    print(f"Aligned depth written to {aligned_depth_dir}")
else:
    print("Step 2: Skipping (aligned depth cached)")

# ---------------------------------------------------------------------------
# Step 3: FP tracking with aligned depth + depth masking
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 3: FoundationPose tracking (aligned DA3 depth, mask_depth=True)...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=aligned_depth_dir,
        masks_folder=masks_dir,
        camera_intrinsics_path=intrinsics_stable,
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
        mask_depth=True,
    )
    print(f"Poses written to {poses_dir}")
else:
    print("Step 3: Skipping (poses cached)")

# ---------------------------------------------------------------------------
# Step 4: EKF smoothing
# ---------------------------------------------------------------------------
if not _has_files(poses_smooth_dir):
    print("Step 4: EKF smoothing...")
    run_ekf_smoothing(
        poses_dir=poses_dir,
        mesh_path=MESH_PATH,
        intrinsics_path=intrinsics_stable,
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
    print("Step 4: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 5: Render raw FP poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 5: Rendering raw FP poses...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=intrinsics_stable,
        output_dir=renders_dir,
    )
    print(f"Renders written to {renders_dir}")
else:
    print("Step 5: Skipping (renders cached)")

# ---------------------------------------------------------------------------
# Step 6: Render smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_smooth_dir):
    print("Step 6: Rendering smoothed poses...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_smooth_dir,
        frames_dir=frames_dir,
        intrinsics_path=intrinsics_stable,
        output_dir=renders_smooth_dir,
    )
    print(f"Smoothed renders written to {renders_smooth_dir}")
else:
    print("Step 6: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 7: Encode + stitch comparison video
# ---------------------------------------------------------------------------
print("Step 7: Encoding videos...")
aligned_depth_mp4  = f"{OUTPUT_DIR}/depth_da3metric_aligned.mp4"
renders_mp4        = f"{OUTPUT_DIR}/renders_da3metric_aligned.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_da3metric_aligned_smoothed.mp4"
comparison_mp4     = f"{OUTPUT_DIR}/comparison_da3metric_aligned.mp4"

if not os.path.exists(aligned_depth_mp4):
    frames_to_video(aligned_depth_dir, aligned_depth_mp4)
if not os.path.exists(renders_mp4):
    frames_to_video(renders_dir, renders_mp4)
if not os.path.exists(renders_smooth_mp4):
    frames_to_video(renders_smooth_dir, renders_smooth_mp4)

if not os.path.exists(comparison_mp4):
    print("Step 7: Creating comparison video...")
    stitch_videos([renders_mp4, renders_smooth_mp4, aligned_depth_mp4], comparison_mp4)

print(f"\nDone.")
print(f"  Aligned depth video         : {aligned_depth_mp4}")
print(f"  FP renders video            : {renders_mp4}")
print(f"  FP smoothed renders video   : {renders_smooth_mp4}")
print(f"  Comparison video            : {comparison_mp4}")
