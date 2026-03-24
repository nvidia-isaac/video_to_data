"""
Pipeline: airplane rayban_airplane_1 — DA3 metric depth + FP tracking

  DA3 metric depth (MoGe stable intrinsics as input)
  -> FP tracking -> EKF smoothing -> render

Assumes frames and masks are already computed (run airplane_rayban_airplane_1.py first).

Run from reconstruction/:
    python experiments/airplane_rayban_airplane_1_da3metric.py
"""

import os

from v2d.common.utils import frames_to_video, stitch_videos
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.depth_anything.docker.run_video_to_depth import run_video_to_depth as run_da3_depth
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

DA3METRIC_WEIGHTS = "data/weights/depth_anything_metric"
FP_WEIGHTS        = "data/weights/foundation_pose"

frames_dir        = f"{OUTPUT_DIR}/frames"
masks_dir         = f"{OUTPUT_DIR}/masks/{OBJECT_ID}"
moge_intrinsics   = f"{OUTPUT_DIR}/intrinsics_moge_stable.json"

depth_dir         = f"{OUTPUT_DIR}/depth_da3metric"
intrinsics_dir    = f"{OUTPUT_DIR}/intrinsics_da3metric"
intrinsics_stable = f"{OUTPUT_DIR}/intrinsics_da3metric_stable.json"

poses_dir          = f"{OUTPUT_DIR}/poses_da3metric"
poses_smooth_dir   = f"{OUTPUT_DIR}/poses_da3metric_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_da3metric"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_da3metric_smoothed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


# ---------------------------------------------------------------------------
# Step 1: DA3 metric depth
# ---------------------------------------------------------------------------
if not _has_files(depth_dir):
    print("Step 1: DA3 metric depth estimation...")
    run_da3_depth(
        video_path=VIDEO_PATH,
        depth_folder=depth_dir,
        intrinsics_folder=intrinsics_dir,
        weights_path=DA3METRIC_WEIGHTS,
        model="metric",
        input_intrinsics_path=moge_intrinsics,
        dev=True,
    )
else:
    print("Step 1: Skipping (DA3 metric depth cached)")

if not os.path.exists(intrinsics_stable):
    print("Step 1b: Stabilising DA3 metric intrinsics...")
    stabilize_intrinsics(intrinsics_dir, intrinsics_stable)
else:
    print("Step 1b: Skipping (stable intrinsics cached)")

# ---------------------------------------------------------------------------
# Step 2: FP tracking
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 2: FoundationPose tracking (DA3 metric depth)...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=depth_dir,
        masks_folder=masks_dir,
        camera_intrinsics_path=intrinsics_stable,
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
    )
    print(f"Poses written to {poses_dir}")
else:
    print("Step 2: Skipping (poses cached)")

# ---------------------------------------------------------------------------
# Step 3: EKF smoothing
# ---------------------------------------------------------------------------
if not _has_files(poses_smooth_dir):
    print("Step 3: EKF smoothing...")
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
    print("Step 3: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 4: Render raw FP poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 4: Rendering raw FP poses...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=intrinsics_stable,
        output_dir=renders_dir,
    )
    print(f"Renders written to {renders_dir}")
else:
    print("Step 4: Skipping (renders cached)")

# ---------------------------------------------------------------------------
# Step 5: Render smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_smooth_dir):
    print("Step 5: Rendering smoothed poses...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_smooth_dir,
        frames_dir=frames_dir,
        intrinsics_path=intrinsics_stable,
        output_dir=renders_smooth_dir,
    )
    print(f"Smoothed renders written to {renders_smooth_dir}")
else:
    print("Step 5: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 6: Encode + stitch comparison video
# ---------------------------------------------------------------------------
print("Step 6: Encoding videos...")
depth_mp4          = f"{OUTPUT_DIR}/depth_da3metric.mp4"
renders_mp4        = f"{OUTPUT_DIR}/renders_da3metric.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_da3metric_smoothed.mp4"
comparison_mp4     = f"{OUTPUT_DIR}/comparison_da3metric.mp4"

if not os.path.exists(depth_mp4):
    frames_to_video(depth_dir, depth_mp4)
if not os.path.exists(renders_mp4):
    frames_to_video(renders_dir, renders_mp4)
if not os.path.exists(renders_smooth_mp4):
    frames_to_video(renders_smooth_dir, renders_smooth_mp4)

if not os.path.exists(comparison_mp4):
    print("Step 6: Creating comparison video...")
    stitch_videos([renders_mp4, renders_smooth_mp4, depth_mp4], comparison_mp4)

print(f"\nDone.")
print(f"  DA3 metric depth video      : {depth_mp4}")
print(f"  FP renders video            : {renders_mp4}")
print(f"  FP smoothed renders video   : {renders_smooth_mp4}")
print(f"  Comparison video            : {comparison_mp4}")
