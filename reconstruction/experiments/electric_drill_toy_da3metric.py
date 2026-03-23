"""
Pipeline: electric_drill_toy Session_20260310_133326_f50
  DA3 metric depth (focal post-processing) -> FoundationPose -> EKF smoothing -> render

Run from reconstruction/:
    python experiments/electric_drill_toy_da3metric.py
"""

import os

from v2d.common.utils import frames_to_video, stitch_videos
from v2d.depth_anything.docker.run_video_to_depth import run_video_to_depth
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_render_poses import run_render_poses

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME            = "electric_drill_toy"
SESSION         = "Session_20260310_133326_f50"
OBJECT_ID       = 1
REFERENCE_FRAME = 0

MESH_PATH  = f"data/objects/{NAME}/mesh/textured_mesh.obj"
VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/{SESSION}_color.mp4"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"

DA3METRIC_WEIGHTS = "data/weights/depth_anything_metric"
FP_WEIGHTS        = "data/weights/foundation_pose"

frames_dir        = f"{OUTPUT_DIR}/frames"
masks_dir         = f"{OUTPUT_DIR}/masks"
depth_dir         = f"{OUTPUT_DIR}/depth_da3metric"
intrinsics_dir    = f"{OUTPUT_DIR}/intrinsics_da3metric"
intrinsics_stable = f"{OUTPUT_DIR}/intrinsics_da3metric_stable.json"
moge_stable       = f"{OUTPUT_DIR}/intrinsics_moge_stable.json"

poses_dir          = f"{OUTPUT_DIR}/poses_da3metric"
poses_smoothed_dir = f"{OUTPUT_DIR}/poses_da3metric_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_da3metric"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_da3metric_smoothed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


# ---------------------------------------------------------------------------
# Step 1: DA3 metric depth estimation
# ---------------------------------------------------------------------------
if not _has_files(depth_dir):
    print("Step 1: DA3 metric depth estimation...")
    run_video_to_depth(
        video_path=VIDEO_PATH,
        depth_folder=depth_dir,
        intrinsics_folder=intrinsics_dir,
        weights_path=DA3METRIC_WEIGHTS,
        model="metric",
        input_intrinsics_path=moge_stable,
        dev=True,
    )
else:
    print("Step 1: Skipping (depth_da3metric already computed)")

# ---------------------------------------------------------------------------
# Step 2: Stabilise intrinsics
# ---------------------------------------------------------------------------
if not os.path.exists(intrinsics_stable):
    print("Step 2: Stabilising intrinsics...")
    stabilize_intrinsics(intrinsics_dir, intrinsics_stable)
else:
    print("Step 2: Skipping (stable intrinsics cached)")

# ---------------------------------------------------------------------------
# Step 3: FoundationPose tracking
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 3: FoundationPose tracking...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=depth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        camera_intrinsics_path=intrinsics_stable,
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
    )
else:
    print("Step 3: Skipping (poses already computed)")

# ---------------------------------------------------------------------------
# Step 4: EKF smoothing
# ---------------------------------------------------------------------------
if not _has_files(poses_smoothed_dir):
    print("Step 4: EKF smoothing...")
    run_ekf_smoothing(
        poses_dir=poses_dir,
        mesh_path=MESH_PATH,
        intrinsics_path=intrinsics_stable,
        weights_dir=FP_WEIGHTS,
        output_dir=poses_smoothed_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        process_noise_xy=0.01,
        process_noise_z=0.01,
        process_noise_r=0.02,
        measurement_noise_xy=0.01,
        measurement_noise_z=0.04,
        measurement_noise_r=0.02,
    )
else:
    print("Step 4: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 5: Render raw poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 5: Rendering raw pose overlays...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=intrinsics_stable,
        output_dir=renders_dir,
    )
else:
    print("Step 5: Skipping (raw renders cached)")

# ---------------------------------------------------------------------------
# Step 6: Render smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_smooth_dir):
    print("Step 6: Rendering smoothed pose overlays...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_smoothed_dir,
        frames_dir=frames_dir,
        intrinsics_path=intrinsics_stable,
        output_dir=renders_smooth_dir,
    )
else:
    print("Step 6: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 7: Encode + stitch comparison video
# ---------------------------------------------------------------------------
print("Step 7: Encoding videos...")
renders_mp4        = f"{OUTPUT_DIR}/renders_da3metric.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_da3metric_smoothed.mp4"
depth_mp4          = f"{OUTPUT_DIR}/depth_da3metric.mp4"

frames_to_video(renders_dir,        renders_mp4)
frames_to_video(renders_smooth_dir, renders_smooth_mp4)
frames_to_video(depth_dir,          depth_mp4)

print("Step 8: Stitching comparison video...")
stitch_videos(
    [renders_mp4, renders_smooth_mp4, depth_mp4],
    f"{OUTPUT_DIR}/comparison_da3metric.mp4",
)

print(f"Done. Output: {OUTPUT_DIR}/comparison_da3metric.mp4")
