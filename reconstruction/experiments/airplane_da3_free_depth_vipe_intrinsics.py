"""
Pipeline: airplane Session_20260310_132206 with Depth Anything 3 (free intrinsics)
          but GT (vipe) intrinsics for FP tracking and rendering.

Reuses depth_da3_free/ computed by airplane_da3_free_intrinsics.py.
GT intrinsics (intrinsics_vipe.json) are used for FoundationPose and rendering.

Prerequisite: run airplane_da3_free_intrinsics.py first (needs depth_da3_free/ to exist).

Run from reconstruction/:
    python experiments/airplane_da3_free_depth_vipe_intrinsics.py
"""

import os

from v2d.common.utils import frames_to_video
from v2d.common.utils import stitch_videos
from v2d.depth_anything.docker.run_video_to_depth import run_video_to_depth
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_render_poses import run_render_poses

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME      = "airplane"
SESSION   = "Session_20260310_132206"
OBJECT_ID = 1
REFERENCE_FRAME = 0

MESH_PATH  = f"data/objects/{NAME}/mesh/textured_mesh.obj"
VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/{SESSION}_color.mp4"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"

GT_INTRINSICS = f"{OUTPUT_DIR}/intrinsics_vipe.json"

DA3_WEIGHTS = "data/weights/depth_anything"
FP_WEIGHTS  = "data/weights/foundation_pose"

frames_dir = f"{OUTPUT_DIR}/frames"
masks_dir  = f"{OUTPUT_DIR}/masks"

# Reuse depth from the free-intrinsics run
depth_dir      = f"{OUTPUT_DIR}/depth_da3_free"
intrinsics_dir = f"{OUTPUT_DIR}/intrinsics_da3_free"

poses_dir          = f"{OUTPUT_DIR}/poses_da3_free_vipe"
poses_smoothed_dir = f"{OUTPUT_DIR}/poses_da3_free_vipe_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_da3_free_vipe"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_da3_free_vipe_smoothed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


if not _has_files(frames_dir):
    raise RuntimeError(f"frames/ not found at {frames_dir}")
if not _has_files(f"{masks_dir}/{OBJECT_ID}"):
    raise RuntimeError(f"masks/{OBJECT_ID}/ not found at {masks_dir}/{OBJECT_ID}")
if not _has_files(depth_dir):
    raise RuntimeError(f"depth_da3_free/ not found — run airplane_da3_free_intrinsics.py first")

# ---------------------------------------------------------------------------
# Step 1: FoundationPose — DA3 free depth, GT (vipe) intrinsics
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 1: FoundationPose tracking (DA3 free depth, vipe intrinsics)...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=depth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        camera_intrinsics_path=GT_INTRINSICS,
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
    )
else:
    print("Step 1: Skipping (poses_da3_free_vipe already computed)")

# ---------------------------------------------------------------------------
# Step 2: EKF smoothing
# ---------------------------------------------------------------------------
if not _has_files(poses_smoothed_dir):
    print("Step 2: EKF smoothing...")
    run_ekf_smoothing(
        poses_dir=poses_dir,
        mesh_path=MESH_PATH,
        intrinsics_path=GT_INTRINSICS,
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
    print("Step 2: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 3: Render raw poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 3: Rendering raw pose overlays...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=GT_INTRINSICS,
        output_dir=renders_dir,
    )
else:
    print("Step 3: Skipping (raw renders cached)")

# ---------------------------------------------------------------------------
# Step 4: Render smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_smooth_dir):
    print("Step 4: Rendering smoothed pose overlays...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_smoothed_dir,
        frames_dir=frames_dir,
        intrinsics_path=GT_INTRINSICS,
        output_dir=renders_smooth_dir,
    )
else:
    print("Step 4: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 5: Encode + stitch comparison video
# ---------------------------------------------------------------------------
print("Step 5: Encoding videos...")
renders_mp4        = f"{OUTPUT_DIR}/renders_da3_free_vipe.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_da3_free_vipe_smoothed.mp4"
depth_mp4          = f"{OUTPUT_DIR}/depth_da3_free.mp4"

frames_to_video(renders_dir,        renders_mp4)
frames_to_video(renders_smooth_dir, renders_smooth_mp4)
frames_to_video(depth_dir,          depth_mp4)

print("Step 6: Stitching comparison video...")
stitch_videos(
    [renders_mp4, renders_smooth_mp4, depth_mp4],
    f"{OUTPUT_DIR}/comparison_da3_free_vipe.mp4",
)

print(f"Done. Output: {OUTPUT_DIR}/comparison_da3_free_vipe.mp4")
