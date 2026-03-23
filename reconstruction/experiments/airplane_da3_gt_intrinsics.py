"""
Pipeline: airplane Session_20260310_132206 with Depth Anything 3 (GT intrinsics) + FP

GT intrinsics (intrinsics_vipe.json) are passed to DA3 as conditioning, and to
FoundationPose and rendering.

Prerequisite: frames/ and masks/ must already exist in the session outputs dir.

Run from reconstruction/:
    python experiments/airplane_da3_gt_intrinsics.py
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

depth_dir      = f"{OUTPUT_DIR}/depth_da3_gt"
intrinsics_dir = f"{OUTPUT_DIR}/intrinsics_da3_gt"

poses_dir          = f"{OUTPUT_DIR}/poses_da3_gt"
poses_smoothed_dir = f"{OUTPUT_DIR}/poses_da3_gt_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_da3_gt"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_da3_gt_smoothed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


if not _has_files(frames_dir):
    raise RuntimeError(f"frames/ not found at {frames_dir}")
if not _has_files(f"{masks_dir}/{OBJECT_ID}"):
    raise RuntimeError(f"masks/{OBJECT_ID}/ not found at {masks_dir}/{OBJECT_ID}")

# ---------------------------------------------------------------------------
# Step 1: DA3 depth — GT intrinsics provided as conditioning
# ---------------------------------------------------------------------------
if not _has_files(depth_dir):
    print("Step 1: Depth Anything 3 depth estimation (GT intrinsics provided)...")
    run_video_to_depth(
        video_path=VIDEO_PATH,
        depth_folder=depth_dir,
        intrinsics_folder=intrinsics_dir,
        weights_path=DA3_WEIGHTS,
        input_intrinsics_path=GT_INTRINSICS,
        process_res=256,
    )
else:
    print("Step 1: Skipping (depth_da3_gt already computed)")

# ---------------------------------------------------------------------------
# Step 2: FoundationPose
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 2: FoundationPose tracking...")
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
    print("Step 2: Skipping (poses_da3_gt already computed)")

# ---------------------------------------------------------------------------
# Step 3: EKF smoothing — tuned to follow measurements closely (less smoothing)
# ---------------------------------------------------------------------------
if not _has_files(poses_smoothed_dir):
    print("Step 3: EKF smoothing...")
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
    print("Step 3: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 4: Render raw poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 4: Rendering raw pose overlays...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=GT_INTRINSICS,
        output_dir=renders_dir,
    )
else:
    print("Step 4: Skipping (raw renders cached)")

# ---------------------------------------------------------------------------
# Step 5: Render smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_smooth_dir):
    print("Step 5: Rendering smoothed pose overlays...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_smoothed_dir,
        frames_dir=frames_dir,
        intrinsics_path=GT_INTRINSICS,
        output_dir=renders_smooth_dir,
    )
else:
    print("Step 5: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 6: Encode + stitch comparison video
# ---------------------------------------------------------------------------
print("Step 6: Encoding videos...")
renders_mp4        = f"{OUTPUT_DIR}/renders_da3_gt.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_da3_gt_smoothed.mp4"
depth_mp4          = f"{OUTPUT_DIR}/depth_da3_gt.mp4"

frames_to_video(renders_dir,        renders_mp4)
frames_to_video(renders_smooth_dir, renders_smooth_mp4)
frames_to_video(depth_dir,          depth_mp4)

print("Step 7: Stitching comparison video...")
stitch_videos(
    [renders_mp4, renders_smooth_mp4, depth_mp4],
    f"{OUTPUT_DIR}/comparison_da3_gt.mp4",
)

print(f"Done. Output: {OUTPUT_DIR}/comparison_da3_gt.mp4")
