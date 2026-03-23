"""
Pipeline: paper_box with MoGe depth + MoGe stable intrinsics for FP

Tests the fully MoGe-driven case: depth and intrinsics both come from MoGe
(no GT intrinsics involved anywhere).

Prerequisite: run paper_box_moge_free_intrinsics.py first
(needs depth_moge_free/ and intrinsics_moge_free_stable.json to exist).

Run from reconstruction/:
    python experiments/paper_box_moge_own_intrinsics.py
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
NAME            = "paper_box"
SESSION         = "paper_box_demo"
OBJECT_ID       = 1
REFERENCE_FRAME = 0

MESH_PATH  = f"data/objects/{NAME}/mesh/textured_simple.obj"
VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/{SESSION}_color.mp4"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"

FP_WEIGHTS = "data/weights/foundation_pose"

frames_dir = f"{OUTPUT_DIR}/frames"
masks_dir  = f"{OUTPUT_DIR}/masks"

# Reuse MoGe free-intrinsics depth + its own stable intrinsics
depth_dir         = f"{OUTPUT_DIR}/depth_moge_free"
intrinsics_stable = f"{OUTPUT_DIR}/intrinsics_moge_free_stable.json"

poses_dir          = f"{OUTPUT_DIR}/poses_moge_own"
poses_smoothed_dir = f"{OUTPUT_DIR}/poses_moge_own_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_moge_own"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_moge_own_smoothed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


if not _has_files(depth_dir):
    raise RuntimeError(f"depth_moge_free/ not found — run paper_box_moge_free_intrinsics.py first")
if not os.path.exists(intrinsics_stable):
    raise RuntimeError(f"intrinsics_moge_free_stable.json not found — run paper_box_moge_free_intrinsics.py first")

# ---------------------------------------------------------------------------
# Step 6: FoundationPose — MoGe depth + MoGe stable intrinsics
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 6: FoundationPose tracking (MoGe depth + MoGe intrinsics)...")
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
    print("Step 6: Skipping (poses_moge_own already computed)")

# ---------------------------------------------------------------------------
# Step 7: EKF smoothing
# ---------------------------------------------------------------------------
if not _has_files(poses_smoothed_dir):
    print("Step 7: EKF smoothing...")
    run_ekf_smoothing(
        poses_dir=poses_dir,
        mesh_path=MESH_PATH,
        intrinsics_path=intrinsics_stable,
        weights_dir=FP_WEIGHTS,
        output_dir=poses_smoothed_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
    )
else:
    print("Step 7: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 8: Render raw poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 8: Rendering raw pose overlays...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=intrinsics_stable,
        output_dir=renders_dir,
    )
else:
    print("Step 8: Skipping (raw renders cached)")

# ---------------------------------------------------------------------------
# Step 9: Render smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_smooth_dir):
    print("Step 9: Rendering smoothed pose overlays...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_smoothed_dir,
        frames_dir=frames_dir,
        intrinsics_path=intrinsics_stable,
        output_dir=renders_smooth_dir,
    )
else:
    print("Step 9: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 10: Encode + stitch comparison video
# ---------------------------------------------------------------------------
print("Step 10: Encoding videos...")
renders_mp4        = f"{OUTPUT_DIR}/renders_moge_own.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_moge_own_smoothed.mp4"
depth_mp4          = f"{OUTPUT_DIR}/depth_moge_free.mp4"

frames_to_video(renders_dir,        renders_mp4)
frames_to_video(renders_smooth_dir, renders_smooth_mp4)
if not os.path.exists(depth_mp4):
    from v2d.common.utils import frames_to_video as ftv
    ftv(depth_dir, depth_mp4)

print("Step 11: Stitching comparison video...")
stitch_videos(
    [renders_mp4, renders_smooth_mp4, depth_mp4],
    f"{OUTPUT_DIR}/comparison_moge_own.mp4",
)

print(f"Done. Output: {OUTPUT_DIR}/comparison_moge_own.mp4")
