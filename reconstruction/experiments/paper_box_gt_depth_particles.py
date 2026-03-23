"""
Pipeline: paper_box with ground-truth depth + 64 particles

Prerequisite: run paper_box_pipeline.py first (needs frames/ and masks/ to exist).

Same as paper_box_gt_depth_pipeline.py but runs FoundationPose with n_particles=64.
Outputs are saved with a _p64 suffix to sit alongside other runs.

Run from reconstruction/ after paper_box_pipeline.py:
    python experiments/paper_box_gt_depth_particles.py
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

GT_DEPTH_DIR  = "data/paper_box/converted/depth"
GT_INTRINSICS = "data/paper_box/converted/intrinsics.json"

FP_WEIGHTS = "data/weights/foundation_pose"

frames_dir = f"{OUTPUT_DIR}/frames"
masks_dir  = f"{OUTPUT_DIR}/masks"

poses_dir          = f"{OUTPUT_DIR}/poses_gt_p64"
poses_smoothed_dir = f"{OUTPUT_DIR}/poses_gt_p64_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_gt_p64"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_gt_p64_smoothed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


if not _has_files(frames_dir):
    raise RuntimeError(f"frames/ not found at {frames_dir} — run paper_box_pipeline.py first")
if not _has_files(f"{masks_dir}/{OBJECT_ID}"):
    raise RuntimeError(f"masks/{OBJECT_ID}/ not found — run paper_box_pipeline.py first")

# ---------------------------------------------------------------------------
# Step 6: FoundationPose tracking (GT depth, 64 particles)
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 6: FoundationPose tracking (GT depth, 64 particles)...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=GT_DEPTH_DIR,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        camera_intrinsics_path=GT_INTRINSICS,
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
        n_particles=64,
    )
else:
    print("Step 6: Skipping (poses_gt_p64 already computed)")

# ---------------------------------------------------------------------------
# Step 7: EKF smoothing
# ---------------------------------------------------------------------------
if not _has_files(poses_smoothed_dir):
    print("Step 7: EKF smoothing...")
    run_ekf_smoothing(
        poses_dir=poses_dir,
        mesh_path=MESH_PATH,
        intrinsics_path=GT_INTRINSICS,
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
        intrinsics_path=GT_INTRINSICS,
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
        intrinsics_path=GT_INTRINSICS,
        output_dir=renders_smooth_dir,
    )
else:
    print("Step 9: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 10: Encode + stitch comparison video
# ---------------------------------------------------------------------------
print("Step 10: Encoding videos...")
renders_mp4        = f"{OUTPUT_DIR}/renders_gt_p64.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_gt_p64_smoothed.mp4"
gt_depth_mp4       = f"{OUTPUT_DIR}/depth_gt.mp4"

frames_to_video(renders_dir,        renders_mp4)
frames_to_video(renders_smooth_dir, renders_smooth_mp4)
if not os.path.exists(gt_depth_mp4):
    frames_to_video(GT_DEPTH_DIR, gt_depth_mp4)

print("Step 11: Stitching comparison video...")
stitch_videos(
    [renders_mp4, renders_smooth_mp4, gt_depth_mp4],
    f"{OUTPUT_DIR}/comparison_gt_p64.mp4",
)

print(f"Done. Output: {OUTPUT_DIR}/comparison_gt_p64.mp4")
