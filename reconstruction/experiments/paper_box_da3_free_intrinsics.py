"""
Pipeline: paper_box with Depth Anything 3 depth (no input intrinsics) + GT intrinsics for FP

DA3 runs unconstrained (estimates its own intrinsics freely). GT intrinsics are passed
to FoundationPose and rendering so the intrinsics are not a confound.

Prerequisite: run paper_box_pipeline.py first (needs frames/ and masks/ to exist).

Run from reconstruction/:
    python experiments/paper_box_da3_free_intrinsics.py
"""

import os

from v2d.pipelines.frames_to_video import frames_to_video
from v2d.pipelines.stitch_videos import stitch_videos
from v2d.depth_anything.docker.run_video_to_depth import run_video_to_depth
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

# GT intrinsics used for FP + rendering (not passed to DA3)
GT_INTRINSICS = "data/paper_box/converted/intrinsics.json"

DA3_WEIGHTS = "data/weights/depth_anything"
FP_WEIGHTS  = "data/weights/foundation_pose"

frames_dir = f"{OUTPUT_DIR}/frames"
masks_dir  = f"{OUTPUT_DIR}/masks"

# DA3 outputs (free intrinsics — not used downstream)
depth_dir      = f"{OUTPUT_DIR}/depth_da3_free"
intrinsics_dir = f"{OUTPUT_DIR}/intrinsics_da3_free"

# FP outputs
poses_dir          = f"{OUTPUT_DIR}/poses_da3_free"
poses_smoothed_dir = f"{OUTPUT_DIR}/poses_da3_free_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_da3_free"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_da3_free_smoothed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


if not _has_files(frames_dir):
    raise RuntimeError(f"frames/ not found at {frames_dir} — run paper_box_pipeline.py first")
if not _has_files(f"{masks_dir}/{OBJECT_ID}"):
    raise RuntimeError(f"masks/{OBJECT_ID}/ not found — run paper_box_pipeline.py first")

# ---------------------------------------------------------------------------
# Step 4: DA3 depth — no input intrinsics
# ---------------------------------------------------------------------------
if not _has_files(depth_dir):
    print("Step 4: Depth Anything 3 depth estimation (free intrinsics)...")
    run_video_to_depth(
        video_path=VIDEO_PATH,
        depth_folder=depth_dir,
        intrinsics_folder=intrinsics_dir,
        weights_path=DA3_WEIGHTS,
        process_res=256,  # lower res to fit all 728 frames in one call (avoids scale drift from chunking)
    )
else:
    print("Step 4: Skipping (depth_da3_free already computed)")

# ---------------------------------------------------------------------------
# Step 5: FoundationPose — DA3 depth, GT intrinsics
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 5: FoundationPose tracking (DA3 depth, GT intrinsics)...")
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
    print("Step 5: Skipping (poses_da3_free already computed)")

# ---------------------------------------------------------------------------
# Step 6: EKF smoothing
# ---------------------------------------------------------------------------
if not _has_files(poses_smoothed_dir):
    print("Step 6: EKF smoothing...")
    run_ekf_smoothing(
        poses_dir=poses_dir,
        mesh_path=MESH_PATH,
        intrinsics_path=GT_INTRINSICS,
        weights_dir=FP_WEIGHTS,
        output_dir=poses_smoothed_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
    )
else:
    print("Step 6: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 7: Render raw poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 7: Rendering raw pose overlays...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=GT_INTRINSICS,
        output_dir=renders_dir,
    )
else:
    print("Step 7: Skipping (raw renders cached)")

# ---------------------------------------------------------------------------
# Step 8: Render smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_smooth_dir):
    print("Step 8: Rendering smoothed pose overlays...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_smoothed_dir,
        frames_dir=frames_dir,
        intrinsics_path=GT_INTRINSICS,
        output_dir=renders_smooth_dir,
    )
else:
    print("Step 8: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 9: Encode + stitch comparison video
# ---------------------------------------------------------------------------
print("Step 9: Encoding videos...")
renders_mp4        = f"{OUTPUT_DIR}/renders_da3_free.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_da3_free_smoothed.mp4"
depth_mp4          = f"{OUTPUT_DIR}/depth_da3_free.mp4"

frames_to_video(renders_dir,        renders_mp4)
frames_to_video(renders_smooth_dir, renders_smooth_mp4)
frames_to_video(depth_dir,          depth_mp4)

print("Step 10: Stitching comparison video...")
stitch_videos(
    [renders_mp4, renders_smooth_mp4, depth_mp4],
    f"{OUTPUT_DIR}/comparison_da3_free.mp4",
)

print(f"Done. Output: {OUTPUT_DIR}/comparison_da3_free.mp4")
