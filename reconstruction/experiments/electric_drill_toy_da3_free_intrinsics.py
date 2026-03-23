"""
Pipeline: electric_drill_toy Session_20260310_133326
  GDino -> SAM2 -> DA3 (free intrinsics) -> stabilize intrinsics
  -> FoundationPose -> EKF smoothing -> render raw + smoothed poses

Run from reconstruction/:
    python experiments/electric_drill_toy_da3_free_intrinsics.py
"""

import json
import os

from v2d.common.utils import extract_images
from v2d.common.utils import frames_to_video
from v2d.common.utils import stitch_videos
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.depth_anything.docker.run_video_to_depth import run_video_to_depth
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_render_poses import run_render_poses

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME             = "electric_drill_toy"
SESSION          = "Session_20260310_133326"
OBJECT_ID        = 1
REFERENCE_FRAME  = 0
DETECTION_PROMPT = "electric hand drill toy"

MESH_PATH  = f"data/objects/{NAME}/mesh/textured_mesh.obj"
VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/{SESSION}_color.mp4"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"

DA3_WEIGHTS  = "data/weights/depth_anything"
FP_WEIGHTS   = "data/weights/foundation_pose"
SAM2_WEIGHTS = "data/weights/sam2"
DINO_WEIGHTS = "data/weights/grounding_dino"

frames_dir        = f"{OUTPUT_DIR}/frames"
masks_dir         = f"{OUTPUT_DIR}/masks"
depth_dir         = f"{OUTPUT_DIR}/depth_da3_free"
intrinsics_dir    = f"{OUTPUT_DIR}/intrinsics_da3_free"
intrinsics_stable = f"{OUTPUT_DIR}/intrinsics_da3_free_stable.json"
dino_detections   = f"{OUTPUT_DIR}/dino_detections.json"
sam2_prompts_path = f"{OUTPUT_DIR}/sam2_prompts.json"

poses_dir          = f"{OUTPUT_DIR}/poses_da3_free"
poses_smoothed_dir = f"{OUTPUT_DIR}/poses_da3_free_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_da3_free"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_da3_free_smoothed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


# ---------------------------------------------------------------------------
# Step 1: Extract frames
# ---------------------------------------------------------------------------
if not _has_files(frames_dir):
    print("Step 1: Extracting frames...")
    extract_images(VIDEO_PATH, frames_dir)
else:
    print("Step 1: Skipping (frames already extracted)")

# ---------------------------------------------------------------------------
# Step 2: Grounding DINO detection
# ---------------------------------------------------------------------------
if not os.path.exists(dino_detections):
    print("Step 2: Grounding DINO detection...")
    run_image_to_object_bboxes(
        image_path=f"{frames_dir}/{REFERENCE_FRAME:06d}.png",
        output_path=dino_detections,
        prompt=DETECTION_PROMPT,
        model_dir=DINO_WEIGHTS,
    )
else:
    print("Step 2: Skipping (DINO detections cached)")

if not os.path.exists(sam2_prompts_path):
    with open(dino_detections) as f:
        detections = json.load(f)
    if not detections:
        raise RuntimeError(f"Grounding DINO found no detections — check prompt: {DETECTION_PROMPT!r}")
    box = BoundingBox.from_dict(detections[0]["box"])
    prompts = Sam2Prompts(prompts=[Sam2Prompt(frame_index=REFERENCE_FRAME, object_id=OBJECT_ID, box=box)])
    with open(sam2_prompts_path, "w") as f:
        json.dump(prompts.to_dict(), f, indent=2)

# ---------------------------------------------------------------------------
# Step 3: SAM2 segmentation
# ---------------------------------------------------------------------------
if not _has_files(f"{masks_dir}/{OBJECT_ID}"):
    print("Step 3: SAM2 segmentation...")
    run_video_to_masks(
        video_path=VIDEO_PATH,
        prompts_path=sam2_prompts_path,
        masks_dir=masks_dir,
        weights_dir=SAM2_WEIGHTS,
    )
else:
    print("Step 3: Skipping (masks already computed)")

# ---------------------------------------------------------------------------
# Step 4: DA3 depth — free intrinsics
# ---------------------------------------------------------------------------
if not _has_files(depth_dir):
    print("Step 4: Depth Anything 3 depth estimation (free intrinsics)...")
    run_video_to_depth(
        video_path=VIDEO_PATH,
        depth_folder=depth_dir,
        intrinsics_folder=intrinsics_dir,
        weights_path=DA3_WEIGHTS,
        process_res=256,
    )
else:
    print("Step 4: Skipping (depth_da3_free already computed)")

# ---------------------------------------------------------------------------
# Step 5: Stabilise per-frame DA3 intrinsics
# ---------------------------------------------------------------------------
if not os.path.exists(intrinsics_stable):
    print("Step 5: Stabilising DA3 intrinsics (temporal median)...")
    stabilize_intrinsics(intrinsics_dir, intrinsics_stable)
else:
    print("Step 5: Skipping (stable intrinsics cached)")

# ---------------------------------------------------------------------------
# Step 6: FoundationPose tracking
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 6: FoundationPose tracking...")
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
    print("Step 6: Skipping (poses already computed)")

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
        process_noise_xy=0.01,
        process_noise_z=0.01,
        process_noise_r=0.02,
        measurement_noise_xy=0.01,
        measurement_noise_z=0.04,
        measurement_noise_r=0.02,
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
renders_mp4        = f"{OUTPUT_DIR}/renders_da3_free.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_da3_free_smoothed.mp4"
depth_mp4          = f"{OUTPUT_DIR}/depth_da3_free.mp4"

frames_to_video(renders_dir,        renders_mp4)
frames_to_video(renders_smooth_dir, renders_smooth_mp4)
frames_to_video(depth_dir,          depth_mp4)

print("Step 11: Stitching comparison video...")
stitch_videos(
    [renders_mp4, renders_smooth_mp4, depth_mp4],
    f"{OUTPUT_DIR}/comparison_da3_free.mp4",
)

print(f"Done. Output: {OUTPUT_DIR}/comparison_da3_free.mp4")
