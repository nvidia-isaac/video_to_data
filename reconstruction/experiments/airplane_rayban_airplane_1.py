"""
Pipeline: airplane rayban_airplane_1

  extract frames -> GDino -> SAM2
  -> MoGe depth
  -> align_depth_to_object (frame 0 reference)
  -> align_depth_to_reference_depth (all frames)
  -> FP tracking (mask_depth=True)
  -> EKF smoothing
  -> render raw + smoothed poses
  -> comparison video

Run from reconstruction/:
    python experiments/airplane_rayban_airplane_1.py
"""

import json
import os

from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.common.utils import extract_images, frames_to_video, stitch_videos
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.foundation_pose.docker.run_align_depth_to_object import run_align_depth_to_object
from v2d.foundation_pose.docker.run_align_depth_to_reference_depth import run_align_depth_to_reference_depth
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME             = "airplane"
SESSION          = "rayban_airplane_1"
OBJECT_ID        = 1
REFERENCE_FRAME  = 0
DETECTION_PROMPT = "airplane"

MESH_PATH  = f"data/objects/{NAME}/mesh/textured_mesh.obj"
VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/airplane.mp4"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"

MOGE_WEIGHTS = "data/weights/moge"
FP_WEIGHTS   = "data/weights/foundation_pose"
SAM2_WEIGHTS = "data/weights/sam2"
DINO_WEIGHTS = "data/weights/grounding_dino"

frames_dir        = f"{OUTPUT_DIR}/frames"
masks_dir         = f"{OUTPUT_DIR}/masks"
dino_detections   = f"{OUTPUT_DIR}/dino_detections.json"
sam2_prompts_path = f"{OUTPUT_DIR}/sam2_prompts.json"

moge_depth_dir         = f"{OUTPUT_DIR}/depth_moge"
moge_intrinsics_dir    = f"{OUTPUT_DIR}/intrinsics_moge"
moge_intrinsics_stable = f"{OUTPUT_DIR}/intrinsics_moge_stable.json"

aligned_depth_dir   = f"{OUTPUT_DIR}/depth_moge_aligned"
rgb_frame0          = f"{frames_dir}/{REFERENCE_FRAME:06d}.png"
depth_frame0        = f"{moge_depth_dir}/{REFERENCE_FRAME:06d}.png"
mask_frame0         = f"{masks_dir}/{OBJECT_ID}/{REFERENCE_FRAME:06d}.png"
depth_reference_out = f"{aligned_depth_dir}/depth_reference.png"

poses_dir          = f"{OUTPUT_DIR}/poses_moge_aligned"
poses_smooth_dir   = f"{OUTPUT_DIR}/poses_moge_aligned_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_moge_aligned"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_moge_aligned_smoothed"

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
    print("Step 1: Skipping (frames cached)")

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
    print("Step 3: Skipping (masks cached)")

# ---------------------------------------------------------------------------
# Step 4: MoGe depth
# ---------------------------------------------------------------------------
if not _has_files(moge_depth_dir):
    print("Step 4: MoGe depth estimation...")
    run_moge_depth(
        video_path=VIDEO_PATH,
        depth_folder=moge_depth_dir,
        intrinsics_folder=moge_intrinsics_dir,
        weights_path=MOGE_WEIGHTS,
    )
else:
    print("Step 4: Skipping (MoGe depth cached)")

if not os.path.exists(moge_intrinsics_stable):
    print("Step 4b: Stabilising MoGe intrinsics...")
    stabilize_intrinsics(moge_intrinsics_dir, moge_intrinsics_stable)
else:
    print("Step 4b: Skipping (stable intrinsics cached)")

# ---------------------------------------------------------------------------
# Step 5: Align reference frame depth to object mesh
# ---------------------------------------------------------------------------
if not os.path.exists(depth_reference_out):
    print("Step 5: Aligning reference frame depth to object mesh...")
    run_align_depth_to_object(
        mesh_path=MESH_PATH,
        rgb_path=rgb_frame0,
        depth_path=depth_frame0,
        mask_path=mask_frame0,
        intrinsics_path=moge_intrinsics_stable,
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
    print("Step 5: Skipping (reference depth cached)")

# ---------------------------------------------------------------------------
# Step 6: Align all frames to reference depth via ICP + affine
# ---------------------------------------------------------------------------
if not os.path.exists(f"{aligned_depth_dir}/{REFERENCE_FRAME:06d}.png"):
    print("Step 6: Aligning all frames to reference depth via ICP...")
    run_align_depth_to_reference_depth(
        depth_folder=moge_depth_dir,
        depth_reference_path=depth_reference_out,
        intrinsics_path=moge_intrinsics_stable,
        output_folder=aligned_depth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        reference_mask_path=mask_frame0,
        n_iterations=3,
        outlier_trim_ratio=0.2,
        max_points=20000,
    )
    print(f"Aligned depth written to {aligned_depth_dir}")
else:
    print("Step 6: Skipping (aligned depth cached)")

# ---------------------------------------------------------------------------
# Step 7: FP tracking with aligned depth + depth masking
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 7: FoundationPose tracking (aligned depth, mask_depth=True)...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=aligned_depth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        camera_intrinsics_path=moge_intrinsics_stable,
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
        mask_depth=True,
    )
    print(f"Poses written to {poses_dir}")
else:
    print("Step 7: Skipping (poses cached)")

# ---------------------------------------------------------------------------
# Step 8: EKF smoothing
# ---------------------------------------------------------------------------
if not _has_files(poses_smooth_dir):
    print("Step 8: EKF smoothing...")
    run_ekf_smoothing(
        poses_dir=poses_dir,
        mesh_path=MESH_PATH,
        intrinsics_path=moge_intrinsics_stable,
        weights_dir=FP_WEIGHTS,
        output_dir=poses_smooth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        process_noise_xy=0.01,
        process_noise_z=0.01,
        process_noise_r=0.02,
        measurement_noise_xy=0.01,
        measurement_noise_z=0.04,
        measurement_noise_r=0.02,
    )
    print(f"Smoothed poses written to {poses_smooth_dir}")
else:
    print("Step 8: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 9: Render raw FP poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 9: Rendering raw FP poses...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=moge_intrinsics_stable,
        output_dir=renders_dir,
    )
    print(f"Renders written to {renders_dir}")
else:
    print("Step 9: Skipping (renders cached)")

# ---------------------------------------------------------------------------
# Step 10: Render smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_smooth_dir):
    print("Step 10: Rendering smoothed poses...")
    run_render_poses(
        mesh_path=MESH_PATH,
        poses_dir=poses_smooth_dir,
        frames_dir=frames_dir,
        intrinsics_path=moge_intrinsics_stable,
        output_dir=renders_smooth_dir,
    )
    print(f"Smoothed renders written to {renders_smooth_dir}")
else:
    print("Step 10: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 11: Encode + stitch comparison video
# ---------------------------------------------------------------------------
print("Step 11: Encoding videos...")
aligned_depth_mp4  = f"{OUTPUT_DIR}/depth_moge_aligned.mp4"
moge_depth_mp4     = f"{OUTPUT_DIR}/depth_moge.mp4"
renders_mp4        = f"{OUTPUT_DIR}/renders_moge_aligned.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_moge_aligned_smoothed.mp4"
comparison_mp4     = f"{OUTPUT_DIR}/comparison.mp4"

if not os.path.exists(aligned_depth_mp4):
    frames_to_video(aligned_depth_dir, aligned_depth_mp4)
if not os.path.exists(moge_depth_mp4):
    frames_to_video(moge_depth_dir, moge_depth_mp4)
if not os.path.exists(renders_mp4):
    frames_to_video(renders_dir, renders_mp4)
if not os.path.exists(renders_smooth_mp4):
    frames_to_video(renders_smooth_dir, renders_smooth_mp4)

if not os.path.exists(comparison_mp4):
    print("Step 11: Creating comparison video...")
    stitch_videos([renders_mp4, renders_smooth_mp4, moge_depth_mp4, aligned_depth_mp4], comparison_mp4)

print(f"\nDone.")
print(f"  Aligned depth video         : {aligned_depth_mp4}")
print(f"  FP renders video            : {renders_mp4}")
print(f"  FP smoothed renders video   : {renders_smooth_mp4}")
print(f"  Comparison video            : {comparison_mp4}")
