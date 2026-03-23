"""
Pipeline: yellow_spray Session_20260310_141158_f50 (video cropped at frame 50)

  GDino -> SAM2 -> DA3 depth (free intrinsics) + MoGe depth
  -> FP tracking with DA3 depth -> EKF smoothing -> render
  -> FP tracking with MoGe depth -> EKF smoothing -> render

Run from reconstruction/:
    python experiments/yellow_spray_f50.py
"""

import json
import os

from v2d.common.utils import extract_images
from v2d.common.utils import frames_to_video
from v2d.common.utils import stitch_videos
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.depth_anything.docker.run_video_to_depth import run_video_to_depth as run_da3_depth
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_render_poses import run_render_poses

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME             = "yellow_spray"
SESSION          = "Session_20260310_141158_f50"
OBJECT_ID        = 1
REFERENCE_FRAME  = 0
DETECTION_PROMPT = "yellow spray can"

MESH_PATH  = f"data/objects/{NAME}/mesh/textured_mesh.obj"
VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/{SESSION}_color.mp4"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"

DA3_WEIGHTS  = "data/weights/depth_anything"
MOGE_WEIGHTS = "data/weights/moge"
FP_WEIGHTS   = "data/weights/foundation_pose"
SAM2_WEIGHTS = "data/weights/sam2"
DINO_WEIGHTS = "data/weights/grounding_dino"

frames_dir        = f"{OUTPUT_DIR}/frames"
masks_dir         = f"{OUTPUT_DIR}/masks"
dino_detections   = f"{OUTPUT_DIR}/dino_detections.json"
sam2_prompts_path = f"{OUTPUT_DIR}/sam2_prompts.json"

# DA3 depth
da3_depth_dir         = f"{OUTPUT_DIR}/depth_da3"
da3_intrinsics_dir    = f"{OUTPUT_DIR}/intrinsics_da3"
da3_intrinsics_stable = f"{OUTPUT_DIR}/intrinsics_da3_stable.json"

# MoGe depth
moge_depth_dir         = f"{OUTPUT_DIR}/depth_moge"
moge_intrinsics_dir    = f"{OUTPUT_DIR}/intrinsics_moge"
moge_intrinsics_stable = f"{OUTPUT_DIR}/intrinsics_moge_stable.json"

# FP + renders for DA3
da3_poses_dir          = f"{OUTPUT_DIR}/poses_da3"
da3_poses_smooth_dir   = f"{OUTPUT_DIR}/poses_da3_smoothed"
da3_renders_dir        = f"{OUTPUT_DIR}/renders_da3"
da3_renders_smooth_dir = f"{OUTPUT_DIR}/renders_da3_smoothed"

# FP + renders for MoGe
moge_poses_dir          = f"{OUTPUT_DIR}/poses_moge"
moge_poses_smooth_dir   = f"{OUTPUT_DIR}/poses_moge_smoothed"
moge_renders_dir        = f"{OUTPUT_DIR}/renders_moge"
moge_renders_smooth_dir = f"{OUTPUT_DIR}/renders_moge_smoothed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


def _ekf_kwargs(poses_dir, intrinsics, output_dir):
    return dict(
        poses_dir=poses_dir,
        mesh_path=MESH_PATH,
        intrinsics_path=intrinsics,
        weights_dir=FP_WEIGHTS,
        output_dir=output_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        process_noise_xy=0.01,
        process_noise_z=0.01,
        process_noise_r=0.02,
        measurement_noise_xy=0.01,
        measurement_noise_z=0.04,
        measurement_noise_r=0.02,
    )


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
# Step 4: DA3 depth (free intrinsics)
# ---------------------------------------------------------------------------
if not _has_files(da3_depth_dir):
    print("Step 4: DA3 depth estimation...")
    run_da3_depth(
        video_path=VIDEO_PATH,
        depth_folder=da3_depth_dir,
        intrinsics_folder=da3_intrinsics_dir,
        weights_path=DA3_WEIGHTS,
        process_res=256,
    )
else:
    print("Step 4: Skipping (DA3 depth cached)")

if not os.path.exists(da3_intrinsics_stable):
    print("Step 4b: Stabilising DA3 intrinsics...")
    stabilize_intrinsics(da3_intrinsics_dir, da3_intrinsics_stable)

# ---------------------------------------------------------------------------
# Step 5: MoGe depth
# ---------------------------------------------------------------------------
if not _has_files(moge_depth_dir):
    print("Step 5: MoGe depth estimation...")
    run_moge_depth(
        video_path=VIDEO_PATH,
        depth_folder=moge_depth_dir,
        intrinsics_folder=moge_intrinsics_dir,
        weights_path=MOGE_WEIGHTS,
    )
else:
    print("Step 5: Skipping (MoGe depth cached)")

if not os.path.exists(moge_intrinsics_stable):
    print("Step 5b: Stabilising MoGe intrinsics...")
    stabilize_intrinsics(moge_intrinsics_dir, moge_intrinsics_stable)

# ---------------------------------------------------------------------------
# Step 6: FP tracking — DA3 depth
# ---------------------------------------------------------------------------
if not _has_files(da3_poses_dir):
    print("Step 6: FoundationPose tracking (DA3 depth)...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=da3_depth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        camera_intrinsics_path=da3_intrinsics_stable,
        mesh_path=MESH_PATH,
        poses_dir=da3_poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
    )
else:
    print("Step 6: Skipping (DA3 poses cached)")

# ---------------------------------------------------------------------------
# Step 7: EKF smoothing — DA3
# ---------------------------------------------------------------------------
if not _has_files(da3_poses_smooth_dir):
    print("Step 7: EKF smoothing (DA3)...")
    run_ekf_smoothing(**_ekf_kwargs(da3_poses_dir, da3_intrinsics_stable, da3_poses_smooth_dir))
else:
    print("Step 7: Skipping (DA3 smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 8: FP tracking — MoGe depth
# ---------------------------------------------------------------------------
if not _has_files(moge_poses_dir):
    print("Step 8: FoundationPose tracking (MoGe depth)...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=moge_depth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        camera_intrinsics_path=moge_intrinsics_stable,
        mesh_path=MESH_PATH,
        poses_dir=moge_poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
    )
else:
    print("Step 8: Skipping (MoGe poses cached)")

# ---------------------------------------------------------------------------
# Step 9: EKF smoothing — MoGe
# ---------------------------------------------------------------------------
if not _has_files(moge_poses_smooth_dir):
    print("Step 9: EKF smoothing (MoGe)...")
    run_ekf_smoothing(**_ekf_kwargs(moge_poses_dir, moge_intrinsics_stable, moge_poses_smooth_dir))
else:
    print("Step 9: Skipping (MoGe smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 10: Render DA3 poses (raw + smoothed)
# ---------------------------------------------------------------------------
if not _has_files(da3_renders_dir):
    print("Step 10: Rendering DA3 raw poses...")
    run_render_poses(mesh_path=MESH_PATH, poses_dir=da3_poses_dir, frames_dir=frames_dir,
                     intrinsics_path=da3_intrinsics_stable, output_dir=da3_renders_dir)
else:
    print("Step 10: Skipping (DA3 raw renders cached)")

if not _has_files(da3_renders_smooth_dir):
    print("Step 10b: Rendering DA3 smoothed poses...")
    run_render_poses(mesh_path=MESH_PATH, poses_dir=da3_poses_smooth_dir, frames_dir=frames_dir,
                     intrinsics_path=da3_intrinsics_stable, output_dir=da3_renders_smooth_dir)
else:
    print("Step 10b: Skipping (DA3 smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 11: Render MoGe poses (raw + smoothed)
# ---------------------------------------------------------------------------
if not _has_files(moge_renders_dir):
    print("Step 11: Rendering MoGe raw poses...")
    run_render_poses(mesh_path=MESH_PATH, poses_dir=moge_poses_dir, frames_dir=frames_dir,
                     intrinsics_path=moge_intrinsics_stable, output_dir=moge_renders_dir)
else:
    print("Step 11: Skipping (MoGe raw renders cached)")

if not _has_files(moge_renders_smooth_dir):
    print("Step 11b: Rendering MoGe smoothed poses...")
    run_render_poses(mesh_path=MESH_PATH, poses_dir=moge_poses_smooth_dir, frames_dir=frames_dir,
                     intrinsics_path=moge_intrinsics_stable, output_dir=moge_renders_smooth_dir)
else:
    print("Step 11b: Skipping (MoGe smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 12: Encode + stitch
# ---------------------------------------------------------------------------
print("Step 12: Encoding and stitching comparison video...")

da3_raw_mp4    = f"{OUTPUT_DIR}/renders_da3.mp4"
da3_smooth_mp4 = f"{OUTPUT_DIR}/renders_da3_smoothed.mp4"
moge_raw_mp4   = f"{OUTPUT_DIR}/renders_moge.mp4"
moge_smooth_mp4 = f"{OUTPUT_DIR}/renders_moge_smoothed.mp4"
da3_depth_mp4  = f"{OUTPUT_DIR}/depth_da3.mp4"
moge_depth_mp4 = f"{OUTPUT_DIR}/depth_moge.mp4"

frames_to_video(da3_renders_dir,        da3_raw_mp4)
frames_to_video(da3_renders_smooth_dir, da3_smooth_mp4)
frames_to_video(moge_renders_dir,        moge_raw_mp4)
frames_to_video(moge_renders_smooth_dir, moge_smooth_mp4)
frames_to_video(da3_depth_dir,  da3_depth_mp4)
frames_to_video(moge_depth_dir, moge_depth_mp4)

stitch_videos(
    [da3_raw_mp4, da3_smooth_mp4, moge_raw_mp4, moge_smooth_mp4],
    f"{OUTPUT_DIR}/comparison.mp4",
)

print(f"Done. Output: {OUTPUT_DIR}/comparison.mp4")
