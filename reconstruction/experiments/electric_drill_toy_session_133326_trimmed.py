"""
Pipeline: electric_drill_toy Session_133326_trimmed

  extract frames
  -> GDino detection
  -> SAM2 segmentation
  -> MoGe depth + stable intrinsics
  -> SAM3D mesh reconstruction (reference frame)
  -> estimate + apply mesh scale to MoGe depth (run_estimate_mesh_scale)
  -> FP tracking (mask_depth=True)
  -> EKF smoothing
  -> render raw + smoothed poses
  -> hand alignment:
       world -> camera space (DynHaMR world_results)
       + intrinsics reprojection (hand -> MoGe)
       + per-hand per-frame z-depth alignment
       + temporal centroid smoothing
  -> comparison video

Run from reconstruction/:
    python experiments/electric_drill_toy_session_133326_trimmed.py

NOTE: Hand intrinsics are required at HAND_INTRINSICS below.
      Provide the intrinsics JSON used by DynHaMR for this session,
      or set to None to skip intrinsics reprojection.
"""

import json
import os
import subprocess
import sys

from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.common.utils import extract_images, frames_to_video, stitch_videos
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME             = "electric_drill_toy"
SESSION          = "Session_133326_trimmed"
SESSION_BASENAME = "trimmed_Session_20260310_133326_color_25s"
OBJECT_ID        = 1
REFERENCE_FRAME  = 0
DETECTION_PROMPT = "electric drill"

MESH_PATH  = f"data/objects/{NAME}/sessions/{SESSION}/sam3d_mesh.obj"
VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/{SESSION_BASENAME}.mp4"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"

# Hand input files (from DynHaMR)
SESSION_DIR     = f"data/objects/{NAME}/sessions/{SESSION}"
HAND_MESH_IN    = f"{SESSION_DIR}/{SESSION_BASENAME}_hand_mesh_traj_000300.npz"
WORLD_RESULTS   = f"{SESSION_DIR}/{SESSION_BASENAME}_000300_world_results.npz"
HAND_INTRINSICS = f"{SESSION_DIR}/hand/intrinsics/intrinsics.json"  # provide DynHaMR intrinsics here
HAND_DIR        = f"{SESSION_DIR}/hand"

SMOOTH_SIGMA = 5.0  # Gaussian sigma in frames for centroid smoothing

MOGE_WEIGHTS  = "data/weights/moge"
FP_WEIGHTS    = "data/weights/foundation_pose"
SAM2_WEIGHTS  = "data/weights/sam2"
DINO_WEIGHTS  = "data/weights/grounding_dino"
SAM3D_WEIGHTS = "data/weights/sam3d"

frames_dir        = f"{OUTPUT_DIR}/frames"
masks_dir         = f"{OUTPUT_DIR}/masks"
dino_detections   = f"{OUTPUT_DIR}/dino_detections.json"
sam2_prompts_path = f"{OUTPUT_DIR}/sam2_prompts.json"

moge_depth_dir         = f"{OUTPUT_DIR}/depth_moge"
moge_intrinsics_dir    = f"{OUTPUT_DIR}/intrinsics_moge"
moge_intrinsics_stable = f"{OUTPUT_DIR}/intrinsics_moge_stable.json"

sam3d_transform  = f"{OUTPUT_DIR}/sam3d_transform.json"
sam3d_intrinsics = f"{OUTPUT_DIR}/sam3d_intrinsics.json"
scale_path       = f"{OUTPUT_DIR}/sam3d_scale.json"
scaled_mesh      = f"data/objects/{NAME}/sessions/{SESSION}/sam3d_mesh_scaled.obj"

rgb_frame0  = f"{frames_dir}/{REFERENCE_FRAME:06d}.png"
depth_frame0 = f"{moge_depth_dir}/{REFERENCE_FRAME:06d}.png"
mask_frame0  = f"{masks_dir}/{OBJECT_ID}/{REFERENCE_FRAME:06d}.png"

poses_dir          = f"{OUTPUT_DIR}/poses_moge"
poses_smooth_dir   = f"{OUTPUT_DIR}/poses_moge_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_moge"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_moge_smoothed"

# Hand mesh outputs
hand_mesh_aligned = f"{HAND_DIR}/hand_mesh/{SESSION_BASENAME}_hand_mesh_traj_000300_moge_aligned.npz"
hand_mesh_perhand = f"{HAND_DIR}/hand_mesh/{SESSION_BASENAME}_hand_mesh_traj_000300_moge_aligned_perhand.npz"
hand_mesh_smooth  = f"{HAND_DIR}/hand_mesh/{SESSION_BASENAME}_hand_mesh_traj_000300_moge_aligned_perhand_smooth.npz"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{HAND_DIR}/hand_mesh", exist_ok=True)


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
# Step 4: MoGe depth estimation + stable intrinsics
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
# Step 5: SAM3D mesh reconstruction from reference frame
# ---------------------------------------------------------------------------
if not os.path.exists(MESH_PATH):
    print("Step 5: SAM3D mesh reconstruction...")
    run_image_to_mesh(
        image_path=rgb_frame0,
        mask_path=mask_frame0,
        mesh_path=MESH_PATH,
        transform_path=sam3d_transform,
        intrinsics_path=sam3d_intrinsics,
        weights_dir=SAM3D_WEIGHTS,
        with_layout_postprocess=True,
        with_texture_baking=True,
    )
    print(f"  SAM3D mesh saved to {MESH_PATH}")
else:
    print("Step 5: Skipping (SAM3D mesh cached)")

# ---------------------------------------------------------------------------
# Step 6: Estimate SAM3D mesh scale from MoGe depth + produce scaled mesh
#   The raw SAM3D mesh canonical scale is ~0.27x, so lo=0.1 covers it.
# ---------------------------------------------------------------------------
if not os.path.exists(scaled_mesh):
    print("Step 6: Estimating SAM3D mesh scale from MoGe depth...")
    run_estimate_mesh_scale(
        mesh_path=MESH_PATH,
        rgb_path=rgb_frame0,
        depth_path=depth_frame0,
        mask_path=mask_frame0,
        intrinsics_path=moge_intrinsics_stable,
        weights_dir=FP_WEIGHTS,
        scale_path=scale_path,
        rescaled_mesh_path=scaled_mesh,
        lo=0.1,
        hi=2.0,
        n_samples=7,
        n_levels=3,
        iou_weight=1.0,
        depth_weight=1.0,
        registration_iterations=5,
    )
    print(f"  Scale saved to {scale_path}, scaled mesh to {scaled_mesh}")
else:
    print("Step 6: Skipping (scaled mesh cached)")

# ---------------------------------------------------------------------------
# Step 8: FP tracking (raw MoGe depth, mask_depth=True)
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 8: FoundationPose tracking...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=moge_depth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        camera_intrinsics_path=moge_intrinsics_stable,
        mesh_path=scaled_mesh,
        poses_dir=poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
        mask_depth=True,
    )
    print(f"  Poses written to {poses_dir}")
else:
    print("Step 8: Skipping (poses cached)")

# ---------------------------------------------------------------------------
# Step 9: EKF smoothing
# ---------------------------------------------------------------------------
if not _has_files(poses_smooth_dir):
    print("Step 9: EKF smoothing...")
    run_ekf_smoothing(
        poses_dir=poses_dir,
        mesh_path=scaled_mesh,
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
    print(f"  Smoothed poses written to {poses_smooth_dir}")
else:
    print("Step 9: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 10: Render raw FP poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 10: Rendering raw FP poses...")
    run_render_poses(
        mesh_path=scaled_mesh,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=moge_intrinsics_stable,
        output_dir=renders_dir,
    )
    print(f"  Renders written to {renders_dir}")
else:
    print("Step 10: Skipping (raw renders cached)")

# ---------------------------------------------------------------------------
# Step 11: Render smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_smooth_dir):
    print("Step 11: Rendering smoothed poses...")
    run_render_poses(
        mesh_path=scaled_mesh,
        poses_dir=poses_smooth_dir,
        frames_dir=frames_dir,
        intrinsics_path=moge_intrinsics_stable,
        output_dir=renders_smooth_dir,
    )
    print(f"  Smoothed renders written to {renders_smooth_dir}")
else:
    print("Step 11: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 12: Hand — world → camera + intrinsics reprojection (hand → MoGe)
# ---------------------------------------------------------------------------
if not os.path.exists(hand_mesh_aligned):
    print("Step 12: Hand world→camera + intrinsics reprojection...")
    cmd = [
        sys.executable, "scripts/reproject_hand_mesh.py",
        "--input",               HAND_MESH_IN,
        "--world_results",       WORLD_RESULTS,
        "--target_intrinsics",   moge_intrinsics_stable,
        "--output",              hand_mesh_aligned,
    ]
    if os.path.exists(HAND_INTRINSICS):
        cmd += ["--hand_intrinsics", HAND_INTRINSICS]
    else:
        print("  NOTE: HAND_INTRINSICS not found — skipping intrinsics reprojection")
    subprocess.run(cmd, check=True)
    print(f"  Saved: {hand_mesh_aligned}")
else:
    print("Step 12: Skipping (aligned hand mesh cached)")

# ---------------------------------------------------------------------------
# Step 13: Per-hand per-frame z-depth alignment against MoGe depth
# ---------------------------------------------------------------------------
if not os.path.exists(hand_mesh_perhand):
    print("Step 13: Per-hand per-frame z-depth alignment...")
    subprocess.run([
        sys.executable, "scripts/align_hand_depth.py",
        "--input",        hand_mesh_aligned,
        "--depth",        moge_depth_dir,
        "--intrinsics",   moge_intrinsics_stable,
        "--output",       hand_mesh_perhand,
        "--per_hand",
        "--align",        "offset",
    ], check=True)
    print(f"  Saved: {hand_mesh_perhand}")
else:
    print("Step 13: Skipping (depth-aligned hand mesh cached)")

# ---------------------------------------------------------------------------
# Step 14: Temporal smoothing of hand centroid
# ---------------------------------------------------------------------------
if not os.path.exists(hand_mesh_smooth):
    print("Step 14: Temporal smoothing of hand centroid...")
    subprocess.run([
        sys.executable, "scripts/smooth_hand_mesh.py",
        "--input",  hand_mesh_perhand,
        "--output", hand_mesh_smooth,
        "--sigma",  str(SMOOTH_SIGMA),
    ], check=True)
    print(f"  Saved: {hand_mesh_smooth}")
else:
    print("Step 14: Skipping (smoothed hand mesh cached)")

# ---------------------------------------------------------------------------
# Step 15: Encode + stitch comparison video
# ---------------------------------------------------------------------------
print("Step 15: Encoding videos...")
moge_depth_mp4     = f"{OUTPUT_DIR}/depth_moge.mp4"
renders_mp4        = f"{OUTPUT_DIR}/renders_moge.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_moge_smoothed.mp4"
comparison_mp4     = f"{OUTPUT_DIR}/comparison.mp4"

if not os.path.exists(moge_depth_mp4):
    frames_to_video(moge_depth_dir, moge_depth_mp4)
if not os.path.exists(renders_mp4):
    frames_to_video(renders_dir, renders_mp4)
if not os.path.exists(renders_smooth_mp4):
    frames_to_video(renders_smooth_dir, renders_smooth_mp4)

if not os.path.exists(comparison_mp4):
    print("Step 15: Creating comparison video...")
    stitch_videos([renders_mp4, renders_smooth_mp4, moge_depth_mp4], comparison_mp4)

print(f"\nDone.")
print(f"  SAM3D mesh (scaled)         : {scaled_mesh}")
print(f"  MoGe depth video            : {moge_depth_mp4}")
print(f"  FP renders video            : {renders_mp4}")
print(f"  FP smoothed renders video   : {renders_smooth_mp4}")
print(f"  Comparison video            : {comparison_mp4}")
print(f"  Hand mesh (MoGe aligned)    : {hand_mesh_aligned}")
print(f"  Hand mesh (depth aligned)   : {hand_mesh_perhand}")
print(f"  Hand mesh (smoothed)        : {hand_mesh_smooth}")
