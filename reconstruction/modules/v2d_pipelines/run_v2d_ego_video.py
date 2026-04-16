"""
Full e2e reconstruction pipeline for V2D ego hand reconstruction videos.

Each numbered folder under data/V2D_ego_reconstruction_10_videos/ contains:
  - trimmed RGB video
  - hand_mesh/  (DynHaMR hand mesh trajectory + world_results)
  - mesh/       (textured object mesh, already copied)

Pipeline — shared steps:
  1.  Extract frames
  2.  Grounding DINO detection → bounding box
  3.  SAM2 video segmentation → per-frame object masks
  4.  MoGe depth + intrinsics → stabilise intrinsics
  5.  Create hand intrinsics JSON from DynHaMR world_results

Then two parallel branches (MoGe / DA3 metric):
  [M/D]1.  Scale estimation (coarse-to-fine) → scaled mesh
  [M/D]2.  FoundationPose tracking
  [M/D]3.  EKF smoothing
  [M/D]4.  Render raw + smoothed poses
  [M/D]5.  Reproject hand mesh: world → camera space + branch intrinsics
  [M/D]6.  Per-hand per-frame z-depth alignment
  [M/D]7.  Temporal centroid smoothing
  [M/D]8.  Encode comparison + multiview render video

Usage:
    python experiments/run_v2d_ego_video.py --video_num 01
    python experiments/run_v2d_ego_video.py --video_num 07

Run from reconstruction/.
"""

import argparse
import glob
import json
import os

import numpy as np

from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.common.utils import extract_images, frames_to_video, stitch_videos
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.depth_anything.docker.run_video_to_depth import run_video_to_depth as run_da3_depth
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.hand_alignment.docker.run_align_hand_depth import run_align_hand_depth
from v2d.hand_alignment.docker.run_render_multiview_video import run_render_multiview_video
from v2d.hand_alignment.docker.run_reproject_hand_mesh import run_reproject_hand_mesh
from v2d.hand_alignment.docker.run_smooth_hand_mesh import run_smooth_hand_mesh
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks

# ---------------------------------------------------------------------------
# Per-video config: detection prompt for Grounding DINO
# ---------------------------------------------------------------------------
VIDEO_CONFIG = {
    "01": "yellow spray can",
    "02": "yellow spray can",
    "03": "dust brush",
    "04": "dust brush",
    "05": "wooden spatula",
    "06": "wooden spatula",
    "07": "electric hand drill toy",
    "08": "electric hand drill toy",
    "09": "airplane",
    "10": "airplane",
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="E2E reconstruction for V2D ego videos")
parser.add_argument("--video_num", required=True, choices=list(VIDEO_CONFIG.keys()),
                    help="Two-digit video number (01–10)")
parser.add_argument("--output_base", default=None,
                    help="Base dir for outputs (default: data/V2D_ego_reconstruction_10_videos/<num>/outputs). "
                         "When set, outputs go to <output_base>/<num>/")
args = parser.parse_args()

NUM = args.video_num
DETECTION_PROMPT = VIDEO_CONFIG[NUM]
OBJECT_ID = 1
REFERENCE_FRAME = 0
SMOOTH_SIGMA = 5.0

# ---------------------------------------------------------------------------
# Paths — inputs
# ---------------------------------------------------------------------------
BASE_DIR   = f"data/V2D_ego_reconstruction_10_videos/{NUM}"
OUTPUT_DIR = f"{args.output_base}/{NUM}" if args.output_base else f"{BASE_DIR}/outputs"

_videos = [f for f in glob.glob(f"{BASE_DIR}/*.mp4")
           if "smooth_fit" not in f and "preview" not in f]
if len(_videos) != 1:
    raise RuntimeError(f"Expected exactly 1 trimmed video in {BASE_DIR}, found: {_videos}")
VIDEO_PATH = _videos[0]

MESH_PATH = f"{BASE_DIR}/mesh/textured_mesh.obj"

_hand_mesh_files = glob.glob(f"{BASE_DIR}/hand_mesh/*_hand_mesh_traj_000300.npz")
if len(_hand_mesh_files) != 1:
    raise RuntimeError(f"Expected 1 hand mesh npz, found: {_hand_mesh_files}")
HAND_MESH_IN = _hand_mesh_files[0]

_world_results_files = glob.glob(f"{BASE_DIR}/*_000300_world_results.npz")
if len(_world_results_files) != 1:
    raise RuntimeError(f"Expected 1 world_results npz, found: {_world_results_files}")
WORLD_RESULTS = _world_results_files[0]

MOGE_WEIGHTS = "data/weights/moge"
DA3_WEIGHTS  = "data/weights/depth_anything_metric"
FP_WEIGHTS   = "data/weights/foundation_pose"
SAM2_WEIGHTS = "data/weights/sam2"
DINO_WEIGHTS = "data/weights/grounding_dino"

# ---------------------------------------------------------------------------
# Paths — shared outputs
# ---------------------------------------------------------------------------
frames_dir           = f"{OUTPUT_DIR}/frames"
masks_dir            = f"{OUTPUT_DIR}/masks"
dino_detections_path = f"{OUTPUT_DIR}/dino_detections.json"
sam2_prompts_path    = f"{OUTPUT_DIR}/sam2_prompts.json"
hand_intrinsics_path = f"{BASE_DIR}/hand_mesh/intrinsics.json"

rgb_frame0  = f"{frames_dir}/{REFERENCE_FRAME:06d}.png"
mask_frame0 = f"{masks_dir}/{OBJECT_ID}/{REFERENCE_FRAME:06d}.png"

# ---------------------------------------------------------------------------
# Paths — MoGe branch
# ---------------------------------------------------------------------------
moge_depth_dir         = f"{OUTPUT_DIR}/depth_moge"
moge_intrinsics_dir    = f"{OUTPUT_DIR}/intrinsics_moge"
moge_intrinsics_stable = f"{OUTPUT_DIR}/intrinsics_moge_stable.json"
moge_depth_frame0      = f"{moge_depth_dir}/{REFERENCE_FRAME:06d}.png"

moge_scale_path        = f"{OUTPUT_DIR}/scale.json"
moge_scaled_mesh       = f"{OUTPUT_DIR}/mesh_scaled.obj"
moge_poses_dir         = f"{OUTPUT_DIR}/poses_moge"
moge_poses_smooth_dir  = f"{OUTPUT_DIR}/poses_moge_smoothed"
moge_renders_dir       = f"{OUTPUT_DIR}/renders_moge"
moge_renders_smooth_dir= f"{OUTPUT_DIR}/renders_moge_smoothed"
moge_hand_aligned      = f"{BASE_DIR}/hand_mesh/hand_mesh_moge_aligned.npz"
moge_hand_perhand      = f"{BASE_DIR}/hand_mesh/hand_mesh_moge_aligned_perhand.npz"
moge_hand_smooth       = f"{BASE_DIR}/hand_mesh/hand_mesh_moge_aligned_perhand_smooth.npz"
moge_comparison_mp4    = f"{OUTPUT_DIR}/comparison_moge.mp4"
moge_multiview_mp4     = f"{OUTPUT_DIR}/multiview_moge.mp4"

# ---------------------------------------------------------------------------
# Paths — DA3 metric branch
# ---------------------------------------------------------------------------
da3_depth_dir         = f"{OUTPUT_DIR}/depth_da3"
da3_intrinsics_dir    = f"{OUTPUT_DIR}/intrinsics_da3"
da3_intrinsics_stable = f"{OUTPUT_DIR}/intrinsics_da3_stable.json"
da3_depth_frame0      = f"{da3_depth_dir}/{REFERENCE_FRAME:06d}.png"

da3_scale_path        = f"{OUTPUT_DIR}/da3_scale.json"
da3_scaled_mesh       = f"{OUTPUT_DIR}/da3_mesh_scaled.obj"
da3_poses_dir         = f"{OUTPUT_DIR}/poses_da3"
da3_poses_smooth_dir  = f"{OUTPUT_DIR}/poses_da3_smoothed"
da3_renders_dir       = f"{OUTPUT_DIR}/renders_da3"
da3_renders_smooth_dir= f"{OUTPUT_DIR}/renders_da3_smoothed"
da3_hand_aligned      = f"{BASE_DIR}/hand_mesh/hand_mesh_da3_aligned.npz"
da3_hand_perhand      = f"{BASE_DIR}/hand_mesh/hand_mesh_da3_aligned_perhand.npz"
da3_hand_smooth       = f"{BASE_DIR}/hand_mesh/hand_mesh_da3_aligned_perhand_smooth.npz"
da3_comparison_mp4    = f"{OUTPUT_DIR}/comparison_da3.mp4"
da3_multiview_mp4     = f"{OUTPUT_DIR}/multiview_da3.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


def _multiview_render(scaled_mesh, poses_smooth_dir, hand_mesh_smooth,
                      intrinsics_stable, frames_dir, multiview_mp4):
    """Run multiview render via docker container."""
    run_render_multiview_video(
        mesh_path=scaled_mesh,
        poses_dir=poses_smooth_dir,
        hand_mesh_path=hand_mesh_smooth,
        intrinsics_path=intrinsics_stable,
        output_path=multiview_mp4,
        fps=25.0,
        frames_folder=frames_dir,
    )


print(f"\n{'='*60}")
print(f"  Video {NUM}: {os.path.basename(VIDEO_PATH)}")
print(f"  Detection prompt: {DETECTION_PROMPT!r}")
print(f"{'='*60}\n")

# ===========================================================================
# SHARED STEPS
# ===========================================================================

# Step 1: Extract frames
if not _has_files(frames_dir):
    print("Step 1: Extracting frames...")
    extract_images(VIDEO_PATH, frames_dir)
else:
    print("Step 1: Skipping (frames cached)")

# Step 2: Grounding DINO detection
if not os.path.exists(dino_detections_path):
    print("Step 2: Grounding DINO detection...")
    run_image_to_object_bboxes(
        image_path=rgb_frame0,
        output_path=dino_detections_path,
        prompt=DETECTION_PROMPT,
        model_dir=DINO_WEIGHTS,
    )
else:
    print("Step 2: Skipping (DINO cached)")

if not os.path.exists(sam2_prompts_path):
    with open(dino_detections_path) as f:
        detections = json.load(f)
    if not detections:
        raise RuntimeError(f"Grounding DINO found no detections — check prompt: {DETECTION_PROMPT!r}")
    box = BoundingBox.from_dict(detections[0]["box"])
    prompts = Sam2Prompts(prompts=[Sam2Prompt(frame_index=REFERENCE_FRAME, object_id=OBJECT_ID, box=box)])
    with open(sam2_prompts_path, "w") as f:
        json.dump(prompts.to_dict(), f, indent=2)

# Step 3: SAM2 segmentation
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

# Step 4: MoGe depth (also required as focal-length input for DA3 metric)
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
    print("Step 4b: Skipping (MoGe stable intrinsics cached)")

# Step 5: Hand intrinsics JSON from DynHaMR world_results
if not os.path.exists(hand_intrinsics_path):
    print("Step 5: Creating hand intrinsics JSON...")
    wr = np.load(WORLD_RESULTS, allow_pickle=True)
    fx, fy, cx, cy = wr["intrins"].tolist()
    with open(hand_intrinsics_path, "w") as f:
        json.dump({"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": 1280, "height": 800}, f, indent=2)
    print(f"  → {hand_intrinsics_path}")
else:
    print("Step 5: Skipping (hand intrinsics cached)")

# ===========================================================================
# MOGE BRANCH
# ===========================================================================
print(f"\n--- MoGe branch ---")

# M1: Scale estimation
if not os.path.exists(moge_scaled_mesh):
    print("M1: Scale estimation (MoGe)...")
    run_estimate_mesh_scale(
        mesh_path=MESH_PATH,
        rgb_path=rgb_frame0,
        depth_path=moge_depth_frame0,
        mask_path=mask_frame0,
        intrinsics_path=moge_intrinsics_stable,
        weights_dir=FP_WEIGHTS,
        scale_path=moge_scale_path,
        rescaled_mesh_path=moge_scaled_mesh,
        lo=0.5, hi=2.0, n_samples=9, n_levels=4,
        iou_weight=1.0, depth_weight=1.0, registration_iterations=5,
    )
else:
    print("M1: Skipping (MoGe scaled mesh cached)")

# M2: FP tracking
if not _has_files(moge_poses_dir):
    print("M2: FoundationPose tracking (MoGe)...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=moge_depth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        camera_intrinsics_path=moge_intrinsics_stable,
        mesh_path=moge_scaled_mesh,
        poses_dir=moge_poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
        mask_depth=True,
    )
else:
    print("M2: Skipping (MoGe poses cached)")

# M3: EKF smoothing
if not _has_files(moge_poses_smooth_dir):
    print("M3: EKF smoothing (MoGe)...")
    run_ekf_smoothing(
        poses_dir=moge_poses_dir,
        mesh_path=moge_scaled_mesh,
        intrinsics_path=moge_intrinsics_stable,
        weights_dir=FP_WEIGHTS,
        output_dir=moge_poses_smooth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        process_noise_xy=0.01, process_noise_z=0.01, process_noise_r=0.02,
        measurement_noise_xy=0.01, measurement_noise_z=0.04, measurement_noise_r=0.02,
    )
else:
    print("M3: Skipping (MoGe smoothed poses cached)")

# M4: Render
if not _has_files(moge_renders_dir):
    print("M4a: Rendering raw poses (MoGe)...")
    run_render_poses(mesh_path=moge_scaled_mesh, poses_dir=moge_poses_dir,
                     frames_dir=frames_dir, intrinsics_path=moge_intrinsics_stable,
                     output_dir=moge_renders_dir)
else:
    print("M4a: Skipping (MoGe raw renders cached)")

if not _has_files(moge_renders_smooth_dir):
    print("M4b: Rendering smoothed poses (MoGe)...")
    run_render_poses(mesh_path=moge_scaled_mesh, poses_dir=moge_poses_smooth_dir,
                     frames_dir=frames_dir, intrinsics_path=moge_intrinsics_stable,
                     output_dir=moge_renders_smooth_dir)
else:
    print("M4b: Skipping (MoGe smoothed renders cached)")

# M5: Hand reprojection → MoGe camera + intrinsics
if not os.path.exists(moge_hand_aligned):
    print("M5: Hand reprojection (MoGe)...")
    run_reproject_hand_mesh(
        input_path=HAND_MESH_IN,
        target_intrinsics_path=moge_intrinsics_stable,
        output_path=moge_hand_aligned,
        world_results_path=WORLD_RESULTS,
        hand_intrinsics_path=hand_intrinsics_path,
    )
else:
    print("M5: Skipping (MoGe hand reprojection cached)")

# M6: Per-hand depth alignment
if not os.path.exists(moge_hand_perhand):
    print("M6: Per-hand depth alignment (MoGe)...")
    run_align_hand_depth(
        input_path=moge_hand_aligned,
        depth_path=moge_depth_dir,
        intrinsics_path=moge_intrinsics_stable,
        output_path=moge_hand_perhand,
        per_hand=True,
        align='offset',
    )
else:
    print("M6: Skipping (MoGe hand depth alignment cached)")

# M7: Centroid smoothing
if not os.path.exists(moge_hand_smooth):
    print("M7: Centroid smoothing (MoGe)...")
    run_smooth_hand_mesh(
        input_path=moge_hand_perhand,
        output_path=moge_hand_smooth,
        sigma=SMOOTH_SIGMA,
    )
else:
    print("M7: Skipping (MoGe hand smooth cached)")

# M8: Videos + multiview
print("M8: Encoding MoGe videos...")
moge_renders_mp4        = f"{OUTPUT_DIR}/renders_moge.mp4"
moge_renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_moge_smoothed.mp4"
moge_depth_mp4          = f"{OUTPUT_DIR}/depth_moge.mp4"
if not os.path.exists(moge_renders_mp4):        frames_to_video(moge_renders_dir, moge_renders_mp4)
if not os.path.exists(moge_renders_smooth_mp4): frames_to_video(moge_renders_smooth_dir, moge_renders_smooth_mp4)
if not os.path.exists(moge_depth_mp4):          frames_to_video(moge_depth_dir, moge_depth_mp4)
if not os.path.exists(moge_comparison_mp4):
    stitch_videos([moge_renders_mp4, moge_renders_smooth_mp4, moge_depth_mp4], moge_comparison_mp4)
if not os.path.exists(moge_multiview_mp4):
    print("M8b: Multiview render (MoGe)...")
    _multiview_render(moge_scaled_mesh, moge_poses_smooth_dir, moge_hand_smooth,
                      moge_intrinsics_stable, frames_dir, moge_multiview_mp4)
else:
    print("M8b: Skipping (MoGe multiview cached)")

# ===========================================================================
# DA3 METRIC BRANCH
# ===========================================================================
print(f"\n--- DA3 metric branch ---")

# D1: DA3 metric depth (uses MoGe stable intrinsics for focal length)
if not _has_files(da3_depth_dir):
    print("D1: DA3 metric depth estimation...")
    run_da3_depth(
        video_path=VIDEO_PATH,
        depth_folder=da3_depth_dir,
        intrinsics_folder=da3_intrinsics_dir,
        weights_path=DA3_WEIGHTS,
        model="metric",
        input_intrinsics_path=moge_intrinsics_stable,
        dev=True,
    )
else:
    print("D1: Skipping (DA3 depth cached)")

if not os.path.exists(da3_intrinsics_stable):
    print("D1b: Stabilising DA3 intrinsics...")
    stabilize_intrinsics(da3_intrinsics_dir, da3_intrinsics_stable)
else:
    print("D1b: Skipping (DA3 stable intrinsics cached)")

# D2: Scale estimation
if not os.path.exists(da3_scaled_mesh):
    print("D2: Scale estimation (DA3)...")
    run_estimate_mesh_scale(
        mesh_path=MESH_PATH,
        rgb_path=rgb_frame0,
        depth_path=da3_depth_frame0,
        mask_path=mask_frame0,
        intrinsics_path=da3_intrinsics_stable,
        weights_dir=FP_WEIGHTS,
        scale_path=da3_scale_path,
        rescaled_mesh_path=da3_scaled_mesh,
        lo=0.5, hi=2.0, n_samples=9, n_levels=4,
        iou_weight=1.0, depth_weight=1.0, registration_iterations=5,
    )
else:
    print("D2: Skipping (DA3 scaled mesh cached)")

# D3: FP tracking
if not _has_files(da3_poses_dir):
    print("D3: FoundationPose tracking (DA3)...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=da3_depth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        camera_intrinsics_path=da3_intrinsics_stable,
        mesh_path=da3_scaled_mesh,
        poses_dir=da3_poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
        mask_depth=True,
    )
else:
    print("D3: Skipping (DA3 poses cached)")

# D4: EKF smoothing
if not _has_files(da3_poses_smooth_dir):
    print("D4: EKF smoothing (DA3)...")
    run_ekf_smoothing(
        poses_dir=da3_poses_dir,
        mesh_path=da3_scaled_mesh,
        intrinsics_path=da3_intrinsics_stable,
        weights_dir=FP_WEIGHTS,
        output_dir=da3_poses_smooth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        process_noise_xy=0.01, process_noise_z=0.01, process_noise_r=0.02,
        measurement_noise_xy=0.01, measurement_noise_z=0.04, measurement_noise_r=0.02,
    )
else:
    print("D4: Skipping (DA3 smoothed poses cached)")

# D5: Render
if not _has_files(da3_renders_dir):
    print("D5a: Rendering raw poses (DA3)...")
    run_render_poses(mesh_path=da3_scaled_mesh, poses_dir=da3_poses_dir,
                     frames_dir=frames_dir, intrinsics_path=da3_intrinsics_stable,
                     output_dir=da3_renders_dir)
else:
    print("D5a: Skipping (DA3 raw renders cached)")

if not _has_files(da3_renders_smooth_dir):
    print("D5b: Rendering smoothed poses (DA3)...")
    run_render_poses(mesh_path=da3_scaled_mesh, poses_dir=da3_poses_smooth_dir,
                     frames_dir=frames_dir, intrinsics_path=da3_intrinsics_stable,
                     output_dir=da3_renders_smooth_dir)
else:
    print("D5b: Skipping (DA3 smoothed renders cached)")

# D6: Hand reprojection → DA3 camera + intrinsics
if not os.path.exists(da3_hand_aligned):
    print("D6: Hand reprojection (DA3)...")
    run_reproject_hand_mesh(
        input_path=HAND_MESH_IN,
        target_intrinsics_path=da3_intrinsics_stable,
        output_path=da3_hand_aligned,
        world_results_path=WORLD_RESULTS,
        hand_intrinsics_path=hand_intrinsics_path,
    )
else:
    print("D6: Skipping (DA3 hand reprojection cached)")

# D7: Per-hand depth alignment
if not os.path.exists(da3_hand_perhand):
    print("D7: Per-hand depth alignment (DA3)...")
    run_align_hand_depth(
        input_path=da3_hand_aligned,
        depth_path=da3_depth_dir,
        intrinsics_path=da3_intrinsics_stable,
        output_path=da3_hand_perhand,
        per_hand=True,
        align='offset',
    )
else:
    print("D7: Skipping (DA3 hand depth alignment cached)")

# D8: Centroid smoothing
if not os.path.exists(da3_hand_smooth):
    print("D8: Centroid smoothing (DA3)...")
    run_smooth_hand_mesh(
        input_path=da3_hand_perhand,
        output_path=da3_hand_smooth,
        sigma=SMOOTH_SIGMA,
    )
else:
    print("D8: Skipping (DA3 hand smooth cached)")

# D9: Videos + multiview
print("D9: Encoding DA3 videos...")
da3_renders_mp4        = f"{OUTPUT_DIR}/renders_da3.mp4"
da3_renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_da3_smoothed.mp4"
da3_depth_mp4          = f"{OUTPUT_DIR}/depth_da3.mp4"
if not os.path.exists(da3_renders_mp4):        frames_to_video(da3_renders_dir, da3_renders_mp4)
if not os.path.exists(da3_renders_smooth_mp4): frames_to_video(da3_renders_smooth_dir, da3_renders_smooth_mp4)
if not os.path.exists(da3_depth_mp4):          frames_to_video(da3_depth_dir, da3_depth_mp4)
if not os.path.exists(da3_comparison_mp4):
    stitch_videos([da3_renders_mp4, da3_renders_smooth_mp4, da3_depth_mp4], da3_comparison_mp4)
if not os.path.exists(da3_multiview_mp4):
    print("D9b: Multiview render (DA3)...")
    _multiview_render(da3_scaled_mesh, da3_poses_smooth_dir, da3_hand_smooth,
                      da3_intrinsics_stable, frames_dir, da3_multiview_mp4)
else:
    print("D9b: Skipping (DA3 multiview cached)")

# ===========================================================================
print(f"\nDone — video {NUM}")
print(f"  MoGe comparison : {moge_comparison_mp4}")
print(f"  MoGe multiview  : {moge_multiview_mp4}")
print(f"  DA3 comparison  : {da3_comparison_mp4}")
print(f"  DA3 multiview   : {da3_multiview_mp4}")
