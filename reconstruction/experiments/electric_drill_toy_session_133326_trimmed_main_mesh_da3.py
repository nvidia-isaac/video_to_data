"""
Pipeline: electric_drill_toy Session_133326_trimmed — main mesh + DA3 metric depth

Same as the main_mesh variant but replaces MoGe depth with Depth Anything 3 metric.
All segmentation steps (frames, SAM2) are reused from the SAM3D run.
Depth, intrinsics, scale estimation, FP tracking, smoothing, rendering, and hand
alignment all run fresh against DA3 metric depth.

Steps run here (new outputs only):
  -> DA3 metric depth + intrinsics
  -> stabilise intrinsics
  -> estimate mesh scale (narrow range, main mesh)
  -> FP tracking
  -> EKF smoothing
  -> render raw + smoothed poses
  -> hand alignment (reuses cached world_results / DynHaMR outputs)
  -> multiview render

Run from reconstruction/:
    python experiments/electric_drill_toy_session_133326_trimmed_main_mesh_da3.py
"""

import os
import subprocess
import sys

from v2d.common.utils import frames_to_video, stitch_videos
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.depth_anything.docker.run_video_to_depth import run_video_to_depth as run_da3_depth
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME             = "electric_drill_toy"
SESSION          = "Session_133326_trimmed"
SESSION_BASENAME = "trimmed_Session_20260310_133326_color_25s"
OBJECT_ID        = 1
REFERENCE_FRAME  = 0

# Main pre-existing mesh (already at real-world scale ~10×16×14 cm)
MESH_PATH = f"data/objects/{NAME}/mesh/textured_mesh.obj"

VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/{SESSION_BASENAME}.mp4"

# Reuse frames + masks from the SAM3D run
SHARED_OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"
OUTPUT_DIR        = f"data/objects/{NAME}/sessions/{SESSION}/outputs_main_mesh_da3"

# Hand input files (from DynHaMR) — unchanged
SESSION_DIR     = f"data/objects/{NAME}/sessions/{SESSION}"
HAND_MESH_IN    = f"{SESSION_DIR}/{SESSION_BASENAME}_hand_mesh_traj_000300.npz"
WORLD_RESULTS   = f"{SESSION_DIR}/{SESSION_BASENAME}_000300_world_results.npz"
HAND_INTRINSICS = f"{SESSION_DIR}/hand/intrinsics/intrinsics.json"
HAND_DIR        = f"{SESSION_DIR}/hand"

SMOOTH_SIGMA = 5.0  # Gaussian sigma in frames for centroid smoothing

DA3_WEIGHTS = "data/weights/depth_anything_metric"
FP_WEIGHTS  = "data/weights/foundation_pose"

# Shared perception outputs (reused, never rewritten)
frames_dir = f"{SHARED_OUTPUT_DIR}/frames"
masks_dir  = f"{SHARED_OUTPUT_DIR}/masks"

# DA3-specific depth + intrinsics (new)
da3_depth_dir         = f"{OUTPUT_DIR}/depth_da3"
da3_intrinsics_dir    = f"{OUTPUT_DIR}/intrinsics_da3"
da3_intrinsics_stable = f"{OUTPUT_DIR}/intrinsics_da3_stable.json"

rgb_frame0   = f"{frames_dir}/{REFERENCE_FRAME:06d}.png"
depth_frame0 = f"{da3_depth_dir}/{REFERENCE_FRAME:06d}.png"
mask_frame0  = f"{masks_dir}/{OBJECT_ID}/{REFERENCE_FRAME:06d}.png"

# New outputs for this run
scale_path  = f"{OUTPUT_DIR}/scale.json"
scaled_mesh = f"{OUTPUT_DIR}/mesh_scaled.obj"

poses_dir          = f"{OUTPUT_DIR}/poses_da3"
poses_smooth_dir   = f"{OUTPUT_DIR}/poses_da3_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_da3"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_da3_smoothed"

# Hand mesh outputs
hand_mesh_aligned = f"{HAND_DIR}/hand_mesh/{SESSION_BASENAME}_main_mesh_da3_aligned.npz"
hand_mesh_perhand = f"{HAND_DIR}/hand_mesh/{SESSION_BASENAME}_main_mesh_da3_aligned_perhand.npz"
hand_mesh_smooth  = f"{HAND_DIR}/hand_mesh/{SESSION_BASENAME}_main_mesh_da3_aligned_perhand_smooth.npz"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{HAND_DIR}/hand_mesh", exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


# ---------------------------------------------------------------------------
# Step 1: DA3 metric depth + intrinsics
# ---------------------------------------------------------------------------
if not _has_files(da3_depth_dir):
    print("Step 1: DA3 metric depth estimation...")
    moge_intrinsics_stable = f"{SHARED_OUTPUT_DIR}/intrinsics_moge_stable.json"
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
    print("Step 1: Skipping (DA3 depth cached)")

if not os.path.exists(da3_intrinsics_stable):
    print("Step 1b: Stabilising DA3 intrinsics...")
    stabilize_intrinsics(da3_intrinsics_dir, da3_intrinsics_stable)
else:
    print("Step 1b: Skipping (stable intrinsics cached)")

# ---------------------------------------------------------------------------
# Step 2: Estimate scale of main mesh against DA3 depth
# ---------------------------------------------------------------------------
if not os.path.exists(scaled_mesh):
    print("Step 2: Estimating main mesh scale from DA3 depth...")
    run_estimate_mesh_scale(
        mesh_path=MESH_PATH,
        rgb_path=rgb_frame0,
        depth_path=depth_frame0,
        mask_path=mask_frame0,
        intrinsics_path=da3_intrinsics_stable,
        weights_dir=FP_WEIGHTS,
        scale_path=scale_path,
        rescaled_mesh_path=scaled_mesh,
        lo=0.5,
        hi=2.0,
        n_samples=9,
        n_levels=4,
        iou_weight=1.0,
        depth_weight=1.0,
        registration_iterations=5,
    )
    print(f"  Scale saved to {scale_path}, scaled mesh to {scaled_mesh}")
else:
    print("Step 2: Skipping (scaled mesh cached)")

# ---------------------------------------------------------------------------
# Step 3: FP tracking
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 3: FoundationPose tracking...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=da3_depth_dir,
        masks_folder=f"{masks_dir}/{OBJECT_ID}",
        camera_intrinsics_path=da3_intrinsics_stable,
        mesh_path=scaled_mesh,
        poses_dir=poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
        mask_depth=True,
    )
    print(f"  Poses written to {poses_dir}")
else:
    print("Step 3: Skipping (poses cached)")

# ---------------------------------------------------------------------------
# Step 4: EKF smoothing
# ---------------------------------------------------------------------------
if not _has_files(poses_smooth_dir):
    print("Step 4: EKF smoothing...")
    run_ekf_smoothing(
        poses_dir=poses_dir,
        mesh_path=scaled_mesh,
        intrinsics_path=da3_intrinsics_stable,
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
    print("Step 4: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 5: Render raw FP poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 5: Rendering raw FP poses...")
    run_render_poses(
        mesh_path=scaled_mesh,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=da3_intrinsics_stable,
        output_dir=renders_dir,
    )
    print(f"  Renders written to {renders_dir}")
else:
    print("Step 5: Skipping (raw renders cached)")

# ---------------------------------------------------------------------------
# Step 6: Render smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_smooth_dir):
    print("Step 6: Rendering smoothed poses...")
    run_render_poses(
        mesh_path=scaled_mesh,
        poses_dir=poses_smooth_dir,
        frames_dir=frames_dir,
        intrinsics_path=da3_intrinsics_stable,
        output_dir=renders_smooth_dir,
    )
    print(f"  Smoothed renders written to {renders_smooth_dir}")
else:
    print("Step 6: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 7: Hand — world → camera + intrinsics reprojection (hand → DA3)
# ---------------------------------------------------------------------------
if not os.path.exists(hand_mesh_aligned):
    print("Step 7: Hand world→camera + intrinsics reprojection...")
    cmd = [
        sys.executable, "scripts/reproject_hand_mesh.py",
        "--input",             HAND_MESH_IN,
        "--world_results",     WORLD_RESULTS,
        "--target_intrinsics", da3_intrinsics_stable,
        "--output",            hand_mesh_aligned,
    ]
    if os.path.exists(HAND_INTRINSICS):
        cmd += ["--hand_intrinsics", HAND_INTRINSICS]
    else:
        print("  NOTE: HAND_INTRINSICS not found — skipping intrinsics reprojection")
    subprocess.run(cmd, check=True)
    print(f"  Saved: {hand_mesh_aligned}")
else:
    print("Step 7: Skipping (aligned hand mesh cached)")

# ---------------------------------------------------------------------------
# Step 8: Per-hand per-frame z-depth alignment against DA3 depth
# ---------------------------------------------------------------------------
if not os.path.exists(hand_mesh_perhand):
    print("Step 8: Per-hand per-frame z-depth alignment...")
    subprocess.run([
        sys.executable, "scripts/align_hand_depth.py",
        "--input",      hand_mesh_aligned,
        "--depth",      da3_depth_dir,
        "--intrinsics", da3_intrinsics_stable,
        "--output",     hand_mesh_perhand,
        "--per_hand",
        "--align",      "offset",
    ], check=True)
    print(f"  Saved: {hand_mesh_perhand}")
else:
    print("Step 8: Skipping (depth-aligned hand mesh cached)")

# ---------------------------------------------------------------------------
# Step 9: Temporal smoothing of hand centroid
# ---------------------------------------------------------------------------
if not os.path.exists(hand_mesh_smooth):
    print("Step 9: Temporal smoothing of hand centroid...")
    subprocess.run([
        sys.executable, "scripts/smooth_hand_mesh.py",
        "--input",  hand_mesh_perhand,
        "--output", hand_mesh_smooth,
        "--sigma",  str(SMOOTH_SIGMA),
    ], check=True)
    print(f"  Saved: {hand_mesh_smooth}")
else:
    print("Step 9: Skipping (smoothed hand mesh cached)")

# ---------------------------------------------------------------------------
# Step 10: Encode videos + multiview render
# ---------------------------------------------------------------------------
print("Step 10: Encoding videos...")
renders_mp4        = f"{OUTPUT_DIR}/renders_da3.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_da3_smoothed.mp4"
comparison_mp4     = f"{OUTPUT_DIR}/comparison.mp4"
multiview_mp4      = f"{OUTPUT_DIR}/multiview.mp4"

if not os.path.exists(renders_mp4):
    frames_to_video(renders_dir, renders_mp4)
if not os.path.exists(renders_smooth_mp4):
    frames_to_video(renders_smooth_dir, renders_smooth_mp4)

if not os.path.exists(comparison_mp4):
    da3_depth_mp4 = f"{OUTPUT_DIR}/depth_da3.mp4"
    if not os.path.exists(da3_depth_mp4):
        frames_to_video(da3_depth_dir, da3_depth_mp4)
    stitch_videos([renders_mp4, renders_smooth_mp4, da3_depth_mp4], comparison_mp4)

if not os.path.exists(multiview_mp4):
    print("Step 10b: Multiview render...")
    # Main mesh is already ~4k faces — no simplification needed
    subprocess.run([
        sys.executable, "scripts/render_multiview_video.py",
        "--mesh",          scaled_mesh,
        "--poses",         poses_smooth_dir,
        "--hand_mesh",     hand_mesh_smooth,
        "--intrinsics",    da3_intrinsics_stable,
        "--frames_folder", frames_dir,
        "--output",        multiview_mp4,
        "--fps",           "25",
    ], check=True)

print(f"\nDone.")
print(f"  Scaled mesh                 : {scaled_mesh}")
print(f"  FP renders video            : {renders_mp4}")
print(f"  FP smoothed renders video   : {renders_smooth_mp4}")
print(f"  Comparison video            : {comparison_mp4}")
print(f"  Multiview video             : {multiview_mp4}")
print(f"  Hand mesh (DA3 aligned)     : {hand_mesh_aligned}")
print(f"  Hand mesh (depth aligned)   : {hand_mesh_perhand}")
print(f"  Hand mesh (smoothed)        : {hand_mesh_smooth}")
