"""
Pipeline: airplane rayban_airplane_1 — CAD mesh + MoGe depth + FP tracking

  1. FP mesh scale estimation: CAD mesh × frame-0 RGB/depth/mask → scale factor
  2. Align frame-0 MoGe depth to scaled CAD mesh (affine: scale + shift)
  3. Align all MoGe frames to frame-0 reference depth via ICP
  4. FP tracking: scaled CAD mesh + aligned depth → raw poses
  5. EKF smoothing → smoothed poses
  6. Render raw + smoothed poses
  7. Re-align hand mesh depth against aligned MoGe depth
  8. Multiview render (object + hands)

Assumes frames, masks, and raw MoGe depth already exist.

Run from reconstruction/:
    python experiments/airplane_rayban_airplane_1_cad_mesh.py
"""

import os

from v2d.common.utils import frames_to_video, stitch_videos
from v2d.foundation_pose.docker.run_align_depth_to_object import run_align_depth_to_object
from v2d.foundation_pose.docker.run_align_depth_to_reference_depth import run_align_depth_to_reference_depth
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME            = "airplane"
SESSION         = "rayban_airplane_1"
OBJECT_ID       = 1
REFERENCE_FRAME = 0

MESH_PATH  = f"data/objects/{NAME}/mesh/textured_mesh.obj"
VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/airplane.mp4"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"

FP_WEIGHTS = "data/weights/foundation_pose"

# Inputs shared with base pipeline (must already exist)
frames_dir      = f"{OUTPUT_DIR}/frames"
masks_dir       = f"{OUTPUT_DIR}/masks/{OBJECT_ID}"
moge_depth_dir  = f"{OUTPUT_DIR}/depth_moge"
moge_intrinsics = f"{OUTPUT_DIR}/intrinsics_moge_stable.json"

rgb_frame0   = f"{frames_dir}/{REFERENCE_FRAME:06d}.png"
depth_frame0 = f"{moge_depth_dir}/{REFERENCE_FRAME:06d}.png"
mask_frame0  = f"{masks_dir}/{REFERENCE_FRAME:06d}.png"

# Scale estimation outputs (stored alongside the CAD mesh)
mesh_scale_path  = f"data/objects/{NAME}/mesh/textured_mesh_scale.json"
mesh_scaled_path = f"data/objects/{NAME}/mesh/textured_mesh_scaled.obj"

# Aligned depth
aligned_depth_dir   = f"{OUTPUT_DIR}/depth_moge_aligned_cad"
depth_reference_out = f"{aligned_depth_dir}/depth_reference.png"

# FP tracking + smoothing
poses_dir          = f"{OUTPUT_DIR}/poses_moge_aligned_cad"
poses_smooth_dir   = f"{OUTPUT_DIR}/poses_moge_aligned_cad_smoothed"
renders_dir        = f"{OUTPUT_DIR}/renders_moge_aligned_cad"
renders_smooth_dir = f"{OUTPUT_DIR}/renders_moge_aligned_cad_smoothed"

# Hand mesh (re-aligned against new depth)
hand_mesh_in  = f"data/objects/{NAME}/sessions/{SESSION}/hand/hand_mesh/airplane_hand_mesh_traj_000300_aligned.npz"
hand_mesh_out = f"data/objects/{NAME}/sessions/{SESSION}/hand/hand_mesh/airplane_hand_mesh_traj_000300_depth_aligned_cad.npz"

os.makedirs(aligned_depth_dir, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


# ---------------------------------------------------------------------------
# Step 1: FP mesh scale estimation (CAD mesh vs frame-0 MoGe depth)
# ---------------------------------------------------------------------------
import json

if not os.path.exists(mesh_scale_path) or True:
    print("Step 1: FP mesh scale estimation (CAD mesh)...")
    run_estimate_mesh_scale(
        mesh_path=MESH_PATH,
        rgb_path=rgb_frame0,
        depth_path=depth_frame0,
        mask_path=mask_frame0,
        intrinsics_path=moge_intrinsics,
        weights_dir=FP_WEIGHTS,
        scale_path=mesh_scale_path,
        rescaled_mesh_path=mesh_scaled_path,
        lo=0.5,
        hi=2.0,
        n_samples=7,
        n_levels=3,
        iou_weight=0.0,
        depth_weight=1.0,
        chamfer_weight=0.0,
        registration_iterations=5,
    )

with open(mesh_scale_path) as f:
    scale = json.load(f)["scale"]
print(f"Step 1: scale={scale:.4f}  mesh={mesh_scaled_path}")

# ---------------------------------------------------------------------------
# Step 3: Align all MoGe frames to reference depth
# ---------------------------------------------------------------------------
if not os.path.exists(f"{aligned_depth_dir}/{REFERENCE_FRAME:06d}.png") or True:
    print("Step 3: Aligning all MoGe frames to reference depth via ICP...")
    run_align_depth_to_reference_depth(
        depth_folder=moge_depth_dir,
        depth_reference_path=f"{moge_depth_dir}/{0:06d}.png",
        intrinsics_path=moge_intrinsics,
        output_folder=aligned_depth_dir,
        masks_folder=masks_dir,
        reference_mask_path=mask_frame0,
        n_iterations=3,
        outlier_trim_ratio=0.2,
        max_points=20000,
    )
    print(f"  Aligned depth: {aligned_depth_dir}")
else:
    print("Step 3: Skipping (aligned depth cached)")

# ---------------------------------------------------------------------------
# Step 4: FP tracking — scaled CAD mesh + aligned depth
# ---------------------------------------------------------------------------
if not _has_files(poses_dir):
    print("Step 4: FP tracking (scaled CAD mesh + aligned depth)...")
    run_video_to_poses(
        video_path=VIDEO_PATH,
        depth_folder=aligned_depth_dir,
        masks_folder=masks_dir,
        camera_intrinsics_path=moge_intrinsics,
        mesh_path=mesh_scaled_path,
        poses_dir=poses_dir,
        weights_dir=FP_WEIGHTS,
        reference_frame=REFERENCE_FRAME,
        mask_depth=True,
    )
    print(f"  Poses: {poses_dir}")
else:
    print("Step 4: Skipping (poses cached)")

# ---------------------------------------------------------------------------
# Step 5: EKF smoothing
# ---------------------------------------------------------------------------
if not _has_files(poses_smooth_dir):
    print("Step 5: EKF smoothing...")
    run_ekf_smoothing(
        poses_dir=poses_dir,
        mesh_path=mesh_scaled_path,
        intrinsics_path=moge_intrinsics,
        weights_dir=FP_WEIGHTS,
        output_dir=poses_smooth_dir,
        masks_folder=masks_dir,
        process_noise_xy=0.01,
        process_noise_z=0.01,
        process_noise_r=0.02,
        measurement_noise_xy=0.01,
        measurement_noise_z=0.04,
        measurement_noise_r=0.02,
    )
    print(f"  Smoothed poses: {poses_smooth_dir}")
else:
    print("Step 5: Skipping (smoothed poses cached)")

# ---------------------------------------------------------------------------
# Step 6: Render raw + smoothed poses
# ---------------------------------------------------------------------------
if not _has_files(renders_dir):
    print("Step 6: Rendering raw poses...")
    run_render_poses(
        mesh_path=mesh_scaled_path,
        poses_dir=poses_dir,
        frames_dir=frames_dir,
        intrinsics_path=moge_intrinsics,
        output_dir=renders_dir,
    )
else:
    print("Step 6: Skipping (raw renders cached)")

if not _has_files(renders_smooth_dir):
    print("Step 6b: Rendering smoothed poses...")
    run_render_poses(
        mesh_path=mesh_scaled_path,
        poses_dir=poses_smooth_dir,
        frames_dir=frames_dir,
        intrinsics_path=moge_intrinsics,
        output_dir=renders_smooth_dir,
    )
else:
    print("Step 6b: Skipping (smoothed renders cached)")

# ---------------------------------------------------------------------------
# Step 7: Re-align hand mesh depth against new aligned MoGe depth
# ---------------------------------------------------------------------------
if not os.path.exists(hand_mesh_out):
    print("Step 7: Re-aligning hand mesh depth against aligned MoGe depth...")
    import subprocess, sys
    cmd = [
        sys.executable, "scripts/align_hand_depth.py",
        "--input",           hand_mesh_in,
        "--depth",           aligned_depth_dir,
        "--intrinsics",      moge_intrinsics,
        "--mesh_intrinsics", f"data/objects/{NAME}/sessions/{SESSION}/hand/intrinsics/intrinsics.json",
        "--output",          hand_mesh_out,
    ]
    subprocess.run(cmd, check=True)
    print(f"  Hand mesh: {hand_mesh_out}")
else:
    print("Step 7: Skipping (depth-aligned hand mesh cached)")

# ---------------------------------------------------------------------------
# Step 8: Encode + stitch comparison video
# ---------------------------------------------------------------------------
print("Step 8: Encoding videos...")
renders_mp4        = f"{OUTPUT_DIR}/renders_moge_aligned_cad.mp4"
renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_moge_aligned_cad_smoothed.mp4"
comparison_mp4     = f"{OUTPUT_DIR}/comparison_moge_aligned_cad.mp4"

if not os.path.exists(renders_mp4):
    frames_to_video(renders_dir, renders_mp4)
if not os.path.exists(renders_smooth_mp4):
    frames_to_video(renders_smooth_dir, renders_smooth_mp4)
if not os.path.exists(comparison_mp4):
    stitch_videos([renders_mp4, renders_smooth_mp4], comparison_mp4)

# ---------------------------------------------------------------------------
# Step 9: Multiview render
# ---------------------------------------------------------------------------
multiview_mp4 = f"{OUTPUT_DIR}/multiview_cad_smoothed.mp4"
if not os.path.exists(multiview_mp4):
    print("Step 9: Multiview render...")
    import subprocess, sys
    cmd = [
        sys.executable, "scripts/render_multiview_video.py",
        "--mesh",           mesh_scaled_path,
        "--poses",          poses_smooth_dir,
        "--hand_mesh",      hand_mesh_out,
        "--intrinsics",     moge_intrinsics,
        "--frames_folder",  frames_dir,
        "--output",         multiview_mp4,
        "--fps",            "30",
    ]
    subprocess.run(cmd, check=True)
    print(f"  Multiview: {multiview_mp4}")
else:
    print("Step 9: Skipping (multiview cached)")

print(f"\nDone.")
print(f"  Scaled CAD mesh:       {mesh_scaled_path}")
print(f"  Aligned depth:         {aligned_depth_dir}")
print(f"  Poses (smoothed):      {poses_smooth_dir}")
print(f"  Hand mesh (aligned):   {hand_mesh_out}")
print(f"  Renders comparison:    {comparison_mp4}")
print(f"  Multiview:             {multiview_mp4}")
