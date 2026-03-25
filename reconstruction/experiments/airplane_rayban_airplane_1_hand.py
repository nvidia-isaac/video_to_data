"""
Pipeline: airplane rayban_airplane_1 — hand mesh alignment

  1. World → camera space (per-track w2c from DynHaMR world_results)
     + intrinsics reprojection (hand 636 → MoGe 680)
  2. Per-hand per-frame z-depth alignment against MoGe depth
     (ray-preserving: scales xyz uniformly to avoid lateral drift)
  3. Temporal smoothing of hand centroid position (fingers unchanged)

Produces:
  hand_mesh_moge_aligned.npz               — camera space, MoGe intrinsics
  hand_mesh_moge_aligned_perhand.npz       — + per-hand per-frame z-scaled
  hand_mesh_moge_aligned_perhand_smooth.npz — + centroid temporally smoothed

Run from reconstruction/:
    python experiments/airplane_rayban_airplane_1_hand.py
"""

import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME    = "airplane"
SESSION = "rayban_airplane_1"

HAND_DIR   = f"data/objects/{NAME}/sessions/{SESSION}/hand"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"

# Inputs
hand_mesh_in    = f"{HAND_DIR}/hand_mesh/airplane_hand_mesh_traj_000300.npz"
world_results   = f"{HAND_DIR}/airplane_000300_world_results.npz"
hand_intrinsics = f"{HAND_DIR}/intrinsics/intrinsics.json"
moge_intrinsics = f"{OUTPUT_DIR}/intrinsics_moge_stable.json"
moge_depth_dir  = f"{OUTPUT_DIR}/depth_moge"

SMOOTH_SIGMA = 5.0   # Gaussian sigma in frames for centroid smoothing

# Outputs
hand_mesh_aligned  = f"{HAND_DIR}/hand_mesh/airplane_hand_mesh_traj_000300_moge_aligned.npz"
hand_mesh_perhand  = f"{HAND_DIR}/hand_mesh/airplane_hand_mesh_traj_000300_moge_aligned_perhand.npz"
hand_mesh_smooth   = f"{HAND_DIR}/hand_mesh/airplane_hand_mesh_traj_000300_moge_aligned_perhand_smooth.npz"


# ---------------------------------------------------------------------------
# Step 1: World → camera + intrinsics reprojection (hand → MoGe)
# ---------------------------------------------------------------------------
if not os.path.exists(hand_mesh_aligned):
    print("Step 1: World→camera + intrinsics reprojection (hand 636 → MoGe 680)...")
    subprocess.run([
        sys.executable, "scripts/reproject_hand_mesh.py",
        "--input",             hand_mesh_in,
        "--world_results",     world_results,
        "--hand_intrinsics",   hand_intrinsics,
        "--target_intrinsics", moge_intrinsics,
        "--output",            hand_mesh_aligned,
    ], check=True)
    print(f"  Saved: {hand_mesh_aligned}")
else:
    print("Step 1: Skipping (aligned mesh cached)")

# ---------------------------------------------------------------------------
# Step 2: Per-hand per-frame z-depth alignment against MoGe depth
# ---------------------------------------------------------------------------
if not os.path.exists(hand_mesh_perhand):
    print("Step 2: Per-hand per-frame z-depth alignment...")
    subprocess.run([
        sys.executable, "scripts/align_hand_depth.py",
        "--input",      hand_mesh_aligned,
        "--depth",      moge_depth_dir,
        "--intrinsics", moge_intrinsics,
        "--output",     hand_mesh_perhand,
        "--per_hand",
        "--align",      "offset",
    ], check=True)
    print(f"  Saved: {hand_mesh_perhand}")
else:
    print("Step 2: Skipping (depth-aligned mesh cached)")

# ---------------------------------------------------------------------------
# Step 3: Temporal smoothing of hand centroid
# ---------------------------------------------------------------------------
if not os.path.exists(hand_mesh_smooth):
    print("Step 3: Temporal smoothing of hand centroid...")
    subprocess.run([
        sys.executable, "scripts/smooth_hand_mesh.py",
        "--input",  hand_mesh_perhand,
        "--output", hand_mesh_smooth,
        "--sigma",  str(SMOOTH_SIGMA),
    ], check=True)
    print(f"  Saved: {hand_mesh_smooth}")
else:
    print("Step 3: Skipping (smoothed mesh cached)")

print(f"\nDone.")
print(f"  Aligned (MoGe space):     {hand_mesh_aligned}")
print(f"  Depth-aligned (per-hand): {hand_mesh_perhand}")
print(f"  Smoothed:                 {hand_mesh_smooth}")
