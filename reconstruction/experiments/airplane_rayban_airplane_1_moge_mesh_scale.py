"""
Mesh scale sweep: airplane rayban_airplane_1 — MoGe (non-aligned) depth

Runs FP tracking + EKF smoothing + render for 4 mesh scale factors:
  0.5, 0.75, 1.25, 1.5

For each scale:
  1. Create scaled OBJ (vertices * scale, same MTL/texture)
  2. FP tracking (MoGe depth, non-aligned)
  3. EKF smoothing
  4. Render raw + smoothed poses
  5. Encode videos

Run from reconstruction/:
    python experiments/airplane_rayban_airplane_1_moge_mesh_scale.py
"""

import os
import re

from v2d.common.utils import frames_to_video, stitch_videos
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NAME            = "airplane"
SESSION         = "rayban_airplane_1"
OBJECT_ID       = 1
REFERENCE_FRAME = 0
SCALES          = [0.5, 0.75, 1.25, 1.5]

MESH_PATH  = f"data/objects/{NAME}/mesh/textured_mesh.obj"
VIDEO_PATH = f"data/objects/{NAME}/sessions/{SESSION}/airplane.mp4"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"
FP_WEIGHTS = "data/weights/foundation_pose"

frames_dir        = f"{OUTPUT_DIR}/frames"
masks_dir         = f"{OUTPUT_DIR}/masks/{OBJECT_ID}"
moge_depth_dir    = f"{OUTPUT_DIR}/depth_moge"
moge_intrinsics   = f"{OUTPUT_DIR}/intrinsics_moge_stable.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"data/objects/{NAME}/mesh/scaled", exist_ok=True)


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


def _scale_tag(scale: float) -> str:
    return str(scale).replace(".", "p")


def create_scaled_mesh(src_obj: str, scale: float) -> str:
    """Write a copy of src_obj with all vertex positions multiplied by scale.
    MTL and texture files are symlinked into the scaled/ dir so relative
    paths in the MTL resolve correctly from the OBJ location.
    Returns path to the new OBJ file.
    """
    tag = _scale_tag(scale)
    scaled_dir = f"data/objects/{NAME}/mesh/scaled"
    out_path   = f"{scaled_dir}/textured_mesh_scale_{tag}.obj"
    if os.path.exists(out_path):
        return out_path

    mesh_dir = os.path.dirname(os.path.abspath(src_obj))

    # Symlink MTL and all texture files alongside the scaled OBJ so that
    # relative paths inside the MTL (map_Kd material_0.png etc.) resolve correctly.
    for fname in os.listdir(mesh_dir):
        if fname.endswith(".mtl") or fname.endswith(".png"):
            link = os.path.join(scaled_dir, fname)
            if not os.path.exists(link):
                os.symlink(os.path.join(mesh_dir, fname), link)

    with open(src_obj) as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        if line.startswith("v ") and not line.startswith("vt ") and not line.startswith("vn "):
            parts = line.split()
            x, y, z = float(parts[1]) * scale, float(parts[2]) * scale, float(parts[3]) * scale
            out_lines.append(f"v {x} {y} {z}\n")
        else:
            out_lines.append(line)

    with open(out_path, "w") as f:
        f.writelines(out_lines)

    print(f"  Created scaled mesh (x{scale}): {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Run per-scale pipeline
# ---------------------------------------------------------------------------
comparison_inputs = []

for scale in SCALES:
    tag = _scale_tag(scale)
    print(f"\n{'='*60}")
    print(f"Scale: {scale}  (tag: {tag})")
    print(f"{'='*60}")

    scaled_mesh = create_scaled_mesh(MESH_PATH, scale)

    poses_dir          = f"{OUTPUT_DIR}/poses_moge_scale_{tag}"
    poses_smooth_dir   = f"{OUTPUT_DIR}/poses_moge_scale_{tag}_smoothed"
    renders_dir        = f"{OUTPUT_DIR}/renders_moge_scale_{tag}"
    renders_smooth_dir = f"{OUTPUT_DIR}/renders_moge_scale_{tag}_smoothed"

    # FP tracking
    if not _has_files(poses_dir):
        print(f"FP tracking (scale={scale})...")
        run_video_to_poses(
            video_path=VIDEO_PATH,
            depth_folder=moge_depth_dir,
            masks_folder=masks_dir,
            camera_intrinsics_path=moge_intrinsics,
            mesh_path=scaled_mesh,
            poses_dir=poses_dir,
            weights_dir=FP_WEIGHTS,
            reference_frame=REFERENCE_FRAME,
        )
    else:
        print(f"FP tracking: skipping (cached)")

    # EKF smoothing
    if not _has_files(poses_smooth_dir):
        print(f"EKF smoothing (scale={scale})...")
        run_ekf_smoothing(
            poses_dir=poses_dir,
            mesh_path=scaled_mesh,
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
    else:
        print(f"EKF smoothing: skipping (cached)")

    # Render raw
    if not _has_files(renders_dir):
        print(f"Rendering raw poses (scale={scale})...")
        run_render_poses(
            mesh_path=scaled_mesh,
            poses_dir=poses_dir,
            frames_dir=frames_dir,
            intrinsics_path=moge_intrinsics,
            output_dir=renders_dir,
        )
    else:
        print(f"Render raw: skipping (cached)")

    # Render smoothed
    if not _has_files(renders_smooth_dir):
        print(f"Rendering smoothed poses (scale={scale})...")
        run_render_poses(
            mesh_path=scaled_mesh,
            poses_dir=poses_smooth_dir,
            frames_dir=frames_dir,
            intrinsics_path=moge_intrinsics,
            output_dir=renders_smooth_dir,
        )
    else:
        print(f"Render smoothed: skipping (cached)")

    # Encode
    renders_mp4        = f"{OUTPUT_DIR}/renders_moge_scale_{tag}.mp4"
    renders_smooth_mp4 = f"{OUTPUT_DIR}/renders_moge_scale_{tag}_smoothed.mp4"

    if not os.path.exists(renders_mp4):
        frames_to_video(renders_dir, renders_mp4)
    if not os.path.exists(renders_smooth_mp4):
        frames_to_video(renders_smooth_dir, renders_smooth_mp4)

    comparison_inputs.append(renders_smooth_mp4)
    print(f"Done: scale={scale} -> {renders_smooth_mp4}")

# ---------------------------------------------------------------------------
# Stitch all smoothed renders into one comparison video
# ---------------------------------------------------------------------------
comparison_mp4 = f"{OUTPUT_DIR}/comparison_moge_scale_sweep.mp4"
if not os.path.exists(comparison_mp4):
    print("\nStitching scale sweep comparison video...")
    stitch_videos(comparison_inputs, comparison_mp4)

print(f"\nAll done. Comparison: {comparison_mp4}")
