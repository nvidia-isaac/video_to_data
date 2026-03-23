"""
Depth alignment experiment: electric_drill_toy Session_20260310_133326_f50

Aligns MoGe depth to the object mesh using two steps:
  1. align_depth_to_object   — FP-guided affine grid search on frame 0
                               → depth_moge_aligned/depth_reference.png
  2. align_depth_to_reference_depth — ICP + affine solve for all frames
                               → depth_moge_aligned/

Run from reconstruction/:
    python experiments/electric_drill_toy_depth_alignment.py
"""

from v2d.foundation_pose.docker.run_align_depth_to_object import run_align_depth_to_object
from v2d.foundation_pose.docker.run_align_depth_to_reference_depth import run_align_depth_to_reference_depth

NAME      = "electric_drill_toy"
SESSION   = "Session_20260310_133326_f50"
OBJECT_ID = 1

MESH_PATH  = f"data/objects/{NAME}/mesh/textured_mesh.obj"
OUTPUT_DIR = f"data/objects/{NAME}/sessions/{SESSION}/outputs"
FP_WEIGHTS = "data/weights/foundation_pose"

frames_dir        = f"{OUTPUT_DIR}/frames"
masks_dir         = f"{OUTPUT_DIR}/masks/{OBJECT_ID}"
moge_depth_dir    = f"{OUTPUT_DIR}/depth_moge"
moge_intrinsics   = f"{OUTPUT_DIR}/intrinsics_moge_stable.json"
aligned_depth_dir = f"{OUTPUT_DIR}/depth_moge_aligned"

reference_frame     = 0
rgb_frame0          = f"{frames_dir}/{reference_frame:06d}.png"
depth_frame0        = f"{moge_depth_dir}/{reference_frame:06d}.png"
mask_frame0         = f"{masks_dir}/{reference_frame:06d}.png"
depth_reference_out = f"{aligned_depth_dir}/depth_reference.png"

# ---------------------------------------------------------------------------
# Step 1: Align reference frame depth to object mesh
# ---------------------------------------------------------------------------
print("Step 1: Aligning reference frame depth to object mesh...")
run_align_depth_to_object(
    mesh_path=MESH_PATH,
    rgb_path=rgb_frame0,
    depth_path=depth_frame0,
    mask_path=mask_frame0,
    intrinsics_path=moge_intrinsics,
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

# ---------------------------------------------------------------------------
# Step 2: Align all frames to reference depth via ICP + affine
# ---------------------------------------------------------------------------
print("Step 2: Aligning all frames to reference depth via ICP...")
run_align_depth_to_reference_depth(
    depth_folder=moge_depth_dir,
    depth_reference_path=depth_reference_out,
    intrinsics_path=moge_intrinsics,
    output_folder=aligned_depth_dir,
    masks_folder=masks_dir,
    reference_mask_path=mask_frame0,
    n_iterations=3,
    outlier_trim_ratio=0.2,
    max_points=20000,
)
print(f"Aligned depth written to {aligned_depth_dir}")
