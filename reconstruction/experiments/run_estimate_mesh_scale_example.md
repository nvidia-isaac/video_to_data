# Mesh Scale Estimation (FoundationPose)

Given a mesh and a single reference frame (RGB + depth + mask + intrinsics), finds the scale factor that best aligns the mesh to the observed metric depth via a coarse-to-fine grid search.

## Setup

Run from `reconstruction/`:

```bash
# 1. Install host-side orchestration packages
pip install -e modules/v2d_foundation_pose/docker
pip install -e modules/v2d_common

# 2. Build the Docker image (contains PyTorch + FP native extensions)
python modules/v2d_foundation_pose/docker/build.py

# 3. Download FP weights (~1-2 GB)
python -c "
from v2d.foundation_pose.docker.run_download_weights import run_download
run_download('data/weights/foundation_pose')
"
```

## Example Script

```python
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale

MESH_PATH       = "data/objects/my_object/mesh/textured_mesh.obj"
RGB_PATH        = "data/objects/my_object/sessions/session_1/frames/000000.png"
DEPTH_PATH      = "data/objects/my_object/sessions/session_1/depth/000000.png"
MASK_PATH       = "data/objects/my_object/sessions/session_1/masks/1/000000.png"
INTRINSICS_PATH = "data/objects/my_object/sessions/session_1/intrinsics_stable.json"
FP_WEIGHTS      = "data/weights/foundation_pose"

SCALE_PATH         = "data/objects/my_object/sessions/session_1/outputs/mesh_scale.json"
RESCALED_MESH_PATH = "data/objects/my_object/sessions/session_1/outputs/mesh_scaled.obj"

scale = run_estimate_mesh_scale(
    mesh_path=MESH_PATH,
    rgb_path=RGB_PATH,
    depth_path=DEPTH_PATH,
    mask_path=MASK_PATH,
    intrinsics_path=INTRINSICS_PATH,
    weights_dir=FP_WEIGHTS,
    scale_path=SCALE_PATH,
    rescaled_mesh_path=RESCALED_MESH_PATH,  # omit if you only want the JSON
    # Search range (relative to original mesh scale):
    lo=0.5,
    hi=2.0,
    # Coarse-to-fine grid search config:
    n_samples=7,
    n_levels=3,
    # Scoring weights (iou + depth are usually sufficient):
    iou_weight=0.0,
    depth_weight=1.0,
    chamfer_weight=0.0,
    # FP register() iterations per candidate (lower = faster):
    registration_iterations=5,
)

print(f"Best scale factor: {scale:.4f}")
print(f"Scale saved to:    {SCALE_PATH}")
print(f"Rescaled mesh:     {RESCALED_MESH_PATH}")
```

## Inputs

| Parameter | Format |
|---|---|
| `mesh_path` | `.obj` (or other trimesh-readable format) |
| `rgb_path` | PNG, single reference frame |
| `depth_path` | uint16 PNG (inverse-depth encoding: `pixel = 65535 * (1 / (depth_m + 1))`) |
| `mask_path` | Grayscale PNG segmentation mask |
| `intrinsics_path` | JSON `{fx, fy, cx, cy, width, height}` |

## Outputs

- `scale_path` — JSON `{"scale": <float>}` with the best scale factor relative to the original mesh
- `rescaled_mesh_path` — rescaled mesh file (optional; omit the argument to skip)

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lo` | `0.5` | Lower bound of initial scale search range |
| `hi` | `2.0` | Upper bound of initial scale search range |
| `n_samples` | `7` | Candidate scales evaluated per refinement level |
| `n_levels` | `3` | Number of coarse-to-fine refinement levels |
| `iou_weight` | `1.0` | Weight for mask IoU score component |
| `depth_weight` | `1.0` | Weight for depth consistency score component |
| `chamfer_weight` | `0.0` | Weight for Chamfer distance component (disabled by default) |
| `registration_iterations` | `5` | FP `register()` iterations per candidate (lower = faster) |
