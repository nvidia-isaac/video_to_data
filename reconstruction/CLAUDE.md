# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

**Install host-side packages (lightweight orchestration wrappers):**
```bash
# From reconstruction/
./scripts/install_pacakages.sh
# Or selectively:
pip install -e modules/v2d_moge/docker
```

**Build Docker images:**
```bash
# All modules (from reconstruction/):
./scripts/build_containers.sh

# Single module (from reconstruction/modules/):
python v2d_moge/docker/build.py
# Or: python -m v2d.moge.docker.build
```

**Download model weights (per module):**
```bash
python -c "from v2d.moge.docker.run_download_weights import run_download; run_download('data/weights/moge')"
```

**Run inference (programmatic):**
```python
from v2d.moge.docker.run_video_to_depth import run_video_to_depth
run_video_to_depth(
    video_path='modules/v2d_moge/assets/test_video.mp4',
    depth_folder='data/outputs/moge/depth',
    intrinsics_folder='data/outputs/moge/intrinsics',
    weights_path='data/weights/moge',
)
```

**Dev mode** (mounts local `/workspace` for live code editing without rebuilding):
```python
run_video_to_depth(..., dev=True)
```

**Run inference (CLI):**
```bash
python -m v2d.moge.docker.run_video_to_depth --video_path ... --depth_folder ... --intrinsics_folder ... --weights_path ...
```

**CI:** Triggered manually via `workflow_dispatch` on GitHub Actions (self-hosted GPU runner). Weights are cached by hash of `download_weights.py`.

## Design Philosophy

### Typed Contracts Between Packages

Packages communicate through **strongly typed dataclasses**, not raw arrays or third-party types. At any package boundary (function signatures, return values, file I/O), use types from `v2d_common` or the relevant shared package — never `np.ndarray`, `trimesh.Trimesh`, `open3d.geometry.PointCloud`, etc.

When wrapping a third-party library (trimesh, open3d, scipy, etc.), create a thin dataclass with `to_<lib>()` / `from_<lib>()` methods. Consumers work with our type; the dependency stays internal:

```python
# Good — typed boundary
@dataclass
class Mesh:
    vertices: np.ndarray  # internal representation is fine
    faces: np.ndarray

    def to_trimesh(self) -> trimesh.Trimesh: ...
    @staticmethod
    def from_trimesh(m: trimesh.Trimesh) -> 'Mesh': ...

# Bad — leaks third-party type across package boundary
def align_mesh(mesh: trimesh.Trimesh, ...) -> trimesh.Trimesh: ...
```

Raw numpy/torch arrays are acceptable **inside** a package (math, intermediate tensors) and at the boundaries of packages whose explicit purpose is tensor/array math. For all other packages, arrays should not appear in inter-package APIs.

### Separation of Concerns / Atomic Packages

Each package does one thing. If logic is needed by more than one package, it belongs in a shared package — not duplicated, not inlined as a private helper. The shared packages (`v2d_common`, `v2d_depth`, `v2d_pointcloud`, `v2d_mesh`, `v2d_smpl`) exist precisely for this: pull functionality up rather than copy it down.

Packages should stay narrow enough to be developed and tested in isolation. It's fine for functionality to live in the package where it first appears — but when a clear pattern of reuse emerges across packages, extract it into the appropriate shared package rather than duplicating it.

### Function Naming

Name functions starting with the primary data type they operate on, so the data flow is clear at a glance:

```python
# Good — primary data type is the subject
mesh_render_image(mesh, camera)
depth_to_pointcloud(depth, intrinsics)
pointcloud_align_icp(source, target)

# Avoid — verb-first obscures what the primary data is
render_mesh_image(mesh, camera)
convert_depth_to_pointcloud(depth, intrinsics)
```

### DRY by Structure

- Shared utility → shared package with a typed API
- Module-internal logic → fine to keep custom and unshared
- Avoid "util.py" grab-bags; group by concept, not by being miscellaneous

## Architecture

### Host / Container Split

The core design pattern: **host Python orchestrates, containers infer**.

- **Host installs** (`modules/v2d_*/docker/`): Zero ML dependencies. Expose `run_*()` functions that build and execute `docker run` commands. Path resolution, volume mounts, and output directory creation happen here.
- **Container installs** (`modules/v2d_*/lib/`): Heavy ML code (PyTorch, ONNX, model-specific deps). Never installed on the host.
- **`v2d_common`**: Shared datatypes used by both layers — `DepthImage`, `CameraIntrinsics`, `Transform3d`, `BoundingBox`, `Point`, `Mask`, plus `mv_config` (multi-camera rig configs, `CameraParam`, EDEX calibration). Installed on both host and in containers.
- **`v2d_mv`**: Shared multi-view utilities (no Docker image). Submodules: `v2d.mv.rig` (rig config), `v2d.mv.io` (video I/O), `v2d.mv.math` (torch/numpy math). Optional dependency groups: `[io]`, `[math]`, `[all]`. Installed inside MV-related containers as a dependency.

### Module Layout

Each module follows this structure:
```
v2d_<module>/
├── lib/               # Core ML inference (installed inside containers only)
│   ├── pyproject.toml # Heavy deps: torch, onnx, model-specific packages
│   └── *.py           # Inference logic, invoked via python -m v2d.<module>.lib.*
├── docker/
│   ├── Dockerfile     # pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel base
│   ├── build.py       # Builds image named v2d_<module>
│   ├── pyproject.toml # No dependencies; pure orchestration package
│   └── run_*.py       # Each exposes run_*() + CLI via __main__
└── assets/            # Test inputs (test_video.mp4, etc.)
```

### Docker Run Pattern

All `run_*.py` files follow the same pattern:
1. Resolve all paths to absolute
2. `os.makedirs(output_dir, exist_ok=True)`
3. Build `docker run --rm --gpus all --user $(uid):$(gid)` with volume mounts mapping host dirs → `/data/*` inside container
4. Pass `dev=True` to additionally mount `/workspace` (the `modules/` dir) for live development
5. `subprocess.run(cmd, check=True)`

### Wrapper Completeness

Every parameter of a `lib/` function must be reachable from both its `lib/run_*.py` wrapper (CLI + programmatic) and its `docker/run_*.py` wrapper (CLI + programmatic). When a lib function gains a new parameter, update both wrappers immediately — the docker wrapper passes it via `extra_args` or `inputs`, and the lib wrapper loads/resolves it and threads it through to the function call. Optional parameters with defaults are exposed as optional args in both wrappers.

### Data Flow

Modules communicate via files, not in-process objects. Outputs are written to folders:
- **Depth**: uint16 PNG (inverse-depth encoded: `pixel = 65535 * (1 / (depth_m + 1))`)
- **Intrinsics**: JSON files with `{fx, fy, cx, cy, width, height}`
- **Masks**: grayscale PNG
- **Poses**: `Transform3d` JSON `{rotation, translation, scale}` per frame (object-to-camera)
- **SMPL**: `.npz` files per frame named `{frame_id:06d}.npz`
- **Bounding box tracks**: `.pt` file (via `torch.save`) containing dict `{det_cat_id, scores, bbox_track}` with numpy arrays
- **Object bounding boxes (Grounding DINO)**: per-camera JSON with frame-stem-keyed detections (`{frame_stem: [{label, box}]}`)
- **Object prompt**: plain text file (`prompt.txt`) — extracted from `hoi_metadata.yaml` by preprocessing
- **Object poses**: `poses.npy` — `(N, 4, 4)` filtered SE(3) world-frame poses

That said, modules can add other module's lib as a direct python dependency to use in-memory utilities.

### Composing Pipelines

`v2d_pipelines` has no Docker image — it's a meta-package that imports and chains docker-layer `run_*` functions.

**`run_mv_hoi_reconstruction.py`** — full multi-view reconstruction pipeline:
```python
from v2d.rosbag.docker.run_rosbag_to_edex import run_rosbag_to_edex
from v2d.mv.preprocess.docker.run_mv_preprocess import run_mv_preprocess
from v2d.foundation_stereo.docker.run_mv_image_list_to_depth import run_mv_image_list_to_depth
from v2d.grounding_dino.docker.run_mv_image_list_to_object_bboxes import run_mv_image_list_to_object_bboxes
from v2d.detectron2.docker.run_mv_track_bboxes import run_mv_track_bboxes
from v2d.sam2.docker.run_mv_videos_to_masks import run_mv_videos_to_masks
from v2d.foundation_pose.docker.run_mv_videos_to_poses import run_mv_videos_to_poses
from v2d.sam3d_body.docker.run_mv_optimize_mhr_params import run_mv_optimize_mhr_params
from v2d.mv.postprocess.docker.run_mv_eval_chamfer_object import run_mv_eval_chamfer_object
from v2d.mv.postprocess.docker.run_mv_eval_chamfer_human import run_mv_eval_chamfer_human
from v2d.mv.postprocess.docker.run_mv_render_hoi_overlay import run_mv_render_hoi_overlay
from v2d.mv.postprocess.docker.run_mv_visualize_wis3d import run_mv_visualize_wis3d
```

**`run_mv_calibration.py`** — chessboard extrinsic calibration pipeline:
```python
from v2d.rosbag.docker.run_rosbag_to_edex import run_rosbag_to_edex
from v2d.mv.calibration.docker.run_calibrate_extrinsics import run_calibrate_extrinsics
```

### Modules at a Glance

| Module | Purpose |
|--------|---------|
| `v2d_common` | Shared datatypes: `DepthImage`, `CameraIntrinsics`, `Transform3d`, `BoundingBox`, `Mask` (no Docker) |
| `v2d_mv` | Multi-view shared utils: rig config (`v2d.mv.rig`), video I/O (`v2d.mv.io`), math (`v2d.mv.math`). Optional deps: `[io]`, `[math]`, `[all]` (no Docker) |
| `v2d_rosbag` | ROS bag extraction → EDEX images + intrinsics |
| `v2d_mv_preprocess` | MV stereo rectification, rescaling, video encoding, HOI bbox remap, prompt extraction (no Docker — shares `v2d_rosbag` or own image) |
| `v2d_mv_calibration` | Chessboard extrinsic calibration: correspondences → PnP → Ceres bundle adjustment |
| `v2d_mv_postprocess` | HOI overlay rendering, Wis3D 3D visualization, chamfer distance evaluation (object + human) |
| `v2d_detectron2` | Person detection + IoU tracking from images/video (Detectron2 ViTDet). MV: `run_mv_track_bboxes` |
| `v2d_moge` | Monocular depth + camera intrinsics from video (MoGe model) |
| `v2d_unidepth` | Monocular depth estimation (UniDepth model) |
| `v2d_anycalib` | Single-view camera calibration (intrinsics + distortion) and undistortion for image / video / image folder (AnyCalib model) |
| `v2d_sam2` | Video segmentation with interactive annotation UI. MV: `run_mv_videos_to_masks` (bbox track `.pt` or grounding dino `.json`) |
| `v2d_sam3d` | 3D mesh reconstruction from image + mask |
| `v2d_sam3d_body` | Human body pose and shape estimation (SAM3D-Body MHR). MV: `run_mv_optimize_mhr_params` |
| `v2d_grounding_dino` | Text-guided object detection → bounding boxes. MV: `run_mv_image_list_to_object_bboxes` (reads prompt from `prompt.txt`) |
| `v2d_mediapipe` | Hand bbox + handedness from a single image (MediaPipe Hands; CPU; Apache 2.0) |
| `v2d_hamer` | Per-frame MANO hand reconstruction (HaMeR ViT) driven by SAM2 hand masks (no internal detector). Verification overlay renderer. MIT |
| `v2d_foundation_stereo` | Stereo depth estimation from left/right image pairs. MV: `run_mv_image_list_to_depth` |
| `v2d_foundation_pose` | 6D pose tracking + mesh alignment/simplification. MV: `run_mv_videos_to_poses` (shared-weight multi-view tracker) |
| `v2d_nlf` | SMPL body model estimation (Neural Layered Fields) |
| `v2d_cusfm` | Structure-from-motion: stereo image list → camera poses |
| `v2d_bundlesdf` | SDF learning + texture baking from pre-computed poses, depth, and masks |
| `v2d_pipelines` | End-to-end pipelines: `run_mv_hoi_reconstruction`, `run_mv_calibration` (no Docker) |

### Multi-View Config Pattern

Multi-camera programs live in `lib/` as `mv_*.py` files. Each has a same-named `.yaml` config alongside it (e.g. `mv_track_bboxes.py` + `mv_track_bboxes.yaml`).

**YAML config:**
- Pre-populated fields have sensible defaults; unpopulated fields use `???` (required) or `null` (optional)
- Path templates use OmegaConf interpolation for directory roots (`${output_dir}/...`) and Python `str.format()` for per-camera expansion (`{cam_name}`)
- Config specifies `rig_config` (or `rig_name`) to select a rig YAML from `v2d_mv/rig/rigs/`

**CLI / `__main__` block:**
- Only accept unpopulated fields (`???` and `null`) plus `--config_path` and `--debug` (if applicable) as CLI arguments
- Load the default YAML, merge CLI overrides via `OmegaConf.merge`, then call `*_from_config(cfg)`

**`*_from_config` function:**
- Takes only `cfg` (no `rig` parameter); creates the `RigConfig` internally
- For simple programs (e.g. running a model on each camera independently): all logic can live directly in `*_from_config`
- For complex programs (e.g. `mv_preprocess`): `*_from_config` resolves config fields into concrete arguments (paths, dicts, etc.) and calls a main function whose signature has all parameters expanded out. The main function has no config/YAML awareness

**`RigConfig`:**
- Central carrier of camera topology and `CameraParam` objects
- Handles format-dispatched load/save/merge of camera params (e.g. EDEX by index, future formats by name)
- `*_from_config` resolves path templates per camera using `rig.get_camera()` / `rig.get_stereo_pairs()`

**Reference examples:**
- Simple: `mv_track_bboxes.py` — logic lives in `*_from_config`, iterates over cameras and calls `track_bboxes` per camera
- Complex: `mv_preprocess.py` — `*_from_config` resolves templates into dicts, calls `mv_preprocess()` which has a full expanded signature

### CUDA Targets

Dockerfiles build native extensions for `TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"` (Ampere and Hopper). Modules with native CUDA extensions (pybind11/nvdiffrast/pytorch3d/kaolin): `v2d_sam3d`, `v2d_sam3d_body`, `v2d_foundation_pose`.

### Active Refactor (branch: `jwelsh/refactor-shared-packages`)

Extracting shared standalone packages with typed APIs:
- `v2d_depth` — Depth filtering, intrinsics utilities
- `v2d_pointcloud` — Point cloud ops (ICP, conversions)
- `v2d_mesh` — Mesh I/O, alignment, simplification
- `v2d_smpl` — SMPL body pose/shape modeling

Dependency order: `v2d_common → v2d_depth / v2d_pointcloud → v2d_mesh → v2d_smpl`
