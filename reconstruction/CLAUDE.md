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
- **`v2d_io`**, **`v2d_math`**: Shared container-side libraries (no Docker images). Installed inside other modules' containers as dependencies.

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

That said, modules can add other module's lib as a direct python dependency to use in-memory utilities.

### Composing Pipelines

Pipeline scripts live in `experiments/` and compose docker-layer `run_*` functions directly:
```python
from v2d.detectron2.docker.run_mv_track_bboxes import run_mv_track_bboxes
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.moge.docker.run_video_to_depth import run_video_to_depth
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
```

Video utilities (`extract_images`, `frames_to_video`, `stitch_videos`) live in `v2d.common.utils`.

### Modules at a Glance

| Module | Purpose |
|--------|---------|
| `v2d_common` | Shared datatypes + video utils (no Docker) |
| `v2d_io` | Shared I/O: `FrameSource`, video read/write, tiling (no Docker) |
| `v2d_math` | Shared torch math: projective geometry, rotations (no Docker) |
| `v2d_detectron2` | Person detection + IoU tracking from images/video (Detectron2 ViTDet) |
| `v2d_moge` | Monocular depth + camera intrinsics from video (MoGe model) |
| `v2d_unidepth` | Monocular depth estimation (UniDepth model) |
| `v2d_depth_anything` | Monocular depth estimation (Depth Anything V3 model) |
| `v2d_sam2` | Video segmentation with interactive annotation UI |
| `v2d_sam3d` | 3D mesh reconstruction from image + mask |
| `v2d_grounding_dino` | Text-guided object detection → bounding boxes |
| `v2d_foundation_stereo` | Stereo depth estimation from left/right image pairs |
| `v2d_foundation_pose` | 6D pose tracking + mesh alignment/simplification |
| `v2d_nlf` | SMPL body model estimation (Neural Layered Fields) |
| `v2d_cusfm` | Structure-from-motion: stereo image list → camera poses |
| `v2d_bundlesdf` | SDF learning + texture baking from pre-computed poses, depth, and masks |

### Multi-View Config Pattern

Multi-camera modules (e.g. `v2d_detectron2`) use OmegaConf-based `<mv_config>.yaml` files to define rig layout, path templates, and per-module settings. The config uses `???` placeholders for required paths (`weights_dir`, `output_dir`) that are filled at runtime via CLI overrides merged with `OmegaConf.merge`. Path templates like `${output_dir}/{cam_name}_bbox_track.pt` use OmegaConf interpolation for directory roots and Python `str.format()` for per-camera expansion. The `RigConfig` class (from `v2d_common.mv_config`) loads camera topology from YAML files in `v2d_common/mv_config/rigs/`.

### CUDA Targets

Dockerfiles build native extensions for `TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"` (Ampere and Hopper). Modules with native CUDA extensions (pybind11/nvdiffrast/pytorch3d/kaolin): `v2d_sam3d`, `v2d_foundation_pose`.

### Active Refactor (branch: `jwelsh/refactor-shared-packages`)

Extracting shared standalone packages with typed APIs:
- `v2d_depth` — Depth filtering, intrinsics utilities
- `v2d_pointcloud` — Point cloud ops (ICP, conversions)
- `v2d_mesh` — Mesh I/O, alignment, simplification
- `v2d_smpl` — SMPL body pose/shape modeling

Dependency order: `v2d_common → v2d_depth / v2d_pointcloud → v2d_mesh → v2d_smpl`
