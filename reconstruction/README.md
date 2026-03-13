# Video-to-Policy Reconstruction Modules

Docker-based modules for video reconstruction, depth estimation, object detection, segmentation, mesh generation, pose tracking, and human body modeling.

## Packages & Tools (Summary)

| Package | Tools | Description | Build | Execute |
|---------|-------|-------------|-------|---------|
| **v2d_unidepth** | `run_image_to_depth`, `run_video_to_depth`, `run_download_weights`, `run_shell` | Monocular depth estimation | `cd reconstruction/modules/v2d_unidepth/docker && python build.py` | `python -m v2d.unidepth.docker.run_<tool> --args` |
| **v2d_moge** | `run_image_to_depth`, `run_video_to_depth`, `run_download_weights`, `run_shell` | Video-to-depth (Midas + MoG) | `cd reconstruction/modules/v2d_moge/docker && python build.py` | `python -m v2d.moge.docker.run_<tool> --args` |
| **v2d_sam2** | `run_video_to_masks`, `run_annotate`, `run_download_weights`, `run_shell` | SAM2 video segmentation | `cd reconstruction/modules/v2d_sam2/docker && python build.py` | `python -m v2d.sam2.docker.run_<tool> --args` |
| **v2d_sam3d** | `run_image_to_mesh`, `run_render_debug_image`, `run_download_weights`, `run_shell` | 3D mesh from image+mask | `cd reconstruction/modules/v2d_sam3d/docker && python build.py` | `python -m v2d.sam3d.docker.run_<tool> --args` |
| **v2d_grounding_dino** | `run_image_to_object_bboxes`, `run_image_list_to_object_bboxes`, `run_video_to_object_bboxes`, `run_download_weights`, `run_shell` | Text-guided object detection | `cd reconstruction/modules/v2d_grounding_dino/docker && python build.py` | `python -m v2d.grounding_dino.docker.run_<tool> --args` |
| **v2d_foundation_stereo** | `run_image_to_depth`, `run_image_list_to_depth`, `run_export_engine`, `run_download_weights`, `run_shell` | Stereo depth (left/right pairs) | `cd reconstruction/modules/v2d_foundation_stereo/docker && python build.py` | `python -m v2d.foundation_stereo.docker.run_<tool> --args` |
| **v2d_foundation_pose** | `run_video_to_poses`, `run_render_overlay`, `run_estimate_scale`, `run_align_mesh_scale`, `run_transform_mesh`, `run_simplify_mesh`, `run_download_weights`, `run_shell` | 6D pose tracking, mesh ops | `cd reconstruction/modules/v2d_foundation_pose/docker && python build.py` | `python -m v2d.foundation_pose.docker.run_<tool> --args` |
| **v2d_nlf** | `run_video_to_smpl`, `run_render_smpl_overlay`, `run_render_smpl_depth`, `run_align_depth_to_smpl`, `run_align_nlf_to_depth`, `run_download_weights`, `run_shell` | Video → SMPL body model | `cd reconstruction/modules/v2d_nlf/docker && python build.py` | `python -m v2d.nlf.docker.run_<tool> --args` |

---

## Modules & Tools

### v2d_unidepth

Monocular depth estimation using UniDepth.

| Tool | Function | Description |
|------|----------|-------------|
| `run_image_to_depth` | `run_image_to_depth(image_path, depth_path, intrinsics_path, weights_path, dev=False)` | Estimate depth and camera intrinsics from a single image |
| `run_video_to_depth` | `run_video_to_depth(video_path, depth_folder, intrinsics_folder, weights_path, dev=False)` | Estimate depth and intrinsics for each video frame |
| `run_download_weights` | `run_download(output_dir, dev=False)` | Download UniDepth model weights |
| `run_shell` | `run_shell(dev=False)` | Interactive bash shell in container |

**Build:** `cd reconstruction/modules/v2d_unidepth/docker && python build.py`  
**Execute:** `python -m v2d.unidepth.docker.run_image_to_depth --image_path ... --depth_path ... --intrinsics_path ... --weights_path ...`

---

### v2d_moge

Video-to-depth using Midas with Grounded MoG prior.

| Tool | Function | Description |
|------|----------|-------------|
| `run_image_to_depth` | `run_image_to_depth(image_path, depth_path, intrinsics_path, weights_path, dev=False)` | Single image to depth map |
| `run_video_to_depth` | `run_video_to_depth(video_path, depth_folder, intrinsics_folder, weights_path, dev=False)` | Video to per-frame depth + intrinsics |
| `run_download_weights` | `run_download(output_dir, dev=False)` | Download MoGE model weights |
| `run_shell` | `run_shell(dev=False)` | Interactive bash shell in container |

**Build:** `cd reconstruction/modules/v2d_moge/docker && python build.py`  
**Execute:** `python -m v2d.moge.docker.run_video_to_depth --video_path ... --depth_folder ... --intrinsics_folder ... --weights_path ...`

---

### v2d_sam2

Segment Anything Model 2 for video segmentation.

| Tool | Function | Description |
|------|----------|-------------|
| `run_video_to_masks` | `run_video_to_masks(video_path, prompts_path, masks_dir, weights_dir, dev=False)` | Generate masks from video using prompts JSON |
| `run_annotate` | `run_annotate(video_path, prompts_path, port=8080, dev=False)` | Web UI to annotate video and save prompts JSON |
| `run_download_weights` | `run_download(output_dir, dev=False)` | Download SAM2 model weights |
| `run_shell` | `run_shell(dev=False)` | Interactive bash shell in container |

**Build:** `cd reconstruction/modules/v2d_sam2/docker && python build.py`  
**Execute:** `python -m v2d.sam2.docker.run_video_to_masks --video_path ... --prompts_path ... --masks_dir ... --weights_dir ...`

---

### v2d_sam3d

3D mesh reconstruction from single images with masks.

| Tool | Function | Description |
|------|----------|-------------|
| `run_image_to_mesh` | `run_image_to_mesh(image_path, mask_path, mesh_path, transform_path, intrinsics_path, weights_dir, ...)` | Generate 3D mesh (GLB), transform, and intrinsics from image+mask |
| `run_render_debug_image` | `run_render_debug_image(image_path, mesh_path, transform_path, intrinsics_path, output_image_path, ...)` | Render mesh overlay for debugging |
| `run_download_weights` | `run_download(output_dir, dev=False)` | Download SAM3D model weights |
| `run_shell` | `run_shell(dev=False)` | Interactive bash shell in container |

**Build:** `cd reconstruction/modules/v2d_sam3d/docker && python build.py`  
**Execute:** `python -m v2d.sam3d.docker.run_image_to_mesh --image_path ... --mask_path ... --mesh_path ... --transform_path ... --intrinsics_path ... --weights_dir ...`

---

### v2d_grounding_dino

Text-guided object detection using Grounding DINO.

| Tool | Function | Description |
|------|----------|-------------|
| `run_image_to_object_bboxes` | `run_image_to_object_bboxes(image_path, output_path, prompt, model_dir, ...)` | Detect objects in a single image by text prompt |
| `run_image_list_to_object_bboxes` | `run_image_list_to_object_bboxes(image_dir, output_path, prompt, model_dir, ...)` | Batch object detection on image directory |
| `run_video_to_object_bboxes` | `run_video_to_object_bboxes(video_path, output_path, prompt, model_dir, ...)` | Per-frame object detection on video |
| `run_download_weights` | `run_download(output_dir, dev=False)` | Download Grounding DINO model weights |
| `run_shell` | `run_shell(dev=False)` | Interactive bash shell in container |

**Build:** `cd reconstruction/modules/v2d_grounding_dino/docker && python build.py`  
**Execute:** `python -m v2d.grounding_dino.docker.run_image_to_object_bboxes --image_path ... --output_path ... --prompt "person" --model_dir ...`

---

### v2d_foundation_stereo

Stereo depth estimation from left/right image pairs.

| Tool | Function | Description |
|------|----------|-------------|
| `run_image_to_depth` | `run_image_to_depth(left_image_path, right_image_path, depth_path, intrinsics_path, model_dir, calibration_file|fx,fy,cx,cy,baseline, ...)` | Single stereo pair → depth map |
| `run_image_list_to_depth` | `run_image_list_to_depth(left_dir, right_dir, depth_folder, intrinsics_folder, model_dir, ...)` | Batch stereo pairs → depth maps |
| `run_export_engine` | `run_export_engine(model_dir, dev=False)` | Export ONNX model to TensorRT engine |
| `run_download_weights` | `run_download(output_dir, dev=False)` | Download Foundation Stereo model weights |
| `run_shell` | `run_shell(dev=False)` | Interactive bash shell in container |

**Build:** `cd reconstruction/modules/v2d_foundation_stereo/docker && python build.py`  
**Execute:** `python -m v2d.foundation_stereo.docker.run_image_to_depth --left_image_path ... --right_image_path ... --depth_path ... --intrinsics_path ... --model_dir ... --calibration_file ...`

---

### v2d_foundation_pose

6D object pose estimation and mesh alignment from video.

| Tool | Function | Description |
|------|----------|-------------|
| `run_video_to_poses` | `run_video_to_poses(video_path, depth_folder, masks_folder, camera_intrinsics_path, mesh_path, poses_dir, weights_dir, ...)` | Track object pose per frame using mesh, depth, masks |
| `run_render_overlay` | `run_render_overlay(video_path, poses_dir, mesh_path, camera_intrinsics_path, output_dir, dev=False)` | Render mesh overlay on video given poses |
| `run_estimate_scale` | `run_estimate_scale(mesh_path, rgb_path, depth_path, mask_path, intrinsics_path, transform_path, output_transform_path, weights_dir, ...)` | Estimate mesh scale from RGBD alignment |
| `run_align_mesh_scale` | `run_align_mesh_scale(mesh_path, depth_path, mask_path, intrinsics_path, transform_path, output_transform_path, dev=False)` | Align mesh scale to depth map |
| `run_transform_mesh` | `run_transform_mesh(input_mesh, output_mesh, transform_path, dev=False)` | Apply transform matrix to mesh |
| `run_simplify_mesh` | `run_simplify_mesh(input_mesh, output_mesh, faces=None, factor=None, dev=False)` | Simplify mesh (reduce polygon count) |
| `run_download_weights` | `run_download(output_dir, dev=False)` | Download FoundationPose model weights |
| `run_shell` | `run_shell(dev=False)` | Interactive bash shell in container |

**Build:** `cd reconstruction/modules/v2d_foundation_pose/docker && python build.py`  
**Execute:** `python -m v2d.foundation_pose.docker.run_video_to_poses --video_path ... --depth_folder ... --masks_folder ... --camera_intrinsics_path ... --mesh_path ... --poses_dir ... --weights_dir ...`

---

### v2d_nlf

Neural layered field: video to SMPL body parameters.

| Tool | Function | Description |
|------|----------|-------------|
| `run_video_to_smpl` | `run_video_to_smpl(video_path, masks_dir, intrinsics_path, gender, output_path, weights_dir, model_type="smplh", chunk_size=32, dev=False)` | Extract SMPL body parameters from video |
| `run_render_smpl_overlay` | `run_render_smpl_overlay(video_path, smpl_params_path, intrinsics_path, output_dir, weights_dir, dev=False)` | Render SMPL body overlay on video |
| `run_render_smpl_depth` | `run_render_smpl_depth(smpl_params_path, intrinsics_path, output_depth_folder, output_mask_folder, weights_dir, dev=False)` | Render SMPL as depth and mask maps |
| `run_align_depth_to_smpl` | `run_align_depth_to_smpl(depth_folder, smpl_depth_folder, output_depth_folder, masks_folder, smpl_masks_folder=None, dev=False)` | Align scene depth to SMPL depth |
| `run_align_nlf_to_depth` | `run_align_nlf_to_depth(smpl_results_path, depth_folder, masks_dir, intrinsics_path, output_path, weights_dir, dev=False)` | Align NLF/SMPL results to depth |
| `run_download_weights` | `run_download(output_dir, dev=False)` | Download NLF and SMPL weights |
| `run_shell` | `run_shell(dev=False)` | Interactive bash shell in container |

**Build:** `cd reconstruction/modules/v2d_nlf/docker && python build.py`  
**Execute:** `python -m v2d.nlf.docker.run_video_to_smpl --video_path ... --masks_dir ... --intrinsics_path ... --gender male --output_path ... --weights_dir ...`

---

## Build & Execute (Summary)

All modules share the same build pattern. Each Dockerfile uses `reconstruction/modules` as build context (parent of each `v2d_*` folder).

| Action | Command |
|--------|---------|
| **Build** | `cd reconstruction/modules/<module>/docker && python build.py` |
| **Execute via Python** | `python -m v2d.<module>.docker.run_<tool> --arg1 ... --arg2 ...` |
| **Execute via CLI** | Same as above when run as main in each `run_*.py` |
| **Dev mode** | Add `--dev` to mount local modules at `/workspace` |

Example pipeline usage (see `reconstruction/example_pipeline.py`):

```python
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
from v2d.moge.docker.run_video_to_depth import run_video_to_depth
# ... etc.
```

---

## Dependencies

- Docker with GPU support (`--gpus all`)
- NVIDIA Container Toolkit
- Weights must be downloaded per-module via `run_download_weights` before first use

---

## Contributing

Rough guidelines for adding new packages to the reconstruction pipeline:

### 1. Module layout

Create a new module under `reconstruction/modules/`:

```
v2d_<name>/
├── lib/                    # Core logic (Python package)
│   ├── __init__.py
│   ├── <tool>.py           # One module per tool/entry point
│   └── download_weights.py # If model weights are needed
└── docker/
    ├── Dockerfile
    ├── build.py
    ├── run_<tool>.py       # One wrapper per lib entry point
    ├── run_download_weights.py  # If applicable
    └── run_shell.py
```

### 2. Docker conventions

- **Build context:** Use `reconstruction/modules` (parent of `v2d_*`) as the Docker build context so `v2d_datatypes` and sibling modules are available.
- **Image name:** `v2d_<name>` (matches folder).
- **Base image:** Use `pytorch/pytorch` variants for GPU workloads.
- **Install:** `pip install -e /workspace/v2d_datatypes -e /workspace/v2d_<name>/lib` (or equivalent).

### 3. Run scripts

Each `run_*.py` should:

- Accept `dev: bool = False` to mount `reconstruction/modules` at `/workspace`.
- Use `--user $(id -u):$(id -g)` and `--gpus all` where appropriate.
- Use `os.path.abspath()` for all host paths before constructing volume mounts.
- Include an `if __name__ == "__main__"` block with `argparse` for CLI use.
- Expose a single callable (e.g. `run_<tool>`) for programmatic use.

### 4. Python package naming

- Folder: `v2d_snake_case` (e.g. `v2d_grounding_dino`).
- Python module: `v2d.snake_case` (e.g. `v2d.grounding_dino`).
- Ensure the lib is installable with `pip install -e .` (or via the Dockerfile).

### 5. Checklist

- [ ] Add `run_download_weights` if the module needs external model weights.
- [ ] Add `run_shell` for debugging (interactive `bash` in the container).
- [ ] Update this README: add the package to the summary table and create a detailed section.
- [ ] Add any shared types to `v2d_datatypes` if other modules will consume them.
