# Video-to-Policy Reconstruction Modules

Docker-based modules for video reconstruction, depth estimation, object detection, segmentation, mesh generation, pose tracking, and human body modeling.

## Quickstart (MOGE)

Minimal example using MOGE. From `reconstruction/`:

```bash
# One-time setup
pip install -e modules/v2d_moge/docker
python -m v2d.moge.docker.build

# 1. Download weights (~1.5GB)
python -m v2d.moge.docker.run_download_weights --output_dir data/weights/moge

# 2. Run video→depth (repo includes sample at modules/v2d_moge/assets/test_video.mp4)
python -m v2d.moge.docker.run_video_to_depth \
  --video_path modules/v2d_moge/assets/test_video.mp4 \
  --depth_folder data/outputs/moge/depth \
  --intrinsics_folder data/outputs/moge/intrinsics \
  --weights_path data/weights/moge
```

Equivalent pipeline using Python imports (run from `reconstruction/`):

```python
from v2d.moge.docker.run_download_weights import run_download
from v2d.moge.docker.run_video_to_depth import run_video_to_depth

run_download(output_dir="data/weights/moge")
run_video_to_depth(
    video_path="modules/v2d_moge/assets/test_video.mp4",
    depth_folder="data/outputs/moge/depth",
    intrinsics_folder="data/outputs/moge/intrinsics",
    weights_path="data/weights/moge",
)
```

**Output:** Per-frame depth maps in `data/outputs/moge/depth/`, intrinsics in `data/outputs/moge/intrinsics/`. The repo includes sample data at `modules/v2d_moge/assets/test_video.mp4`.

---

## Packages & Tools (Summary)

| Package | Tools | Description | Build | Execute |
|---------|-------|-------------|-------|---------|
| **v2d_unidepth** | `run_image_to_depth`, `run_video_to_depth`, `run_download_weights`, `run_shell` | Monocular depth estimation | `python -m v2d.unidepth.docker.build` | `python -m v2d.unidepth.docker.run_<tool> --args` |
| **v2d_moge** | `run_image_to_depth`, `run_video_to_depth`, `run_download_weights`, `run_shell` | Video-to-depth (Midas + MoG) | `python -m v2d.moge.docker.build` | `python -m v2d.moge.docker.run_<tool> --args` |
| **v2d_sam2** | `run_video_to_masks`, `run_annotate`, `run_download_weights`, `run_shell` | SAM2 video segmentation | `python -m v2d.sam2.docker.build` | `python -m v2d.sam2.docker.run_<tool> --args` |
| **v2d_sam3d** | `run_image_to_mesh`, `run_render_debug_image`, `run_download_weights`, `run_shell` | 3D mesh from image+mask | `python -m v2d.sam3d.docker.build` | `python -m v2d.sam3d.docker.run_<tool> --args` |
| **v2d_grounding_dino** | `run_image_to_object_bboxes`, `run_image_list_to_object_bboxes`, `run_video_to_object_bboxes`, `run_download_weights`, `run_shell` | Text-guided object detection | `python -m v2d.grounding_dino.docker.build` | `python -m v2d.grounding_dino.docker.run_<tool> --args` |
| **v2d_foundation_stereo** | `run_image_to_depth`, `run_image_list_to_depth`, `run_export_engine`, `run_download_weights`, `run_shell` | Stereo depth (left/right pairs) | `python -m v2d.foundation_stereo.docker.build` | `python -m v2d.foundation_stereo.docker.run_<tool> --args` |
| **v2d_foundation_pose** | `run_video_to_poses`, `run_render_overlay`, `run_estimate_scale`, `run_align_mesh_scale`, `run_transform_mesh`, `run_simplify_mesh`, `run_download_weights`, `run_shell` | 6D pose tracking, mesh ops | `python -m v2d.foundation_pose.docker.build` | `python -m v2d.foundation_pose.docker.run_<tool> --args` |
| **v2d_nlf** | `run_video_to_smpl`, `run_render_smpl_overlay`, `run_render_smpl_depth`, `run_align_depth_to_smpl`, `run_align_nlf_to_depth`, `run_download_weights`, `run_shell` | Video → SMPL body model | `python -m v2d.nlf.docker.build` | `python -m v2d.nlf.docker.run_<tool> --args` |
| **v2d_hoi_object_reconstruction** | `run_reconstruction`, `run_fp_tracking` | End-to-end textured mesh reconstruction from hand-object interaction video (two-stage scan) | `python v2d_hoi_object_reconstruction/docker/build.py` | `python v2d_hoi_object_reconstruction/docker/run_reconstruction.py --args` |
| **v2d_cusfm** | `run_image_list_to_sfm` | Structure-from-motion: stereo image list → camera poses | `python v2d_cusfm/docker/build.py` | `python v2d_cusfm/docker/run_image_list_to_sfm.py --input_dir ... --output_dir ...` |
| **v2d_bundlesdf** | `run_reconstruct`, `run_download_weights` | SDF learning + texture baking from pre-computed poses, depth, and masks | `python v2d_bundlesdf/docker/build.py` | `python v2d_bundlesdf/docker/run_reconstruct.py --output_path ... --weights_dir ...` |
| **v2d_detectron2** | `run_track_bboxes`, `run_mv_track_bboxes`, `run_download_weights`, `run_shell` | Person detection + IoU tracking (ViTDet) | `python -m v2d.detectron2.docker.build` | `python -m v2d.detectron2.docker.run_<tool> --args` |
| **v2d_sam3d_body** | `run_estimate_mhr_params`, `run_mv_optimize_mhr_params`, `run_mv_render_mhr_mesh`, `run_download_weights`, `run_shell` | Human body pose & shape (SAM3D-Body MHR) | `python -m v2d.sam3d_body.docker.build` | `python -m v2d.sam3d_body.docker.run_<tool> --args` |

**Shared packages** (no Docker images — installed inside other modules' containers as dependencies):

| Package | Description |
|---------|-------------|
| **v2d_common** | Shared datatypes (`DepthImage`, `CameraIntrinsics`, `Transform3d`, `BoundingBox`, `Mask`, etc.) |
| **v2d_mv** | Multi-view utilities: rig config (`v2d.mv.rig`), video I/O (`v2d.mv.io`), math (`v2d.mv.math`). Optional deps: `[io]`, `[math]`, `[all]` |

---

## Setup

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with GPU support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Python 3.10+

### 1. Install the Docker orchestration packages

Each module exposes a **docker** package (lightweight Python wrappers that build and run containers).

**Quick install (all packages):**

```bash
cd reconstruction
./scripts/install_pacakages.sh
```

Install only the modules you need:

```bash
# From the repo root
cd reconstruction

# Install docker packages for the modules you want to use
pip install -e modules/v2d_sam3d/docker
pip install -e modules/v2d_sam3d_body/docker
pip install -e modules/v2d_moge/docker
pip install -e modules/v2d_sam2/docker
pip install -e modules/v2d_foundation_pose/docker
pip install -e modules/v2d_nlf/docker
pip install -e modules/v2d_detectron2/docker
# ... and/or: v2d_unidepth, v2d_grounding_dino, v2d_foundation_stereo
```

Or install all docker packages at once (including the example pipeline):

```bash
# From reconstruction/ - install all docker packages + v2d_pipelines in one command
pip install -e modules/v2d_sam2/docker -e modules/v2d_sam3d/docker -e modules/v2d_sam3d_body/docker \
  -e modules/v2d_unidepth/docker -e modules/v2d_moge/docker -e modules/v2d_nlf/docker \
  -e modules/v2d_foundation_pose/docker -e modules/v2d_foundation_stereo/docker \
  -e modules/v2d_grounding_dino/docker -e modules/v2d_cusfm/docker -e modules/v2d_bundlesdf/docker \
  -e modules/v2d_hoi_object_reconstruction/docker -e modules/v2d_detectron2/docker \
  -e modules/v2d_pipelines
```

This installs `v2d-pipelines` and all docker packages (v2d_pipelines declares them as dependencies). Alternatively:

```bash
for d in modules/v2d_*/docker; do pip install -e "$d"; done
```

### 2. Build Docker images

Each module has its own image.

**Build all images:**

```bash
cd reconstruction
./build_containers.sh
```

Or build individually (run from any directory after install):

```bash
python -m v2d.sam3d.docker.build
python -m v2d.moge.docker.build
# ... repeat for each module you use
```

### 3. Download weights

Run `run_download_weights` for each module that requires model weights (e.g. via `python -m v2d.sam3d.docker.run_download_weights --output_dir data/weights/sam3d`).

### Design pattern: host orchestration, containerized inference

This project separates **orchestration** (run on the host) from **inference** (run inside Docker):

| Layer | Location | Role |
|-------|----------|------|
| **Docker package** | Host (`pip install -e modules/v2d_*/docker`) | Thin Python wrappers that construct `docker run` commands, mount volumes, and invoke the container. No heavy ML deps on the host. |
| **Lib package** | Container (installed in Dockerfile) | Actual inference code (PyTorch, ONNX, etc.) and model logic. Runs only inside the built image. |

**Why `pip install -e <module>/docker`?**

- The `docker/` folder is a **pip-installable package** (`v2d.<module>.docker`) that exposes callables like `run_image_to_mesh`, `run_video_to_depth`, etc.
- Installing it makes `from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh` work, so you can compose pipelines in Python.
- The package is lightweight (subprocess, paths, argparse) and has no ML dependencies. It simply spawns containers.
- `build` is part of the same package; after install run `python -m v2d.<module>.docker.build` to build the image (from any directory).

**Flow:** Host Python → calls `run_*()` → spawns Docker container → container runs `lib/` code → results written to mounted volumes.

**Why this design?**

- **Isolation:** Each module has its own environment (CUDA, PyTorch) without host pollution.
- **Reproducibility:** Containers pin exact dependency versions.
- **Portability:** Run the same containers on different hosts; only Docker + GPU are required on the host.
- **Composable pipelines:** The host can import and chain multiple `run_*` functions (e.g. `v2d.pipelines.example_pipeline`) without installing heavy ML stacks locally.

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

**Build:** `python -m v2d.unidepth.docker.build`  
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

**Build:** `python -m v2d.moge.docker.build`  
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

**Build:** `python -m v2d.sam2.docker.build`  
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

**Build:** `python -m v2d.sam3d.docker.build`  
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

**Build:** `python -m v2d.grounding_dino.docker.build`  
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

**Build:** `python -m v2d.foundation_stereo.docker.build`  
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

**Build:** `python -m v2d.foundation_pose.docker.build`  
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

**Build:** `python -m v2d.nlf.docker.build`  
**Execute:** `python -m v2d.nlf.docker.run_video_to_smpl --video_path ... --masks_dir ... --intrinsics_path ... --gender male --output_path ... --weights_dir ...`

---

### v2d_hoi_object_reconstruction

End-to-end textured 3D mesh reconstruction from hand-object interaction video using a two-stage scan (object stationary → rotated → stationary).

| Tool | Description |
|------|-------------|
| `run_reconstruction` | Full pipeline: CuSFM → depth → mask → Stage-1 NeRF → FoundationPose → Stage-2 NeRF → textured mesh |
| `run_fp_tracking` | FoundationPose tracking only: center mesh → depth → mask → FP tracking → render overlay |

**Build (from `reconstruction/modules/`):** `python v2d_hoi_object_reconstruction/docker/build.py` (builds the HOI image); `python v2d_bundlesdf/docker/build.py` and `python v2d_cusfm/docker/build.py` build their respective images separately.

**Example (from `reconstruction/`):**

The repo includes example data at `modules/v2d_hoi_object_reconstruction/assets/basketball_example/` (203 stereo frames, 960×600) for quick testing.

```bash
python modules/v2d_hoi_object_reconstruction/docker/run_reconstruction.py \
    --mapping_data_dir modules/v2d_hoi_object_reconstruction/assets/basketball_example \
    --job_dir          data/outputs/hoi_recon/basketball_example \
    --prompt           "basketball"
```

**Inputs:** `mapping_data_dir/` — stereo images (`front_stereo_camera_left/`, `front_stereo_camera_right/`), `frames_meta.json`, `frame_metadata.jsonl`
**Outputs:** `job_dir/merged_recon/textured_mesh.obj` — final textured mesh; `job_dir/stage1_recon/textured_mesh.obj` — Stage-1 mesh

See [`modules/v2d_hoi_object_reconstruction/README.md`](modules/v2d_hoi_object_reconstruction/README.md) for full pipeline details and troubleshooting.

---

### v2d_cusfm

Structure-from-motion using CuSFM. Produces camera poses for a stereo image sequence.

**Inputs:**
- `input_dir/` — stereo images (e.g. `front_stereo_camera_left/*.jpeg`) and `frames_meta.json` with camera calibration

**Outputs:**
- `output_dir/keyframes/frames_meta.json` — camera poses for each keyframe

**Build:** `python modules/v2d_cusfm/docker/build.py`

**Example (from `reconstruction/`):**

The repo includes example data at `modules/v2d_hoi_object_reconstruction/assets/basketball_example/` (shared with the HOI reconstruction pipeline).

```bash
python modules/v2d_cusfm/docker/run_image_list_to_sfm.py \
    --input_dir  modules/v2d_hoi_object_reconstruction/assets/basketball_example \
    --output_dir data/outputs/cusfm/basketball_example
```

---

### v2d_bundlesdf

SDF learning and texture baking from pre-computed camera poses. Takes keyframes with depth and masks, outputs a textured mesh.

> **Note:** This module only covers SDF learning and texture baking. Pose tracking (BundleTrack) is not supported — camera poses must be pre-computed externally (e.g. via `v2d_cusfm`).

**Inputs:**
- `recon_dir/keyframes.yml` — camera-to-object poses per keyframe
- `recon_dir/left/` — RGB images (one per keyframe)
- `recon_dir/depth/` — depth maps (uint16 PNG, one per keyframe)
- `recon_dir/masks/` — object masks (grayscale PNG, one per keyframe)
- `config.yaml` — NeRF/SDF config (e.g. `theseus_optimizer_hawk.yaml`)

**Outputs:**
- `recon_dir/textured_mesh.obj` — final textured mesh (+ `.mtl`, `_0.png` texture atlas)
- `recon_dir/mesh_cleaned.obj` — untextured SDF mesh

**Build:** `python modules/v2d_bundlesdf/docker/build.py`

**Example (from `reconstruction/`):**

The repo includes example data at `modules/v2d_bundlesdf/assets/stage1_example/` (19 keyframes with RGB, depth, and masks) for quick testing. A per-dataset config with correct intrinsics is required (the default config has Hawk 1920×1200 intrinsics):

```bash
python modules/v2d_bundlesdf/docker/run_reconstruct.py \
    --output_path modules/v2d_bundlesdf/assets/stage1_example \
    --weights_dir data/weights \
    --config     modules/v2d_bundlesdf/assets/stage1_example/config.yaml
```

---

### v2d_detectron2

Person detection and IoU-based tracking using Detectron2 ViTDet models. Supports single-camera and multi-view operation. Outputs `.pt` files containing `{det_cat_id, scores, bbox_track}`.

| Tool | Function | Description |
|------|----------|-------------|
| `run_track_bboxes` | `run_track_bboxes(weights_dir, output_path, image_dir=None, video_path=None, model_size="b", bbox_thr=0.5, iou_threshold=0.3, max_lost=30, min_hits=3, batch_size=1, debug=0, dev=False)` | Detect + track person bboxes on a single camera (image dir or video) |
| `run_mv_track_bboxes` | `run_mv_track_bboxes(weights_dir, output_dir, image_dir=None, video_dir=None, config_path=..., debug=-1, dev=False)` | Multi-view detection + tracking across cameras defined by rig config |
| `run_download_weights` | `run_download_weights(output_dir, model_sizes=["b"], dev=False)` | Download ViTDet checkpoint(s) (b/l/h) |
| `run_shell` | `run_shell(dev=False)` | Interactive bash shell in container |

**Build:** `python -m v2d.detectron2.docker.build`

**Execute (single-cam):** `python -m v2d.detectron2.docker.run_track_bboxes --video_path ... --weights_dir ... --output_path ... --model_size b`

**Execute (multi-view):** `python -m v2d.detectron2.docker.run_mv_track_bboxes --video_dir ... --weights_dir data/weights/detectron2 --output_dir data/outputs/detectron2`

**Example (single-cam):** From `reconstruction/`:

```bash
# 1. Download weights
python -m v2d.detectron2.docker.run_download_weights --output_dir data/weights/detectron2

# 2. Run person detection + tracking (sample video at modules/v2d_detectron2/assets/test_video.mp4)
python -m v2d.detectron2.docker.run_track_bboxes \
  --video_path modules/v2d_detectron2/assets/test_video.mp4 \
  --weights_dir data/weights/detectron2 \
  --output_path data/outputs/detectron2/bbox_track.pt \
  --debug 2
```

**Output:** `bbox_track.pt` — a dict with `{det_cat_id, scores, bbox_track}` (numpy arrays). `bbox_track` has shape `(N, 4)` in `[x1, y1, x2, y2]` format, one row per frame.

---

### v2d_sam3d_body

Human body pose and shape estimation using SAM3D-Body with multi-view MHR (Multi-Hypothesis Recovery) optimization.

| Tool | Function | Description |
|------|----------|-------------|
| `run_estimate_mhr_params` | `run_estimate_mhr_params(cam_intrinsics_path, weights_dir, bbox_path, output_params_path, image_dir=None, video_path=None, output_mesh_path=None, debug=-1, dev=False)` | Estimate MHR body parameters from a single camera (image dir or video) |
| `run_mv_optimize_mhr_params` | `run_mv_optimize_mhr_params(camera_params_path, weights_dir, bbox_dir, output_dir, image_dir=None, video_dir=None, config_path=..., debug=-1, dev=False)` | Multi-view MHR parameter optimization across cameras |
| `run_mv_render_mhr_mesh` | `run_mv_render_mhr_mesh(data_path, config_path=None, dev=False)` | Render MHR mesh overlay for all cameras |
| `run_download_weights` | `run_download(output_dir, dev=False)` | Download SAM3D-Body and MoGe-2 weights |
| `run_shell` | `run_shell(dev=False)` | Interactive bash shell in container |

**Build:** `python -m v2d.sam3d_body.docker.build`

**Execute (single-cam):** `python -m v2d.sam3d_body.docker.run_estimate_mhr_params --image_dir ... --cam_intrinsics_path ... --weights_dir ... --bbox_path ... --output_params_path ...`

**Execute (multi-view):** `python -m v2d.sam3d_body.docker.run_mv_optimize_mhr_params --image_dir ... --camera_params_path ... --weights_dir ... --bbox_dir ... --output_dir ...`

**Example (single-cam):** From `reconstruction/`:

```bash
# 1. Download weights
python -m v2d.sam3d_body.docker.run_download_weights --output_dir data/weights/sam3d_body

# 2. Run MHR body estimation (uses bundled test assets)
python -m v2d.sam3d_body.docker.run_estimate_mhr_params \
  --video_path modules/v2d_sam3d_body/assets/test_video.mp4 \
  --cam_intrinsics_path modules/v2d_sam3d_body/assets/cam_intrinsics.json \
  --weights_dir data/weights/sam3d_body \
  --bbox_path modules/v2d_sam3d_body/assets/bbox_track.pt \
  --output_params_path data/outputs/sam3d_body/params \
  --output_mesh_path data/outputs/sam3d_body/meshes \
  --debug 2
```

---

## Build & Execute (Summary)

All modules share the same build pattern. Each Dockerfile uses `reconstruction/modules` as build context (parent of each `v2d_*` folder).

| Action | Command |
|--------|---------|
| **Build** | `python -m v2d.<module>.docker.build` |
| **Execute** | `python -m v2d.<module>.docker.run_<tool> --arg1 ... --arg2 ...` |
| **Dev mode** | Add `--dev` to mount local modules at `/workspace` |

Example pipeline usage (install `v2d-pipelines` and docker packages; see Setup above):

```bash
# From reconstruction/ - install all in one command
pip install -e modules/v2d_sam2/docker -e modules/v2d_sam3d/docker -e modules/v2d_sam3d_body/docker \
  -e modules/v2d_unidepth/docker -e modules/v2d_moge/docker -e modules/v2d_nlf/docker \
  -e modules/v2d_foundation_pose/docker -e modules/v2d_foundation_stereo/docker \
  -e modules/v2d_grounding_dino/docker -e modules/v2d_cusfm/docker -e modules/v2d_bundlesdf/docker \
  -e modules/v2d_hoi_object_reconstruction/docker -e modules/v2d_detectron2/docker \
  -e modules/v2d_pipelines
```

Then run from `reconstruction/` or repo root:

```bash
python -m v2d.pipelines.example_pipeline
```

Or import in Python:

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

- **Build context:** Use `reconstruction/modules` (parent of `v2d_*`) as the Docker build context so `v2d_common` and sibling modules are available.
- **Image name:** `v2d_<name>` (matches folder).
- **Base image:** Use `pytorch/pytorch` variants for GPU workloads.
- **Install:** `pip install -e /workspace/v2d_common -e /workspace/v2d_<name>/lib` (or equivalent).

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
- [ ] Add any shared types to `v2d_common` if other modules will consume them.
