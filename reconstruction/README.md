# Modular Infrastructure

Modular infrastructure for various vision tasks, containerized with Docker and scaled with Celery.

## Shared Data Volume
All modules share a common data directory at `./data` in the project root.
- **Host Path:** `./data`
- **Container Path:** `/data`

Place your input files in `./data` and reference them using the `/data/` prefix in commands.

## HTTP API Servers
Each module provides an HTTP API server for programmatic access. To start an API server:

```bash
docker compose --profile api up <module>-api
```

API servers run on the following ports:
- **SAM2:** Port 8001
- **MoGe:** Port 8002
- **UniDepth:** Port 8003
- **SAM3D:** Port 8004
- **FoundationPose:** Port 8005

All endpoints accept file uploads via multipart/form-data and return zipped result files. Each request creates a unique job directory under `/data/jobs/<job_id>/` for processing.

---

## Table of Contents

- [SAM2](#sam2-segment-anything-model-2)
  - [`video_to_masks`](#video_to_masks)
- [MoGe](#moge-monocular-geometry-estimation)
  - [`image_to_depth`](#image_to_depth)
  - [`video_to_depth`](#video_to_depth)
- [UniDepth](#unidepth)
  - [`image_to_depth`](#image_to_depth-1)
  - [`video_to_depth`](#video_to_depth-1)
- [SAM3D](#sam3d-3d-mesh-generation)
  - [`image_to_mesh`](#image_to_mesh)
  - [`render_debug`](#render_debug)
- [FoundationPose](#foundationpose-object-pose--mesh-utils)
  - [`video_to_poses`](#video_to_poses)
  - [`align_mesh`](#align_mesh)
  - [`transform_mesh`](#transform_mesh)
  - [`simplify_mesh`](#simplify_mesh)

---

## SAM2 (Segment Anything Model 2)
Video object segmentation and tracking.

### `video_to_masks`
Process a video with SAM2 prompts and save masks to files.
```bash
docker compose --profile exec run sam2-video-to-masks \
  --video_path /data/video.mp4 \
  --prompts_path /data/prompts.json \
  --masks_dir /data/output/masks
```

---

## MoGe (Monocular Geometry Estimation)
Estimates depth and camera intrinsics from images or video.

### `image_to_depth`
Process a single image to estimate depth and camera intrinsics.
```bash
docker compose --profile exec run moge-image-to-depth \
  --image_path /data/image.jpg \
  --depth_path /data/output/depth.png \
  --intrinsics_path /data/output/intrinsics.json
```

### `video_to_depth`
Process a video to estimate depth frames and camera intrinsics.
```bash
docker compose --profile exec run moge-video-to-depth \
  --video_path /data/video.mp4 \
  --depth_folder /data/output/depth \
  --intrinsics_folder /data/output/intrinsics
```

---

## UniDepth
Universal Monocular Depth Estimation.

### `image_to_depth`
Process a single image to estimate depth and camera intrinsics.
```bash
docker compose --profile exec run unidepth-image-to-depth \
  --image_path /data/image.jpg \
  --depth_path /data/output/depth.png \
  --intrinsics_path /data/output/intrinsics.json
```

### `video_to_depth`
Process a video to estimate depth frames and camera intrinsics.
```bash
docker compose --profile exec run unidepth-video-to-depth \
  --video_path /data/video.mp4 \
  --depth_folder /data/output/depth \
  --intrinsics_folder /data/output/intrinsics
```

---

## SAM3D (3D Mesh Generation)
Generating 3D meshes from images and masks.

### `image_to_mesh`
Generate a 3D mesh from an image and a corresponding object mask.
```bash
docker compose --profile exec run sam3d-image-to-mesh \
  --image_path /data/image.jpg \
  --mask_path /data/mask.png \
  --mesh_path /data/output/mesh.glb \
  --transform_path /data/output/transform.json \
  --intrinsics_path /data/output/intrinsics.json
```

### `render_debug`
Render a debug visualization showing projected mesh vertices on the image.
```bash
docker compose --profile exec run sam3d-render-debug \
  /data/image.jpg \
  /data/output/mesh.glb \
  /data/output/transform.json \
  /data/output/intrinsics.json \
  /data/output/debug.jpg
```

---

## FoundationPose (Object Pose & Mesh Utils)
6D object pose estimation and tracking, along with mesh utilities.

### `video_to_poses`
Track 6D object poses through a video sequence given a 3D mesh.
```bash
docker compose --profile exec run foundationpose-video-to-poses \
  --video_path /data/video.mp4 \
  --depth_folder /data/output/depth \
  --masks_folder /data/output/masks \
  --camera_intrinsics_path /data/output/intrinsics.json \
  --mesh_path /data/output/mesh.glb \
  --poses_dir /data/output/poses
```

### `align_mesh`
Refine the scale and translation of a mesh to match observed depth data.
```bash
docker compose --profile exec run foundationpose-align-mesh \
  --mesh /data/mesh.glb \
  --depth /data/depth.png \
  --mask /data/mask.png \
  --intrinsics /data/intrinsics.json \
  --transform /data/transform.json \
  --output-transform /data/refined_transform.json
```

### `transform_mesh`
Apply scaling from a transform JSON to a mesh.
```bash
docker compose --profile exec run foundationpose-transform-mesh \
  --input-mesh /data/mesh.glb \
  --output-mesh /data/scaled_mesh.glb \
  --transform /data/transform.json
```

### `simplify_mesh`
Simplify a 3D mesh using quadratic decimation.
```bash
docker compose --profile exec run foundationpose-simplify-mesh \
  --input-mesh /data/mesh.glb \
  --output-mesh /data/simplified_mesh.glb \
  --faces 5000 \
  --factor 0.5
```
