# Video to Data

Monorepo for **Video to Data (V2D)** — an end-to-end pipeline that converts human demonstration videos into simulation-ready assets and physics-grounded robot training data.

## End-to-End Workflow

```
 ┌───────────────┐   ┌──────────────────┐   ┌────────────────────────────┐
 │ Human demo    │ → │ 1. Reconstruction│ → │ 2. Robotic Grounding       │
 │ video / rosbag│   │ depth · masks ·  │   │ retargeting → Isaac Lab    │
 │               │   │ meshes · 6D pose │   │ RL training (RSL-RL PPO)   │
 │               │   │ · SMPL body      │   │                            │
 └───────────────┘   └──────────────────┘   └────────────────────────────┘
                         reconstruction/         robotic_grounding/
```

1. **Reconstruction** — containerized vision modules turn RGB (or stereo) video into per-frame depth, object masks, textured meshes, 6-DoF object poses, and SMPL human body parameters. Multi-view pipelines (`run_mv_hoi_reconstruction`, `run_mv_calibration`) orchestrate the full reconstruction from a rosbag.
2. **Robotic Grounding** — human motion (e.g. Arctic) is retargeted onto the target robot embodiment (Sharpa), then the reconstructed scene and retargeted motion drive Isaac Lab environments trained with RSL-RL PPO to produce deployable policies.

## Packages

| Package | Role | Runtime |
|---|---|---|
| [`reconstruction/`](reconstruction/) | Video → depth, masks, meshes, 6D poses, human body. 18 containerized modules + multi-view pipelines. | Docker (per-module images) |
| [`robotic_grounding/`](robotic_grounding/) | RL training on NVIDIA Isaac Lab 2.3.1 with RSL-RL (PPO); motion retargeting utilities. | Docker (`nvcr.io/nvstaging/isaac-amr`) |

## Prerequisites

- Docker with GPU support ([install](https://docs.docker.com/engine/install/ubuntu/))
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Python 3.10+
- NVIDIA driver 580.126.09 / CUDA 13.0 recommended (for `robotic_grounding`)

## Quickstart

### Reconstruction (video → 3D data)

```bash
cd reconstruction

# Install host-side orchestration wrappers (lightweight, no ML deps)
./scripts/install_pacakages.sh

# Build per-module Docker images
./scripts/build_containers.sh

# Run a minimal video→depth example (MoGe)
python -m v2d.moge.docker.run_download_weights --output_dir data/weights/moge
python -m v2d.moge.docker.run_video_to_depth \
  --video_path modules/v2d_moge/assets/test_video.mp4 \
  --depth_folder data/outputs/moge/depth \
  --intrinsics_folder data/outputs/moge/intrinsics \
  --weights_path data/weights/moge
```

Full multi-view HOI pipeline (rosbag → textured object mesh + SMPL body):

```bash
python -m v2d.pipelines.run_mv_hoi_reconstruction \
  --rosbag_path /data/rosbags/session1 \
  --output_dir  /data/datasets/session1 \
  --extrinsics_camera_params_path /data/datasets/calibration/extrinsics/edex \
  --obj_mesh_path /data/meshes/object.glb
```

See [reconstruction/README.md](reconstruction/README.md) for the complete module reference, including [Grounding DINO](reconstruction/README.md#v2d_grounding_dino), [SAM2](reconstruction/README.md#v2d_sam2), [FoundationPose](reconstruction/README.md#v2d_foundation_pose), [SAM3D-Body](reconstruction/README.md#v2d_sam3d_body), and others.

### Robotic Grounding (data → RL policy)

```bash
cd robotic_grounding

# One-time host setup (git-lfs, pre-commit)
bash workflow/setup_deps.sh

# Build + enter the Isaac Lab container
./workflow/run.sh build  [version]
./workflow/run.sh start  [version] [gpu_id]

# Inside the container — train a policy
python scripts/rsl_rl/train.py --task Sharpa-V2P-v0
```

See [robotic_grounding/README.md](robotic_grounding/README.md) for retargeting, debug environments, and task definitions.

### Visualizer (retargeting gallery)

Browse retargeted sequences as 3D animations at **http://10.111.83.14:8080/**

See [robotic_grounding/README.md#visualizer](robotic_grounding/README.md#visualizer) for setup instructions.

## Design Philosophy

- **Host orchestration, containerized inference.** The host runs thin Python wrappers that `docker run` each module; all ML dependencies live inside their respective images. No CUDA or PyTorch is ever installed on the host.
- **Typed contracts between packages.** Modules communicate through strongly-typed dataclasses in [`v2d_common`](reconstruction/modules/v2d_common/) (`DepthImage`, `CameraIntrinsics`, `Transform3d`, `BoundingBox`, `Mask`) — never raw arrays across package boundaries.
- **File-based dataflow.** Modules write intermediate artifacts to disk (depth PNGs, pose JSONs, mask PNGs, etc.), enabling independent execution, caching, and pipeline composition via [`v2d_pipelines`](reconstruction/modules/v2d_pipelines/).

## Contributing

See the contributing guide in [reconstruction/README.md](reconstruction/README.md#contributing) for adding new reconstruction modules. Each new module must expose a Docker image, a `run_download_weights` entry point (if weights are required), a `run_shell` entry point, and a typed API surface consistent with `v2d_common`.
