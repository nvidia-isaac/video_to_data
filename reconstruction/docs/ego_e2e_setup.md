# Ego Reconstruction Setup Guide

Setup for the consolidated entrypoint:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py
```

The older scripts remain available, but new runs should use
`run_ego_reconstruction.py`.

## Agent Skills

- **Codex:** `ego-reconstruction-setup` prepares the environment;
  `run-ego-reconstruction-video` runs and verifies a video reconstruction.
- **Claude:** `.claude/skills/ego-reconstruction-setup` prepares the environment;
  `.claude/skills/run-ego-reconstruction-video` runs and verifies a video reconstruction.

All commands below run from `reconstruction/`.

## 1. Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-smi` accessible inside containers)
- Python 3.10+
- `ffmpeg` on `PATH`
- A Hugging Face token for SAM3D gated model access when using prompt-based mesh reconstruction

## 2. Install Host Packages

```bash
./scripts/install_ego_reconstruction_packages.sh
```

## 3. Build Docker Images

```bash
./scripts/build_ego_reconstruction_packages.sh
```

## 4. Download Model Weights

```bash
./scripts/download_ego_reconstruction_weights.sh
```

The downloader supports narrower modes if you do not want every optional model:

```bash
./scripts/download_ego_reconstruction_weights.sh --mode dynhamr_prompt
./scripts/download_ego_reconstruction_weights.sh --mode hamer_prompt
./scripts/download_ego_reconstruction_weights.sh --mode hamer_mesh
```

SAM3D requires a Hugging Face token for gated model access. Either set `HF_TOKEN`
in your environment or log in with `huggingface-cli login` before downloading or
running SAM3D.

MANO assets are licensed separately and still manual. Download
`MANO_RIGHT.pkl` from https://mano.is.tue.mpg.de/ and place it here:

```text
data/weights/hand/
├── models/
│   └── MANO_RIGHT.pkl
└── BMC/
    └── *.npy
```

This is the canonical source for both hand-tracking paths. The HaMeR pipeline
automatically copies it into its container-visible `models/` layout before the
run; do not manually place files under `data/weights/hamer/_DATA/data/mano/`.
DynHaMR additionally requires its `BMC/*.npy` files.

## 5. Get The Sample Video

A ready-to-run sample, `assets/airplane.mp4`, ships with the repo via Git LFS.
If it is still a small pointer file, install LFS and pull it:

```bash
git lfs install
git lfs pull --include reconstruction/assets/airplane.mp4
```

Confirm it materialized as a real video file before running the examples:

```bash
ls -lh assets/airplane.mp4
```

## 6. Run The Pipeline

DynHaMR hand tracking with prompt-based SAM3D object reconstruction, DROID-SLAM,
gravity alignment, and Three.js export:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
    --video assets/airplane.mp4 \
    --object_prompt "A toy airplane" \
    --output_dir data/outputs/airplane_dynhamr \
    --reference_frame 0 \
    --undistort \
    --hand_tracking dynhamr \
    --run_droid_slam \
    --run_gravity_alignment \
    --export_threejs_result \
    --dev
```


> **Agent prompt:** In Claude or Codex, ask: “Run the DynHaMR ego reconstruction pipeline on `assets/airplane.mp4` for a toy airplane with undistortion, DROID-SLAM, gravity alignment, and Three.js export.” The matching run skill supplies this command.

New full pipeline with HaMeR hand tracking, prompt-based SAM3D object
reconstruction, DROID-SLAM, gravity alignment, and gsplat refinement:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
    --video assets/airplane.mp4 \
    --object_prompt "A toy airplane" \
    --output_dir data/outputs/airplane_hamer \
    --reference_frame 0 \
    --undistort \
    --hand_tracking hamer \
    --run_droid_slam \
    --run_gravity_alignment \
    --run_gsplat_refinement \
    --export_threejs_result \
    --dev
```


> **Agent prompt:** In Claude or Codex, ask: “Run the HaMeR ego reconstruction pipeline on `assets/airplane.mp4` for a toy airplane with undistortion, DROID-SLAM, gravity alignment, gsplat refinement, and Three.js export.” The matching run skill supplies this command.

New full pipeline with a provided object mesh override. The prompt is still used
for object detection, masks, and FoundationPose tracking initialization; the mesh
only replaces SAM3D geometry and scale estimation:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
    --video assets/airplane.mp4 \
    --object_prompt "A toy airplane" \
    --object_mesh assets/textured_mesh.obj \
    --skip_object_scale_estimation \
    --output_dir data/outputs/airplane_hamer_mesh \
    --reference_frame 0 \
    --undistort \
    --hand_tracking hamer \
    --run_droid_slam \
    --run_gravity_alignment \
    --run_gsplat_refinement \
    --export_threejs_result \
    --dev
```


> **Agent prompt:** In Claude or Codex, ask: “Run HaMeR ego reconstruction on `assets/airplane.mp4` using `assets/textured_mesh.obj` for a toy airplane; skip mesh scale estimation and enable undistortion, DROID-SLAM, gravity alignment, gsplat refinement, and Three.js export.” The matching run skill supplies this command.

## Legacy Command

The old e2e script is intentionally left untouched. Existing commands like this
still run through the legacy path:

```bash
python modules/v2d_pipelines/run_v2d_ego_e2e.py \
    --video_path assets/airplane.mp4 \
    --prompt "airplane" \
    --output_dir data/outputs/airplane_legacy \
    --depth_source moge
```

## Outputs

The base portable result bundle is written to:

```text
<output_dir>/result/
```

Optional post-processing writes suffixed bundles so stages can be cached and
compared:

```text
<output_dir>/result_slam/
<output_dir>/result_gravity_aligned/
<output_dir>/result_slam_gravity_aligned/
```

When `--export_threejs_result` is enabled, the viewer is written under the final
selected bundle:

```text
<final_result_dir>/threejs_scene/index.html
```

The pipeline is cache-aware. Re-run the same command to resume from completed
stage outputs.
