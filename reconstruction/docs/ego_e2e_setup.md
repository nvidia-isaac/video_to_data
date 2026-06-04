# Ego E2E Pipeline — Setup Guide

Setup for `modules/v2d_pipelines/run_v2d_ego_e2e.py`.
All commands run from `reconstruction/`.

## Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-smi` accessible inside containers)
- Python 3.10+
- `ffmpeg` on `PATH`

## 1. Install host packages

Install just the packages this pipeline needs:

```bash
./scripts/install_ego_e2e_packages.sh
```

Or install every host package in the repo:

```bash
./scripts/install_packages.sh
```

## 2. Build Docker images

Build just the containers this pipeline needs:

```bash
./scripts/build_ego_e2e_containers.sh
```

Or build every container in the repo:

```bash
./scripts/build_containers.sh
```

## 3. Download model weights

```bash
python -m v2d.moge.docker.run_download_weights           --output_dir data/weights/moge
python -m v2d.grounding_dino.docker.run_download_weights --output_dir data/weights/grounding_dino
python -m v2d.sam2.docker.run_download_weights           --output_dir data/weights/sam2
python -m v2d.sam3d.docker.run_download_weights          --output_dir data/weights/sam3d
python -m v2d.foundation_pose.docker.run_download_weights --output_dir data/weights/foundation_pose
```

**SAM3D requires a Hugging Face token** for gated model access. Either set `HF_TOKEN` in your
environment or log in with `huggingface-cli login` beforehand.

**AnyCalib (optional)** — only needed if you run with `--undistort` (Step 0) to calibrate and
undistort fisheye / wide-angle footage. Skip this download otherwise:

```bash
python -m v2d.anycalib.docker.run_download_weights --output_dir data/weights/anycalib
```

## 4. Prepare MANO weights (for DynHaMR + hand alignment)

Download `MANO_RIGHT.pkl` from https://mano.is.tue.mpg.de/ and generate BMC data
following https://github.com/MengHao666/Hand-BMC-pytorch (run up to `python calculate_bmc.py`).

Place both in the weights directory. The layout follows the manotorch
convention so the same directory works for both DynHaMR and v2d_hamer:

```
data/weights/hand/
├── models/
│   └── MANO_RIGHT.pkl
└── BMC/
    └── *.npy
```

## 5. Get the sample video

A ready-to-run sample, `assets/airplane.mp4` (~25 MB), ships with the repo via
[Git LFS](https://git-lfs.com/). If you cloned with Git LFS installed it is already
present. Otherwise install LFS and pull it:

```bash
git lfs install                                              # one-time, enables LFS filters
git lfs pull --include reconstruction/assets/airplane.mp4    # paths are repo-root relative
```

Confirm it materialized — it should be ~25 MB, not a ~130-byte pointer file:

```bash
ls -lh assets/airplane.mp4
```

## 6. Run the pipeline

```bash
python modules/v2d_pipelines/run_v2d_ego_e2e.py \
    --video_path  assets/airplane.mp4 \
    --prompt      "airplane" \
    --output_dir  data/outputs/airplane \
    --depth_source moge
```

Substitute your own `--video_path` / `--prompt` to run on a different clip.

All weight paths default to `data/weights/<model>` relative to cwd so no extra flags are needed
if you followed the layout above.

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--undistort` | off | Run AnyCalib first (Step 0) to estimate intrinsics + distortion and undistort the video before all other steps. Recommended for fisheye / wide-angle footage. Requires the AnyCalib weights from Step 3. |
| `--anycalib_weights` | `data/weights/anycalib` | AnyCalib weights directory (used only with `--undistort`) |
| `--depth_source` | `moge` | Depth for FP tracking: `moge` or `vipe` (DynHaMR) |
| `--reference_frame` | `0` | Frame used for DINO, SAM3D, and FP registration |
| `--reregister_iou_thresh` | `0.3` | IoU threshold for FP re-registration; `0` to disable |
| `--smooth_sigma` | `5.0` | Gaussian sigma (frames) for hand translation smoothing |
| `--dev` | off | Mount local module source into containers (live-edit) |

### Resuming a partial run

Every step checks whether its output already exists and skips if so. Simply re-run the same
command to resume from where it stopped.
