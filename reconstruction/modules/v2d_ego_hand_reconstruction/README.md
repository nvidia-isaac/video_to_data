<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Egocentric Hand Reconstruction

Automated pipeline for 4D hand and camera pose reconstruction from egocentric videos. Integrates ViPE and Dyn-HaMR in containerized environments.

## Setup

Fetch vendored sources from [IsaacCapture](https://github.com/NVIDIA/IsaacCapture) (branch `ego4robo/0.1`, see `sync.sh`):

```bash
./sync.sh
```

Install the host-side orchestration package (use a virtualenv; system Python is externally managed on recent Ubuntu/Debian):

```bash
uv venv .venv && source .venv/bin/activate   # or: python3 -m venv .venv
uv pip install -e docker/                    # or: pip install -e docker/
```

Build both Docker images (ViPE + Dyn-HaMR, tagged `ego_vipe:latest` and `ego_dynhamr:latest`). The first build downloads several GB of model checkpoints and takes a while:

```bash
python -m v2d_ego_hand_reconstruction.docker.build
```

Place required data in your weights directory before running (see [IsaacCapture egocentric hand reconstruction](https://github.com/NVIDIA/IsaacCapture/tree/ego4robo/0.1/src/postprocessing/egocentric_hand_reconstruction)).
The layout follows the manotorch convention so the same directory is shared with `v2d_hamer`:

```
<weights_dir>/
├── models/
│   └── MANO_RIGHT.pkl     # from https://mano.is.tue.mpg.de/
└── BMC/
    └── *.npy              # from the Hand-BMC-pytorch repo
```

## Usage

**Python (programmatic):**

```python
from v2d_ego_hand_reconstruction.docker.run_reconstruction import run_reconstruction

run_reconstruction(
    video_input="path/to/video.mp4",
    output_dir="data/outputs/ego_hand",
    weights_dir="data/weights",  # contains models/MANO_RIGHT.pkl and BMC/
)
```

**CLI:**

```bash
python -m v2d_ego_hand_reconstruction.docker.run_reconstruction \
    --video_input path/to/video.mp4 \
    --output_dir data/outputs/ego_hand \
    --weights_dir data/weights
```

Remote videos (S3/Swift) are also supported:

Set environment variables ACCESS_KEY_ID and SECRET_ACCESS_KEY for S3/Swift permission.

```bash
export ACCESS_KEY_ID=XXX SECRET_ACCESS_KEY=XXX
```

Please check [IsaacCapture egocentric hand reconstruction](https://github.com/NVIDIA/IsaacCapture/tree/ego4robo/0.1/src/postprocessing/egocentric_hand_reconstruction) for detail.

```bash
python -m v2d_ego_hand_reconstruction.docker.run_reconstruction \
    --video_input s3://bucket/video.mp4 \
    --output_dir data/outputs/ego_hand \
    --weights_dir data/weights
```

Results are saved to `<output_dir>/logs/`. ViPE camera estimates are written to `<output_dir>/vipe/`.

### Hand mesh generation

After running reconstruction, export per-track MANO hand mesh trajectories from every run under `<output_dir>/logs/`:

```python
from v2d_ego_hand_reconstruction.docker.run_mesh_generation import run_mesh_generation

run_mesh_generation(output_dir="data/outputs/ego_hand")
```

CLI:

```bash
python -m v2d_ego_hand_reconstruction.docker.run_mesh_generation \
    --output_dir data/outputs/ego_hand
```

Results land at `<output_dir>/logs/.../smooth_fit/<seq>_hand_mesh_traj_<iter>.npz`.

## Upstream Diff

To see local modifications vs upstream IsaacCapture:

```bash
./diff.sh          # summary
./diff.sh --full   # full unified diff
```

## Structure

```
docker/          Native Python orchestration (tracked in git)
vendor/          Upstream content from IsaacCapture (gitignored, populated by sync.sh)
sync.sh          Fetch/update vendored sources
diff.sh          Compare vendor/ against upstream
```
