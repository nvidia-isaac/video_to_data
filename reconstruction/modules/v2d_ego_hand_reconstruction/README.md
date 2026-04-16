<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Egocentric Hand Reconstruction

Automated pipeline for 4D hand and camera pose reconstruction from egocentric videos. Integrates ViPE and Dyn-HaMR in containerized environments.

## Setup

Fetch vendored sources from [IsaacTeleop](https://github.com/NVIDIA/IsaacTeleop):

```bash
./sync.sh
```

Install the host-side orchestration package:

```bash
pip install -e docker/
```

Build both Docker images (ViPE + Dyn-HaMR):

```bash
python -m v2d_ego_hand_reconstruction.docker.build
```

Place required data in your output directory before running (see `vendor/doc/quickstart.md`):
- `MANO_RIGHT.pkl` from <https://mano.is.tue.mpg.de/>
- `BMC/*.npy` from the Hand-BMC-pytorch repo

## Usage

**Python (programmatic):**

```python
from v2d_ego_hand_reconstruction.docker.run_reconstruction import run_reconstruction

run_reconstruction(
    video_input="path/to/video.mp4",
    output_dir="data/outputs/ego_hand",
)
```

**CLI:**

```bash
python -m v2d_ego_hand_reconstruction.docker.run_reconstruction \
    --video_input path/to/video.mp4 \
    --output_dir data/outputs/ego_hand
```

Remote videos (S3/Swift) are also supported:

Set environment variables ACCESS_KEY_ID and SECRET_ACCESS_KEY for S3/Swift permission.

```bash
export ACCESS_KEY_ID=XXX SECRET_ACCESS_KEY=XXX
```

Please check vendor/doc/quickstart.md for detail.

```bash
python -m v2d_ego_hand_reconstruction.docker.run_reconstruction \
    --video_input s3://bucket/video.mp4 \
    --output_dir data/outputs/ego_hand
```

Results are saved to `<output_dir>/logs/`.

## Upstream Diff

To see local modifications vs upstream IsaacTeleop:

```bash
./diff.sh          # summary
./diff.sh --full   # full unified diff
```

## Structure

```
docker/          Native Python orchestration (tracked in git)
vendor/          Upstream content from IsaacTeleop (gitignored, populated by sync.sh)
sync.sh          Fetch/update vendored sources
diff.sh          Compare vendor/ against upstream
```
