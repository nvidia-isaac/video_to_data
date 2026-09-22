# FlashCHORD

GPU-parallel robotic reference tracking with Newton, Warp, JAX, PPO, and FlashSAC.

**[Setup](#setup) → [Data](#example-data) → [View robot](#1-view-robot) → [View scene](#2-view-scene) → [Replay](#3-replay) → [Train](#4-train) → [Evaluate](#5-evaluate) → [View policy](#6-view-policy)** · **[Configuration](#configuration)**

## Setup

### Host prerequisites

- Linux and NVIDIA driver **580 or newer**.
- [Docker](https://docs.docker.com/engine/install/ubuntu/) with [non-root access](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- Git LFS; access to the repository, `nvidia/cuda`, and `ghcr.io/astral-sh/uv`.
- Full training: **48 GiB GPU with ≥46,000 MiB free**, as validated in EVT-07. Replay and one-world viewing use less memory.

### Run in the container

```bash
# Host, from the repository root.
git lfs install
git lfs pull
cd robotic_grounding/flash_chord
docker/run.sh build
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
docker/run.sh start latest 0   # Replace 0 with your host GPU index.
```

`start` opens an interactive shell at `/workspace`. Run the workflow below there.

| Host command | Action |
| --- | --- |
| `docker/run.sh shell latest 0` | Open another shell |
| `docker/run.sh exec latest 0 -- nvidia-smi` | Run a command |
| `docker/run.sh stop latest 0` | Stop and remove the container |

The container uses the pinned [Dockerfile](docker/Dockerfile) and `uv.lock`.

### Host installation (without Docker)

**Skip this section if using Docker.** The image includes Python and the FlashCHORD dependencies; the host prerequisites above still apply.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[train,dev]"
```

### Open Viser

- Open **http://localhost:8080** on the host; for remote machines, forward the viewer port over SSH.
- Use the port in Viser's **`listening *:PORT`** banner. A busy port moves the server elsewhere; Newton's later URL can still show 8080.
- Stop each viewer with **Ctrl+C** before starting the next. Closing the browser tab leaves the process running.

### Debug visualization

Change `markers.enabled=false` to `markers.enabled=true` in replay/policy commands to show axes, keypoints, and contacts.

| Markers | Reference target | Simulation |
| --- | --- | --- |
| Hand keypoints | Green | Blue |
| Contacts / normals | Yellow | Magenta |

- **Tracking tolerances:** see [Sharpa](src/flash_chord/configs/termination/tracking.yaml) and [G1](src/flash_chord/configs/termination/recon_body.yaml) for wrist/palm, object, and body tracking limits.

## Example data

Both examples ship under the sibling `source/robotic_grounding/robotic_grounding/assets` tree, mounted read-only at `/v2d/assets`:

```text
/v2d/assets/human_motion_data/
├── ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave/
└── whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/
```

```bash
# Inside the container. Keep these variables for the following steps.
SHARPA_PARQUET=/v2d/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave
G1_PARQUET=/v2d/assets/human_motion_data/whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet
```

- External data: set `V2D_ASSETS_DIR=/absolute/path/to/assets` on the host **before starting a new container**.
- Include `human_motion_data/` and object/support assets. [Asset resolution](V2D_INTEGRATION.md#local-data-visibility). Full corpora are supplied separately.

## 1. View robot

```bash
python scripts/debug/view_robot.py --embodiment sharpa_hands --viewer viser
# Stop the first viewer, then:
python scripts/debug/view_robot.py --embodiment g1_dex3 --viewer viser
```

| Floating Sharpa | G1 + Dex3 |
| --- | --- |
| ![Floating Sharpa robot, camera orbit](docs/media/sharpa_robot.gif) | ![G1 and Dex3 robot, camera orbit](docs/media/g1_robot.gif) |

Default robot poses; only the camera moves.

## 2. View scene

```bash
python scripts/debug/view_scene.py --parquet "${SHARPA_PARQUET}" \
  --embodiment sharpa_hands --frame 0 --viewer viser

# G1: open the SONIC scene, paused at frame 0.
python scripts/debug/replay.py --config-name=sonic_replay \
  "task.parquet='${G1_PARQUET}'" viewer.backend=viser start_paused=true markers.enabled=false
```

| Sharpa tissue-box, frame 0 | G1 snack-box, frame 0 |
| --- | --- |
| ![Sharpa tissue-box reference scene at frame zero](docs/media/sharpa_scene.gif) | ![G1 snack-box reference scene at frame zero](docs/media/g1_scene.gif) |

Check the robot, object, and support surface. Only the camera moves.

## 3. Replay

```bash
python scripts/debug/replay.py "task.parquet='${SHARPA_PARQUET}'" \
  viewer.backend=viser +start_paused=true markers.enabled=false

python scripts/debug/replay.py --config-name=sonic_replay \
  "task.parquet='${G1_PARQUET}'" viewer.backend=viser start_paused=true markers.enabled=false
```

Wait for **“Replay kernels are ready”**, then click **Show kinematic reference** to play the recorded robot/object poses without physics.

| Button | Result |
| --- | --- |
| **Show kinematic reference** | Recorded robot/object poses, without physics |
| **Run configured replay** | Controller-driven physical replay; task success is not guaranteed |

- Each button starts at frame 0 and pauses at the final pose; `start_paused` requires Viser.
- Generic config: `+start_paused=true` adds the key. `sonic_replay`: `start_paused=true` overrides its existing default.

| Sharpa kinematic reference | G1 kinematic reference |
| --- | --- |
| ![Sharpa tissue-box kinematic reference](docs/media/sharpa_reference.gif) | ![G1 snack-box kinematic reference](docs/media/g1_reference.gif) |

## 4. Train

Run one full recipe at a time on the selected GPU:

```bash
python scripts/train_flash_sac.py experiment=sharpa_flash_sac \
  "task.parquet='${SHARPA_PARQUET}'" reset.seed=42 logging.mode=disabled \
  output_dir=outputs/sharpa_full

python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac \
  "task.parquet='${G1_PARQUET}'" reset.seed=42 logging.mode=disabled \
  output_dir=outputs/g1_full
```

| Recipe | Terminal environment steps | Final actor checkpoint |
| --- | ---: | --- |
| Sharpa FlashSAC | 250,003,456 | `outputs/sharpa_full/policy_250003456.safetensors` |
| G1 FlashSAC | 349,999,104 | `outputs/g1_full/policy_349999104.safetensors` |

- Expect compilation, progress through **100%**, and the final checkpoint path.
- `/workspace/outputs` persists under the host's `robotic_grounding/flash_chord/outputs`.
- Without `output_dir`, checkpoints use Hydra's run directory, printed at startup.
- FlashSAC: `policy_<steps>.safetensors` for evaluation; `state_<steps>.safetensors` for resume. PPO: `model_<iteration>.safetensors`.

## 5. Evaluate

```bash
# Select the checkpoint produced above; repeat for the other task.
CHECKPOINT=outputs/sharpa_full/policy_250003456.safetensors
# CHECKPOINT=outputs/g1_full/policy_349999104.safetensors

python scripts/evaluate_policy.py "evaluation.checkpoint=${CHECKPOINT}" \
  evaluation.world_count=4096 "evaluation.metrics_output=${CHECKPOINT}.metrics.json"
```

The checkpoint restores its training configuration and reference; incompatible metadata is rejected.

**MPPE (`metrics.mppe_cm`):** mean per-frame object pose error in cm, using six body-local keypoints at ±5 cm along X/Y/Z. Average keypoint distances per body, take the maximum across bodies per frame, then average over frames and finite simulation worlds. Lower is better.

## 6. View policy

```bash
python scripts/view_policy.py "evaluation.checkpoint=${CHECKPOINT}" \
  viewer.backend=viser markers.enabled=false
```

| Trained Sharpa tissue-box policy | Trained G1 snack-box policy |
| --- | --- |
| ![Trained Sharpa tissue-box pick-and-place policy](docs/media/sharpa_policy.gif) | ![Trained G1 snack-box pick-and-place policy](docs/media/g1_policy.gif) |

Watch **at least five full loops**. Expect pick-and-place followed by a restart at frame 0. [Reset options](docs/configuration.md#policy-resets).

GIFs show the seed-42 terminal actors from release validation. [Media provenance](docs/media/README.md).

## Configuration

Start with an [experiment recipe](src/flash_chord/configs/experiment/); its `defaults` select the Hydra configuration groups below.

| What to change | YAML under [configs/](src/flash_chord/configs/) | Implementation |
| --- | --- | --- |
| Reference data and motion timing | [task/](src/flash_chord/configs/task/) | [data/](src/flash_chord/data/) |
| Rewards: terms, weights, shaping | [objective/](src/flash_chord/configs/objective/) | [objectives/](src/flash_chord/objectives/) |
| Tracking tolerances and terminations | [Sharpa](src/flash_chord/configs/termination/tracking.yaml), [G1](src/flash_chord/configs/termination/recon_body.yaml) | [termination.py](src/flash_chord/lifecycle/termination.py) |
| Episode resets and reset events | [reset/](src/flash_chord/configs/reset/) | [reset.py](src/flash_chord/lifecycle/reset.py) |
| Curriculum and scheduled changes | [curriculum/](src/flash_chord/configs/curriculum/) | [curriculum.py](src/flash_chord/lifecycle/curriculum.py) |
| Scene randomization, objects, collisions | [scene/](src/flash_chord/configs/scene/), [collision/](src/flash_chord/configs/collision/) | [scene/](src/flash_chord/scene/) |
| Physics, control rate, `njmax`, `nconmax` | [sim/](src/flash_chord/configs/sim/) | [sim.py](src/flash_chord/runtime/sim.py) |
| Replay mode and assistance | [replay/](src/flash_chord/configs/replay/) | [replay.py](src/flash_chord/runtime/replay.py) |
| Robot assets and policy actions | [embodiment/](src/flash_chord/configs/embodiment/), [action/](src/flash_chord/configs/action/) | [embodiments/](src/flash_chord/embodiments/), [actions/](src/flash_chord/runtime/actions/) |
| Policy observations | [observation/](src/flash_chord/configs/observation/) | [envs/](src/flash_chord/envs/) |
| Environment and object assistance | [env/](src/flash_chord/configs/env/) | [envs/](src/flash_chord/envs/), [object_control.py](src/flash_chord/runtime/object_control.py) |
| Learner, checkpoints, logging | [training/](src/flash_chord/configs/training/), [logging/](src/flash_chord/configs/logging/) | [training/](src/flash_chord/training/) |
| Evaluation, viewers, debug markers | [evaluation/](src/flash_chord/configs/evaluation/), [viewer/](src/flash_chord/configs/viewer/), [markers/](src/flash_chord/configs/markers/) | [evaluation/](src/flash_chord/evaluation/), [visualization/](src/flash_chord/visualization/) |

- **Events:** use `reset/` for episode initialization, `scene/` for scene-build randomization, and `curriculum/` for scheduled changes.
- **Reward weights/reset probabilities:** curriculum stages can override the base `objective/` and `reset/` settings.
- **Overrides:** append `key=value`, e.g. `scene.world_count=1024`. Sharpa and G1 use different groups; inspect the selected recipe first.

Print the composed configuration without starting training:

```bash
python scripts/train_flash_sac.py experiment=sharpa_flash_sac --cfg job --resolve
python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac --cfg job --resolve
```

More: [recipes and overrides](docs/configuration.md) · [checkpoints](src/flash_chord/training/README.md) · [package structure](docs/package_structure.md) · [V2D integration](V2D_INTEGRATION.md).

## Development

```bash
pytest
ruff check .
```

Sequence-backed tests require local `src/flash_chord/assets/human_motion_data`; otherwise they skip.

## License

Copyright © 2026 NVIDIA Corporation and affiliates. Licensed under the Apache License, Version 2.0.
