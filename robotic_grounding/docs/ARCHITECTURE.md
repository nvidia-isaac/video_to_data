# Robotic Grounding — Architecture & Contents

This is the architecture and contents reference for the `robotic_grounding` package. For run commands see [../README.md](../README.md).

## Table of contents

1. [What this package is](#1-what-this-package-is)
2. [The big picture (data flow)](#2-the-big-picture-data-flow)
3. [Top-level directory map](#3-top-level-directory-map)
4. [The source package](#4-the-source-package)
5. [scripts](#5-scripts)
6. [Workflows & deployment](#6-workflows--deployment)
7. [Datasets](#7-datasets)
8. [RL training & tasks](#8-rl-training--tasks)
9. [Visualizer, tests & CI](#9-visualizer-tests--ci)
10. [Where do I go for X?](#10-where-do-i-go-for-x)
11. [Conventions & gotchas](#11-conventions--gotchas)

---

## 1. What this package is

`robotic_grounding` turns mocap/video **hand-object and whole-body human motion** into **retargeted robot motion** and uses it to train **RL policies in Isaac Lab**. Supported robots: **Sharpa Wave** and **Dex3** hands, **G1** whole-body. The package owns the full pipeline from raw dataset to trained policy — dataset ingestion via MANO forward-kinematics, IK-based retargeting to the target robot, support-surface reconstruction for sim, and RSL-RL training inside Isaac Lab. For [run commands](../README.md) see the top-level README.

---

## 2. The big picture (data flow)

The pipeline takes a raw mocap/video dataset plus MANO hand models and produces retargeted robot motion ready for RL training:

```
raw dataset  +  MANO models
          │
          ▼
╔═══════════════════════════════════════════════╗
║  IMAGE 1 · v2d_task_library_loader              ║
║  stage: load   —   MANO forward-kinematics      ║
╚═══════════════════════════════════════════════╝
          │
          ▼
     <ds>_loaded
  Parquet — per-frame MANO joints + object poses
          │
          ▼
╔═══════════════════════════════════════════════╗
║  IMAGE 2 · robotic-grounding                    ║
║  segment → urdf → processed → support → vis     ║
║  (IK retarget · support surfaces · viz)         ║
╚═══════════════════════════════════════════════╝
          │
          ▼
       outputs
  <ds>_processed                retargeted robot motion
  object_assets/urdfs/<ds>      object URDFs
  reconstructed_stage/*.usda    support surfaces
  <ds>_html/                    viser recording + MP4
          │
          ▼
    Isaac Lab RL training
```

**IMAGE 1** (`v2d_task_library_loader`) reads the raw dataset and MANO models and runs MANO forward-kinematics to produce `<ds>_loaded` — a Parquet of per-frame MANO joints and object poses. **IMAGE 2** (`robotic-grounding`) takes that Parquet and runs: optional `segment` (atomic clip splitting, hot3d only), `urdf` (object URDF generation), `processed` (IK retargeting to the robot), `support` (support-surface reconstruction for sim collision geometry), and `vis` (viser recording + MP4). Both images are driven from the host by a single orchestrator script — `scripts/run_pipeline_docker.py` — with no need to shell into a container.

Full end-to-end flow: [`../workflow/data_pipeline.md`](../workflow/data_pipeline.md); dataset download/layout: [`SETUP.md`](SETUP.md).

---

## 3. Top-level directory map

| Path | What it is |
|------|------------|
| `source/` | The installable `robotic_grounding` Python package (Isaac Lab extension). |
| `scripts/` | Runnable entry points: pipeline orchestration, retargeting, RL training, data-quality checks, and asset generation. |
| [`workflow/`](../workflow/) | Docker + OSMO deployment definitions for containerised pipeline runs. |
| `docs/` | This document plus setup and dataset guides. |
| [`visualizer/`](../visualizer/) | 3D gallery server for browsing retargeted sequences interactively. |
| `tests/` | pytest suite covering schema correctness, retarget end-to-end, replay, and training end-to-end. |
| `ci/` | CI configuration. |
| `run_example_sequences.sh`, `run_load_local.sh`, `run_pipeline_local.sh`, `run_retarget_local.sh` | Local convenience wrappers that mirror the container pipeline stages without Docker. |
| `pyproject.toml` | Package metadata (`name = "robotic_grounding"`) and lint configuration. |
| `.pre-commit-config.yaml` | Pre-commit hooks including license-header enforcement. |

---

## 4. The source package

The installable Python package lives at `source/robotic_grounding/robotic_grounding/` and is registered as an Isaac Lab extension in [`../source/robotic_grounding/config/extension.toml`](../source/robotic_grounding/config/extension.toml).

### motion_schema

The versioned Parquet **motion interchange format** (`motion_v1`) shared across the whole pipeline. Producers (the retargeter and planner) write it; consumers (training loaders, replay, support reconstruction, the visualiser) read it through a single stable API. Key files: `schema.py`, `reader.py`, `writer.py`. See [`motion_schema/README.md`](../source/robotic_grounding/robotic_grounding/motion_schema/README.md).

### retarget

**Hand and whole-body IK retargeting** from MANO/SOMA representations to target robots, plus the dataset registry, ground alignment, and support-surface reconstruction. Per-robot configuration lives under `retarget/configs/<robot_name>/` and is loaded via `robot_config.py`. Key files: `dataset_registry.py`, `hand_kinematics.py`, `whole_body_kinematics.py`, `support_recon.py`, `ground_alignment.py`, `robot_config.py`. See [`retarget/configs/README.md`](../source/robotic_grounding/robotic_grounding/retarget/configs/README.md).

### tasks

**Isaac Lab RL environments** organised into three subpackages:

- `v2d/` — dual floating-hand environments for Sharpa Wave and Dex3; entry-point config: `v2d_hand_env_cfg.py`.
- `v2d_whole_body/` — G1 whole-body humanoid environments (`SonicG1-v0`, `SonicG1-ReconBody-v0`, `SonicG1-ReconHand-v0`, `SonicG1-ReconHand-EpisodeTimeout-v0`) using the SONIC controller with RL residuals. See [`tasks/v2d_whole_body/README.md`](../source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/README.md).
- `scene_utils/` — scene assembly helpers and the replay driver shared across task types.

Each task subpackage contains an `mdp/` submodule (actions, commands, rewards, terminations) that defines the Markov Decision Process. The whole-body `mdp/` subdirectories include their own README files with reward and observation details.

### planner

**G1 trajectory / motion planning** ("motionbricks") that takes end-effector targets from V2D retargeting and runs a learned motion model to produce full-body joint trajectories, emitted as `motion_v1` Parquet. Key files: `g1_planner.py`, `trajectory.py`, `cli.py`, and the `motionbricks/` submodule. See [`planner/README.md`](../source/robotic_grounding/robotic_grounding/planner/README.md).

### assets

Bundled **robot models and data** used across the package:

- `xmls/` — MJCF models for `sharpawave` and `g1`.
- `urdfs/` — URDF robot descriptions.
- `meshes/` — collision and visual mesh files.
- `actuators/` — actuator configuration data.
- `policies/` — bundled policy assets: `grasp` (grasp helpers) and `sonic` (SONIC neural-controller ONNX weights).
- `body_models/` — SOMA body model files.
- `human_motion_data/` — mount point for processed motion sequences produced by the pipeline.

---

## 5. scripts/

All runnable entry points live under `scripts/`. They are grouped by function below; for exact invocation commands see [`../README.md`](../README.md) and [`SETUP.md`](SETUP.md).

**Pipeline orchestration**
- `run_pipeline_docker.py` — host-side orchestrator; runs each pipeline stage (`load`, `segment`, `urdf`, `processed`, `support`, `vis`) in the appropriate Docker image without manual container entry.
- `run_osmo.py` — submits the pipeline as an OSMO cloud job.

**Retarget (`retarget/`)**
- Per-dataset converters from MANO/SOMA representations to robot joint space: `arctic_to_sharpa.py`, `arctic_to_dex3.py`, `dexycb_to_sharpa.py`, `grab_to_sharpa.py`, `h2o_to_sharpa.py`, `hot3d_to_sharpa.py`, `oakink2_to_sharpa.py`, `taco_to_sharpa.py`, `taco_to_dex3.py`, `soma_to_g1.py`.
- `run_retarget.py` — unified retarget launcher that dispatches to the appropriate per-dataset converter.
- `vis_retargeted.py` — viser-based interactive viewer for retargeted sequences.

**RL (`rsl_rl/`)**
- `train.py`, `eval.py`, `dummy_agent.py`, `cli_args.py` — RSL-RL training, evaluation, dummy-agent baseline, and shared CLI argument definitions.

**Data quality (`data_quality_checks/`)**
- `hand_penetration.py`, `arctic_support_disks.py`, `dummy_agent_success.py` — per-sequence quality checks for hand–object penetration, support-disk validity, and dummy-agent success rate.
- `data_assessor.py`, `filter_penetrations.py` — top-level scripts that apply quality checks across an entire processed dataset and filter out failing sequences.

**Asset generation**
- `generate_rigid_urdfs.py` — generates per-object rigid URDFs from mesh assets.
- `reconstruct_support_surfaces.py` — reconstructs support-surface meshes (`.usda`) from processed motion data.
- `setup_soma_assets.py` — prepares SOMA body model assets required for whole-body retargeting.

**Download / replay / view**
- `replay_motion.py`, `replay_motion_viser.py` — replay a `motion_v1` Parquet sequence (headless and viser-based, respectively).
- `view_scene.py` — inspect a reconstructed scene (support surfaces + object URDFs) in a viser window.

For exact commands see [`../README.md`](../README.md) and [`SETUP.md`](SETUP.md).

---

## 6. Workflows & deployment

### Docker

Container lifecycle is managed by `workflow/run.sh`, which accepts `build`, `start`, `shell`, and `stop` subcommands. The image is defined in `workflow/Dockerfile`; Python and system dependencies are installed by `workflow/setup_deps.sh`. The convention is: **development and pipeline execution happen inside the container; git operations (commit, push, branch) are done on the host outside the container**. See the [Docker Usage section of `../README.md`](../README.md#docker-usage) for details.

### OSMO cloud runs

For multi-GPU or scheduled cloud execution, OSMO job manifests are provided:
- `workflow/train.yaml` — RL training job.
- `workflow/retarget.yaml` — retarget pipeline job.
- `workflow/dev_env.yaml` — interactive dev environment.

### Workflow documentation

| Document | What it covers |
|----------|----------------|
| [`../workflow/README.md`](../workflow/README.md) | OSMO and NGC setup, image registry, how to launch jobs. |
| [`../workflow/data_pipeline.md`](../workflow/data_pipeline.md) | End-to-end pipeline walk-through: stage order, inputs/outputs, and how to run each stage. |

---

## 7. Datasets

Seven hand-object motion datasets are supported: **taco**, **hot3d**, **arctic**, **grab**, **h2o**, **dexycb**, and **oakink2**.

### `<HMD>` convention

All dataset sources share a single root directory, referred to as `<HMD>` (human-motion-data root). Its layout is:

```
<HMD>/
  mano/          # MANO hand model weights (license-gated, never committed)
  taco/
  arctic/
  grab/
  h2o/
  hot3d/
  dexycb/
  oakink2/
```

Retargeted output for each dataset lands under `<HMD>/<dataset>/<dataset>_processed/...` (e.g. `<HMD>/arctic/arctic_processed/<sequence_id>/sharpa_wave`).

### MANO requirement

MANO weights are **license-gated**: they must be obtained directly from the MANO website and placed under `<HMD>/mano/`. They are read only at load time and must never be committed to the repository. See [`SETUP.md`](SETUP.md) for download and placement instructions.

### Per-dataset setup guides

| Dataset | Setup guide | Notes |
|---------|-------------|-------|
| arctic | [`ARCTIC_SETUP.md`](ARCTIC_SETUP.md) | |
| dexycb | [`DEXYCB_SETUP.md`](DEXYCB_SETUP.md) | |
| grab | [`GRAB_SETUP.md`](GRAB_SETUP.md) | |
| h2o | [`H2O_SETUP.md`](H2O_SETUP.md) | |
| hot3d | [`HOT3D_SETUP.md`](HOT3D_SETUP.md) | |
| taco | [`TACO_SETUP.md`](TACO_SETUP.md) | |
| oakink2 | [`SETUP.md`](SETUP.md) | General setup; also covers oakink2 (grab has its own guide above) |

Example sequences (one representative sequence per dataset for quick smoke tests) are documented in [`EXAMPLE_SEQUENCES.md`](EXAMPLE_SEQUENCES.md).

> **Note:** This release ships the dataset setup guides and the retarget → train pipeline described here. The task library also includes additional dataset processing scripts (further data curation and conversion) that are **not part of this release** and will be published in a later version.

---

## 8. RL training & tasks

### Registered Gymnasium tasks

**Hand — dual floating-hand (Sharpa Wave), `tasks/v2d/`:**

| Task ID | Purpose |
|---------|---------|
| `Sharpa-V2D-v0` | Standard training task |
| `Sharpa-V2D-v0-Play` | Evaluation / playback task |

**Whole-body — G1 humanoid (SONIC controller + RL residuals), `tasks/v2d_whole_body/`:**

| Task ID | Purpose |
|---------|---------|
| `SonicG1-v0` | Base env (JOINT_RESIDUAL action, reward weights zeroed) — a scaffold for custom reward configs via Hydra |
| `SonicG1-ReconBody-v0` | Body-accurate reference from third-person video reconstruction (MHR) |
| `SonicG1-ReconHand-v0` | Hand-accurate reference from the EE-based planner pipeline |
| `SonicG1-ReconHand-EpisodeTimeout-v0` | `ReconHand` variant using a fixed episode-length timeout instead of trajectory-end termination |

See [`tasks/v2d_whole_body/README.md`](../source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/README.md) for the full whole-body task, observation, and reward reference.

### Entry points

Training and evaluation scripts live in `scripts/rsl_rl/`:

| Script | Purpose |
|--------|---------|
| `train.py` | Launch RSL-RL training |
| `eval.py` | Run evaluation against a trained checkpoint |
| `dummy_agent.py` | Dummy-agent baseline (always-zero actions) |

### Motion file shorthand

Motion sequences are addressed with the shorthand path:

```
<dataset>/<dataset>_processed/<sequence_id>/sharpa_wave
```

For asset-free smoke tests, pass `--use_primitive_urdfs` to replace per-object URDFs with primitive collision shapes.

For full run commands (smoke tests, full training, evaluation) see the "Agent Smoke Tests" and "RL training" sections of [`../README.md`](../README.md).

---

## 9. Visualizer, tests & CI

### Visualizer

The `visualizer/` package provides a lightweight 3D gallery server for browsing retargeted sequences. Each sequence renders a split-screen view: a viser 3D playback panel and an MP4 camera-feed panel. See [`../visualizer/README.md`](../visualizer/README.md) for setup, data-sync instructions, and usage.

### Tests

`tests/` contains the following test modules:

| File | What it covers |
|------|----------------|
| `test_motion_schema.py` | Unit tests for the `motion_v1` Parquet schema |
| `test_motion_schema_parquet_integration.py` | Integration tests for Parquet round-trips |
| `test_retarget_pipeline_e2e.py` | End-to-end retarget pipeline |
| `test_replay_data.py` | Motion replay correctness |
| `test_train_e2e.py` | End-to-end RL training smoke test |
| `test_reference_plane.py` | Reference-plane geometry |
| `test_whole_body_kinematics_baseline.py` | Whole-body kinematics baseline |

### CI

`ci/` holds CI job configuration. The repository also ships `.pre-commit-config.yaml` (at the `robotic_grounding/` root) which enforces license headers and code formatting on every commit.

---

## 10. Where do I go for X?

| I want to… | Go to |
|------------|-------|
| Understand the data flow | [§2 The big picture](#2-the-big-picture-data-flow) · [`../workflow/data_pipeline.md`](../workflow/data_pipeline.md) |
| Add a new dataset | [§4 `retarget` (`dataset_registry.py`)](#4-the-source-package) · [`SETUP.md`](SETUP.md) |
| Download / lay out a dataset | [§7 Datasets](#7-datasets) · per-dataset `*_SETUP.md` in [`docs/`](./) |
| Retarget one sequence | [§5 scripts/](#5-scripts) · [`../README.md`](../README.md) (Retargeting section) |
| Run a training smoke test | [§8 RL training & tasks](#8-rl-training--tasks) · [`../README.md`](../README.md) (RL training section) |
| Understand the motion Parquet schema | [§4 `motion_schema`](#4-the-source-package) · [`../source/robotic_grounding/robotic_grounding/motion_schema/README.md`](../source/robotic_grounding/robotic_grounding/motion_schema/README.md) |
| Visualize a retargeted result | [§9 Visualizer, tests & CI](#9-visualizer-tests--ci) · [`../visualizer/README.md`](../visualizer/README.md) |
| Run on OSMO / the cloud | [§6 Workflows & deployment](#6-workflows--deployment) · [`../workflow/README.md`](../workflow/README.md) |

---

## 11. Conventions & gotchas

- **MANO is never committed.** MANO weights are license-gated and read only at the `load` stage. Always pass them via `--mano-dir`; do not add them to the repository.
- **Motion-file shorthand.** The path `<dataset>/<dataset>_processed/<sequence_id>/sharpa_wave` resolves under `assets/human_motion_data/` inside the container (or the local `<HMD>` root on the host).
- **`--use_primitive_urdfs` for asset-free smoke tests.** This flag replaces per-object URDFs with primitive collision shapes so that `dummy_agent` and training smoke tests can run without first completing the `urdf` pipeline stage.
- **Two Docker images — don't confuse them.** `v2d_task_library_loader` runs the `load` stage (MANO forward-kinematics). `robotic-grounding` runs all subsequent stages (`segment`, `urdf`, `processed`, `support`, `vis`). See [§2](#2-the-big-picture-data-flow) for the flow.
- **Single source of truth.** Run commands live in `README.md` and `SETUP.md`. This document links to those files rather than duplicating commands; update the source files, not this doc.
