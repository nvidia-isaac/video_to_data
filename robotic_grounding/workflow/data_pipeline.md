# Data Pipeline: Raw Motion Data to RL Training

This document describes how hand-object motion capture data flows through the
robotic_grounding pipeline, from raw dataset files to a trained RL policy.

Two parquet formats are in use today:

- **`motion_v1`** (`robotic_grounding.motion_schema`) — the unified whole-body /
  bimanual format used by `soma_to_g1`, `arctic_to_dex3`, and the
  `g1_planner` output. See
  [../source/robotic_grounding/robotic_grounding/motion_schema/README.md](../source/robotic_grounding/robotic_grounding/motion_schema/README.md).
- **`ManoSharpaData`** — the legacy dual-hand V2P retarget pipeline described
  below. Still used end-to-end by arctic/taco/oakink2/hot3d/h2o/grab/dexycb
  loaders and the dual-hand v2p tracking command. Not on `motion_v1` yet.

## Pipeline Overview

```
Raw Data on CSS (pkl/jsonl/csv/npy/...)
    |
    v  Stage 1: Load  (reconstruction v2d_{dataset}_load workflow; GPL MANO FK)
ManoSharpaData Parquet (MANO hands + object poses)  -->  {dataset}_loaded/ swift prefix
    |
    |-- Stage 1.5: Generate URDFs         --> {dataset}_urdfs/*.urdf
    |
    v  Stage 2: Retarget (IK)
ManoSharpaData Parquet (+ robot joint trajectories)
    |
    |-- Stage 3: Support surfaces         --> reconstructed_stage/*.usda
    |-- Stage 4: Visualize (optional)     --> {dataset}_html/recordings/{seq}.viser (+ .mp4)
    |-- Stage 5: Record video (optional)  --> {dataset}_videos/{seq}.mp4 (Isaac Sim)
    |
    v  (All stages' outputs written to the swift output_url prefix)
swift://…/human_motion_data/{dataset}/{dataset}_processed/ (+ _urdfs, reconstructed_stage, _html, _videos)
    |
    v  Stage 6: sync_css_data.py / aws s3 sync
Local assets/human_motion_data/{dataset}/
    |
    v  Stage 7: RL Training (Isaac Sim)
Trained policy checkpoint
```

Stage 1 (Load) runs as a **separate workflow in the `reconstruction` repo**
(`v2d_{dataset}_load`): it carries the GPL-3.0 MANO/manotorch forward-kinematics
code, contained in its own image, and writes a `{dataset}_loaded` Parquet to a
swift prefix. Stages 1.5–5 are the robotic_grounding `retarget.yaml` workflow,
which **consumes** that `{dataset}_loaded` data (it is manotorch-free); any
subset can run via `--set stages=<stage>`. All `retarget.yaml` artifacts from one
run are written to the swift `output_url` prefix (see [data_storage.md](data_storage.md)).

## Stage 1: Load

**Purpose:** Parse raw dataset files into a standardized Parquet schema.

> **Home:** Stage 1 lives in the **`reconstruction`** repo, not here. It uses
> MANO forward kinematics (manotorch, GPL-3.0), so it is contained in the
> `v2d_task_library_loader` image and run as the `v2d_{dataset}_load` OSMO
> workflow. robotic_grounding consumes the resulting `{dataset}_loaded` dataset
> and never imports manotorch. The description below documents the contract.

**Module:** `reconstruction/modules/v2d_task_library_loader/lib/{dataset}_loader.py`
(dispatched via `lib/run_loader.py`, i.e. `python -m v2d.task_library_loader.lib.run_loader`)

Each dataset stores motion data differently (pickles, JSONL, CSV, NPY). The
loader reads the raw format and writes a `ManoSharpaData` Parquet containing:

- **MANO hand parameters** -- wrist position/orientation, 45-DOF finger pose,
  shape betas, fitting error, per frame, per hand (left + right).
- **Object poses** -- per-body position and quaternion, per frame.
- **Object metadata** -- mesh file paths, URDF paths, body names, mesh radius.
- **Contact data** (optional) -- per-link contact positions and normals between
  hand surface and object surface (16 MANO links per hand).

The loader also handles dataset-specific preprocessing:

- Coordinate frame transforms (Y-up to Z-up for Quest3, OakInk2).
- MANO PCA expansion (15 PCA coefficients to 45-DOF axis-angle for Hot3D).
- Timestamp deduplication (Hot3D multi-camera streams to ~30 Hz).
- Mesh scale normalization (centimeters to meters for TACO).

**Output:** `human_motion_data/{dataset}/{dataset}_loaded/` partitioned by
`sequence_id` and `robot_name`.

**In-container command** (what runs inside the loader image):
```bash
python -m v2d.task_library_loader.lib.run_loader \
  --dataset <name> \
  --dataset_root <raw_dataset_dir> \
  --object_model_root <dir with the dataset's *.urdf> \
  --mesh_dir <dir with the dataset's object meshes> \
  --mano_model_dir <dir with models/MANO_{LEFT,RIGHT}.pkl> \
  --output_dir <path> \
  --device cuda:0 --save
```

**Run it locally** (host wrapper — mounts the inputs and runs the in-container
command above). Raw data lives under `<human_motion_data_dir>/<dataset>/dataset/`;
`--object_assets_dir` is the root holding `urdfs/<dataset>/` + `meshes/<dataset>/`:

```bash
python -m v2d.task_library_loader.docker.run_loader \
  --dataset arctic \
  --human_motion_data_dir <dir containing arctic/dataset/...> \
  --object_assets_dir <dir containing urdfs/arctic + meshes/arctic> \
  --mano_model_dir <dir with models/MANO_{LEFT,RIGHT}.pkl> \
  --output_dir <out> \
  --max_sequences 2 --save
```

Omit `--object_assets_dir` for h2o/grab/dexycb (meshes come from the raw dataset).

Or submit `reconstruction/workflows/task_library_load/osmo/load.yaml` on OSMO.

## Stage 1.5: Generate Rigid URDFs

**Purpose:** Create URDF files that Isaac Sim can load for each object mesh.

**Script:** `scripts/generate_rigid_urdfs.py`

Isaac Sim requires URDF (or USD) descriptions to spawn objects. This stage
converts raw object meshes (OBJ, GLB) into single-link rigid URDFs:

1. Load source mesh via trimesh.
2. Export a clean visual STL (avoids Isaac Sim OBJ parser issues).
3. Generate a URDF with visual + collision geometry and default inertia.

The mesh scale comes from the dataset registry (`mesh_vertex_scale`). This
stage is idempotent -- it skips objects that already have URDFs. Runs
automatically inside the `process` stage on OSMO.

Locally, the source meshes are not committed — make sure they're in place first
(see [Object Assets](#object-assets-urdfs--meshes)).

**Output:** `assets/urdfs/{dataset}/{object_id}_rigid.urdf` locally; in the
OSMO workflow these are copied to `{dataset}_urdfs/` in the published
dataset version.

**Command:**
```bash
python scripts/generate_rigid_urdfs.py --dataset <name>
```

## Stage 2: Retarget (IK)

**Purpose:** Convert human MANO hand poses into robot joint trajectories.

**Script:** `scripts/retarget/{dataset}_to_sharpa.py` (dispatched via `scripts/retarget/run_retarget.py`)

For each frame in the loaded Parquet:

1. Read MANO wrist position, orientation, and finger joint angles.
2. Run inverse kinematics to solve the Sharpa Wave robot's 22 finger joints
   (per hand) to match MANO fingertip positions.
3. Compute 67 task-space reference "frames" (robot link poses) used for
   reward computation during training.
4. Record IK error metrics and optimization iteration counts.

The retarget script appends robot-specific fields (wrist pose, finger joints,
frames) to the existing Parquet alongside the original MANO and object data.

**Output:** `human_motion_data/{dataset}/{dataset}_processed/` (same partition
structure, now includes SHARPA robot fields).

**Command** (in-container — robotic_grounding image):
```bash
python scripts/retarget/run_retarget.py \
  --dataset <name> --robot sharpa_wave \
  --input_dir <loaded_dir> \
  --output_dir <processed_dir> \
  --device cuda:0 --save
```

**Run it locally** (override the image's Isaac Sim entrypoint — retarget is
pinocchio IK, no sim):

```bash
docker run --rm --gpus all --entrypoint /bin/bash \
  -v "$PWD/..:/workspace/video_to_data" \
  -v <loaded_dir>:/data/loaded -v <processed_dir>:/data/processed \
  -w /workspace/video_to_data/robotic_grounding \
  robotic-grounding:<tag> \
  -c "python scripts/retarget/run_retarget.py --dataset <name> --robot sharpa_wave \
       --input_dir /data/loaded --output_dir /data/processed --device cuda:0 --save"
```

(Or `./workflow/run.sh start` then run the in-container command.)

## Stage 3: Reconstruct Support Surfaces (optional)

**Purpose:** Build collision meshes for surfaces that objects rest on.

**Script:** `scripts/reconstruct_support_surfaces.py`

Analyzes object still-poses across frames to identify the flat surface each
object contacts (e.g., a table). Generates a thin USD disk mesh at the
estimated support plane. These surfaces improve contact reward accuracy during
training by providing a physical ground for the objects.

**Output:** `human_motion_data/{dataset}/reconstructed_stage/*.usda`

**Command:**
```bash
python scripts/reconstruct_support_surfaces.py \
  --input_dir <loaded_dir> --dataset <name>
```

## Stage 4: Visualize (optional)

**Purpose:** Produce inspection artifacts alongside the retargeted data —
both interactive (`.viser`) and offline video (`.mp4`). Drives quality
reviews without downloading the full parquets locally.

**Script:** `scripts/retarget/vis_retargeted.py`

Running with `--save_html --save_mp4` writes, for every sequence:

- `<seq>.viser` — web-based playback, shows MANO hands + Sharpa robot +
  objects + support surfaces; open in `viser-client/?playbackPath=...`.
- `<seq>.mp4` — headless pyrender render of the same scene, auto-framed on
  the object trajectory. No browser or Isaac Sim required. Useful for
  quick QA at scale, embedding in reviews, or sanity-checking new datasets.

Pyrender MP4s are skipped for OakInk2 (120 FPS × long sequences OOMs the
pod); Stage 5 already produces Isaac Sim MP4s for those.

**Output:** `{dataset}_html/recordings/{seq}.viser` (+ `.mp4` where not
skipped) under the swift output prefix.

## Stage 5: Record Isaac Sim Video (optional)

**Purpose:** Record a playback MP4 of each retargeted sequence in Isaac Sim
with the Sharpa robot physically grasping the object — the closest thing
to a training-time render without actually training. Used for catching
regressions that don't show up in the MANO-only pyrender output (e.g.
collision-group misconfigurations, self-penetrations, physics blow-ups).

**Script:** `scripts/rsl_rl/dummy_agent.py` (driven by the workflow's
`video` stage, not typically invoked directly)

For each processed sequence the workflow spawns a single-env Isaac Sim
instance with `dummy_agent.py --record_video`, steps 300 frames of the
motion, and saves the MP4. Runs sequentially on one GPU — parallel Isaac
Sim startup contention (shader cache, EGL drivers, VRAM) made the sharded
version strictly slower in practice.

**Output:** `{dataset}_videos/{sequence_id}.mp4` under the swift output prefix.

## Stage 6: Sync Results Locally

**Purpose:** Pull the workflow's swift outputs to the local repo so training
and local visualization scripts can read them.

**Command:** `sync_css_data.py` / `aws s3 sync` — full details and
component-filter examples are in
[data_storage.md](data_storage.md#pulling-retarget-outputs-locally).

Quick version:

```bash
source scripts/setup_css_env.sh

# Pull the pieces training needs (processed / loaded / support_surfaces)
python scripts/sync_css_data.py --dataset <dataset> --component processed

# Other components (urdfs/html/videos) via the aws CLI against the CSS endpoint
aws s3 sync \
  s3://datasets/v2d/human_motion_data/<dataset>/<dataset>_urdfs/ \
  source/robotic_grounding/robotic_grounding/assets/human_motion_data/<dataset>/<dataset>_urdfs/ \
  --endpoint-url ${CSS_ENDPOINT_URL} --region us-east-1
```

## Stage 7: RL Training

**Purpose:** Train a reinforcement learning policy to control the robot hand.

**Script:** `scripts/rsl_rl/train.py` (runs inside Docker with Isaac Sim)

### Scene Setup

1. `SceneConfig` reads the processed Parquet, resolves object URDFs (via
   object registry, parquet paths, or mesh-derived fallback), and loads
   support surfaces if available.
2. Isaac Sim spawns the Sharpa Wave robot and objects in N parallel
   environments.

### Motion Command System

3. The `CommandManager` loads retargeted motion data from the Parquet,
   interpolates it to the simulation timestep, and provides per-step
   target poses for both hands and all objects.
4. Contact tensors are initialized from Parquet data (or set to zero if the
   dataset has no contact annotations).

### Training Loop (PPO)

5. Each simulation step:
   - The command manager provides target robot joint positions, wrist poses,
     and object poses from the motion trajectory.
   - The policy network outputs joint position deltas.
   - The simulation advances one physics step.

6. Rewards measure how well the robot follows the reference motion:

   | Reward | What it tracks |
   |--------|---------------|
   | `hand_keypoints_tracking` | Wrist + fingertip position error |
   | `hand_joint_pos_tracking` | Finger joint angle error |
   | `object_keypoints_tracking` | Object position/orientation error |
   | `contact_force_reward` | Matching reference contact patterns |
   | `contact_wrench_support_reward` | Wrench-space contact quality |
   | `action_rate_l2` / `action_l1` | Motion smoothness penalties |

   Contact-based rewards gracefully degrade to zero for datasets without
   contact data (e.g., Hot3D).

7. Episodes reset on timeout or when the hand/object drifts too far from
   the reference trajectory.

**Output:** Policy checkpoints in `logs/`.

**Command:**
```bash
python scripts/rsl_rl/train.py \
  --task Sharpa-V2P-v0 \
  --headless \
  --motion_file <dataset>/<dataset>_processed/<sequence_id>/sharpa_wave \
  --num_envs 8
```

## OSMO Workflow

Stages 1.5 through 5 are submitted as a single OSMO workflow (`workflow/retarget.yaml`) running on the GPU cluster (Stage 1 Load runs separately in `reconstruction`). All artifacts from one run are written to the swift `output_url` prefix (`…/human_motion_data/{dataset}/`). See [README.md](README.md) for submission options and [data_storage.md](data_storage.md) for the output layout; `/osmo-retarget` skill covers usage patterns.

```bash
python scripts/run_osmo.py \
  --experiment-name retarget-<dataset> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=<name>

# Run a subset — e.g. skip the expensive video stage
python scripts/run_osmo.py \
  --experiment-name retarget-<dataset>-fast \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=<name> --set stages=process
```

## Dataset Inventory

Candidates considered for Robot Grounding Task Library. Source:
[dataset-candidates spreadsheet](https://docs.google.com/spreadsheets/d/1aG_m4I1vuFdM5LkeHmSbhlF5zBxTRo-3Qu7TGThVt1M/edit).

### In the pipeline

Outputs live under the swift `…/human_motion_data/<name>/` prefix.

| Dataset | Sequences | GT Fidelity | Hands | Notes |
|---------|-----------|-------------|-------|-------|
| **TACO** | 2,317 | High (NOKOV MoCap) | Bimanual | — |
| **Arctic** | 554 | Very High (Vicon MoCap) | Bimanual, articulated objects | — |
| **OakInk2** | 627 | High (OptiTrack MoCap) | Bimanual | Stage 4 MP4s skipped (long sequences OOM pyrender) |
| **HOT3D** | 294 | Very High (OptiTrack MoCap) | Bimanual | — |
| **H2O** | 137 | Medium-High (RGB-D opt.) | Bimanual | cam4 egocentric; known ~45° pitch (initial head tilt baked into PnP world frame) |
| **GRAB** | 1,335 | Very High (Vicon MoCap) | Bimanual + SMPL-X full body | — |
| **DexYCB** | 1,000 | Medium-High (multi-view RGB-D) | Single (per-session) | cam `932122062010`; master-frame gravity inferred per session |

Current total: **~6,264 retargeted sequences** across 7 datasets and 150+ unique objects.

### Next candidates (have MANO + tracked objects)

Ordered by expected onboarding effort.

| Dataset | Sequences | GT Fidelity | Notes |
|---------|-----------|-------------|-------|
| **HOI4D** | 4,000+ | Medium | 16 categories, rigid + articulated. Manual BBox annotation every 10 frames. Would roughly double the sequence count — biggest single-dataset gain available. |
| **FPHA** | 1,175 | Medium | 45 action categories. Older (2018), uses magnetic sensors (~2-3mm accuracy). Only 4/26 objects have meshes — most of the sequences would land without a mesh for training, limiting value. |

### Not usable without extra preprocessing

These datasets have MANO hands (or pseudo-labels) but no reliable object pose
ground truth, so they can't flow through the retarget pipeline as-is.

| Dataset | Sequences | Why |
|---------|-----------|-----|
| HoloAssist | 2,200 | HaMeR pseudo-labels for hands, no object tracking |
| Taste-Rob | ~100K | Same — pseudo-labels + no object poses |
| EgoDex | 338K | ARKit skeleton only, no object tracking |
| ContactPose | 2,306 | Static grasps only (no temporal sequences) |

## Adding a New Dataset

See the `/add-dataset` skill or run through these steps:

1. Add a `DatasetConfig` entry in `source/robotic_grounding/robotic_grounding/retarget/dataset_registry.py`.
2. Write a **loader** in the `reconstruction` repo at
   `reconstruction/modules/v2d_task_library_loader/lib/<name>_loader.py` (this is
   the GPL/MANO stage) and register it in `lib/loader_registry.py`.
3. Write a **retarget** script here at `scripts/retarget/<name>_to_sharpa.py`.

Everything else (workflow dispatch, swift output publish, URDF generation,
training validation) is driven by the dataset registry automatically.

## Validating Before Training

Run the asset validator to catch missing URDFs or meshes before Isaac Sim
starts (saves the 45-second startup wait):

```bash
python scripts/validate_training_assets.py --dataset <name>
```

## Object Assets (URDFs + Meshes)

Stage 1.5 (URDF gen), Stage 2 (retarget), Stage 3/4 (reconstruct/visualize), and
Stage 7 (training) load object geometry from:

- `source/robotic_grounding/robotic_grounding/assets/meshes/<dataset>/` — object meshes
- `source/robotic_grounding/robotic_grounding/assets/urdfs/<dataset>/`  — rigid URDFs

**Obtaining them.** Download the dataset's object models from its original
distribution (per the dataset's own license/terms; see [Dataset Inventory](#dataset-inventory)),
copy the meshes into `assets/meshes/<dataset>/`, then generate the rigid URDFs:

```bash
python scripts/generate_rigid_urdfs.py --dataset <dataset>
```

For h2o/grab/dexycb the object meshes ship inside the raw dataset, so the loaders
read them from the raw tree — no separate mesh placement needed.

<!-- INTERNAL-ONLY:START — remove before public release (TODO(public-release)) -->
**NVIDIA-internal shortcut.** The prebuilt meshes + URDFs are mirrored on CSS;
pull them straight into place instead of re-downloading + regenerating:

```bash
python scripts/fetch_object_assets.py --dataset <name>   # or --dataset all
```

(upload with `scripts/upload_object_assets.py`). OSMO retarget outputs also carry
a regenerated `{dataset}_urdfs/` tree.
<!-- INTERNAL-ONLY:END -->

> Still open: wire object-asset provisioning into the Docker image build / OSMO
> retarget so the cloud path doesn't rely on a committed copy.
