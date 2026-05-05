# Data Pipeline: Raw Motion Data to RL Training

This document describes how hand-object motion capture data flows through the
robotic_grounding pipeline, from raw dataset files to a trained RL policy.

Two parquet formats are in use today:

- **`motion_v1`** (`robotic_grounding.motion_schema`) — the unified whole-body /
  bimanual format used by `nvhuman_to_g1`, `nvhuman_to_dex3`, and the
  `g1_planner` output. See
  [../source/robotic_grounding/robotic_grounding/motion_schema/README.md](../source/robotic_grounding/robotic_grounding/motion_schema/README.md).
- **`ManoSharpaData`** — the legacy dual-hand V2P retarget pipeline described
  below. Still used end-to-end by arctic/taco/oakink2/hot3d/h2o/grab/dexycb
  loaders and the dual-hand v2p tracking command. Not on `motion_v1` yet.

## Pipeline Overview

```
Raw Data on CSS (pkl/jsonl/csv/npy/...)
    |
    v  Stage 1: Load
ManoSharpaData Parquet (MANO hands + object poses)
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
    v  (All stages' outputs published as one OSMO dataset version)
v2d_{dataset}_retarget_exp_200:<version>
    |
    v  Stage 6: osmo dataset download
Local assets/human_motion_data/{dataset}/
    |
    v  Stage 7: RL Training (Isaac Sim)
Trained policy checkpoint
```

Stages 1–5 are the OSMO `retarget.yaml` workflow; any subset can run via `--set stages=<stage>`. All artifacts from one run are published as a single versioned OSMO dataset (see [data_storage.md](data_storage.md)).

## Stage 1: Load

**Purpose:** Parse raw dataset files into a standardized Parquet schema.

**Script:** `scripts/retarget/{dataset}_loader.py` (dispatched via `scripts/retarget/run_loader.py`)

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

**Command:**
```bash
python scripts/retarget/run_loader.py \
  --dataset <name> \
  --output_dir <path> \
  --device cuda:0 --save
```

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

**Command:**
```bash
python scripts/retarget/run_retarget.py \
  --dataset <name> \
  --input_dir <loaded_dir> \
  --output_dir <processed_dir> \
  --device cuda:0 --save
```

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
skipped) in the OSMO dataset.

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

**Output:** `{dataset}_videos/{sequence_id}.mp4` in the OSMO dataset.

## Stage 6: Sync Results Locally

**Purpose:** Pull the published OSMO dataset to the local repo so training
and local visualization scripts can read it.

**Command:** `osmo dataset download` — full details, component-filter
examples, and the legacy CSS-swift fallback are in
[data_storage.md](data_storage.md#pulling-retarget-outputs-locally).

Quick version:

```bash
# Pick a published version
osmo dataset info v2d_<dataset>_retarget_exp_200 --order desc

# Pull just the pieces training needs
osmo dataset download v2d_<dataset>_retarget_exp_200:<version> \
  source/robotic_grounding/robotic_grounding/assets/human_motion_data/<dataset>/ \
  --regex '(<dataset>_processed|<dataset>_urdfs|reconstructed_stage)/.*'
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

Stages 1 through 5 are submitted as a single OSMO workflow (`workflow/retarget.yaml`) running on the GPU cluster. All artifacts from one run are snapshotted into a new version of the `v2d_{dataset}_retarget_exp_200` OSMO dataset. See [README.md](README.md) for submission options and [data_storage.md](data_storage.md) for the output layout; `/osmo-retarget` skill covers usage patterns.

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

Outputs live under `v2d_<name>_retarget_exp_200` OSMO datasets.

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
2. Write a loader script at `scripts/retarget/<name>_loader.py`.
3. Write a retarget script at `scripts/retarget/<name>_to_sharpa.py`.

Everything else (workflow dispatch, OSMO dataset publish, URDF generation,
training validation) is driven by the dataset registry automatically.

## Validating Before Training

Run the asset validator to catch missing URDFs or meshes before Isaac Sim
starts (saves the 45-second startup wait):

```bash
python scripts/validate_training_assets.py --dataset <name>
```

## Planned Refactor: Consolidate URDFs + Meshes

Object assets still live in inconsistent places between datasets:

| Dataset | Meshes location | URDFs location |
|---------|-----------------|----------------|
| hot3d, h2o | committed `assets/meshes/{name}/` | regenerated in image from committed meshes |
| arctic, taco, oakink2 | committed `assets/meshes/{name}/` | committed `assets/urdfs/{name}/` |
| grab, dexycb | CSS only (mounted at run time) | regenerated per workflow run from CSS meshes |

This bakes ~2.7 GB of meshes into every Docker image. Progress so far:

- [x] **URDFs published with retarget outputs** — the `process` stage
  generates URDFs into `assets/urdfs/{dataset}/` and the workflow copies
  them to `{dataset}_urdfs/` inside the OSMO dataset version, so every
  versioned run carries a regenerated URDF tree.
- [ ] **Local sync for URDFs** — `scripts/sync_css_data.py` still only
  knows about `loaded`/`processed`/`support_surfaces`; `osmo dataset
  download --regex '{dataset}_urdfs/.*'` is the current workaround. Adding
  `--component urdfs` to `sync_css_data.py` would close the gap for anyone
  still pulling from CSS.
- [ ] **Read-time fallback** — update `SceneConfig` path resolution to
  check `assets/urdfs/{name}/` first, then the per-dataset download dir.
- [ ] **Phase out committed meshes** — once the runtime URDF/mesh path is
  the source of truth, stop committing `assets/meshes/{name}/` for
  hot3d/h2o/arctic/taco/oakink2 and trim the image size.
