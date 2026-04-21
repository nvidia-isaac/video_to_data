# Data Pipeline: Raw Motion Data to RL Training

This document describes how hand-object motion capture data flows through the
robotic_grounding pipeline, from raw dataset files to a trained RL policy.

## Pipeline Overview

```
Raw Data (pkl/jsonl/csv/npy/...)
    |
    v  Stage 1: Load
ManoSharpaData Parquet (MANO hands + object poses)
    |
    |-- Stage 1.5: Generate URDFs         --> urdfs/{dataset}/*.urdf
    |
    v  Stage 2: Retarget (IK)
ManoSharpaData Parquet (+ robot joint trajectories)
    |
    |-- Stage 3:   Support surfaces        --> reconstructed_stage/*.usda
    |-- Stage 3.5: Visualization (opt.)    --> {dataset}_html/recordings/{seq}.viser + .mp4
    |
    v  Stage 4: Sync from CSS (after OSMO)
Local assets/human_motion_data/
    |
    v  Stage 5: RL Training (Isaac Sim)
Trained policy checkpoint
```

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
stage is idempotent -- it skips objects that already have URDFs.

**Output:** `assets/urdfs/{dataset}/{object_id}_rigid.urdf`

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

## Stage 3.5: Visualization (optional, OSMO-side)

**Purpose:** Produce inspection artifacts alongside the retargeted data —
both interactive (`.viser`) and offline video (`.mp4`). Drives quality
reviews without downloading the full parquets locally.

**Scripts:** `scripts/retarget/vis_retargeted.py`

Running with `--save_html --save_mp4` writes, for every sequence:

- `<seq>.viser` — web-based playback, shows MANO hands + Sharpa robot +
  objects + support surfaces; open in `viser-client/?playbackPath=...`.
- `<seq>.mp4` — headless pyrender render of the same scene, auto-framed on
  the object trajectory. No browser or Isaac Sim required. Useful for
  quick QA at scale, embedding in reviews, or sanity-checking new datasets.

The OSMO workflow produces both under `{dataset}_html/recordings/` (see
`workflow/retarget.yaml` Stage 4). For local iteration see the
`/add-dataset` skill.

## Stage 4: Sync Results

**Purpose:** Pull processed data from cloud storage to the local machine.

**Scripts:** `scripts/sync_css_data.py`, `scripts/list_css_sequences.py`

When stages 1-3 run on OSMO (GPU cluster), the outputs are stored on CSS
(S3-compatible cloud storage). This stage downloads them to the local
`assets/human_motion_data/` directory for training.

**Output:** Local copy of processed Parquets, URDFs, and support surfaces.

**Commands:**
```bash
# Browse available data
python scripts/list_css_sequences.py --dataset <name>

# Download processed data
python scripts/sync_css_data.py --dataset <name> --component processed
```

## Stage 5: RL Training

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

Stages 1 through 3 can be submitted as a single OSMO workflow that runs on
the GPU cluster. See `workflow/retarget.yaml` for the workflow definition and
the `/osmo-retarget` skill for usage instructions.

```bash
python scripts/run_osmo.py \
  --experiment-name retarget-<dataset> \
  --workflow-yaml workflow/retarget.yaml \
  --set dataset=<name>
```

## Dataset Inventory

Candidates considered for Robot Grounding Task Library. Source:
[dataset-candidates spreadsheet](https://docs.google.com/spreadsheets/d/1aG_m4I1vuFdM5LkeHmSbhlF5zBxTRo-3Qu7TGThVt1M/edit).

### In the pipeline

| Dataset | Sequences | GT Fidelity | Hands | Status |
|---------|-----------|-------------|-------|--------|
| **TACO** | 2,317 | High (NOKOV MoCap) | Bimanual | Retargeted on CSS |
| **Arctic** | 554 | Very High (Vicon MoCap) | Bimanual, articulated objects | Retargeted on CSS |
| **OakInk2** | 627 | High (OptiTrack MoCap) | Bimanual | Retargeted on CSS |
| **HOT3D** | 294 | Very High (OptiTrack MoCap) | Bimanual | Retargeted on CSS |
| **H2O** | 137 | Medium-High (RGB-D opt.) | Bimanual | Retargeted on CSS (cam4 egocentric) |
| **GRAB** | 1,335 | Very High (Vicon MoCap) | Bimanual + SMPL-X full body | Retargeted on CSS |
| **DexYCB** | 1,000 | Medium-High (multi-view RGB-D) | Single (per-session) | Retargeted on CSS (cam `932122062010`) |

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

Everything else (workflow dispatch, CSS sync, URDF generation, training
validation) is driven by the dataset registry automatically.

## Validating Before Training

Run the asset validator to catch missing URDFs or meshes before Isaac Sim
starts (saves the 45-second startup wait):

```bash
python scripts/validate_training_assets.py --dataset <name>
```

## Planned Refactor: Consolidate URDFs + Meshes on CSS

Today the pipeline is inconsistent about where per-dataset object assets live:

| Dataset | Meshes location | URDFs location |
|---------|-----------------|----------------|
| hot3d, h2o | committed `assets/meshes/{name}/` | regenerated in image from committed meshes |
| arctic, taco, oakink2 | committed `assets/meshes/{name}/` | committed `assets/urdfs/{name}/` |
| grab, dexycb | CSS only (mounted at run time) | regenerated per workflow run from CSS meshes |

This bakes ~2.7 GB of meshes into every Docker image and drifts between
committed and on-CSS artifacts.  Unify on a single pattern:

1. **Write URDFs to CSS in the workflow** — add a `{dataset}_urdfs` output
   in `workflow/retarget.yaml` (or fold it into `{dataset}_processed`).
   Update `scripts/generate_rigid_urdfs.py` to write under
   `${OUTPUT}/{dataset}_urdfs/` instead of `assets/urdfs/`.
2. **Sync on demand** — add `--component urdfs` to `scripts/sync_css_data.py`.
   Training flow becomes `sync_css_data.py --dataset X --component urdfs`.
3. **Prefer local → CSS fallback** at read time — update
   `SceneConfig` path resolution to check `assets/urdfs/{name}/` first and
   hit CSS if missing.
4. **Phase out committed meshes** — once the CSS path is the source of
   truth, stop committing `assets/meshes/{name}/` for hot3d/h2o/arctic/
   taco/oakink2 and trim ~2.7 GB from the image.

OSMO dataset versioning (already enabled via the `dataset:` output in
`retarget.yaml`) will then cover URDF regeneration out of the box.
