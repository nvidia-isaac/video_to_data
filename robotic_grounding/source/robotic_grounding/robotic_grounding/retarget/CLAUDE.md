# `retarget/` — Human-motion-to-robot retargeting

Per-frame IK retargeting from human motion (MANO hands, NVHuman whole-body, SOMA-X) to robots (Sharpa Wave, Unitree Dex3, Unitree G1). Built on **Pink + Pinocchio** for IK; produces either the legacy `ManoSharpaData` parquet (dual-hand V2P) or the unified `motion_v1` parquet (whole-body), depending on the producer script.

Two parallel pipelines share this package:

```
dual-hand:      MANO source → HandKinematics → ManoSharpaData (this package)
                                              └─ ManoDex3Data
whole-body:     NVHuman/SOMA source → WholeBodyKinematics → motion_v1
                                              (motion_v1 lives in robotic_grounding.motion_schema)
```

## File map

### Path constants (`__init__.py`)
- `ASSETS_DIR`, `BODY_MODELS_DIR`, `HUMAN_MOTION_DATA_DIR` (overridable via `HUMAN_MOTION_DATA_DIR` env var), `SHARPA_WAVE_XMLS_DIR`, `G1_URDF_DIR`, `MESHES_DIR`. Anchor for every path-derived asset in this package.

### Dataset registry — single source of truth
- **`dataset_registry.py`** — `DatasetConfig` dataclass + `DATASET_CONFIGS` registry for `taco`, `arctic`, `oakink2`, `hot3d`, `h2o`, `grab`, `dexycb`. Per entry: `fps`, `mano_kwargs`, `mesh_vertex_scale`, `mesh_format`, `has_articulated_objects`, `has_contact_data`, `has_support_surfaces`, `link_to_site_quat_wxyz`, `loaded_suffix`, `processed_suffix`, `css_raw_prefix`, `loader_script`, `retarget_scripts` (per-robot dispatch). Public API: `get_dataset_config(name)`, `get_all_dataset_names()`, `get_css_stage_prefixes(name)`. **Loaders / CSS tools / URDF generators must import from here — never hard-code dataset constants.**

### Parquet schemas / loggers
- **`data_logger.py`** — Schema + serialization for the **legacy `ManoSharpaData` pipeline only**. `BASE_FIELDS`, `MANO_FIELDS`, `SHARPA_FIELDS` (`SHARPA_NUM_FRAMES=67`), `DEX3_MANO_FIELDS` (`DEX3_MANO_NUM_FRAMES=23`), `OBJECT_FIELDS` are combined by `create_data_logger_class(...)` into:
  - `ManoSharpaData` — BASE + MANO + SHARPA + OBJECT
  - `ManoDex3Data` — BASE + MANO + DEX3_MANO + OBJECT
  Each generated class exposes `log_timestep(...)`, `save_to_parquet(root, partition_cols)`, `load_from_parquet(root, filters, trajectory_id)`. Also exports sequence-filtering helpers (`list_sequence_ids`, `shard_matches`, `add_sequence_filter_args`, `filter_sequence_ids`). **The whole-body `motion_v1` schema is _not_ here** — it lives in `robotic_grounding.motion_schema`.

### Loader abstraction
- **`dataset_loader_base.py`** — `DatasetLoaderBase` ABC implemented by each `scripts/retarget/{dataset}_loader.py`. Required overrides: `list_sequences`, `load_mano_data`, `load_object_data`, `load_object_meshes`, `get_mano_kwargs`, `get_fps`. Default `get_frame_object_poses` composes per-body world poses from `(pose, root_position, root_axis_angle, articulation)` tuples. Helpers: `make_usd_safe`, `poses_to_root_position_and_axis_angle`, `load_meshes_to_device`, `build_combined_object_surface` (concatenates per-part surface samples with part IDs), `SequenceInfo`, `FrameObjectPoses`.

### Body-model readers (source side)
- **`read_mano.py`** — `MANO` wraps `manotorch.ManoLayer` + `AxisLayerFK`. Loads from `BODY_MODELS_DIR/mano/`. Used by every dual-hand dataset loader.
- **`read_nvhuman.py`** — `NVHuman` wraps `NVHumanLayer`. Loads from `BODY_MODELS_DIR/nvhuman/models/nvHuman_shape_TPose.npz`. Used by `nvhuman_to_{g1,dex3}.py`.
- **`read_soma.py`** — `SOMA` reads `soma_params.npz` from the SOMA exporter; `SOMAMotion` holds reconstructed `(joints, joints_wxyz, vertices)`. Mirrors `NVHuman.load_motion` so SOMA scripts can drop in. Used by `soma_to_g1.py`.

### IK kinematics (Pink + Pinocchio)
- **`hand_kinematics.py`**:
  - `HandKinematics` (ABC) — daqp solver, `frequency=200 Hz` default, `max_iter=200`, builds `ConfigurationLimit` + `VelocityLimit`, owns FrameTask / RelativeFrameTask, exposes `compute(joints, joints_wxyz, source_to_robot_scale, qpos) -> {q, ...}`.
  - `SharpaHandKinematics` — MJCF via `pin.RobotWrapper.BuildFromMJCF` (`{side}_sharpawave.xml`), free-flyer root, applies `SHARPA_TO_MANO_ROTATION_OFFSET` per wrist frame.
  - `Dex3HandKinematics` — URDF via `BuildFromURDF` (`G1_URDF_DIR/dex3_{side}.urdf`), uses `DEX3_TO_MANO_MAPPING`.
- **`whole_body_kinematics.py`**:
  - `WholeBodyKinematics` (ABC) — full-body counterpart with frame + posture tasks.
  - `G1WholeBodyKinematics` — **legacy NVHuman → G1 path**. Hard-codes `R_NVHUMAN_TO_ROBOT` and palm corrections. Refuses `source_model="soma"`.
  - `ConfigDrivenWholeBodyKinematics` — **the SOMA-to-robot core**. All robot-specific data (URDF, ik_map, world swap, per-bone/per-link rotations, posture costs) comes from a `RobotRetargetConfig`.

### Robot configs (JSON-driven)
- **`configs/<robot>/{frame_alignment,retargeter}.json`** — Per-robot configuration consumed by `ConfigDrivenWholeBodyKinematics`. `g1/` is the shipped bundle. **See `configs/README.md`** for the full schema, conventions (row-major 3×3, `xyzw` quats, meters, X-fwd/Y-left/Z-up), q_offset tuning, and the "adding a new robot" runbook.
- **`robot_config.py`** — `load_robot_config(name)` reads both JSONs and returns a `RobotRetargetConfig` with materialized matrices. Key dataclasses:
  - `IkMapEntry(soma_joint, position_cost, orientation_cost)`
  - `PostureTaskConfig(q0_cost=0.0, q_prev_cost=0.0, lm_damping=0.0, gain=1.0)` — both costs default to 0 (no posture term); `q0_cost` pulls toward `robot.q0`, `q_prev_cost` pulls toward previous frame's `q`.
  - `RobotRetargetConfig` carries `urdf_path`, `ik_map`, `foot_frames`, `ankle_roll_offset`, `r_world` / `r_per_bone` / `r_per_link` / `t_per_link`, `auto_derived_joints`, `posture_task`.

### Per-frame retargeting (dual-hand)
- **`retarget_utils.py`** — Setup factories + per-frame driver:
  - `setup_sharpa_kinematics(side, ...)`, `setup_dex3_kinematics(side, ...)` — wire URDF/MJCF paths from the package constants.
  - `run_frame_ik(right_kinematics, left_kinematics, mano_joints, ..., qpos_prev=None, wrist_position=None, wrist_quat_xyzw=None)` — runs both hands at one frame. On frame 0 init `qpos` from the wrist (`xyzw`); after that, pass the returned `q` back in as `qpos_prev`.
  - `compute_tip_to_object_surface_distance(mano_joints, surface_points)` — 5 fingertip-to-surface min-distances (used for the ManipTrans contact reward and tips_distance cache).
  - `wrist_pose_from_mano_joint0(joint0_pos, joint0_wxyz, link_to_site_quat_xyzw=None)` — robot wrist init from MANO joint 0; ARCTIC uses `link_to_site_quat_xyzw=(0.5,-0.5,0.5,0.5)` (in wxyz).

### Mapping / convention constants
- **`params.py`** — Frozen lookup tables. MANO: `MANO_JOINTS_ORDER` (21), `MANO_TRANSFORMS_ORDER` (16), `TRANSFORMS_TO_JOINTS`, `MANO_FINGERTIP_INDICES`, `MANO_HAND_LINKS`, `NUM_MANO_LINKS`. Source skeletons: `NVHUMAN_JOINTS_ORDER`, `SOMA_JOINTS_ORDER`. Frame mappings: `SHARPA_TO_MANO_MAPPING`, `SHARPA_RELATIVE_FRAMES`, `SHARPA_TO_MANO_ROTATION_OFFSET`, `DEX3_TO_MANO_MAPPING`, `DEX3_TO_NVHUMAN_MAPPING`, `G1_WHOLEBODY_TO_NVHUMAN_MAPPING`. Coordinate transforms: `R_NVHUMAN_TO_ROBOT`, `R_PALM_CORRECTION_LEFT`, `R_PALM_CORRECTION_RIGHT`.

### Contact / distance utilities
- **`contact_utils.py`** — Dexmachina-style hand–object contact. `compute_hand_link_contact_positions` maps MANO surface vertices to per-link contact positions/normals (16 links per hand); `find_object_contact_positions` maps each link to its nearest object surface point with part IDs; `approximate_contact_with_id` is the underlying NN search.
- **`distance_utils.py`** — `compute_tips_distance(mano_joints, mesh_verts, mesh_faces)` returns `(T, F)` fingertip-to-surface distances by sampling mesh surface points (PyTorch3D when available, trimesh fallback). `load_object_mesh(path)` returns `(verts, faces)` as torch tensors.

### Whole-body post-processing
- **`ground_alignment.py`** — Two-pass drift correction that runs **after** the IK loop in `nvhuman_to_g1.py`:
  1. `compute_plane_alignment_offsets` drags the lowest contact point (e.g. foot sole) onto a `ReferencePlane` every frame.
  2. `correct_object_trajectory` applies two rules: (a) **interaction** segments inherit the robot's Z-delta so the hand-object relative pose is preserved; (b) **no-interaction** segments are anchored to the release pose of the preceding contact (or first frame of the next contact). Configurable via `PlaneAlignmentConfig`, `InteractionMaskConfig`, `ObjectCorrectionConfig`. Pure NumPy — no Pinocchio / Isaac dependency, unit-testable.
- **`load_ground_plane_robot_frame`** — reads a precomputed ground plane and transforms it into the robot's frame.

### Visualization
- **`pinocchio_viser_visualizer.py`** — `ViserVisualizer` renders a Pinocchio model in viser; used inside the IK classes when `visualize=True`.
- **`viser_playback.py`** — `ViserPlayback` — two entry points sharing one scene graph:
  - `ViserPlayback(motion_file=...)` — replay a `motion_v1` parquet with Frame slider / Play / FPS / Loop. Used by `scripts/replay_viser.py`. **Deliberately does not import from `tasks.scene_utils.replay_data`** to avoid pulling in Isaac Lab (so it works in plain Python envs).
  - `ViserPlayback.for_live_retarget(server, pin_model, ...)` — attaches to an existing viser server to overlay per-frame retarget output. Used by `nvhuman_to_g1.py`.
  - `LiveFrameState` is the optional-fields bundle for live drawing.

### Quat / frame math
- **`utils.py`** — Torch-only quat utilities: `quat_mul`, `quat_apply`, `quat_from_matrix`, `quat_conjugate`, `quat_inv`, `subtract_frame_transforms`. Mirror `isaaclab.utils.math` semantics so this package can run without Isaac Lab.

## Public API at a glance

```python
# Dataset metadata
from robotic_grounding.retarget.dataset_registry import (
    get_dataset_config, get_all_dataset_names, get_css_stage_prefixes,
)

# Dual-hand parquet
from robotic_grounding.retarget.data_logger import (
    ManoSharpaData, ManoDex3Data,
    list_sequence_ids, add_sequence_filter_args, filter_sequence_ids,
)

# Per-frame dual-hand IK
from robotic_grounding.retarget.retarget_utils import (
    setup_sharpa_kinematics, setup_dex3_kinematics, run_frame_ik,
    compute_tip_to_object_surface_distance, wrist_pose_from_mano_joint0,
)

# Whole-body IK
from robotic_grounding.retarget.whole_body_kinematics import (
    G1WholeBodyKinematics,             # legacy NVHuman → G1
    ConfigDrivenWholeBodyKinematics,   # SOMA / configurable
)
from robotic_grounding.retarget.robot_config import load_robot_config

# Visualizers
from robotic_grounding.retarget.viser_playback import ViserPlayback, LiveFrameState
```

## Conventions (do not break)

- **Quaternions on disk** (`ManoSharpaData`, `motion_v1`, frame_alignment JSON `*_wxyz`) are **wxyz**.
- **Quaternions inside Pinocchio `qpos`** are **xyzw** (free-flyer slot `q[3:7]`). `retarget_utils.run_frame_ik` and `wrist_pose_from_mano_joint0` deal in xyzw; everywhere else is wxyz.
- **Distances are meters.** TACO meshes are centimeters → loader multiplies by `mesh_vertex_scale=0.01` (set in `dataset_registry`).
- **Whole-body world frame**: X forward, Y left, Z up (robot world).
- **Pose entries on disk** combine position + orientation as `[x, y, z, qw, qx, qy, qz]` (length 7).
- **MANO fingertip transforms** are 4 per finger including the tip (see `MANO_TRANSFORMS_ORDER`). The tip index is reused from the last joint; use `MANO_FINGERTIP_INDICES` to pull the 5 tips out of the (21, 3) joint tensor.
- **Rotation composition** for source→robot: `target_rot = R_world @ source_rot @ correction`. `correction` is left- or right-multiplied depending on whether it lives in `r_per_bone` (right-mul on source bone) or `r_per_link` (right-mul on URDF link).
- **`motion_v1` is the future** — the schema for whole-body / planner output lives in `robotic_grounding.motion_schema`, not here. New whole-body producers must emit `motion_v1` directly, not the retired `NvhumanG1*Data` loggers.
- **Pinocchio model surface**: free-flyer root contributes 7 entries to `q` (`[xyz, xyzw]`); finger DoFs start at index 7. `HandKinematics.__init__` enumerates finger joints from `model.names[2:nq-7+2]`.

## Where things actually run

| Producer script (in `scripts/retarget/`) | Uses | Writes |
|---|---|---|
| `{dataset}_loader.py` (taco, arctic, oakink2, hot3d, h2o, grab, dexycb) | `DatasetLoaderBase` + `MANO` + `dataset_registry` | `ManoSharpaData` parquet (MANO + object only) |
| `{dataset}_to_sharpa.py`, `{dataset}_to_dex3.py` | `setup_{sharpa,dex3}_kinematics` + `run_frame_ik` | `ManoSharpaData` / `ManoDex3Data` parquet (+ robot joint trajectories) |
| `nvhuman_to_g1.py`, `nvhuman_to_dex3.py` | `read_nvhuman.NVHuman` + `G1WholeBodyKinematics` / `Dex3HandKinematics` + `ground_alignment` | `motion_v1` parquet |
| `soma_to_g1.py` | `read_soma.SOMA` + `ConfigDrivenWholeBodyKinematics(load_robot_config("g1"))` | `motion_v1` parquet |
| `verify_tips_distance.py` | `distance_utils.compute_tips_distance` | (validation) |

Consumers of the resulting parquets live in `tasks/v2p` (dual-hand) and `tasks/v2p_whole_body` (motion_v1).
