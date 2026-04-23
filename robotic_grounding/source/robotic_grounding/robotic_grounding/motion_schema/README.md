# `motion_schema` — unified motion parquet (version `motion_v1`)

Single schema shared by:

- **Producers**: `scripts/retarget/nvhuman_to_g1.py`, `scripts/retarget/nvhuman_to_dex3.py`,
  `source/robotic_grounding/robotic_grounding/planner/g1_planner.py::save_planner_parquet`.
- **Consumers**: training loader at
  `source/robotic_grounding/robotic_grounding/tasks/v2p_whole_body/mdp/commands/tracking_utils.py`,
  `SceneConfig.from_motion_file()`, `scripts/rsl_rl/dummy_agent.py::_autoframe_viewer`,
  `scripts/reconstruct_support_surfaces.py`, `tasks/scene_utils/replay_data.py`.

Design context lives alongside this doc in
[motion_schema.md](motion_schema.md).

## Public API

```python
from robotic_grounding.motion_schema import (
    MotionData,              # in-memory dataclass
    SCHEMA_VERSION,          # "motion_v1"
    SchemaVersionMismatch,   # raised by reader on mismatch
    MissingRequiredField,    # raised by reader on missing required field
    save_motion_parquet,     # writer
    load_motion_data_parquet,# reader
    build_schema,            # pyarrow schema factory
)
```

Reader: `md = load_motion_data_parquet(path, device="cuda")`. Accepts a file
path, Hive partition directory, or dataset root; resolves to the first parquet
file and backfills partition columns (`sequence_id`, `robot_name`) from the
directory names. Fills both the on-disk view (`robot_root_position`, `ee_pose_w`,
`hand_sides`-indexed lists, etc.) and the flattened view (`left_wrist_position`,
`right_hand_frames`, etc.) so `tracking_command.py` consumes `MotionData` without
changes.

Writer: `save_motion_parquet(md, root_path, partition_cols=["sequence_id", "robot_name"])`.
Runs required-field checks (fail-fast when any training-eligible column is
missing) and a lightweight `wxyz`-convention guard on `robot_root_wxyz`.

## Required fields (training contract)

The writer / reader will reject a file missing any of:

| Field | Shape |
|---|---|
| `schema_version` | string (`"motion_v1"`) |
| `sequence_id` | string |
| `robot_name` | string |
| `fps` | float32 (> 0) |
| `robot_joint_names` | list[str] (length `J`) |
| `robot_root_position` | (T, 3) |
| `robot_root_wxyz` | (T, 4), wxyz |
| `robot_joint_positions` | (T, J) aligned with `robot_joint_names` |
| `ee_link_names` | list[str] (length `E`) |
| `ee_pose_w` | (T, E, 7), each entry `[x, y, z, qw, qx, qy, qz]` |
| `object_body_names` | list[str] (length `B`) |
| `object_body_position` | (T, B, 3) |
| `object_body_wxyz` | (T, B, 4), wxyz |

## Optional groups

Left empty when the producer does not have the data; training guards each
field.

- **Hands** — `hand_sides`, `hand_frame_names`, `hand_frames_w`,
  `hand_finger_joint_names`, `hand_finger_joints`. Per-side lists aligned with
  the `hand_sides` index.
- **Contacts** — `hand_contact_link_names`, `hand_link_contact_positions`,
  `hand_link_contact_normals`, `hand_object_contact_positions`,
  `hand_object_contact_normals`, `hand_object_contact_part_ids`,
  `hand_contact_active`. Per-side, aligned by `hand_sides`.
- **Object metadata** — `object_name`, `safe_object_name`, `safe_object_body_names`,
  `object_mesh_paths`, `object_urdf_paths`, `object_mesh_radius`,
  `object_articulation`, `object_root_axis_angle`, `object_root_position`.
- **Source raw motion** — `source_kind`, `source_payload` (pickled bytes),
  `source_joint_names`.
- **Retarget diagnostics** — `ik_error_per_frame`, `ik_num_iterations`,
  `frame_task_errors`.

## Conventions (do not break)

- Quaternions are **wxyz** everywhere. No xyzw, no Euler angles in the on-disk
  schema.
- Pose entries combine position + orientation as `[x, y, z, qw, qx, qy, qz]`
  (length 7).
- Time series are `(T, ...)` with T as the leading axis. Per-timestep entries
  only — no transposed layouts.
- World frame for wrist and object body poses. If a producer uses a different
  convention it must record that in `coord_frame`.

## Versioning

- On-disk files carry `schema_version` as a string; the reader raises
  `SchemaVersionMismatch` if it does not match `SCHEMA_VERSION`.
- Breaking changes bump the version (e.g. `motion_v2`). Additive changes
  inside a version are backward-compatible.
- Run `scripts/motion_schema/migrate_to_v1.py <path>` to upgrade legacy
  `NvhumanG1Data` / `NvhumanDex3Data` / planner parquets in place.

The migrator is **idempotent** — rerunning on a `motion_v1` file prints
`SKIP` and exits 0 without rewriting.

## Tests

- `tests/test_motion_schema.py` — U1–U6 unit tests (round-trip, minimal-file,
  version enforcement, `hand_sides` alignment, quaternion guard, variable E)
  and M1–M4 migrator tests (idempotency, planner adapter, nvhuman_g1 adapter,
  source payload).
- `tests/test_motion_schema_parquet_integration.py` — round-trip via the
  reader + `SceneConfig.from_motion_file()`.
- `tests/test_replay_data.py` — `load_replay_trajectory` on `motion_v1`
  whole-body and `ManoSharpaData` dual-hand inputs.

Run:

```bash
python tests/test_motion_schema.py
python tests/test_motion_schema_parquet_integration.py
python tests/test_replay_data.py
```
