# `motion_schema` — unified motion parquet (version `motion_v1`)

Single schema shared by:

- **Producers**: `scripts/retarget/soma_to_g1.py`,
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
    SINGLE_ROBOT,            # motion_kind value: whole-body
    DUAL_HAND,               # motion_kind value: floating dual hands
    KNOWN_MOTION_KINDS,      # frozenset of accepted motion_kind values
    save_motion_parquet,     # writer
    load_motion_data_parquet,# reader
    build_schema,            # pyarrow schema factory
    resolve_motion_kind,     # resolves and validates `motion_kind`
    required_fields_for,     # required fields for a given motion_kind
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
Runs required-field checks (fail-fast when any training-eligible column for the
file's `motion_kind` is missing) and a lightweight `wxyz`-convention guard on
`robot_root_wxyz`.

## Motion kinds

Every file carries an explicit `motion_kind` discriminator. Required fields
branch on this value, so producers must tag the file at write time.

- **`single_robot`** — whole-body humanoid trajectories (e.g. G1, G1+Dex3
  planner output). Requires `robot_joint_names`, `robot_root_position`,
  `robot_root_wxyz`, `robot_joint_positions` in addition to the common fields.
- **`dual_hand`** — floating dual-wrist trajectories (e.g. Dex3 from
  MANO source). Requires `hand_sides`, `hand_frame_names`, `hand_frames_w`,
  `hand_finger_joint_names`, `hand_finger_joints` in addition to the common
  fields. Whole-body joint state may be left empty.

Files predating `motion_kind` will fail to load with `MissingRequiredField`;
no migrator is shipped, so producers must be re-run.

## Common required fields (every motion_kind)

The writer / reader will reject a file missing any of:

| Field | Shape |
|---|---|
| `schema_version` | string (`"motion_v1"`) |
| `sequence_id` | string |
| `robot_name` | string |
| `motion_kind` | string (`"single_robot"` or `"dual_hand"`) |
| `fps` | float32 (> 0) |
| `ee_link_names` | list[str] (length `E`) |
| `ee_pose_w` | (T, E, 7), each entry `[x, y, z, qw, qx, qy, qz]` |
| `object_body_names` | list[str] (length `B`) |
| `object_body_position` | (T, B, 3) |
| `object_body_wxyz` | (T, B, 4), wxyz |

### Additional required fields for `motion_kind="single_robot"`

| Field | Shape |
|---|---|
| `robot_joint_names` | list[str] (length `J`) |
| `robot_root_position` | (T, 3) |
| `robot_root_wxyz` | (T, 4), wxyz |
| `robot_joint_positions` | (T, J) aligned with `robot_joint_names` |

### Additional required fields for `motion_kind="dual_hand"`

| Field | Shape |
|---|---|
| `hand_sides` | list[str] (length `S`) |
| `hand_frame_names` | list[list[str]] aligned with `hand_sides` |
| `hand_frames_w` | per-side `(T, K, 7)` aligned with `hand_sides` |
| `hand_finger_joint_names` | list[list[str]] aligned with `hand_sides` |
| `hand_finger_joints` | per-side `(T, J_f)` aligned with `hand_sides` |

The writer additionally enforces that the outer length of every per-side
field equals `len(hand_sides)`, catching producers that set `hand_sides` but
forget to populate one side's payload.

## Optional groups

Left empty when the producer does not have the data; training guards each
field.

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
- No migrator is shipped. Files written before the `motion_kind` discriminator
  was added must be regenerated by re-running the producing retarget or
  planner script.

## Tests

- `tests/test_motion_schema.py` — U1–U6 unit tests (round-trip, minimal-file,
  version enforcement, `hand_sides` alignment, quaternion guard, variable E)
  and K1–K5 `motion_kind` tests (dual-hand round-trip, single/dual required-
  field enforcement, per-side alignment, missing/unknown kind rejection).
- `tests/test_motion_schema_parquet_integration.py` — round-trip via the
  reader + `SceneConfig.from_motion_file()`.
- `tests/test_replay_data.py` — `load_replay_trajectory` on `motion_v1`
  single-robot and dual-hand inputs plus `ManoSharpaData` legacy dual-hand.

Run:

```bash
python tests/test_motion_schema.py
python tests/test_motion_schema_parquet_integration.py
python tests/test_replay_data.py
```
