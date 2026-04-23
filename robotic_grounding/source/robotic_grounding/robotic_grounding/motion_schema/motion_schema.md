# MotionData: Unified Motion Parquet Schema

*Design doc for team review. Implementation to follow.*

## TL;DR

Every retargeting and planning script will write the same parquet layout, `MotionData` (`schema_version = "motion_v1"`), and training will load that layout directly. One interface covers single-arm, dual-hand, bimanual, whole-body, body-only, and contact-annotated sequences. Optional field groups keep the files lean; an explicit `schema_version` lets us evolve it safely.

## Why we need this

The whole-body pipeline currently has two incompatible parquet layouts and a growing divergence between them:

- **Planner output** (`g1_planner.py`) writes `qpos` + `qpos_layout` + `joint_names` + `ee_pos_w/ee_quat_w` + `robot_*_wrist_*`. Used by the apple-pick experiment.
- **Nvhuman retarget output** (`nvhuman_to_g1.py`, backed by `NvhumanG1Data`) writes `robot_root_position` + `robot_root_wxyz` + `robot_joint_positions` + `robot_frames`. No `qpos`, no `ee_pos_w`. Used by the bottle experiment.

The symptoms we hit last week are not one-offs; they are the predictable result of having two schemas:

1. Training's `load_motion_data` was written for planner output, so the bottle parquet crashes with `KeyError: 'qpos'`.
2. We've been adding adapter branches inside `load_motion_data` to paper over the differences (`if "ee_pos_w" in data: … elif "robot_left_wrist_position" in data: …`). Every new retarget script adds another branch.
3. Every new robot (Dex3 today, others later) spawns another per-robot dataclass (`NvhumanG1Data`, `NvhumanDex3Data`) that effectively duplicates 80% of the same fields with slightly different shapes and quaternion conventions.
4. Downstream tools (`SceneConfig`, `dummy_agent._autoframe_viewer`, `reconstruct_support_surfaces.py`, replay) each implement their own parquet reads with their own assumptions about which columns exist.

This is the kind of divergence that only gets worse. Since this parquet is the contract between retargeting and training — the entire data pipeline hinges on it — the cost of not fixing it scales with the team, the number of robots, and the number of datasets.

## Goals

- **One schema, one reader.** Every retarget script and the planner produce the same file; training reads it through one code path.
- **Handle any robot configuration.** Single-arm, dual-hand (floating), bimanual whole-body, future full-humanoid — all represented without per-robot subclasses.
- **Long-lived and evolvable.** We can add fields over time without breaking old files, and detect at load time when a file is too old to use.
- **Fail fast and loud.** Unreadable files error out at load time with an actionable message, never silently default to zeros or skip rewards.
- **Minimal blast radius.** Columns that downstream tools already read (object poses, support surface, `fps`) keep the same names so `SceneConfig`, `dummy_agent`, and `reconstruct_support_surfaces.py` keep working unchanged.

## Non-goals

- Replacing `ManoSharpaData` or the dual-hand V2P pipeline (separate, working, out of scope).
- Changing asset storage conventions (URDFs on CSS vs. local, etc.).
- Rewriting retargeting math or the planner model. This is a pure data-interface change.

## Design principles

- **Optional groups, not schema variants.** There's exactly one `MotionData` schema. Retargeters without contacts simply leave the contact group empty; single-arm robots leave `hand_sides = []`. No per-robot dataclasses.
- **Canonical conventions, chosen once.**
  - Quaternions are **wxyz** everywhere. No xyzw, no Euler angles in the on-disk schema. (Producers that internally use Euler or xyzw convert on write.)
  - Pose entries are `[x, y, z, qw, qx, qy, qz]` for each frame.
  - Time series are `(T, ...)`. Per-timestep lists only.
  - Joint ordering is always carried alongside values via a companion name list.
- **Keep raw source co-located.** MANO params, NVHuman joints, and any other upstream motion data travel with the file as an optional opaque `source_payload` blob, so one parquet carries provenance end-to-end without bloating the strongly-typed columns that training actually reads.
- **Explicit versioning.** Every file carries `schema_version` (e.g. `"motion_v1"`). The reader checks it and raises on mismatch with a clear migration hint. Breaking changes bump the version; additive changes are backward-compatible.
- **Additive evolution only within a version.** New optional columns are fine; renaming or repurposing existing columns requires a version bump.

## Proposed schema

The schema is grouped here for readability; in code it's one flat pyarrow schema.

### Metadata (required)

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | `"motion_v1"`; checked by the reader |
| `sequence_id` | string | partition key |
| `robot_name` | string | partition key (`g1`, `dex3`, …) |
| `source_dataset` | string | `nvhuman`, `arctic`, `hot3d`, … |
| `raw_motion_file` | string | provenance |
| `fps` | float32 | trajectory frame rate |
| `coord_frame` | string | producer-declared convention tag, e.g. `"robot_base_z_up"` |

### Robot state (required for training-eligible files)

| Field | Shape | Notes |
|---|---|---|
| `robot_joint_names` | list[str] | static; J entries |
| `robot_root_position` | (T, 3) | world frame |
| `robot_root_wxyz` | (T, 4) | wxyz |
| `robot_joint_positions` | (T, J) | aligned with `robot_joint_names` |

These four fields are the minimum training needs. They replace the planner's `qpos` + `qpos_layout` slice machinery. The training loader internally concatenates them to `[root_pos(3), root_quat_wxyz(4), joints(J)]` — the shape `TrackingCommand` already consumes.

### End-effector frames (required for tracking tasks)

| Field | Shape | Notes |
|---|---|---|
| `ee_link_names` | list[str] | static; E entries |
| `ee_pose_w` | (T, E, 7) | `[x, y, z, qw, qx, qy, qz]` in world frame |

One field replaces the current tangle of `ee_pos_w` / `ee_quat_w` + `robot_left_wrist_*` + `robot_right_wrist_*`. `E` is arbitrary, so the same schema covers:

- single-arm: `ee_link_names = ["right_wrist_yaw_link"]`, `E = 1`
- bimanual G1: `ee_link_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]`, `E = 2`
- Dex3 per side, future multi-end-effector robots: `E = N`

### Hands (optional)

Aligned by `hand_sides` index so single-hand, left-only, right-only, and bimanual are all first-class without left-empty shadow columns.

| Field | Shape | Notes |
|---|---|---|
| `hand_sides` | list[str] | subset of `["left", "right"]` |
| `hand_frame_names` | list[list[str]] | per side (K per side, varies) |
| `hand_frames_w` | list[(T, K, 7)] | per side, world frame |
| `hand_finger_joint_names` | list[list[str]] | per side |
| `hand_finger_joints` | list[(T, J_f)] | per side |

### Object (required when the scene has an object; unchanged from today)

Kept verbatim (`object_name`, `safe_object_name`, `object_body_names`, `safe_object_body_names`, `object_mesh_paths`, `object_urdf_paths`, `object_mesh_radius`, `object_articulation`, `object_root_axis_angle`, `object_root_position`, `object_body_position`, `object_body_wxyz`). `SceneConfig.from_motion_file()`, `reconstruct_support_surfaces.py`, and `dummy_agent._autoframe_viewer` read these names today; leaving them alone keeps the blast radius of this change small.

### Contacts (optional, `hand_sides`-indexed)

Generalized from today's `mano_{side}_*` column family.

| Field | Shape | Notes |
|---|---|---|
| `hand_contact_link_names` | list[list[str]] | per side |
| `hand_link_contact_positions` | list[(T, N, 3)] | per side |
| `hand_link_contact_normals` | list[(T, N, 3)] | per side |
| `hand_object_contact_positions` | list[(T, N, 3)] | per side |
| `hand_object_contact_normals` | list[(T, N, 3)] | per side |
| `hand_object_contact_part_ids` | list[(T, N)] | per side; int32 |
| `hand_contact_active` | list[(T,)] | per side; float32 binary |

### Source raw motion (optional)

| Field | Shape | Notes |
|---|---|---|
| `source_kind` | string | `"nvhuman"`, `"mano"`, or `""` |
| `source_payload` | bytes | pickled dict (betas, finger pose, joints, head/root trajectories, etc.) |
| `source_joint_names` | list[str] | optional, if `source_payload` carries joint data |

A single opaque blob keeps the strongly-typed part of the schema small and avoids exploding it with e.g. `nvhuman_joints (T, 93, 3)` shapes that only retarget validation ever needs. **Training never reads `source_payload`**; it exists for reproducibility, debugging, and re-retargeting.

### Retarget diagnostics (optional)

| Field | Shape | Notes |
|---|---|---|
| `ik_error_per_frame` | (T,) | |
| `ik_num_iterations` | (T,) | |
| `frame_task_errors` | (T, K_tasks) | per-task residuals |

## Pipeline

```
Raw motion (NVHuman / MANO / …)
            │
            ▼
Dataset loader          scripts/retarget/*_loader.py
            │
            ▼
Retarget IK             nvhuman_to_g1 / nvhuman_to_dex3 / …
            │
            ▼
(optional) Planner      g1_planner
            │
            ▼
┌─────────────────────────────────────────────┐
│ MotionData parquet (schema_version=motion_v1)│
└─────────────────────────────────────────────┘
            │
            ▼
MotionData.from_parquet (schema_version check)
            │
   ┌────────┴────────────────┐
   ▼                         ▼
SceneConfig        TrackingCommand.load_motion_data
   │                         │
   └────────────┬────────────┘
                ▼
          Isaac Lab env
```

Any step can populate only the groups it knows about. The planner and the retarget scripts fill the robot + ee + object groups; retarget scripts also fill hands/contacts when they have them; raw source stays attached as `source_payload` for whatever produced the first parquet.

## Benefits

**For the team:**

- New robot = new entry in the robot registry + a retarget script that writes `MotionData`. No new dataclass, no new training-loader branch.
- New dataset = new loader that writes `MotionData`. Training sees no difference between datasets.
- Reading a parquet file only requires knowing one schema.

**For the training side:**

- `load_motion_data` has one code path. Consumers (`tracking_command.py`, rewards, observations) keep their existing attribute names, so this refactor is invisible to them.
- Missing optional data surfaces as `None` attributes on the in-memory `MotionData` object, which is already how `tracking_command.py` checks for hands and contacts today — nothing changes there.
- `SchemaVersionMismatch` at load time instead of `KeyError: 'qpos'` 30 seconds into env startup.

**For the data pipeline:**

- Retarget → planner → training is one format end-to-end. No schema adapters in the middle.
- Provenance travels with the file via `source_payload`, so we can trace any training run back to the raw upstream motion without maintaining a separate sidecar.
- Hive-partitioned path layout is unchanged (`<dataset>/sequence_id=<seq>/robot_name=<robot>/data.parquet`), so existing experiment configs keep their `motion_file` strings.

**For future evolution:**

- Additive fields within `motion_v1` are backward-compatible; old readers tolerate new columns.
- Any future breaking change bumps to `motion_v2` with a migration script and an immediate, obvious error on old readers.
- The migration script we write for `motion_v1` also serves as the template for `v2`, `v3`, …

## Costs / trade-offs

- One-time migration pass over existing on-disk `NvhumanG1Data`, `NvhumanDex3Data`, and planner parquets. The migrator handles this idempotently; it's a single offline run per dataset.
- Retarget scripts get slightly longer (they now populate more structured groups). This is a net-positive: the structure is explicit instead of implicit in column naming conventions.
- Breaking change vs. the current bottle parquet on disk. The migrator covers this; the disruption window is "one afternoon" not "gradual debt across the codebase."

## Relationship to existing code

- `ManoSharpaData` and the dual-hand V2P pipeline are **untouched**. Their schema stays as-is.
- `NvhumanG1Data` and `NvhumanDex3Data` are **retired**. Their producers switch to `MotionData`; their consumers read via the unified reader.
- The `create_data_logger_class` factory stays, used by `ManoSharpaData` only.

---

Feedback welcome, especially on:

- Any field we're missing for robots or experiments not represented today.
- Naming: `MotionData` vs. something more specific (e.g. `WholeBodyMotionData`) — leaning toward the generic name since it covers both whole-body and dual-hand trajectory data.
- Whether `source_payload` should be pickled bytes or a list of typed sub-fields. Bytes keeps the schema small but is opaque; typed sub-fields would grow the schema.
