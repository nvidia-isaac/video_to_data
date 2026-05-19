# Retarget config bundle

Per-robot configuration for the SOMA → robot retargeter. One robot = one
sub-directory under `configs/`, containing exactly two files:

```
configs/
└── <robot_name>/
    ├── frame_alignment.json   # source-to-robot frame change (world + per-bone)
    └── retargeter.json        # URDF, IK map, foot framing, posture-task
```

`load_robot_config(<robot_name>)` in
`source/.../retarget/robot_config.py` reads both files together and
returns a fully-materialized `RobotRetargetConfig` whose matrices are
already composed for runtime consumption. Runtime code does not need to
know whether a value was supplied as a 3×3 matrix or an `xyzw`
quaternion; the loader normalizes both forms.

## Convention rules

These hold across both files:

- **Matrices** are row-major 3×3 lists.
- **Quaternions** are `xyzw`.
- **Distances** are meters.
- **Robot world frame**: X = forward, Y = left, Z = up.
- **Rotation composition** is right-multiply, applied as
  `target_rot = R_world @ source_rot @ correction`.

## `frame_alignment.json`

The frame change from the source skeleton (e.g. SOMA-X / NVHuman) into
the robot world. Two layers:

- **World layer**: `world_axis_swap_matrix` (or the equivalent
  `world_axis_swap_xyzw`) — a single rotation that maps source-world
  vectors into robot-world vectors. Applied left-multiplied to every
  source position and source-world rotation.
- **Per-bone layer**: `joint_offsets[<source_joint>]` — for each source
  joint that drives an IK target, the basis change between the
  source-joint local frame and the robot-link local frame. Applied
  right-multiplied onto the source bone's global rotation:
  `target_rot = R_world @ source_rot @ joint_offsets[joint].q_offset`.

### Schema

```jsonc
{
  "schema_version": 1,                   // bump on incompatible changes
  "robot_name": "g1",                    // must match retargeter.json
  "source_model": "soma",                // must match retargeter.json
  "_description": "...",                 // optional human-readable summary
  "_provenance": "...",                  // optional: where values came from
  "_upstream_reference": "...",          // optional: original table URL

  // World-frame axis swap (one of these is required):
  "world_axis_swap_matrix": [[...],[...],[...]],
  // or "world_axis_swap_xyzw": [x, y, z, w]

  // Per-bone corrections, keyed by source joint name:
  "joint_offsets": {
    "Hips": {
      "q_offset_matrix": [[...],[...],[...]],   // or q_offset_xyzw
      "t_offset": [0.0, 0.0, 0.01]              // optional, default zeros
    },
    // ...
  },

  // Optional: SOMA joint names whose q_offset was algorithmically
  // derived (rather than hand-tuned). Consumers may use this list to
  // assert tight q0 invariants; absent values default to []:
  "_auto_derived_joints": ["LeftLeg", "LeftShin", "LeftFoot", ...]
}
```

`t_offset` is applied in the corrected effector frame at runtime as
`target_pos += target_rot @ t_offset`. Only the `joint_offsets` whose
key appears in `retargeter.json:ik_map`'s `soma_joint` field are
projected onto a `t_per_link` entry; offsets for other joints are
parsed but unused.

## `retargeter.json`

What the IK actually does on the robot.

### Schema

```jsonc
{
  "schema_version": 1,
  "robot_name": "g1",
  "source_model": "soma",
  "_description": "...",
  "_provenance": "...",
  "_upstream_reference": "...",

  "urdf": "../../../assets/urdfs/g1/main_with_hand.urdf",
  "package_dirs": ["../../../assets/urdfs/g1"],
  "base_source_joint": "Hips",            // anchor used for source -> robot scaling

  "ik_map": {
    "<robot_frame_name>": {
      "soma_joint": "<source_joint_name>",
      "position_cost": 1.0,
      "orientation_cost": 0.2
    },
    // ...
  },

  "foot_frames": ["left_ankle_roll_link", "right_ankle_roll_link"],
  "ankle_roll_offset": 0.037,             // meters, ankle-roll origin -> sole

  "posture_task": {                       // optional, defaults to all-zero
    "q0_cost": 0.05,
    "q_prev_cost": 0.5,
    "lm_damping": 0.0,
    "gain": 1.0
  }
}
```

`urdf` and `package_dirs` are resolved relative to the directory the
JSON lives in (i.e. `configs/<robot>/`). `posture_task` is consumed by
`ConfigDrivenWholeBodyKinematics`; absent or all-zero costs disable the
posture term entirely (no-op for the legacy `G1WholeBodyKinematics`
path, which never reads it).

## Adding a new robot

1. **Copy an existing bundle** as a starting point:
   ```
   cp -r configs/g1 configs/<new_robot>
   ```
2. **Update `retargeter.json`**:
   - `robot_name`: set to `<new_robot>`.
   - `urdf`, `package_dirs`: point at the new robot's URDF.
   - `ik_map`: replace each `<robot_frame_name>` key with a frame name
     that exists in the new URDF (verify with
     `pinocchio.RobotWrapper.BuildFromURDF(...).model.frames`). Tune
     `position_cost` / `orientation_cost` per IK target.
   - `foot_frames`, `ankle_roll_offset`: set from the URDF's foot
     geometry.
3. **Update `frame_alignment.json`**:
   - `robot_name`: set to `<new_robot>`.
   - `world_axis_swap_matrix`: usually identical to a sibling robot
     because both share the source skeleton's world convention. Only
     change if the new URDF uses a different world frame.
   - `joint_offsets[*].q_offset_matrix`: see "Tuning q_offsets" below.
   - `joint_offsets[*].t_offset`: usually zero; set non-zero only for
     joints where the URDF link origin sits at a known offset from the
     source joint origin (e.g. an ankle link offset to the sole).
4. **Validate the bundle** via `load_robot_config("<new_robot>")` (see
   "Verifying a bundle" below).

## Tuning `q_offsets`

Three legitimate sources, in order of preference:

1. **Copy from an upstream reference table.** NVIDIA/soma-retargeter
   ships per-bone `q_offset` quaternions for several humanoid URDFs;
   the G1 bundle was seeded this way. The `_upstream_reference` field in
   each JSON points at the table.
2. **Derive from a known rest pose.** For each `(source_joint,
   robot_frame)` pair, set the source skeleton to a reference rest pose
   and compute the rotation that aligns the source joint's world
   rotation with the URDF link's world rotation:
   ```
   q_offset = (R_world @ R_source_world).T @ R_link_world
   ```
   This is what a "derive corrections" tool would automate. Constraint
   to satisfy: applying `q_offset` makes the IK target match the URDF's
   `q0` configuration at the chosen rest pose.
3. **Hand-tune in a visualizer.** Run the retargeter against a single
   reference pose with the new `q_offset` matrix and inspect the IK
   target frame in `viser_playback`. Iterate until the visual alignment
   is correct.

When values are derived (path 2) rather than hand-tuned (paths 1 or 3),
add the source joint's name to `_auto_derived_joints` so downstream
assertions can be tightened.

## Verifying a bundle

Smoke-test the loader:

```bash
python -c "from robotic_grounding.retarget.robot_config import load_robot_config; \
           c = load_robot_config('<robot_name>'); \
           print(c.robot_name, len(c.r_per_bone), c.posture_task)"
```

For a deeper check, run a short retarget on a fixture sequence and
inspect the result:

```bash
python scripts/retarget/soma_to_g1.py <data_folder> --robot-name <robot_name> --visualize
```

Open the printed viser URL and confirm:
- The robot lands near the source skeleton at frame 0 (no axis flip).
- Each IK end-effector tracks its source joint over time (no per-bone
  twist).
- Feet sit on the ground after the post-process (no float / sink).

If a per-bone twist appears, the corresponding `joint_offsets[*]` entry
is wrong — re-derive or hand-tune.
