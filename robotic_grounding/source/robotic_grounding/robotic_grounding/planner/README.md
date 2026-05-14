# Whole-Body Planner

Generates planned whole-body motion from V2P retargeted hand/object trajectories. Takes EE targets as input, runs a learned motion model to produce full-body joint trajectories, and outputs a single Hive-partitioned parquet containing everything needed for RL training.

## Quick Start

```bash
cd /path/to/video_to_data
PYTHONPATH=robotic_grounding/source/robotic_grounding:$PYTHONPATH \
python -m robotic_grounding.planner.g1_planner \
    --v2p_parquet robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/arctic/arctic_processed \
    --v2p_sequence box_grab \
    --robot sharpa \
    --workspace_offset -0.30 0.0 0.07 \
    --output planner_processed
```

This opens a MuJoCo viewer showing the planned motion with EE axes, object mesh, and support surface.

## Pipeline

```
V2P Retargeted Parquet (arctic_processed/)
│
├── Step 1: Nominal FK
│     Compute G1 nominal wrist positions from standing pose
│
├── Step 2: Load V2P Reference
│     Load hand/object trajectories, interpolate to target FPS
│
├── Step 3-4: Transform to G1 Frame
│     Local frame fix (Sharpa → G1) → yaw correction → position offset
│
├── Step 5: Build Trajectory
│     Hold nominal (5s) → interpolate (5s) → hold start (5s) → reference
│
├── Step 6: Inference
│     MotionInferenceAgent: EE targets → chunked autoregressive → full-body qpos
│
├── Step 7: Build Full Qpos
│     Combine: planner body (29 DOF) + reference fingers + static legs
│
├── Step 8: Save Parquet
│     Hive-partitioned: planner_processed/sequence_id=.../robot_name=.../*.parquet
│     Post-write invariants (utils/validation.py) hard-fail on any contract
│     break before the parquet leaves planning.
│
├── Step 8b: Reconstruct Support Surface
│     support_recon writes <output>/reconstructed_stage/<seq>_support.usda.
│     Disks that sit on another body's trajectory (a tool resting on the
│     target) are filtered so they don't spawn intersecting the target body.
│
└── Step 9: Viewer
      MuJoCo playback with EE axes, object mesh, support surface
```

Pre-plan, `utils/validation.warn_reference_issues` runs over the loaded V2P
motion and prints warnings for reference-owned gaps (missing required fields,
unresolvable asset paths, missing URDF mesh dependencies, off-FPS source).
These are informational — the upstream retargeter / asset pipeline owns them.

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--v2p_parquet` | required | Path to V2P retargeted parquet folder |
| `--v2p_sequence` | `box_grab` | Sequence ID substring filter |
| `--v2p_robot_name` | `sharpa_wave` | Robot name filter |
| `--v2p_trajectory_id` | `0` | Trajectory index within filtered results |
| `--robot` | `sharpa` | Robot type: `sharpa` or `dex3` |
| `--workspace_offset` | `-0.10 0.0 -0.15` | XYZ offset for EE targets |
| `--target_fps` | `150.0` | Resample V2P data to this FPS |
| `--ref_seconds` | `-1` | Seconds of reference to include (-1 = all) |
| `--output` | `planner_processed/` | Output parquet directory |
| `--no_viewer` | false | Skip MuJoCo visualization |
| `--ik_verify` | false | Run IK reachability check |
| `--ik_plan` | false | Use IK solution instead of learned model |

## Output Parquet Schema

The planner writes the unified `motion_v1` schema (see
[../motion_schema/README.md](../motion_schema/README.md) and
[../motion_schema/motion_schema.md](../motion_schema/motion_schema.md)). Hive
layout: `<output>/sequence_id=<seq>/robot_name=<robot>/data.parquet`.

The planner populates:

- Robot state (`robot_root_position`, `robot_root_wxyz`, `robot_joint_positions`,
  `robot_joint_names`) decomposed from the planner's mujoco qpos.
- `ee_link_names` set per robot: `["left_hand_palm_link", "right_hand_palm_link"]`
  for dex3 (where the palm IS the free-flyer URDF root), `["left_wrist_yaw_link",
  "right_wrist_yaw_link"]` for sharpa. `ee_pose_w (T, 2, 7)` is built from the
  reference wrist trajectories.
- Object metadata + trajectory (`object_body_position`, `object_body_wxyz`,
  `object_body_names`, `object_articulation`, mesh/URDF paths copied from the
  upstream ManoSharpaData retarget file).
- `object_root_position` / `object_root_axis_angle` derived from body 0 of the
  planner-frame object pose so the env's articulated scene init lands where the
  trajectory starts.
- `robot_joint_names` / `robot_joint_positions` cover every actuated joint
  (body + fingers) in MuJoCo joint order; the per-side `hand_finger_joints` /
  `hand_finger_joint_names` lists stay populated for callers that want the
  side-segregated view.
- Per-side hand frames + contact groups are transformed by the same per-frame
  rigid transform applied to `ee_pose_w` / `object_body_position`, so every
  field of the output parquet lives in a single coherent planner frame.

Support surfaces are discovered by `SceneConfig.from_motion_file()` from the
sibling `reconstructed_stage/` directory; they are not embedded in the parquet
(previously stored as `support_position` / `support_size`).

## Module Layout

```
planner/
├── g1_planner.py             CLI orchestration; sole consumer of utils/
├── support_recon.py          support-surface reconstruction with phantom-tool
│                             trajectory-overlap filter (drops disks whose
│                             stillness comes from resting on another body)
├── trajectory.py             warmup builder: hold nominal → interp → hold start
│                             → reference
├── visualization.py          MuJoCo viewer with EE axes, object mesh, support
├── motionbricks/             MotionBricks model backend (current planner)
│   ├── inference.py          loads planner_agent.pkg, runs chunked AR inference
│   └── qpos.py               qpos assembly helpers
├── mfm/                      Legacy MFM model backend (kept for reference)
│   ├── inference.py / chunk_runner.py / data_adapters.py / smoothing.py /
│   │ motion_reps.py / mujoco_helper.py / ik_verify.py
└── utils/                    Pure helpers, no planner state
    ├── transforms.py         Quaternion conversions, low-level rigid
    │                         primitives (quat_*, transform_primary_*,
    │                         transform_contact_*_by_part), and the high-level
    │                         transform_reference pipeline (local frame fix →
    │                         heading → position offset → workspace shift)
    ├── loader.py             Resample V2P motion fields to target FPS
    │                         (linear for positions, SLERP for quats, masked
    │                         interp for contacts)
    └── validation.py         Pre-plan warn_reference_issues + post-plan
                              assert_motion_parquet_invariants. The asserts
                              catch every contract task 48 had to patch by
                              hand; running them at planning time means
                              regressions surface before training sees them.
```

The active model backend is `motionbricks/`. `MotionInferenceAgent` in
`motionbricks/inference.py` loads weights from a self-contained `torch.package`
archive bundled under `assets/models/`, so the planner runs without any
training-codebase dependency.
