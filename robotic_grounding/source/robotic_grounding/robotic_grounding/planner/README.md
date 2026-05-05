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
│     Hive-partitioned: planner_processed/sequence_id=.../robot_name=.../data.parquet
│
└── Step 9: Viewer
      MuJoCo playback with EE axes, object mesh, support surface
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--v2p_parquet` | required | Path to V2P retargeted parquet folder |
| `--v2p_sequence` | `box_grab` | Sequence ID substring filter |
| `--v2p_robot_name` | `sharpa_wave` | Robot name filter |
| `--v2p_trajectory_id` | `0` | Trajectory index within filtered results |
| `--robot` | `sharpa` | Robot type: `sharpa` or `dex3` |
| `--workspace_offset` | `-0.10 0.0 -0.15` | XYZ offset for EE targets |
| `--target_fps` | `100.0` | Resample V2P data to this FPS |
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
- `ee_link_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]` with
  `ee_pose_w (T, 2, 7)` built from the reference wrist trajectories.
- Object metadata + trajectory (`object_body_position`, `object_body_wxyz`,
  `object_body_names`, `object_articulation`, mesh/URDF paths copied from the
  upstream ManoSharpaData retarget file).
- Per-side hand frames + finger joints + contact groups carried over verbatim
  from V2P.

Support surfaces are discovered by `SceneConfig.from_motion_file()` from the
sibling `reconstructed_stage/` directory; they are not embedded in the parquet
(previously stored as `support_position` / `support_size`).

## Inference Module

The `MotionInferenceAgent` in `inference.py` loads model weights from a self-contained `torch.package` archive (`planner/assets/models/planner_agent.pkg`). No external dependencies on the training codebase.

```
inference.py           loads planner_agent.pkg, calls predict(), local pre/post processing
data_adapters.py       MuJoCo qpos ↔ model features, T-pose correction, interpolation
chunk_runner.py        chunked autoregressive inference with half-stride blending
smoothing.py           Hamming-window temporal smoothing
motion_reps.py         quaternion / rotation utilities
transforms.py          Sharpa/Dex3 → G1 frame alignment
trajectory.py          warmup trajectory builder (hold → interp → hold → reference)
visualization.py       MuJoCo viewer with EE axes, object mesh, support surface
```
