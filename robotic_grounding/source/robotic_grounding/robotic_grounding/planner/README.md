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

The output parquet is Hive-partitioned by `sequence_id` and `robot_name`. It contains everything needed by `TrackingCommand` and `SceneConfig.from_motion_file()`.

### Planner Body Motion

| Column | Shape | Description |
|--------|-------|-------------|
| `qpos` | (T, 80) | Full qpos: root(7) + body(29) + left_fingers(22) + right_fingers(22) |
| `joint_names` | list | Joint name ordering in qpos |
| `qpos_layout` | JSON | Index ranges: root_pos, root_quat, body_joints, finger_joints |
| `fps` | float | Output frame rate |
| `ee_pos_w` | (T, 2, 3) | Planner EE target positions [left, right] |
| `ee_quat_w` | (T, 2, 4) | Planner EE target orientations [left, right] |
| `ee_link_names` | list | `["left_wrist_yaw_link", "right_wrist_yaw_link"]` |

### Hand Keypoints (from V2P retargeting)

| Column | Shape | Description |
|--------|-------|-------------|
| `robot_{side}_wrist_position` | (T, 3) | Retargeted wrist positions |
| `robot_{side}_wrist_wxyz` | (T, 4) | Retargeted wrist orientations |
| `robot_{side}_finger_joints` | (T, J) | Retargeted finger joint angles |
| `robot_{side}_frames` | (T, K, 7) | Hand FK frames [pos(3), quat(4)] |
| `{side}_robot_frame_names` | list | Body names for hand frames |
| `{side}_robot_finger_joint_names` | list | Finger joint names |

### Contact Data (from V2P retargeting)

| Column | Shape | Description |
|--------|-------|-------------|
| `mano_{side}_link_contact_positions` | (T, N, 4) | Contact positions on hand [xyz, part_id] |
| `mano_{side}_object_contact_positions` | (T, N, 4) | Contact positions on object |
| `mano_{side}_object_contact_normals` | (T, N, 4) | Contact normals |
| `mano_{side}_object_contact_part_ids` | (T, N) | Object body index per contact |

### Object and Scene (for SceneConfig)

| Column | Shape | Description |
|--------|-------|-------------|
| `object_name` | string | Object name for registry lookup |
| `object_body_names` | list | Object body names |
| `object_body_position` | (T, B, 3) | Object body trajectory |
| `object_body_wxyz` | (T, B, 4) | Object body orientations |
| `object_articulation` | (T,) | Articulation parameter (0 = rigid) |
| `support_position` | (3,) | Support surface center XYZ |
| `support_size` | (3,) | Support surface dimensions |

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
