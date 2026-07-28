# Whole-Body Tracking Command

The `TrackingCommand` loads all reference data from a single `motion_v1` parquet (via `robotic_grounding.motion_schema`) and provides command targets and current state observations consumed by rewards, observations, terminations, and actions.

## Data Flow

```
motion_v1 Parquet (Hive-partitioned: sequence_id=.../robot_name=.../*.parquet)
|
+-- robot_root_position (T, 3)              --> root_pos_w
+-- robot_root_wxyz (T, 4)                  --> root_quat_w
+-- robot_joint_positions (T, J)            --> joint_pos, joint_vel (via FD)
+-- robot_joint_names (J,)                  --> joint reordering against live robot
+-- ee_link_names, ee_pose_w (T, E, 7)      --> EE tracking + wrist body ID resolution
+-- object_body_position/wxyz (T, B, ...)   --> object tracking targets
|
+-- hand_frames_w[side] (T, K, 7)           --> _precompute_hand_keypoints_in_object_frame()
+-- hand_finger_joints[side] (T, Jf)        |   wrist + fingertip positions in object frame
|                                           |
+-- hand_link_contact_positions[side]       --> _precompute_contact_positions_in_object_frame()
+-- hand_object_contact_normals[side]       |   contacts + normals in object COM frame
+-- hand_object_contact_part_ids[side]      |
|                                           |
+-- hand_contact_active[side]               --> binary contact labels (for force closure reward)
+-- object_mesh_radius                      --> _precompute_contact_wrench_support_values()
```

At runtime each step:
1. `_update_command()` advances timestep and decays VOC (right-justified within freeze)
2. Properties index precomputed data by `timestep` and transform from object frame to env/world frame using the live object pose
3. `_spread_offset_blended` anneals shoulder spread during freeze period

## Initialization

| Method | Purpose |
|--------|---------|
| `_init_scene_references` | Resolve robot, object, wrist/fingertip/finger IDs, contact sensors |
| `_load_and_process_motion` | Load motion_v1 parquet, populate root/joints/EE tensors, resolve wrist body IDs |
| `_init_buffers` | Per-env counters, VOC scale, action history, spread offset |
| `_init_hand_data` | Fingertip/finger joint IDs, retargeted hand data, contact labels |
| `_init_contact_data` | Contact positions, normals, part IDs, validity masks |
| `_precompute_hand_keypoints_in_object_frame` | Hand frames + wrist poses -> object frame |
| `_precompute_contact_positions_in_object_frame` | Contact positions + normals -> object COM frame |
| `_init_wrench_data` | Wrench basis (512 samples), friction cone, runtime buffers |
| `_precompute_contact_wrench_support_values` | Per-timestep wrench supports from retargeted contacts |

## Properties

### Command Targets (single frame)

| Property | Shape | Frame |
|----------|-------|-------|
| `command_anchor_pos_w` | (E, 3) | world |
| `command_anchor_quat_w` | (E, 4) | world |
| `command_joint_pos` | (E, J) | joint (includes spread blend) |
| `command_object_pos_w` | (E, 3) | world |
| `command_object_quat_w` | (E, 4) | world |
| `command_ee_pos_w` | (E, L, 3) | world |
| `command_ee_quat_w` | (E, L, 4) | world |

### Command Targets (multi-future)

| Property | Shape |
|----------|-------|
| `command_joint_pos_multi_future` | (E, F, J) — includes spread blend |
| `command_joint_vel_multi_future` | (E, F, J) |
| `command_anchor_pos_w_multi_future` | (E, F, 3) |
| `command_anchor_rot_diff_l_multi_future` | (E, F, 6) — 6D rotation deltas |
| `command_anchor_z_multi_future` | (E, F) — root Z positions |
| `command_ee_pos_w_multi_future` | (E, F, L, 3) |
| `command_ee_quat_w_multi_future` | (E, F, L, 4) |
| `command_object_pos_w_multi_future` | (E, F, 3) |
| `command_multi_future` | (E, F, 2J) — joint pos + vel concatenated |

### Simulation State

| Property | Shape | Frame |
|----------|-------|-------|
| `robot_anchor_pos_w` | (E, 3) | world |
| `robot_anchor_quat_w` | (E, 4) | world |
| `robot_joint_pos` | (E, J) | joint |
| `robot_joint_vel` | (E, J) | joint |
| `robot_ee_pos_w` | (E, L, 3) | world |
| `robot_ee_quat_w` | (E, L, 4) | world |
| `object_pos_w` | (E, 3) | world |
| `object_quat_w` | (E, 4) | world |

### Hand Keypoints and Fingers

| Property | Shape | Frame |
|----------|-------|-------|
| `{side}_hand_wrist_pose_command_e` | (E, 7) | env |
| `{side}_hand_wrist_position_e` | (E, 3) | env |
| `{side}_hand_fingertip_position_command_e` | (E, K, 3) | env |
| `{side}_hand_fingertip_position_e` | (E, K, 3) | env |
| `{side}_hand_finger_joint_pos` | (E, Jf) | joint |
| `{side}_hand_finger_joint_pos_command` | (E, Jf) | joint |

### Contact and Wrench

| Property | Shape | Description |
|----------|-------|-------------|
| `{side}_hand_object_contact_command_positions_e` | (E, N, 3) | Target contacts (env frame) |
| `{side}_hand_object_contact_positions_e` | (E, B, N, 3) | Live contacts (env frame) |
| `{side}_hand_object_contact_forces_w` | (E, H, B, N, 3) | Live force history |
| `{side}_hand_contact_wrench_supports_command` | (E, B, S) | Precomputed wrench supports |
| `{side}_hand_contact_wrench_supports` | (E, B, S) | Live wrench supports |
| `{side}_hand_contact_active_command` | (E,) | Binary contact label at timestep |

### VOC and Action History

| Property | Shape | Description |
|----------|-------|-------------|
| `virtual_object_controller_scale_factor` | (1,) | Global VOC curriculum target |
| `virtual_object_controller_scale_factor_per_env` | (E, 1) | Per-env VOC (decays during freeze) |
| `object_position_e` | (E, 1, 3) | Current object position (env frame) |
| `object_orientation_e` | (E, 1, 4) | Current object orientation |
| `object_body_position_command_e` | (E, B, 3) | Target object body positions |
| `object_body_wxyz_command_e` | (E, B, 4) | Target object body orientations |
| `action_history` | (E, H*A) | Flattened past processed actions |

## Shape Legend

- **E** = num_envs, **F** = num_future_frames, **J** = num_tracked_joints
- **Jf** = num_finger_joints (per hand), **L** = num_ee_links, **K** = num_fingertips
- **N** = num_contact_filter_prims (per hand), **H** = history length, **B** = num_object_bodies
- **S** = num_wrench_space_basis_samples (default 512), **A** = action_dim
