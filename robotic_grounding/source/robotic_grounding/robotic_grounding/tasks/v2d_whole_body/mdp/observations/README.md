# Observations

## `policy_observations.py` — RL policy inputs

### Body-frame state (egocentric)

| Function | Shape | Frame | Description |
|----------|-------|-------|-------------|
| `wrist_position_b` | (E, 6) | body | Both wrists relative to pelvis [R3+L3] |
| `wrist_orientation_b` | (E, 12) | body | Both wrist orientations as 6D rotation [R6+L6] |
| `wrist_velocity_b` | (E, 6) | body | Both wrist linear velocities in pelvis frame [R3+L3] |
| `object_position_b` | (E, 3) | body | Object position relative to pelvis |
| `object_orientation_b` | (E, 6) | body | Object orientation as 6D rotation relative to pelvis |

### Future frame deltas

| Function | Shape | Description |
|----------|-------|-------------|
| `motion_anchor_pos_b` | (E, F*3) | Future root position deltas, flattened |
| `motion_joint_pos_delta` | (E, F*J) | Future joint position deltas from current |
| `motion_ee_pos_delta` | (E, F*L*3) | Future EE position deltas |
| `motion_ee_quat_delta` | (E, L*F*6) | Future EE rotation deltas as 6D |
| `object_pos_delta` | (E, F*3) | Future object position deltas |

### Scene and tracking

| Function | Shape | Description |
|----------|-------|-------------|
| `object_pose_delta_6d` | (E, 9) | Object error vs command [pos(3) + 6D rot(6)] |
| `hand_object_transform_6d` | (E, 9) | Hand-object transform [pos(3) + 6D rot(6)], distance-gated |
| `command_trajectory_progress` | (E, 1) | Normalized trajectory progress [0, 1] |
| `action_history` | (E, H*A) | Past H processed actions, flattened. Zeroed on reset |
| `contact_desired_positions_e` | (E, 2*N*3) | Target contact positions (left+right, env frame) |

## `sonic_tokenizer_observations.py` — SONIC encoder inputs

| Function | Shape | Description |
|----------|-------|-------------|
| `encoder_mode` | (E, 4) | One-hot encoder mode selection |
| `command_joint_pos` | (E, F*J) | Future joint positions, optionally SONIC-joints-only |
| `command_joint_vel` | (E, F*J) | Future joint velocities, optionally SONIC-joints-only |
| `motion_anchor_ori_b` | (E, F*6) | Future root orientation diffs as 6D |
| `command_z` | (E, F) | Future root Z (height) positions |
| `encoder_padding` | (E, dim) | Zero padding to match encoder input dimension |

## `sonic_policy_observations.py` — SONIC decoder inputs

| Function | Shape | Description |
|----------|-------|-------------|
| `joint_pos_rel` | (E, J) | Current joint positions relative to default pose |
| `joint_vel_rel` | (E, J) | Current joint velocities, optionally SONIC-only |
| `last_action` | (E, J) | Last SONIC output actions or raw RL actions |

## `rnd_observations.py` — Hand-object and contact

| Function | Shape | Description |
|----------|-------|-------------|
| `hand_object_transform` | (E, B*7) | Transform from hand to object (quat), distance-gated |
| `contact_force` | (E, N*3) | Mean contact forces from sensor history |

## Shape Legend

- **E** = num_envs, **F** = num_future_frames, **J** = num_tracked_joints
- **L** = num_ee_links, **N** = num_contact_links, **B** = num_object_bodies
- **H** = action_history_length, **A** = action_dim
- All orientations use 6D rotation (first two columns of rotation matrix) unless noted
