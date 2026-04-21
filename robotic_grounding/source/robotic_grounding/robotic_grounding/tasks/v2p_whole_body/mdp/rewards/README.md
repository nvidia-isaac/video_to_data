# Rewards

## `tracking_rewards.py` — Motion tracking rewards

| Function | Kernel | Description |
|----------|--------|-------------|
| `motion_global_anchor_position_error_exp` | exp(-err/std^2) | Root position tracking |
| `motion_global_anchor_orientation_error_exp` | exp(-err/std^2) | Root orientation tracking |
| `motion_ee_position_error_exp` | exp(-err/std^2) | EE position tracking (per-link max) |
| `motion_ee_orientation_error_exp` | exp(-err/std^2) | EE orientation tracking (per-link max) |
| `motion_joint_pos_error_exp` | exp(-err/std^2) | Joint position tracking (sum over joints) |
| `motion_object_position_error_exp` | exp(-err/std^2) | Object position tracking |
| `motion_object_orientation_error_exp` | exp(-err/std^2) | Object orientation tracking |
| `motion_object_lifted` | continuous | Height-based lifting progress |
| `motion_progress` | continuous | Trajectory progress relative to reset point |
| `motion_finger_joint_pos_gaussian_exp` | exp(-err/std^2) | Finger joint tracking from retargeted data (sum L+R, max 2.0) |
| `motion_hand_keypoints_gaussian_exp` | exp(-err/std^2) | Hand keypoint tracking from retargeted data (sum L+R, max 2.0) |
| `motion_contact_tracking_gaussian_exp` | exp(-chamfer/std^2) | Contact point Chamfer distance |
| `motion_contact_force_gaussian_exp` | exp(-err/std^2) | Contact force magnitude tracking |

## `contact_rewards.py` — Wrench-based contact rewards

All wrench rewards use per-body wrench space support functions computed by the tracking command. Contact data flows: sensor -> COM frame transform -> `compute_wrench_space` -> `compute_wrench_space_support_function`.

| Function | Description |
|----------|-------------|
| `contact_wrench_support_reward` | Per-direction wrench support matching (command vs sim). Tolerance band, gated by both sides having support. |
| `unintended_contact_penalty` | Penalty for sim contact where command expects none. Binary + continuous magnitude. |
| `missed_contact_penalty` | Proportional penalty for missing expected contact directions per body. |
| `force_closure_reward` | Fraction of wrench basis directions with sim support, gated by binary contact labels. For use without retargeted contact data. |

Note: wrench rewards currently compute per-hand independently. For bimanual grasps, both hands' contacts should be combined into a single wrench space per body (TODO).

## `regularization_rewards.py` — Regularization

| Function | Type | Description |
|----------|------|-------------|
| `body_acc_l2` | Class | Body linear + angular acceleration penalty (velocity history) |
| `body_ang_vel_l2` | Function | Body angular velocity L2 penalty |
| `mmd_similarity_reward` | Class | MMD between current pose distribution and expert motion dataset |
