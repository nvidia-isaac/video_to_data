# Terminations

## `terminations.py`

| Function | Params | Description |
|----------|--------|-------------|
| `timestep_termination` | `command_name` | Terminate when trajectory end is reached |
| `anchor_pos_error` | `command_name`, `threshold` | Root position error exceeds threshold (meters) |
| `anchor_quat_error` | `command_name`, `threshold` | Root orientation error exceeds threshold (radians) |
| `ee_position_error` | `command_name`, `threshold` | Sum of squared EE position errors exceeds threshold |
| `ee_quat_error` | `command_name`, `threshold` | Sum of squared EE orientation errors exceeds threshold |
| `joint_pos_error` | `command_name`, `threshold` | Joint position L2 norm exceeds threshold |
| `object_pos_error` | `command_name`, `threshold` | Object position error exceeds threshold (meters) |
| `object_quat_error` | `command_name`, `threshold` | Object orientation error exceeds threshold (radians) |
