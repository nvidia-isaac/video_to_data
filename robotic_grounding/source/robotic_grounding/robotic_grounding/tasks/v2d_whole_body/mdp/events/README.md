# Events

## `events.py`

| Function | Mode | Description |
|----------|------|-------------|
| `reset_robot_to_trajectory_start` | reset | Reset robot + object to a trajectory frame. Configurable via `TrackingCommandCfg`. |

### Reset behavior (all configurable):

- **Frame selection**: random within `trajectory_time_index`, or frame 0 if `always_reset_to_first_frame=True`
- **Root Z clamp**: `reset_root_height_min` prevents ground penetration at random start frames
- **Yaw-only**: `reset_yaw_only` zeros roll/pitch from root quaternion
- **Shoulder spread**: `reset_shoulder_spread > 0` + `freeze_steps > 0` widens arms and zeros fingers during freeze. Offset stored on command for smooth annealing via `_spread_blend_factor`
- **Object reset**: teleports object to trajectory frame. `RigidObject` guard (TODO: ArticulatedObject support)
