# Naming conventions

Applied throughout `src/flash_chord/` so names are unambiguous about *what* a value is
and *which frame* it is in.

## Quantity
| Name | Meaning |
|---|---|
| `joint_pos` | joint position(s) [rad or m] |
| `pos` | Cartesian position [m] |
| `quat` | rotation as a quaternion |
| `rot_6d` | rotation as 6D (two basis columns) |
| `pose` | position **and** rotation together |
| `_ids` | indices (e.g. `body_ids`, `contact_part_ids`); `_id` for a single index |

## Frame suffix (physical quantities)
| Suffix | Frame |
|---|---|
| `_w` | world |
| `_b` | body |
| `_e` | simulation environment |
| `_o` | object |

Frame suffixes apply to any physical quantity (positions, poses, velocities, forces, …),
e.g. `wrist_pos_w`, `object_body_quat_w`, `contact_pos_o`, `wrist_pose_b`, `finger_joint_pos`,
`fingertip_body_ids`.

Quaternion layout is not implied by `quat` — state it in the docstring: Newton/Warp `wp.quat`
is `xyzw`; the retargeted parquet / `Reference` uses `wxyz`.
