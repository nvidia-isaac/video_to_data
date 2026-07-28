# Generic parallel-jaw renderer

This package renders a complete bimanual parallel-jaw URDF without changing the
existing Vega/Sharpa renderer. It consumes a robot-neutral semantic target NPZ,
an explicit robot bundle, exact TACO calibration, and one fixed shared
`T_world_hub`.

## Target contract

`v2d.inpainting.parallel-jaw-target/v1` contains exactly:

```text
schema_version              scalar string
tracker                     scalar string
coordinate_frame            scalar string, exactly "world"
frame_indices               (N,) int, exactly 0..N-1
left_valid/right_valid      (N,) bool, all true
left_position/right_position (N,3) float, meters
left_wxyz/right_wxyz        (N,4) float, unit scalar-first quaternion
left_aperture_m/right_aperture_m (N,) float, nonnegative inner-jaw opening
```

Invalid frames are rejected. Interpolation, holding, and hand-to-gripper
semantic estimation belong upstream.

## Robot bundle contract

Bundle-relative URDF paths resolve relative to the bundle JSON. Absolute
`/robot_assets/...` visual mesh paths are supported for a read-only container
mount.

```json
{
  "schema_version": "v2d.inpainting.parallel-jaw-robot-bundle/v1",
  "robot_id": "example",
  "render_urdf": "robot_render.urdf",
  "ik_urdf": "robot_arm_only.urdf",
  "tcp_frames": {"left": "left_tcp", "right": "right_tcp"},
  "T_robot_root_hub": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ],
  "semantic_target_to_tcp_rotation": [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ],
  "arm_joint_names": ["left_j1", "right_j1"],
  "gripper_mapping": {
    "kind": "mirrored_prismatic",
    "joint_names": {
      "left": ["left_l_finger", "left_r_finger"],
      "right": ["right_l_finger", "right_r_finger"]
    },
    "params": {
      "closed_aperture_m": 0.0,
      "open_aperture_m": 0.095,
      "closed_joint_position_m": 0.0,
      "open_joint_position_m": -0.0475
    }
  },
  "fixed_root_posture": {
    "joint_values": {},
    "provenance": {"source": "reviewed embodiment posture"}
  },
  "asset_provenance": {"repository": "https://example.invalid", "revision": "pin"}
}
```

`T_robot_root_hub` maps hub-frame points into the robot-root frame. The
renderer computes:

```text
T_world_robot_root = T_world_hub @ inverse(T_robot_root_hub)
R_world_tcp = R_world_semantic @ semantic_target_to_tcp_rotation
```

Exactly one of `T_robot_root_hub` or its inverse `T_hub_robot_root` may be
declared. The IK URDF must contain only the named arm joints. The render URDF's
independent joints must be covered exactly once by arm joints, gripper drivers,
or fixed-root posture. URDF mimic followers are not listed; `yourdfpy` derives
them from each driven parent joint.

`galbot_four_bar` uses the source linkage:

```text
opening = 2 * (inner_pivot_half_gap - pad_inset
                + finger_link_length * sin(knuckle_angle - q))
```

The inverse is applied after clipping physical aperture to the linkage range.
Clipping counts and ranges are recorded per side.

## Container CLI

The host wrapper prints the command by default. `--execute` is required to run:

```bash
PYTHONPATH=video_to_data_internal python3 -m \
  inpainting.parallel_jaw_renderer.container_runner \
  --target /host/parallel_jaw_trajectory.npz \
  --bundle /host/bundle_manifest.json \
  --intrinsics /host/egocentric_intrinsic.txt \
  --world-to-camera /host/egocentric_frame_extrinsic.npy \
  --robot-asset-root /host/pinned_robot_repo \
  --scene-utils-root /host/robotic_grounding/tasks/scene_utils \
  --T-world-hub-metadata /host/gt_vega/render_metadata.json \
  --width 1920 --height 1080 --fps 30 \
  --output-dir /host/output \
  --gpu 0 --execute
```

The metadata option reuses only
`kinematics.arm_center_world` from a completed render and hashes that source
sidecar. Alternatively, pass `--T-world-hub` followed by 16 row-major floats.
No mount search is performed.

For visual QA, run separate one-frame jobs with `--preview-frame-index` at
start, middle, and end. Keeping them separate avoids pretending disjoint source
frames are temporally adjacent.

The renderer writes `robot_rgb.mp4`, `robot_mask.npy`, and `robot_depth.npy`.
It validates video geometry, visibility, TCP position/orientation residuals,
joint limits, and frame-to-frame arm-joint steps. `render_metadata.json` uses
the established `v2d.inpainting.robot-render/v1` compositor contract and is
atomically published last, after all three artifacts and their hashes exist.
