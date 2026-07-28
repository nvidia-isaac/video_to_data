# Stage contracts

The experiment is intentionally split at two narrow, versioned seams. A stage
must fail on incompatible geometry instead of resizing or dropping frames
implicitly.

## Tracking contract (`tracking.npz`, schema `v2d.inpainting.tracking/v1`)

Required arrays:

| Key | Shape | Meaning |
|---|---:|---|
| `schema_version` | scalar string | Exact schema identifier |
| `tracker` | scalar string | `phantom`, `v2d`, or `ground_truth` |
| `coordinate_frame` | scalar string | `camera` or `world` |
| `frame_indices` | `(N,)` int | Original zero-based RGB frame indices |
| `left_valid`, `right_valid` | `(N,)` bool | Per-side observation validity |
| `left_wrist_position`, `right_wrist_position` | `(N,3)` float | Metres in `coordinate_frame` |
| `left_wrist_wxyz`, `right_wrist_wxyz` | `(N,4)` float | Unit quaternion, scalar first |

Optional arrays:

| Key | Shape | Meaning |
|---|---:|---|
| `left_joints_3d`, `right_joints_3d` | `(N,21,3)` | MANO-order hand joints in metres |
| `left_joints_wxyz`, `right_joints_wxyz` | `(N,21,4)` | Unit WXYZ orientations for the MANO-order joints in `coordinate_frame` |
| `left_joints_2d`, `right_joints_2d` | `(N,21,2)` | Pixel coordinates in source RGB geometry |
| `left_finger_joints`, `right_finger_joints` | `(N,J)` | Sharpa joint values in declared name order |
| `left_finger_joint_names`, `right_finger_joint_names` | `(J,)` string | Joint names corresponding to columns |

World-frame tracks also require camera calibration in the condition metadata:
one intrinsic matrix and a frame-aligned world-to-camera transform. Missing
calibration is an explicit blocked state, never an inferred identity transform.
All scalar metadata is stored as a zero-dimensional NumPy string, frame indices
use an integer dtype, validity arrays use exact boolean dtype, and geometric or
joint arrays use a floating-point dtype. Every position, quaternion, joint, and
finger value in a row marked valid must be finite; invalid rows may contain
non-finite sentinels. Every quaternion in a valid `*_joints_wxyz` row must also
have unit norm within the contract tolerance. Although joint orientations are
optional for generic tracking producers, the standalone Video2Data Sharpa
stage requires both sides' arrays so it can consume the exact MANO FK result
without recomputing pose conventions or changing frame alignment.

## Retargeted robot trajectory (`robot_trajectory.npz`)

Tracking and robot retargeting are separate so all three trackers can be
evaluated through the same Sharpa representation. The schema is
`v2d.inpainting.robot-trajectory/v1` and requires:

- scalar `coordinate_frame`, `robot`, and `gripper` strings;
- contiguous `frame_indices` and per-side validity arrays;
- per-side `(N,3)` wrist position and `(N,4)` WXYZ orientation;
- per-side `(N,J)` Sharpa finger joints and `(J,)` matching joint names.

It follows the same integer frame-index, exact boolean validity, scalar string,
floating-point target, and finite-valid-row rules as the tracking archive.

The TACO adapter can copy the already-retargeted Sharpa fields for the GT
condition. Phantom and Video2Data adapters must invoke the same Video2Data
Sharpa retargeter rather than inventing tracker-specific robot semantics.

## Human-removal masks

Every formal human-removal mask is an `(N,H,W)` NumPy array with exact boolean
dtype. `True` means remove this pixel, and `N`, `H`, and `W` must match the
selected decoded source-video window. Producers must normalize binary
integer/image masks to boolean before writing this contract.

The condition-comparison pipeline publishes `arm_masks.npy` for E2FGVI. The
MECKA-to-Panda automatic path publishes `arm_mask/arm_mask.npy` with schema
`v2d.inpainting.mecka-arm-mask-run/v1`, then passes that array to its ProPainter
adapter. The adapter may materialize temporary PNGs for the backend, but those
PNGs are not a second contract. Dilation and inference configuration belong in
the producer's versioned sidecar rather than being implied across backends.

## Robot render contract

The renderer emits:

| Artifact | Shape/encoding | Meaning |
|---|---|---|
| `robot_rgb.mp4` | source `N,H,W,fps` | Vega + Sharpa RGB on a flat background |
| `robot_mask.npy` | `(N,H,W)` bool | Pixels belonging to the robot |
| `robot_depth.npy` | `(N,H,W)` float32 | Positive metric camera-z depth; invalid is `+inf` |

The compositor requires the renderer's `render_metadata.json` commit marker.
It accepts only schema `v2d.inpainting.robot-render/v1` with `state=complete`,
matching frame geometry, exact bundle filenames, and matching declared byte
sizes and SHA-256 values. Immediately before committing its output, the
compositor revalidates the consumed robot and object bundles and confirms that
their sidecars and the base video still match the generation opened at start.

## TACO object-occlusion render contract

The GT TACO parquet exposes both the manipulated `tool` and `target` object in
the same world frame as the hands. The object pass emits:

| Artifact | Shape/encoding | Meaning |
|---|---|---|
| `object_mask.npy` | `(N,H,W)` bool | Pixels covered by either GT object mesh |
| `object_depth.npy` | `(N,H,W)` float32 | Nearest positive metric camera-z object depth; invalid is `+inf` |
| `object_render_metadata.json` | JSON | `v2d.inpainting.taco-object-render/v1` commit marker |

Masked depth values must be finite and strictly positive; every unmasked value
must be positive infinity. The sidecar must be `state=complete`, match source
geometry, and carry exact artifact byte sizes and SHA-256 values.

If the entire object trio is omitted, compositing retains the original hard
robot-mask behavior. If supplied, all three files are mandatory and robot
visibility is:

```text
robot_mask & (~object_mask | robot_depth <= object_depth + depth_guard_m)
```

The default guard is 0.003 m. It favors the robot within 3 mm of an object
surface, preventing small calibration or rasterization differences from
creating unstable edge holes. Composite input/output/metadata paths must be
distinct after resolution, and existing output files require explicit
overwrite authorization.

## Metadata sidecars

Every stage writes JSON containing its configuration, frame geometry, source
fingerprints, implementation/model identity, artifact identity, and completion
state. Operational multi-file bundles may also record absolute host paths and
wall-clock timestamps. Deterministic model stages such as E2FGVI deliberately
omit machine-specific paths and timestamps so identical inputs and options
produce identical sidecars. In either policy, partially written media must
never be reported as complete and consumers must validate the bundle's
completion state before reading its artifacts.
