# Floating-Hand Sharpa Contract

Use this reference for two independently controlled floating Sharpa hands.

## Registered task

| Record task | Inference task | Modality config | Contract |
|---|---|---|---|
| `Sharpa-V2D-Gr00t-Record-v0` | `Sharpa-V2D-Gr00t-Inference-v0` | `groot_finetune/sharpa_dual_hand_three_camera_config.py` | `sharpa_dual_hand_three_camera` |

The recording and inference tasks provide front, right-wrist, and left-wrist videos. All three are
required by the embodiment contract and dataset.

## State and action

Record terms:

```text
wrist_position_e:    right xyz + left xyz = 6
wrist_orientation_e: right wxyz + left wxyz = 8
finger_joint_pos:    right 22 + left 22 = 44
```

State dimension is 58. State key order:

```text
right_wrist_pos
left_wrist_pos
right_wrist_quat
left_wrist_quat
right_finger
left_finger
```

Absolute action key order:

```text
right_wrist_pos
right_wrist_quat
right_finger
left_wrist_pos
left_wrist_quat
left_finger
```

The equal dimensions do not imply equal key order.

Conversion and inference align wrist quaternion signs to the same fixed per-hand reference
quaternions through the `sharpa_dual_hand_three_camera` contract adapter.

## Cameras

| Observation term | Video key |
|---|---|
| `image` | `front` |
| `image_right_wrist` | `right_wrist_view` |
| `image_left_wrist` | `left_wrist_view` |

## Native recording

Use the semantic rerender route:

```bash
python scripts/rsl_rl/rerender_demo_visuals.py --headless \
  --task Sharpa-V2D-Gr00t-Record-v0 \
  --contract sharpa_dual_hand_three_camera \
  --task_profile <TASK_PROFILE_JSON> \
  --checkpoint <CHECKPOINT> \
  --motion_file <MOTION_DIR> \
  --num_demos <COUNT> \
  --record_output <OUTPUT_ROOT>/recording
```

Keep deterministic frame-zero source rollouts by default. If the policy was trained with random
trajectory starts and cannot succeed from frame zero, use the logged seed with
`--random_source_start --min_source_steps <N>`. Record the accepted source range and reject a
trivial near-end timeout.

Native source success means reaching the configured task timeout. Rerendered copies must preserve
source state/action arrays while changing camera pixels.

## Evaluation

Warm cameras with render ticks only so renderer initialization does not advance simulation or
command state.

Use the evaluator declared by the task profile. Lift-and-hold is valid when
`object_position_e` is present; do not hardcode it for unrelated floating-hand tasks.

Closed-loop evaluation requires all three camera terms and rejects mismatched checkpoint modality
metadata before stepping the environment.

## Required tests

Run:

```bash
python -m pytest \
  groot_finetune/test_convert_to_gr00t.py \
  groot_finetune/closed_loop/test_profiles.py \
  groot_finetune/test_recording_contract.py -q
```

Test quaternion alignment, state/action key order, camera keys, semantic recording lengths, and a
short GPU record plus closed-loop smoke.
