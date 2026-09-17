# Vega/Dexmate Sharpa Contract

Use this reference only when the robot and export match the calibrated Vega Sharpa joint-space
contract exactly. A different Dexmate layout requires a new contract.

## Registered tasks and contract

```text
record task:         VegaSharpa-WholeBody-Gr00t-Record-v0
inference task:      VegaSharpa-WholeBody-Gr00t-Joint-Inference-v0
embodiment contract: vega_sharpa_joint
modality config:     groot_finetune/vega_sharpa_joint_config.py
```

## State and action

Record terms:

```text
arm_joint_pos:    right arm 7 + left arm 7 = 14 raw radians
finger_joint_pos: right 22 + left 22 = 44 raw radians
```

State key order:

```text
right_arm
left_arm
right_finger
left_finger
```

Absolute action key order:

```text
right_arm
right_finger
left_arm
left_finger
```

Both total 58 dimensions. Never feed concatenated state directly as action without reordering.
Joint names must match the canonical Vega GR00T order.

## Cameras

| Observation term | Video key |
|---|---|
| `image` | `front` |
| `image_right_wrist` | `right_wrist_view` |
| `image_left_wrist` | `left_wrist_view` |

All are calibrated 240x320 RGB streams in the current record task.

## Source routes

Camera-free source route:

```bash
python scripts/rsl_rl/export_parallel_rollouts.py --headless \
  --task VegaSharpa-WholeBody-Manip-v0 \
  --checkpoint <CHECKPOINT> \
  --motion_file <MOTION_DIR> \
  --num_envs <COUNT> \
  --num_steps <MAX_STEPS> \
  --contract <CONTRACT_JSON> \
  --export_dir <OUTPUT_ROOT>/source
```

For the released tissue-box example, use the repository-owned Git LFS expert at
`robotic_grounding/source/robotic_grounding/robotic_grounding/assets/policies/e2e_example/tissue_box/vega_sharpa_policy.onnx`.
The exporter also accepts compatible RSL-RL `.pt` checkpoints. ONNX support is inference-only and
does not change source-expert training.

The optional reset-noise flags set named environment-event ranges before construction. They
default to zero and perturb simulator state only, never the motion reference or policy action.

External-export route:

```text
manifest.json:
  format: joint_rollout
  schema_version: 1
  embodiment_contract: vega_sharpa_joint
  embodiment_contract_sha256: exact contract hash
  fps: 20
  joint_names: exact canonical order
  object_names: exact SceneConfig order
  timeout_termination: timeout

episode_*.npz:
  joint_pos:     (T, 58)
  action_target: (T, 58)
  object_pose:   (T, B, 7), position + wxyz
  source_success: scalar bool (timeout only)
```

Select exactly the requested successes before rendering. Then:

```bash
python scripts/rsl_rl/replay_record.py --headless \
  --export_dir <SELECTED_EXPORT> \
  --motion_file <MOTION_DIR> \
  --contract <CONTRACT_JSON> \
  --task_profile <TASK_PROFILE_JSON> \
  --num_envs <RENDER_ENVS> \
  --num_demos <COUNT> \
  --save_on source_success \
  --record_output <OUTPUT_ROOT>/recording \
  --verify
```

`--verify` requires post-replay joint and object-position error below `1e-3`.

## Collection and reset

For the validated frame-zero evaluation contract:

```text
reset frame: 0
reset_finger_openness: 0.0
VOC: off
curriculum: off
```

Use a pilot eligibility rate to plan one large camera-free collection wave. Hundreds of camera-free
source environments do not imply that the three-camera inference task can render the same number.

## Closed-loop evaluation

Warm the renderer with render ticks only, without stepping the environment. Reject all-black first
policy frames before evaluating the first action.

Use training-matched per-episode visual randomization. Keep startup light initialization separate
from per-episode visual terms.

Closed-loop thresholds come only from the selected task profile.

Keep timeout and non-finite robot-state termination. Remove source wrist/object trajectory
deviation terminations during free-running evaluation.

Measure rendered capacity for the active renderer, camera count, and runtime.

## Required tests

Run:

```bash
python -m pytest \
  groot_finetune/test_source_policy.py \
  groot_finetune/test_convert_to_gr00t.py \
  groot_finetune/closed_loop/test_profiles.py \
  groot_finetune/test_recording_contract.py \
  groot_finetune/test_replay_export.py \
  groot_finetune/test_evaluation.py -q
```

Construct the registered record and inference task configs in IsaacLab. Run a verified replay or
native record smoke, open-loop gate, one-episode closed-loop smoke, and then the requested
evaluation.
