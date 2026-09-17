# Embodiment Contract

Use this contract to keep collection, conversion, training, and inference consistent. Resolve
every field from code or data and record it in the run report before execution.

## Identity and routing

```text
contract_id
robot_name
source_task
record_task
inference_task
modality_config
source_action_terms
joint_names
reset_joint_groups
source_terminations
evaluation_terminations
```

Task semantics belong in a separate task profile. A source policy and an external export must
both pass explicit source-success validation.

## Data contract

```text
record observation terms and shapes
state modality keys, order, units, transforms, dimension
action modality keys, order, representation, units, dimension
camera observation term -> video modality key
FPS
episode horizon
action horizon
```

State and action may have the same dimension while using different slice orders. Compare key
order and slice boundaries, not only totals. The HDF5, LeRobot `meta/modality.json`, training
config, server-reported config, and closed-loop adapter must agree.

Transforms are part of the contract. Examples include quaternion hemisphere alignment,
joint normalization, coordinate-frame conversion, and action de-normalization.

## Success contract

Define two separate gates:

```text
source_success:
  timeout termination term
  expected episode length

task_success:
  evaluator ID
  observation terms
  thresholds and hold duration
```

Source success determines training eligibility and means the source policy reached the configured
timeout. Task success evaluates the fine-tuned policy. Never substitute process exit or source
trajectory tracking for task success.

For lift-and-hold:

```text
evaluator: lift_hold
object_position_term: object_position_e
lift_threshold_m
hold_threshold_m
min_hold_steps
```

Other tasks must register or implement another evaluator and test it independently.

## Evaluation contract

```text
reset frame or reset distribution
initial finger/gripper state
curriculum and virtual-control state
visual randomization mode
warmup mode
source and evaluation terminations
camera aliases for recordings
```

Warmup must use render ticks only so it cannot change the logical initial condition or delayed
actions. Validate the first policy frame from every required camera.

Free-running evaluation should retain timeout and true safety/non-finite checks while removing
source-reference deviation failures unless the task explicitly requires them.

## Capacity contract

Track separate measured limits:

```text
max_camera_free_envs
max_rendered_envs
camera_count
renderer/device/runtime
```

Do not infer rendered capacity from camera-free collection. Total views scale approximately as
`num_envs * camera_count`, but renderer descriptor limits may be reached before memory limits.

## Artifact invariants

Require:

1. Exactly the requested number of selected timeout-eligible sources.
2. Identical expected horizon for every selected source time series.
3. Identical horizon for HDF5 actions, state terms, and every camera.
4. One Parquet episode per demo with exactly that many rows.
5. One video per selected camera and episode with exactly that many frames.
6. Finite statistics with modality dimensions matching the contract.
7. A complete checkpoint with processor/config/statistics and all model shards.
8. Finite open-loop metrics before closed-loop work.
9. Structured task-level closed-loop results with reset and randomization metadata.
10. Successful recordings selected by the same evaluator used for the reported success rate.

Reject mixed contracts in one dataset.
