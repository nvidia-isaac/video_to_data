# GR00T post-training

This package owns the post-training path from an expert checkpoint to a closed-loop GR00T
evaluation. It does not configure or train the RL expert.

## Contracts

Every run requires two explicit inputs:

- An embodiment contract defines ordered state and action fields, required cameras, control
  frequency, reset-noise groups, inference task, and separate source/evaluation terminations.
- A task profile defines the object prompt, language instruction, target-object selector, and
  closed-loop evaluator.

The built-in Vega contract ID is `vega_sharpa_joint`. Contract and profile IDs use semantic names,
not numbered local experiment names. Serialized JSON has an explicit schema version.

The tissue-box lift task is provided only as an explicit example at
`groot_finetune/task_profiles/tissue_box_lift_hold.json`; its prompt and thresholds are never
pipeline defaults.

## Data flow

1. `scripts/rsl_rl/export_parallel_rollouts.py` runs an expert checkpoint without cameras. It
   accepts inference-only ONNX exports and RSL-RL `.pt` checkpoints; this does not change expert
   training.
   Source eligibility means the episode reached its timeout. Optional arm, finger, and rigid-
   object reset noise is configured on the environment before `gym.make`; defaults are zero.
2. `python -m groot_finetune.tools.select_successful_episodes` selects an exact number of
   timeout-eligible `joint_rollout` episodes.
3. `scripts/rsl_rl/replay_record.py` replays all named rigid objects with calibrated cameras and
   writes semantic HDF5. Replay validates exact contract identity, names, ordering, dimensions,
   frequency, and provenance.
4. `python -m groot_finetune.convert_to_gr00t` converts semantic HDF5 to LeRobot. The converter
   requires `--contract` and `--task-profile`; it does not infer layouts or camera subsets.
5. Isaac-GR00T statistics, fine-tuning, and open-loop evaluation use
   `groot_finetune/vega_sharpa_joint_config.py`, whose keys come from the same contract.
6. `groot_finetune/closed_loop/run_eval.sh` evaluates the served policy. Task success is reported
   independently from termination reason.

Example conversion:

```bash
python -m groot_finetune.convert_to_gr00t \
  --input out/recording/data.h5 \
  --output out/dataset \
  --contract out/contracts/embodiment.json \
  --task-profile out/contracts/task_profile.json
```

## Success semantics

- `source_success`: timeout-only eligibility for expert rollouts used as data.
- Artifact validity: exact structural and provenance checks performed by replay, conversion, and
  audit tools.
- Closed-loop task success: the evaluator in the task profile. The provided `lift_hold`
  evaluator measures maximum rise and consecutive samples above the hold threshold.

The closed-loop evaluator samples pre-action observations. It does not install reset hooks or
use post-action observations that have already been replaced by an automatic reset.

## Validation

From `robotic_grounding/`:

```bash
python -m pytest \
  groot_finetune/test_contracts.py \
  groot_finetune/test_source_policy.py \
  groot_finetune/test_evaluation.py \
  groot_finetune/test_replay_export.py \
  groot_finetune/test_convert_to_gr00t.py \
  groot_finetune/closed_loop/test_profiles.py -q
```

The audit tool derives dimensions and cameras from the contracts:

```bash
python -m groot_finetune.tools.audit_groot_run \
  --root out/run \
  --episodes 100 \
  --frames 519 \
  --contract out/run/contracts/embodiment.json \
  --task-profile out/run/contracts/task_profile.json \
  --selected-export selected \
  --hdf5 recording/data.h5 \
  --dataset gr00t_dataset
```
