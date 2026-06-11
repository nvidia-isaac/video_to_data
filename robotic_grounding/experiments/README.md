# Experiments

This directory keeps the committed experiment runner, shared helpers, and a small
set of checked-in configs. Private experiments and run outputs should stay local.

Run commands from the repository root:

```bash
python robotic_grounding/experiments/run_experiment.py --list
python robotic_grounding/experiments/run_experiment.py example_single --local --dry-run
python robotic_grounding/experiments/run_experiment.py example_single --osmo
```

## Tracked Files

| Path | Purpose |
|------|---------|
| `run_experiment.py` | Main entry point for local training and OSMO submission. |
| `launch_stage2.py` | Helper for stage-2 jobs that fetch stage-1 checkpoints from W&B. No stage-2 example config is committed on this branch. |
| `utils.py` | Shared command and workflow-generation helpers. |
| `registry.yaml` | Committed experiment id to directory mapping. |
| `example_single_run/config.yaml` | Minimal single-sequence smoke example. |
| `recon_body_*/config.yaml` | Whole-body reconstruction example configs currently kept in the repo. |

## Registered Configs

`registry.yaml` currently exposes these committed ids:

| Id | Directory |
|----|-----------|
| `example_single` | `example_single_run` |
| `recon_body_apple` | `recon_body_apple_pick` |
| `recon_body_blue_trash_can_drag_007` | `recon_body_blue_trash_can_drag_007` |
| `recon_body_bottle_pick_transfer` | `recon_body_bottle_pick_transfer` |
| `recon_body_corn_can_right_left_handover_01` | `recon_body_corn_can_right_left_handover_01` |
| `recon_body_snack_box_pick_and_place_01` | `recon_body_snack_box_pick_and_place_01` |

The old committed multi-sequence, sweep, stage-1, stage-2, and two-stage example
directories are intentionally not present on this branch.

## Adding Local Experiments

Create local experiment directories under `experiments/`, then register them in
`experiments/registry.local.yaml`. That file is gitignored and is merged with
`registry.yaml` at runtime, so local entries do not require tracked changes.

```yaml
# robotic_grounding/experiments/registry.local.yaml
my_experiment: my_experiment_dir
```

Then run:

```bash
python robotic_grounding/experiments/run_experiment.py my_experiment --local --dry-run
```

Use `example_single_run/config.yaml` as the starting template for a new config.
For private sweeps or multi-task experiments, add a local `workflow.py` next to
the config and keep the directory untracked.
