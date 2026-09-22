# Configuration

FlashCHORD applications use Hydra configuration groups. Training starts from one of five complete experiment
recipes and requires an explicit reference path:

```bash
python scripts/train_rl.py experiment=sharpa_ppo "task.parquet='${PARQUET}'"
python scripts/train_rl.py experiment=g1_recon_body_ppo "task.parquet='${PARQUET}'"

python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='${PARQUET}'"
python scripts/train_flash_sac.py experiment=dexmate_sharpa_flash_sac "task.parquet='${PARQUET}'"
python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac "task.parquet='${PARQUET}'"
```

The `experiment/` files own the validated embodiment, action, observation, objective, termination, reset,
curriculum, world-count, learner, replay, and logging selections. Override individual values only for an
intentional ablation.

Reusable groups are organized by runtime concern:

| Group | Responsibility |
| --- | --- |
| `embodiment/` | Robot asset, layout, controller, and semantic frames |
| `action/`, `observation/` | Policy actions and observations |
| `objective/`, `termination/`, `reset/` | Rewards, episode endings, and resets |
| `curriculum/` | Environment-step stage schedule |
| `scene/`, `collision/`, `sim/`, `env/` | Simulation and environment assembly |
| `training/` | Learner defaults |
| `viewer/`, `markers/`, `evaluation/` | Evaluation and viewer settings |

`task.parquet` is mandatory. The repository does not publish a default sequence or a sequence corpus.

The configured viewer backends are `gl`, `viser`, `mp4`, and `null`/`none`. MP4 output also requires an `.mp4`
`viewer.output_path` and even image dimensions.

Checkpoint-driven evaluation restores the saved training configuration and then applies the current evaluation,
viewer, and explicit command-line overrides. This release does not migrate older checkpoint configurations or
infer missing policy descriptions.
