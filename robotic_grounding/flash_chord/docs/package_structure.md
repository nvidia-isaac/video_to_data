# Package structure

The release keeps the runtime, training, evaluation, and robot assets needed by the public recipes.

```text
src/flash_chord/
├── assets/          # Sharpa, Dexmate/Vega, G1, and reusable policy assets
├── configs/         # Hydra roots, reusable groups, and five experiment recipes
├── data/            # Reference loading, resampling, and runtime buffers
├── embodiments/     # Robot import, semantic layouts, and reference binding
├── envs/            # Warp environments and policy observations
├── evaluation/      # Rollout harness and CHORD/ADD-AUC/SPIDER/ManipTrans metrics
├── lifecycle/       # Curriculum, reset, and termination logic
├── objectives/      # Tracking, wrench, force-closure, and ReconBody rewards
├── runtime/         # Actions, delay, control, contacts, and replay
├── scene/           # Robot/object/support assembly and collision policy
├── training/        # PPO, FlashSAC, checkpoints, and metadata schemas
└── visualization/   # Markers and GL/Viser/MP4/null viewer integration
```

Top-level entry points are in `scripts/`:

- `train_rl.py` and `train_flash_sac.py` train PPO and FlashSAC;
- `evaluate_policy.py` evaluates checkpoints across many parallel simulation worlds;
- `view_policy.py` visualizes PPO and FlashSAC checkpoints;
- `scripts/debug/view_robot.py` and `view_scene.py` inspect released robots and reference scenes;
- `scripts/debug/replay.py` provides physical, kinematic, Dexmate, and SONIC local replay modes.

`workflow/` contains remote training and evaluation templates. `tests/` separates pure/unit coverage from optional
CUDA and local-sequence integration coverage.

Reference sequence corpora, system-identification data/pipelines, retired experiment matrices, policy exporters,
and checkpoint migration layers are not part of the release branch.
