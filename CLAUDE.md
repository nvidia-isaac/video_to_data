# CLAUDE.md

Guidance for Claude Code / developers working in this repo.

## What this is

A **ground-up Isaac Lab (Isaac Sim 5.1) reimplementation** of the SimToolReal RL task —
an object-centric policy for zero-shot dexterous tool manipulation with an **IIWA14 arm +
left Sharpa hand (29 DOF)**. It mirrors the original Isaac Gym `SimToolReal` env's
observations, actions, reward, and the **SAPG** algorithm (a vendored `rl_games` fork), but
the sim plumbing is rewritten against Isaac Lab's `DirectRLEnv`.

Two things work here:
1. **Train from scratch** in Isaac Lab (SAPG), and **evaluate** the resulting checkpoint.
2. **Zero-shot deploy** the *original Isaac Gym pretrained checkpoint* directly in this Isaac
   Lab env (it transfers — the policy grasps + manipulates the tools), and run faithful
   **demos** reproducing the original `dextoolbench/eval_interactive.py` scenarios.

The vertical slice is the `claw_hammer` object; the demos also cover `long_screwdriver` and
`sharpie_marker`.

## Layout & environment (non-obvious)

- **`~/simtoolreal_isaaclab/`** (this repo, gittable). Top level:
  - `simtoolreal_lab/` — the Python package (import via `simtoolreal_lab.tasks`):
    - `simtoolreal_lab/tasks/simtoolreal/` — the env: `simtoolreal_env.py` (`DirectRLEnv`),
      `simtoolreal_env_cfg.py` (cfg), `robot_gains.py` (exact per-joint Kp/Kd/effort),
      `agents/rl_games_{ppo,sapg}_cfg.yaml`.
    - `simtoolreal_lab/ported/observation_action_utils_sharpa.py` — obs/action layout
      contract copied from the original (the source of truth for tensor layout).
  - `scripts/` — entry points: `train.py`, `play.py`, `deploy_pretrained.py`,
    `convert_object.py`, plus `verify_assets.py` / `inspect_robot.py`.
  - `assets/` — robot + object URDFs and converted USDs. **Gitignored** (large); regenerate
    object USDs with `convert_object.py`.
  - `trajectories/` — fixed-goal task trajectories (world-frame pose sequences).
  - `logs/`, `videos/` — generated outputs (**gitignored**).
  - The scripts add the repo root to `sys.path`, so `import simtoolreal_lab...` works without
    installing the package.
- **`~/isaaclab/`** (outside this repo): `env_isaaclab/` (uv venv, Python 3.11, Isaac Sim
  5.1.0 + Isaac Lab 0.54.4) and `IsaacLab/` (cloned IsaacLab). Moved out so this folder is
  gittable.
- The **SAPG `rl_games` fork** is pip-installed editable from **`~/simtoolreal/rl_games`**
  (the *original* Isaac Gym repo) — that path must exist. Plain `pip install rl-games` is the
  wrong code (no SAPG).
- The **original pretrained policy** (`config.yaml` + `model.pth`) and the **task
  trajectories** are read from the original repo at **`~/simtoolreal/`** (see
  `deploy_pretrained.py`: `ORIG_REPO`, `TRAJ_ROOT`).

**Everything runs through `isaaclab.sh` with the venv active.** The standard prefix:

```bash
source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && \
  OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p <script.py> <args>
```

Gotchas: `num_envs` must be divisible by **6** for training (SAPG block count); the gym task
id is `Isaac-SimToolReal-ClawHammer-Direct-v0`; paths in the scripts are currently absolute
(`/home/cning/...`), so the repo runs in place rather than portably.

## Commands

All commands below assume the run prefix above, i.e.
`... ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/<script>` (so `.../scripts/X` means
`~/simtoolreal_isaaclab/scripts/X`).

### Train (from scratch, SAPG)
```bash
.../scripts/train.py --headless --num_envs 24576 --max_iterations 10000 \
    --agent_cfg rl_games_sapg_cfg.yaml [--no-domain_randomization] [--run_name 00_my_run]
```
- `--agent_cfg rl_games_sapg_cfg.yaml` selects SAPG (default is plain PPO). Run names must
  start with a numeric prefix (`00_`) — the SAPG fork parses `int(name.split('_')[0])`.
- `--domain_randomization` is on by default (`--no-domain_randomization` to disable).
- Auto-scales `minibatch_size = horizon*num_envs/4` and `expl_coef_block_size = num_envs/6`.
- Checkpoints → `logs/simtoolreal/<run_name>/nn/`.

### Evaluate a from-scratch checkpoint
```bash
.../scripts/play.py --headless --checkpoint logs/simtoolreal/<run>/nn/<run>.pth --num_envs 96 \
    [--delta] [--video --cam_env_index N]
```
- Default goals = the fixed `swing_down` trajectory; `--delta` = training delta-goal
  distribution. Metrics → `eval_result.txt`. The checkpoint must be a 6-block SAPG net.

### Zero-shot deploy the ORIGINAL pretrained checkpoint  +  demos
```bash
# numeric eval (swing_down trajectory, 96 envs)
.../scripts/deploy_pretrained.py --headless --num_envs 96 [--delta]

# faithful eval_interactive.py DEMO (1 env, fixed init, stops when all goals reached)
.../scripts/deploy_pretrained.py --headless --demo \
    --object {claw_hammer|long_screwdriver|short_screwdriver|sharpie_marker|staples_marker} \
    --demo_task {swing_down|spin_vertical|spin_horizontal|draw_smile|write_c} \
    --video --steps 1500 --video_length 1500
```
- Builds the net from the original `config.yaml`'s `train.params` (so the checkpoint loads
  bit-for-bit), restores `model.pth`, runs the env with `pretrained_compat=True`.
- `--demo` adds the exact `eval_interactive.py` scenario (fixed object init from the
  trajectory `start_pose`, fixed-size-keypoint success @ tol 0.015, `startArmHigher`, no
  noise) and **stops the rollout/recording when all trajectory goals are reached**.
- Videos → `videos/pretrained_<object>_<task>-step-0.mp4`; metrics → `deploy_result.txt`.

### Convert an object URDF → USD (convex decomposition)
```bash
.../scripts/convert_object.py <input.urdf> <output.usd> --headless
```
Used to (re)generate the gitignored object USDs (collision = convex decomposition).

## Logic flow

### The env (`simtoolreal_env.py`, `DirectRLEnv`)
- **Observation (140-dim)**, in this exact order: `joint_pos`(29, unscaled to [-1,1]),
  `joint_vel`(29), `prev_action_targets`(29), `palm_pos`(3), `palm_rot`(4), `object_rot`(4),
  `fingertip_pos_rel_palm`(15), `keypoints_rel_palm`(12), `keypoints_rel_goal`(12),
  `object_scales`(3). Keypoints are 4 corners of a coarse object bounding box (the
  object-centric abstraction). An asymmetric **critic state** (155-dim) adds privileged
  object velocities / distances for SAPG's central value.
- **Action (29-dim)** → joint position targets: hand (7:29) scaled to limits + EMA; arm (0:7)
  integrated (`prev + dof_speed_scale*dt*action`) + clamp + EMA. Computed once per control
  step in `_pre_physics_step`, held across `decimation` substeps (60 Hz control).
- **Reward**: fingertip-approach (pre-lift) + lifting reward/bonus + keypoint-to-goal delta
  (post-lift) + reach-goal bonus + small action penalties. Success = keypoints within
  tolerance for `success_steps`; `successes` advances the goal.
- **Goals**: training = delta goals (first goal elevated in a target volume → forces lifting;
  subsequent = small body-frame deltas) + a tolerance curriculum. Eval/demo = a fixed
  world-frame trajectory (`trajectories/<cat>/<obj>/<task>.json`), advanced by `successes`.

### Three modes (cfg flags toggle obs/action/init)
- **Training** (default): obs/action in Isaac Lab-native convention; DR + curriculum on.
- **`pretrained_compat`** (set by `deploy_pretrained.py`): aligns everything to the ORIGINAL
  Isaac Gym convention so the pretrained checkpoint runs zero-shot — `object_scales` from the
  dextoolbench scale, **xyzw** quats for `palm_rot`/`object_rot`, the exact `Q_LOWER/UPPER`
  joint-limit constants for unscale/scale, SAPG coef_cond (append the exploit coef at obs idx
  140), and per-shape friction (the 5 `*_DP` fingertip links = 1.5; arm/object/table = 0.5 —
  critical for a reliable grasp).
- **`demo_mode`** (set by `--demo`): on top of compat, reproduces `eval_interactive.py` —
  fixed object init from the trajectory `start_pose`, success measured on **fixed-size**
  keypoints, `startArmHigher` arm pose, no reset noise.

### RL wiring
- **Training** registers the env with the vendored SAPG `rl_games` fork via the agent yaml
  (`rl_games_sapg_cfg.yaml`): LSTM(1024) + MLP[1024,1024,512,512], `coef_cond` sigma,
  leader-follower experience sharing, 6 blocks, asymmetric `central_value_config`.
- **Deploy** builds the player from the *original* `config.yaml`'s `train.params` (guarantees
  the state_dict matches the pretrained `model.pth`), then runs a deterministic rollout.
  Note: rl_games' player `clip_actions` must be `False` here (the env clamps actions itself;
  otherwise rescaling against the env's infinite action-space bounds yields NaN).

## Conventions

- Entry-point scripts use argparse + Isaac Lab's `AppLauncher`. Env config is an Isaac Lab
  `@configclass`.
- Don't reorder the obs concat or the joint canonicalization (`JOINT_NAMES_ISAACGYM`) — the
  layout must match `observation_action_utils_sharpa.py` (shared with the original / real
  deployment).
- Assets, logs, and videos are generated/large and gitignored — never assume they're present
  in a fresh clone.
