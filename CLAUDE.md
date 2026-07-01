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

### Collect a BC dataset → convert to a GR00T training dataset
For training/finetuning a GR00T-N1.7 policy (the stock model **or** the DINOv3 specialist in
`Isaac-GR00T/`), build an imitation dataset by rolling out a policy, then convert it to GR00T's
LeRobot format. Two steps, **two different environments**:

**1. Collect** (rolls out the pretrained SAPG policy; uses the `isaaclab.sh` prefix above). The hammer
task **defaults to the teleport-recovery + expert-faithful-proprio config** (`hammer_bc_teleport`):
```bash
.../scripts/collect_bc_data.py --headless --task hammer --num_envs 16 --num_demos 1000 \
    --cam_width 640 --cam_height 480 --out datasets/hammer_bc_teleport.hdf5
# ^ implies (hammer defaults): --tool_displacement --tool_displace_pregrasp --simtoolreal --max_ep_steps 1500
#   add --no_tool_displacement / --no_simtoolreal for a plain image-BC dataset (the old hammer_bc_success style)
```
Writes a robomimic-style HDF5 (one group per SUCCESSFUL demo): `obs/image` (T,H,W,3) uint8,
`obs/joint_pos` (T,29) f32 (current joints), `actions` (T,29) f32 (**delta** joint targets),
`obs/keypoints` (T,8,3) f32 (4 tool-bbox + 4 dynamic-screw), and — with `--simtoolreal` (default-on for
hammer) — `obs/proprio` (T,109, incl. **joint_vel**) + `obs/keypoints_rel_palm` (T,8,3), @ 60 Hz. With
`--tool_displacement` (or `--joint_displacement`), also a per-step **`teleport` (T,) uint8** flag (1 = a
teleport fired that step — tool OR joint) → the trainers MASK the action-chunk loss at/after a teleport
(those recovery actions are unpredictable from the pre-teleport obs, so they'd be inconsistent chunk
supervision). Masking is automatic when the field is present (`--no_teleport_mask` to ablate); the
`convert_bc_to_gr00t.py` LeRobot path carries it as a `teleport` parquet column for `train_specialist.py`.
**Two teleport perturbations** (independent sampling, both feed the one `teleport` flag): `--tool_displacement`
(jump the OBJECT pose, default-on for hammer) and `--joint_displacement` (jump the robot's 29 arm+hand JOINT
positions, opt-in) — both leave the controller targets unchanged so the recorded action stays the clean expert command. The file
is locked while collecting — wait for the `DONE:` line before converting. **Memory**: 16 envs @ 640×480 /
1500-step episodes peaks ~35 GB — run in a `systemd-run --user --scope -p MemoryMax=48G` cgroup (the long
episodes + per-env frame buffers OOM-killed earlier runs at 100 envs).

**2. Convert** to a GR00T LeRobot-v2.1 dataset. Runs in the **GR00T venv** (has
pandas/pyarrow/av + h5py), *not* through `isaaclab.sh`:
```bash
cd ~/simtoolreal_isaaclab/Isaac-GR00T && source .venv/bin/activate
python ~/simtoolreal_isaaclab/scripts/convert_bc_to_gr00t.py \
    --hdf5 ~/simtoolreal_isaaclab/datasets/hammer_bc_success.hdf5 \
    --out  ~/simtoolreal_isaaclab/datasets/hammer_gr00t_lerobot \
    --task "hammer the screw into the hole"
```
Produces `meta/{info,modality,episodes,tasks,stats}.json[l]` + `data/chunk-000/episode_*.parquet`
(`observation.state[29]`, `action[29]`, `timestamp`, `frame_index`, `episode_index`, `index`,
`task_index`) + `videos/chunk-000/observation.images.front/episode_*.mp4` (H.264). `modality.json`
maps `state/action.joint_pos [0:29]`, `video.front`, and `annotation.human.task_description →
task_index` — one dataset for **both** stock GR00T (uses the task string) and the specialist
(ignores it). `stats.json` is required (the loader asserts it). Validate with
`gr00t.data.dataset.lerobot_episode_loader.LeRobotEpisodeLoader`.

**3a. Finetune the stock GR00T-N1.7** — the IIWA14 + Sharpa robot is **not** a pretrained GR00T
embodiment, so use the `NEW_EMBODIMENT` tag (the converter prints the exact command):
```bash
cd ~/simtoolreal_isaaclab/Isaac-GR00T && uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B --dataset-path <out> \
    --embodiment-tag NEW_EMBODIMENT --output-dir /tmp/finetune_out
```

**3b. Train the DINOv3 specialist** (the lightweight `Gr00tN1d7Specialist`: pretrained frozen
DINOv3 small_plus + ~50M action expert — see `Isaac-GR00T/gr00t/model/.../gr00t_n1d7_specialist.py`).
A self-contained trainer (GR00T venv) that does NOT use `finetune.sh` — it decodes the dataset's
videos once into a disk **memmap** (`uint8 [N,3,256,256]`, ~100 GB for 1000 demos) and trains the
flow-matching loss directly, with a recorded loss/val curve + checkpoints:
```bash
cd ~/simtoolreal_isaaclab/Isaac-GR00T && source .venv/bin/activate
python ~/simtoolreal_isaaclab/scripts/train_specialist.py \
    --dataset ~/simtoolreal_isaaclab/datasets/hammer_gr00t_lerobot_full \
    --name hammer --steps 12000 --batch_size 64 --pretrained_backbone --val_frac 0.05
```
Outputs `logs/gr00t_specialist/<name>/{<name>.pt, loss_curve.csv, loss_curve.png}`. State/action
are z-scored with `meta/stats.json`; images are ImageNet-normalized (DINOv3 default).

**4. Evaluate (host/client)** — the model needs the GR00T venv and the sim needs the isaaclab venv,
so eval is split: a policy **server** (loads the checkpoint, serves `get_action` over TCP) + an env
**client** (runs the hammer env, applies the returned delta-joint chunk as `cur_targets = clamp(
joint_pos + delta)` via a `BCEvalHammerEnv` subclass, scores success via `nail_driven`). Start the
server first, then the client:
```bash
# server (GR00T venv)
cd ~/simtoolreal_isaaclab/Isaac-GR00T && source .venv/bin/activate
python ~/simtoolreal_isaaclab/scripts/eval_specialist_server.py \
    --checkpoint ~/simtoolreal_isaaclab/logs/gr00t_specialist/hammer/hammer.pt --port 5601 &
# client (isaaclab.sh) -- replicates the collection scene; use --replan 1 (closed-loop) + --video
.../scripts/eval_specialist_client.py --headless --num_envs 25 --episodes 100 --replan 1 --port 5601
```
Use `--replan 1` (re-query every control step): open-loop chunk execution (`--replan 8`) drifts and
scores much lower. Note image-BC from a single camera + joints clones a 140-dim **privileged-state**
RL expert, so expect a large success-rate gap vs the expert (compounding error / info bottleneck).
**Image-BC stays near 0% on hammer**: single-view ≈ 0–4%, and a **2-view (table + wrist, `--wrist`)
DINOv3 specialist still scored 0%** @ table_dist 0.15 — extra views don't close the gap. The
**state-based SimToolReal specialist (below) is far stronger (≈61%)**; use it, not image-BC, for hammer.

### State-based policies (no images) — keypoint policy & SimToolReal specialist

Two state-based variants reuse the **same ~50M flow-matching action head** as the DINOv3 specialist but
swap the image+DINOv3 perception for a tiny **KeypointBackbone** (each keypoint → one token). Both train
on a `collect_bc_data.py` HDF5 and eval via the same host/client split (`BCEvalHammerEnv`, `--replan 1`).
The shared obs extractor is `simtoolreal_lab/tasks/simtoolreal/keypoint_utils.py` (used by BOTH the
collector and the eval client, so they compute identical inputs).

- **Keypoint policy** (`Gr00tN1d7KeypointPolicy`): input = 8 object-centric keypoints (4 tool bbox + 4
  dynamic screw, env-local absolute) + 29 `joint_pos`. Scripts: `train_keypoint_policy.py`,
  `eval_keypoint_{server,client}.py` (port **5601**). Records `obs/keypoints (T,8,3)` (default in the
  collector). On hammer it reaches ~7% (vs image ~4%, SAPG expert ~86%) — privileged geometry only
  modestly beats pixels; the BC-of-privileged-RL-expert gap dominates.

- **SimToolReal specialist** (`make_simtoolreal_config`, same `Gr00tN1d7KeypointPolicy` class): **expert-faithful**
  state — PALM-RELATIVE keypoints (tokens) + proprio `joint_pos(29)+joint_vel(29)+prev_targets(29)+palm_pos(3)+
  palm_rot(4)+fingertip_pos_rel_palm(15)` (109). **The BEST variant DROPS joint_vel** (`--no_joint_vel` → 80-dim;
  +~12–14 pts, see table). NO goal: `--with_goal` (appends keypoints_rel_goal → 121) exists but **collapses success
  to ~0%** (goal makes a waypoint-tracker, not a nail-driver). Scripts: collect `collect_bc_data.py --simtoolreal`
  (records `obs/keypoints_rel_palm (T,8,3)` + `obs/proprio (T,109)` [+ `obs/keypoints_rel_goal (T,12)`]),
  `train_simtoolreal_specialist.py`, `eval_simtoolreal_{server,client}.py` (port **5602**). **Eval MUST pass
  `--table_dist 0.15` (matches the data) and `--max_ep_steps 1200`** (default 800 truncates slow successes:
  best model 55%@800 → 61%@1200).

  **CURRENT BEST RECIPE (hammer ≈ 61%)** — all-perturbation data + drop-joint_vel + 100k steps:
  ```bash
  # 1) collect (isaaclab.sh): ALL perturbations + table_dist 0.15, state-only. ~10% yield (perturbations are hard)
  .../scripts/collect_bc_data.py --headless --task hammer --no_image --table_dist 0.15 \
      --goal_diversify --random_action --random_action_prob 0.007 --random_action_steps_std 27 \
      --joint_displacement --force_perturbation --action_noise 0.1 \
      --num_envs 100 --num_demos 1000 --max_ep_steps 1500 --step_cap 50000 \
      --out datasets/hammer_allpert.hdf5      # tool_displacement is default-ON for hammer; --simtoolreal default-ON
  # 2) train (GR00T venv): drop joint_vel; all-pert data is diverse -> 100k steps (50k UNDERtrains: 48% vs 61%)
  python ~/simtoolreal_isaaclab/scripts/train_simtoolreal_specialist.py --hdf5 datasets/hammer_allpert.hdf5 \
      --name hammer_allpert_nojv --no_joint_vel --steps 100000
  # 3) eval (server GR00T venv:5602, then client isaaclab.sh). --no_joint_vel + --table_dist 0.15 + 1200 REQUIRED
  python ~/simtoolreal_isaaclab/scripts/eval_simtoolreal_server.py \
      --checkpoint logs/gr00t_specialist/hammer_allpert_nojv/hammer_allpert_nojv.pt --port 5602 &
  .../scripts/eval_simtoolreal_client.py --headless --num_envs 25 --episodes 400 --replan 1 \
      --no_joint_vel --table_dist 0.15 --max_ep_steps 1200 --port 5602
  ```

  **Per-recipe success** (hammer, no-goal, **no-jv**, @ table_dist 0.15, replan 1, 400 ep, 1200-step budget;
  SAPG expert ≈ 86%):
  | data recipe | success |
  |---|---|
  | no perturbation | ~2% |
  | only tool-teleport | ~14% |
  | diverse_goal only | 19% |
  | random_action only | 39% |
  | diverse_goal + random_action | 40% |
  | + tool-teleport | 48% |
  | **all perturbations (100k steps)** | **61%** |

  Findings: (1) **random_action is the dominant perturbation** (baseline→39%); diverse_goal alone is weak (19%)
  and adds ~nothing on top of random_action — the win is RECOVERY data, not goal diversity. (2) **Dropping
  joint_vel helps ~12–14 pts** (noisy sim-specific signal the BC policy over-fits — the "state-based" variant).
  (3) **Goal-conditioning collapses to ~0%.** (4) **DAgger** (`dagger_{server,client}.py`, SAPG expert teacher) is
  **break-even at best (~60%)**, never improves the strong specialist (the earlier 0% was an LR bug; use chunk
  relabel + `--train_lr 1e-5` + slow/linear β `--beta_step 0.1` + best-on-eval; `--relabel_horizon 1` single-action
  is worse — chunking helps). The remaining gap to ~86% is the fundamental single-step-BC-of-a-recurrent-RL-expert
  gap (compounding error, no recovery, no LSTM memory). *(Superseded: the old `hammer_str` = 13.9% used action_noise-
  only data, WITH joint_vel, table_dist 0, 800-step eval.)*

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

### Hammer task — default config (`HammerEnvCfg.__post_init__`)
The **default** `HammerEnvCfg` *is* the pretrained-deploy / BC-collection / eval config (so
`collect_bc_data.py` and `eval_specialist_client.py` get it for free). Baked-in defaults:
- **mode**: `pretrained_compat=True`, `domain_randomization=False`, `use_tolerance_curriculum=False`
  — zero-shot deploy of the original SAPG policy, clean eval. *For training-from-scratch on hammer,
  flip `pretrained_compat`/`domain_randomization` back.*
- **success / goals**: `success_tolerance=0.01`, `success_steps=1`, `max_consecutive_successes=0`,
  `terminate_on_nail_driven=-0.006` (episode ends when the nail is seated), `screw_contact_clearance=
  -0.04` (strike overshoots 4 cm), `use_tighten_goals=True`, `use_fixed_goal_trajectory=False`,
  `randomize_layout=True`, `physical_screw=True`, `episode_length_s=800/60`.
- **genuine-hammer-strike guards** (recorded successes must be real hammer blows; `HammerEnv._get_dones`):
  success = nail seated **AND** the hammer striking face ≤ `nail_strike_contact_dist=0.030` (3 cm) of the
  head **AND** ≥ `tool_displace_success_block_steps=60` (1 s) since any teleport **AND** the nail never
  moved > `nail_move_eps=0.001` (1 mm)/step while the hammer was far. If the nail seats OR moves while the
  hammer is far, the episode TERMINATES as a failure + resets. (`nail_hand_reject_dist=None` — a "hand must
  be far from the nail" check; off because it was too strict, superseded by the 3 cm head check.)
- **BC-collection defaults vs eval/deploy (CRITICAL)**: `HammerEnvCfg` keeps perturbations OFF
  (`tool_displacement=False`) so the 8 eval/deploy/render scripts that build `HammerEnvCfg()` stay clean.
  But **`collect_bc_data.py --task hammer` defaults to the teleport-recovery dataset config**:
  `tool_displacement` + `tool_displace_pregrasp` (random tool teleports `U[2,10]cm` before & after grasp →
  the expert recovers; success-filtering keeps recoveries) + `--simtoolreal` recording + `max_ep_steps`
  default **1500** (recovery budget). Disable with `--no_tool_displacement` / `--no_simtoolreal`. goal-noise
  / force-perturbation / action-noise stay OFF unless their flags are passed (`--goal_noise`/`--goal_noise_scale`,
  `--force_perturbation`/`--force_scale`, `--action_noise`).
- **reset** (deterministic): `startArmHigher` (`iiwa14_joint_2=1.571−10°`, `iiwa14_joint_4=1.376+10°`),
  all reset noise 0.
- **camera / visuals**: `per_env_camera=True` @ 640×480, eye `(0,−0.65,0.85)` → lookat `(0,0.30,0.55)`,
  `cam_z_far=2.5`, `scene.env_spacing=4.0` (isolates each sub-env in the clipped view),
  `ground_color=(0.12,0.12,0.12)`, `screw_color=(0.55,0.72,0.82)`, `gpu_collision_stack_size=2³⁰`.
  **Run with cameras enabled** (`--enable_cameras`; the deploy/collect/eval scripts set it).
- The optional palm-facing **wrist camera** (`wrist_camera`, a TiledCamera on `iiwa14_link_7`) is **off
  by default** — opt in via `collect_bc_data.py --wrist` (a 2nd view for the multi-cam specialist). It is
  **640×480 (same as the table cam)**, a SIDE view ~8 cm off the wrist base aimed at the grasp
  (`wrist_cam_eye=(0.08,−0.02,0.08)`, `lookat=(−0.02,−0.015,0.18)`, `up=Y`, focal 14).

### Screwdriver task — default config (`ScrewdriverEnvCfg`)
The **044 flat screwdriver tightening a slot screw**. The default cfg *is* the working BC-collection
config (reproduces `videos/best10_screwdriver_6_14`); collect with
`collect_bc_data.py --task screwdriver`. Baked-in defaults:
- **collider (critical)**: `SCREWDRIVER_USD = 044_screwdriver_sdf` — the **SDF** collider whose thin
  blade physically ENTERS the slot. The convex-decomp `044_screwdriver.usd` has a blunt blade that
  can't, so it just shoves the head (the original "tip never in the slot" bug). `__post_init__` bumps
  `gpu_collision_stack_size=2²⁸` (SDF makes many contacts).
- **screw**: `physical_screw=True` (revolute `screw_spin` assembly = thread_test fixed base + spinning
  screw, driven by blade contact). Original 6_14 size (NOT scaled).
- **goal trajectory**: `tighten_traj` (T=74: lift→reorient→over→lower→**TURN 180°** over 24 goals),
  `tighten_turn_degrees=180`. The env targets the screw head via `screw_head_offset_nominal`.
- **success / terminal** (`terminate_on_screw_rotated=radians(150)`, default **150°**, via
  `--success_deg`): the screw must rotate **CLOCKWISE-from-top ≥ threshold** (`screw_tighten_sign=-1`;
  CCW/loosening and back-and-forth do NOT count) **AND** the tip must be in the slot at that step.
  Tip-in-slot = `screw_engage_radius=0.008` (tip ≤ 8 mm from head — successes reach 2–3 mm) **and** `screw_engage_tipdown=0.8`
  (tool within ~37° of vertical) — tightened from the loose 0.03/0.6. The physical screw slips, so
  yield is ~19% at 150° (quality over quantity; collect-until-N just runs more episodes).
- **visuals**: same as hammer — `ground_color=(0.12,0.12,0.12)`, `screw_color=(0.55,0.72,0.82)`,
  640×480 per-env camera. Long episodes (≤1500 steps) → heavy RAM; collect in a memory-capped cgroup.

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
