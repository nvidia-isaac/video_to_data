## Project Overview

Robotic Grounding is an NVIDIA Isaac Lab-based project for robotic manipulation. It combines human motion retargeting (MANO hands or whole-body humans → Sharpa hand, Dex3 hands, or G1+Dex3 whole body via IK / learned planner) with reinforcement learning (PPO via RSL-RL) to train dexterous manipulation and loco-manipulation policies.

## Where to look

Most details live in per-package docs — this file only covers project-wide orientation, the dev loop, and gotchas. **Read the relevant doc before editing that package**:

| Topic | Doc |
|---|---|
| Host setup (Docker, direnv, CSS keys), retargeting & RL CLIs, visualizer | `README.md` |
| OSMO job submission + prerequisites | `workflow/README.md` |
| Raw → retargeted → trained data flow (stages 1–7) | `workflow/data_pipeline.md` |
| OSMO/CSS storage layout + download | `workflow/data_storage.md` |
| `motion_v1` schema (unified whole-body parquet) | `source/robotic_grounding/robotic_grounding/motion_schema/README.md` |
| Retargeters, dataset registry, IK kinematics, schema conventions | `source/robotic_grounding/robotic_grounding/retarget/CLAUDE.md` |
| Per-robot JSON config schema (`frame_alignment`/`retargeter`) | `source/robotic_grounding/robotic_grounding/retarget/configs/README.md` |
| Whole-body G1 planner | `source/robotic_grounding/robotic_grounding/planner/README.md` |
| Dual-hand V2P envs (Sharpa) — actions / obs / rewards / curriculum / PPO | `source/robotic_grounding/robotic_grounding/tasks/v2p/CLAUDE.md` |
| Whole-body SONIC G1 envs | `source/robotic_grounding/robotic_grounding/tasks/v2p_whole_body/README.md` |

## Development Environment

NVIDIA GPU + Docker required. **Code runs inside the container; Git stays on the host.** Inside the container, always invoke Python through the Isaac Lab wrapper.

```bash
./workflow/run.sh build [version]
./workflow/run.sh start <version> <gpu_id>
./workflow/run.sh shell <version> <gpu_id>
./workflow/run.sh exec  <version> <gpu_id> -- <cmd>

# Inside the container, never run bare `python`:
${ISAACLAB_PATH}/isaaclab.sh -p <script.py>
```

CSS credentials are loaded via `direnv` from a gitignored `.envrc.local` in the repo root, or by sourcing `scripts/setup_css_env.sh`. See `README.md` § *Environment & Credentials*.

## Common Commands

```bash
# --- Inside the container ---
${ISAACLAB_PATH}/isaaclab.sh -p scripts/retarget/run_loader.py   --dataset <name> --save   # dual-hand stage 1
${ISAACLAB_PATH}/isaaclab.sh -p scripts/retarget/run_retarget.py --dataset <name> --save   # dual-hand stage 2
${ISAACLAB_PATH}/isaaclab.sh -p scripts/retarget/nvhuman_to_g1.py   <data_folder> --save   # whole-body (motion_v1)
${ISAACLAB_PATH}/isaaclab.sh -p scripts/replay_motion.py             --motion_file <path>  # kinematic replay
${ISAACLAB_PATH}/isaaclab.sh -p scripts/reconstruct_support_surfaces.py --input_dir <d> --sequence_id <s>
${ISAACLAB_PATH}/isaaclab.sh -p scripts/rsl_rl/{dummy_agent,train,play,eval}.py --task <task_id> --motion_file <path>

# --- Outside the container (host) ---
python scripts/run_osmo.py --pool <pool> --workflow-yaml workflow/{train,retarget,dev_env}.yaml \
    --image <image>:<tag> --experiment-name <name>
```

## Registered Gym Task IDs

Auto-imported via the `robotic_grounding.tasks` entry point. Details (cfgs, rewards, PPO) live in the per-task docs.

| Family | IDs | See |
|---|---|---|
| Dual-hand Sharpa (manipulation, tracking, direct, auto-curriculum) | `Sharpa-V2P-v0(-Play)`, `Sharpa-V2P-AutoCurr-v0`, `Sharpa-V2P-Direct-v0(-Play)`, `Sharpa-V2P-Tracking-v0(-Play)` | `tasks/v2p/CLAUDE.md` |
| Debug (interactive GUI — do **not** pass `--headless`) | `Sharpa-V2P-Debug-v0`, `Vega-Sharpa-Debug-v0` | `tasks/debug/` |
| Whole-body G1 SONIC | `SonicG1-v0`, `SonicG1-ReconBody-v0`, `SonicG1-ReconHand-v0` | `tasks/v2p_whole_body/README.md` |

## Package Structure (`source/robotic_grounding/robotic_grounding/`)

- **`motion_schema/`** — Unified `motion_v1` parquet contract for whole-body producers/consumers. wxyz quats, `[x,y,z,qw,qx,qy,qz]` poses, `(T, ...)` leading time axis, world frame.
- **`retarget/`** — Motion retargeting (MANO/NVHuman/SOMA → Sharpa/Dex3/G1). Owns dataset registry, IK solvers, source-format readers, ground-alignment post-process. See `retarget/CLAUDE.md`.
- **`planner/`** — G1 whole-body planner (V2P EE targets → full-body qpos). Emits `motion_v1`. See `planner/README.md`.
- **`tasks/`** — Isaac Lab RL envs (entry point `robotic_grounding.tasks`).
  - `scene_utils/` — `SceneConfig.from_motion_file()` auto-discovers objects, support surfaces, and episode length from a parquet, then `apply_scene_config` patches them into the env cfg.
  - `v2p/` — Dual-hand manipulation. See `tasks/v2p/CLAUDE.md`.
  - `v2p_whole_body/` — SONIC G1 loco-manipulation. See `tasks/v2p_whole_body/README.md`.
  - `debug/` — DearPyGui interactive envs.
- **`assets/`** — Robot configs, URDFs/MJCFs, meshes, motion data, policies.

## Data Flow (two parquet flavors)

1. **Dual-hand** (`ManoSharpaData` / `ManoDex3Data`): `{dataset}_loader.py` → `{dataset}_to_{robot}.py` (IK) → `{dataset}_processed/sequence_id=<s>/robot_name=<r>/`. Consumed by `tasks/v2p`.
2. **Whole-body** (`motion_v1`): NVHuman/SOMA → `nvhuman_to_{g1,dex3}.py` / `soma_to_g1.py` (IK), or `g1_planner` (learned). Consumed by `tasks/v2p_whole_body` and `replay_motion.py`.

Both feed `SceneConfig.from_motion_file()` → Isaac Lab env → PPO.

## Gotchas

- **Never run bare `python`** inside the container — use `${ISAACLAB_PATH}/isaaclab.sh -p`.
- **Quaternions are wxyz on disk**; xyzw only inside Pinocchio `qpos`. Convert at the boundary.
- **`motion_v1` is the only supported whole-body schema.** Files predating `motion_kind` fail to load with `MissingRequiredField` — no migrator; regenerate.
- **Git on host, code in container.** Root container can scramble permissions; `sudo chown -R $(whoami) .` on the host clears them.
- **Debug envs need a display** — do not pass `--headless`.
- **Adding a new dataset:** use the `add-dataset` skill (registers in `dataset_registry.py`, scaffolds loader + retargeter + URDF wiring).
- **Slash commands** in `.claude/commands/`: `/eval-record`, `/getneval`, `/grid-video`, `/launch-two-stage`.

## Tests

Tests in `tests/` are runnable directly (no pytest-collected entry point). Run with the Isaac Lab wrapper from inside the container:

```bash
${ISAACLAB_PATH}/isaaclab.sh -p tests/test_motion_schema.py
${ISAACLAB_PATH}/isaaclab.sh -p tests/test_retarget_pipeline_e2e.py
${ISAACLAB_PATH}/isaaclab.sh -p tests/test_train_e2e.py
```

## Code Style

`pyproject.toml` configures isort (black profile, 120-char), ruff (py311, line 120), pyright (basic, Linux). Pre-commit hooks run on commit — install once with `bash workflow/setup_deps.sh`.
