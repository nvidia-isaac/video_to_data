# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`robotic_grounding` turns mocap/video **hand-object and whole-body human motion** into
**retargeted robot motion** and uses it to train **RL policies in Isaac Lab**. Supported
robots: **Sharpa Wave** and **Dex3** floating hands, and the **G1** whole-body humanoid.

The installable package lives at `source/robotic_grounding/robotic_grounding/` and is registered
as an Isaac Lab extension (`source/robotic_grounding/config/extension.toml`). Runnable entry points
live in `scripts/`.

**Deeper references (don't duplicate them — link to them):**
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — contents map, data-flow diagram, "where do I go for X" table.
- [`README.md`](README.md) — the canonical source of run commands.
- [`docs/SETUP.md`](docs/SETUP.md) + per-dataset `docs/*_SETUP.md` — dataset download & layout.
- [`workflow/data_pipeline.md`](workflow/data_pipeline.md) — end-to-end stage walk-through.
- [`docs/EXAMPLE_SEQUENCES.md`](docs/EXAMPLE_SEQUENCES.md) — the one-command example reproduction.

There are three companion skills (in the monorepo's `.claude/skills/`): **`robotic_grounding_onboard`**
(interactive first-run), **`robotic_grounding_run`** (generates the exact floating-hand command from
your intent), and **`robotic_grounding_doctor`** (troubleshooting). Whole-body (G1) work has its own
**`robotic_grounding_whole_body`** skill.

## Container / host split (read this first)

**Development and pipeline execution happen INSIDE the Docker container; git operations
(commit, push, branch) happen on the HOST.** This is the #1 thing to get right.

```bash
./workflow/run.sh build [version]        # build the image (default version: latest)
./workflow/run.sh start [version] [gpu]  # start + enter container (default: latest, gpu 0)
./workflow/run.sh shell [version] [gpu]  # enter a second shell in a running container
./workflow/run.sh exec  [version] [gpu] -- <cmd>   # run one command in the container
./workflow/run.sh stop  [version] [gpu]
```

The running container is named `robotic-grounding-<version>-gpu<gpu>` (default
`robotic-grounding-latest-gpu0`). To run a single command from the host, prefer the sanctioned
wrapper `./workflow/run.sh exec latest 0 -- <cmd>`; `docker exec robotic-grounding-latest-gpu0 <cmd>`
is equivalent. The working directory inside the container is
`/workspace/video_to_data/robotic_grounding`, and the repo is volume-mounted, so Python edits take
effect on the next invocation — **no rebuild needed** to iterate.

Permission hiccups (Isaac Lab's image runs as root) are fixed on the host with
`sudo chown -R $(whoami) .`.

## The pipeline: two Docker images

```
raw dataset + MANO ──► [IMAGE 1: v2d_task_library_loader] ──► <ds>_loaded (Parquet)
                          stage: load (MANO forward-kinematics)
                                                                      │
<ds>_loaded ──► [IMAGE 2: robotic-grounding] ──► <ds>_processed  ◄────┘
                  segment → urdf → processed → support → vis        (RL-ready motion)
```

- **IMAGE 1** (`v2d_task_library_loader`, lives in the `reconstruction` repo — the MANO/GPL stage)
  runs `load`: MANO forward-kinematics → `<ds>_loaded` Parquet of per-frame joints + object poses.
- **IMAGE 2** (`robotic-grounding`, this repo) runs `segment` (hot3d only) → `urdf` (object URDFs)
  → `processed` (IK retargeting) → `support` (support surfaces) → `vis` (viser/MP4).
- Both are driven from the **host** by one orchestrator: `scripts/run_pipeline_docker.py`
  (no manual container entry needed for pipeline runs).

**Don't confuse the two images.** `load` needs the loader image; everything else needs `robotic-grounding`.

## Common commands

All RL/replay commands assume you are **inside the container** at the package root. When issuing
from the host, wrap them: `./workflow/run.sh exec latest 0 -- <cmd>`.

**Run the full pipeline on a dataset (from the host):**
```bash
python scripts/run_pipeline_docker.py arctic --hmd <HMD> --mano-dir <HMD>/mano --max-sequences 2
python scripts/run_pipeline_docker.py --build-only        # build both images once
```

**Reproduce the fixed example sequences (from the host):**
```bash
HMD=~/datasets/human_motion_data ./run_example_sequences.sh          # arctic + hot3d + taco
ONLY=taco  HMD=~/datasets/human_motion_data ./run_example_sequences.sh
DRY_RUN=1  HMD=~/datasets/human_motion_data ./run_example_sequences.sh   # print, don't run
```

**Asset/scene smoke test — dummy agent (zero actions):**
```bash
python scripts/rsl_rl/dummy_agent.py --task Sharpa-V2D-v0-Play \
  --motion_file arctic/arctic_processed/dataset_s07_box_grab_01/sharpa_wave \
  --num_envs 1 --use_primitive_urdfs                      # add --headless --record_video for CI
```

**Train / evaluate (Sharpa floating-hand):**
```bash
python scripts/rsl_rl/train.py --headless --task Sharpa-V2D-v0 \
  --motion_file arctic/arctic_processed/dataset_s07_box_grab_01/sharpa_wave \
  --num_envs 1 --max_iterations 1 --logger tensorboard --run_name smoke --use_primitive_urdfs

CHECKPOINT=$(find logs/rsl_rl -path '*smoke*/model_*.pt' | sort -V | tail -1)
python scripts/rsl_rl/eval.py --headless --task Sharpa-V2D-v0 \
  --motion_file arctic/arctic_processed/dataset_s07_box_grab_01/sharpa_wave \
  --num_envs 1 --checkpoint "$CHECKPOINT" --eval_episodes 1 --use_primitive_urdfs
```

**Tests & lint (inside container):**
```bash
pytest tests/                                # schema, retarget e2e, replay, train e2e
pytest tests/test_motion_schema.py -v        # one file
pre-commit run --all-files                   # license headers + formatting
```

## dummy_agent vs train vs eval — the three RL entry points

`scripts/rsl_rl/` has three entry points that are easy to conflate. They differ in whether a
**policy/checkpoint** is involved:

| Script | What it does | Checkpoint? | Learns? | Typical task |
|--------|--------------|-------------|---------|--------------|
| `dummy_agent.py` | Runs the env with **zero actions** — verifies assets load, scene builds, sim advances. No policy at all. | No | No | `Sharpa-V2D-v0-Play` |
| `train.py` | RSL-RL training loop; writes checkpoints to `logs/rsl_rl/<run>/model_*.pt`. | No (unless `--resume`) | **Yes** | `Sharpa-V2D-v0` |
| `eval.py` | Loads a **trained checkpoint**, runs `--eval_episodes`, and exports the policy (JIT/ONNX). | **Yes** (`--checkpoint` or `--use_pretrained_checkpoint`) | No | `Sharpa-V2D-v0` |

Use `dummy_agent.py` first to confirm a motion file loads before spending time training. Task IDs
ending in `-Play` are the evaluation/playback variants.

## Motion-file shorthand & the asset-free flag

- **Shorthand:** RL scripts accept `--motion_file <dataset>/<dataset>_processed/<sequence_id>/<robot_name>`
  (e.g. `arctic/arctic_processed/dataset_s07_box_grab_01/sharpa_wave`). It resolves under
  `source/robotic_grounding/robotic_grounding/assets/human_motion_data/` inside the container. You can
  also pass an absolute path to a Parquet partition.
- **`--use_primitive_urdfs`:** replaces per-object URDFs with primitive collision shapes so
  `dummy_agent`/`train`/`eval` run **without** first completing the `urdf` pipeline stage. Always use
  it for asset-free smoke tests; drop it once real object assets are present.

## Conventions & gotchas

- **MANO is never committed.** MANO weights are license-gated, read only at the `load` stage. Pass
  them via `--mano-dir <HMD>/mano`; never add them to the repo. See [`docs/SETUP.md`](docs/SETUP.md).
- **`<HMD>` root.** All datasets live under one human-motion-data root (`<HMD>`) with `mano/` and one
  subdir per dataset. Retargeted output lands at `<HMD>/<dataset>/<dataset>_processed/...`.
- **Registry is the single source of truth.** `source/robotic_grounding/robotic_grounding/retarget/dataset_registry.py`
  drives dataset choices across every script (pipeline, CSS sync, URDF gen, training validation). Add
  a dataset there first — see the `add-dataset` skill.
- **Motion interchange format.** Everything reads/writes the versioned `motion_v1` Parquet schema via
  `motion_schema/{schema,reader,writer}.py`. Producers (retarget, planner) write it; consumers
  (training, replay, support recon, visualizer) read it through that one API.
- **Local iteration without OSMO.** `run_load_local.sh`, `run_retarget_local.sh`, and
  `run_pipeline_local.sh` mirror the container stages on one machine; OSMO (`scripts/run_osmo.py`,
  `workflow/*.yaml`) is only for cloud/multi-GPU runs.
- **Supported datasets:** taco, hot3d, arctic, grab, h2o, dexycb, oakink2. Adding a new one is a
  separate flow — invoke the `add-dataset` skill.

## Where run commands live

`README.md` and `docs/SETUP.md` are the single source of truth for exact commands. This file and
`docs/ARCHITECTURE.md` link to them rather than duplicating; when a command changes, update the
source docs, not the summaries.
