---
name: robotic_grounding_run
description: Generates the exact command to run the robotic_grounding floating-hand (Sharpa Wave / Dex3) pipeline — retargeting, dummy-agent smoke tests, RL training, and evaluation. Use this skill whenever the user wants to RUN something in robotic_grounding and needs the right command: "how do I run a smoke test", "run the dummy agent on <sequence>", "train a policy on <dataset>", "evaluate my checkpoint", "retarget <dataset>", "run the pipeline on arctic/taco/hot3d/grab/h2o/dexycb/oakink2", "what's the command for ...", or when they describe an intent (dataset + stage + GUI/headless) and want the precise invocation. For the G1 whole-body pipeline use robotic_grounding_whole_body instead; for first-time setup use robotic_grounding_onboard; for failures use robotic_grounding_doctor.
---

# robotic_grounding — Floating-Hand Command Generator

This skill turns a user's intent into the **exact command** to run, for the floating-hand
(Sharpa Wave / Dex3) pipeline. It does **not** run long jobs blindly — it asks the few questions
needed to disambiguate, prints the command with a one-line explanation, and tells the user where it
must run (host vs container).

For the whole-body G1 / SONIC pipeline, use the **`robotic_grounding_whole_body`** skill instead.

## Host vs container — state this every time

`robotic_grounding` splits execution:

- **Host** (outside any container): the pipeline orchestrator and the example wrappers —
  `scripts/run_pipeline_docker.py`, `./run_example_sequences.sh`. They spin up the Docker images
  themselves.
- **Container** (inside `robotic-grounding`): the RL scripts and replay/view tools —
  `scripts/rsl_rl/*.py`, `scripts/replay_motion.py`, `scripts/view_scene.py`, `scripts/retarget/*.py`.

When you emit a container command, **make the wrapping explicit** so the user knows what to do. Emit
it in this shape (pick the form that matches where they are):

```bash
# If you are already INSIDE the container (./workflow/run.sh start latest 0):
python scripts/rsl_rl/dummy_agent.py --task ...

# If you are on the HOST, wrap it (runs in the running container named robotic-grounding-latest-gpu0):
./workflow/run.sh exec latest 0 -- python scripts/rsl_rl/dummy_agent.py --task ...
```

Always ask (or infer from the conversation) **"are you already inside the container, or on the host?"**
before finalizing a container command. If unknown, show both forms with the comment lines above.

## Triage questions

Ask only what you can't already infer. Usually 2–4 short questions:

1. **What do you want to do?** — one of:
   - `pipeline` — turn a raw dataset into RL-ready motion (retarget)
   - `dummy` — dummy-agent smoke test (zero actions; verify a motion file loads)
   - `train` — RL training
   - `eval` — evaluate a trained checkpoint
   - `replay` / `view` — replay motion or inspect a scene
2. **Which dataset + sequence?** — dataset is one of `arctic, taco, hot3d, grab, h2o, dexycb, oakink2`;
   the sequence id or a `--sequence-pattern`. (For RL, this becomes the `--motion_file` shorthand.)
3. **Which robot?** — `sharpa_wave` (default) or `dex3`. Only arctic and taco have a `dex3` retarget script.
4. **Assets ready, or asset-free?** — if object URDFs/meshes aren't generated yet, use
   `--use_primitive_urdfs` (works for dummy/train/eval). If they've run the `urdf` stage, drop it.
5. **GUI or headless?** — headless for CI/servers (`--headless`, optionally `--record_video`); GUI for
   an interactive Isaac window on a machine with a display.

If the user names a sequence but you're unsure it exists locally, suggest a `dummy` run first (it's the
cheapest way to confirm a motion file loads).

## Motion-file shorthand

RL scripts take `--motion_file <dataset>/<dataset>_processed/<sequence_id>/<robot_name>`, resolved under
`source/robotic_grounding/robotic_grounding/assets/human_motion_data/` in the container. Example:
`arctic/arctic_processed/arctic_s01_box_grab_01/sharpa_wave`. An absolute path to a Parquet partition
also works.

Known-good example sequences (from `docs/EXAMPLE_SEQUENCES.md`): `arctic_s01_box_grab_01`,
`arctic_s07_box_grab_01`, `arctic_s01_mixer_use_01` (articulated), `arctic_s01_espressomachine_use_01`.

## Command templates

### pipeline — retarget a dataset (HOST)
```bash
# Build both images once (loader + robotic-grounding):
python scripts/run_pipeline_docker.py --build-only

# Run load → [segment] → [urdf] → processed → support on a downloaded dataset:
python scripts/run_pipeline_docker.py <dataset> \
  --hmd <HMD> --mano-dir <HMD>/mano --max-sequences 2

# Add the viser/MP4 visualization stage:
python scripts/run_pipeline_docker.py <dataset> --hmd <HMD> --mano-dir <HMD>/mano --stages vis
```
Output lands at `<HMD>/<dataset>/<dataset>_processed/sequence_id=.../robot_name=<robot>/`.
For the fixed multi-dataset reproduction, prefer `HMD=<HMD> ./run_example_sequences.sh`
(knobs: `ONLY=<ds>`, `DRY_RUN=1`, `EXAMPLE_DIR=<path>`).

### dummy — asset/scene smoke test (CONTAINER)
```bash
# GUI:
python scripts/rsl_rl/dummy_agent.py \
  --task Sharpa-V2D-v0-Play \
  --motion_file <dataset>/<dataset>_processed/<sequence_id>/<robot> \
  --num_envs 1 --use_primitive_urdfs

# Headless with an MP4 (CI):
python scripts/rsl_rl/dummy_agent.py --headless \
  --task Sharpa-V2D-v0-Play \
  --motion_file <dataset>/<dataset>_processed/<sequence_id>/<robot> \
  --num_envs 1 --use_primitive_urdfs \
  --record_video --output_dir /tmp/rg_dummy_agent_video --video_length 300
```
Success = Isaac starts, the task registers, the motion Parquet loads, no missing-asset exception,
the sim advances.

### train — RL training (CONTAINER)
```bash
# One-iteration smoke train (use tensorboard if W&B isn't configured):
python scripts/rsl_rl/train.py --headless \
  --task Sharpa-V2D-v0 \
  --motion_file <dataset>/<dataset>_processed/<sequence_id>/<robot> \
  --num_envs 1 --max_iterations 1 \
  --logger tensorboard --run_name smoke_train --use_primitive_urdfs \
  agent.num_steps_per_env=8 agent.save_interval=1

# Real run: raise --num_envs and --max_iterations, drop the agent.* overrides,
# and drop --use_primitive_urdfs once real object assets exist.
```
Checkpoints write to `logs/rsl_rl/<experiment>/<run>/model_*.pt`.

### eval — evaluate a checkpoint (CONTAINER)
```bash
CHECKPOINT=$(find logs/rsl_rl -path '*<run_name>*/model_*.pt' | sort -V | tail -1)
python scripts/rsl_rl/eval.py --headless \
  --task Sharpa-V2D-v0 \
  --motion_file <dataset>/<dataset>_processed/<sequence_id>/<robot> \
  --num_envs 1 --checkpoint "$CHECKPOINT" --eval_episodes 1 --use_primitive_urdfs
```
`eval.py` also exports the policy (JIT/ONNX). `--use_pretrained_checkpoint` pulls a published
checkpoint instead of a local one.

### replay / view (CONTAINER)
```bash
# Kinematic replay (teleported, no physics forces):
python scripts/replay_motion.py --motion_file <abs_or_shorthand_path>   # --speed 0.5 / --no-loop / --headless

# Static spawn verification of a reconstructed scene:
python scripts/view_scene.py --motion_file <parquet_partition_path>
```

## dummy vs train vs eval — pick the right one

If the user is unsure which entry point they want:

| They want to… | Use | Needs a checkpoint? |
|---------------|-----|---------------------|
| Check a motion file / scene loads at all | `dummy` | no |
| Produce a policy | `train` | no (unless `--resume`) |
| Measure / export an existing policy | `eval` | **yes** |

## After emitting a command

- State **where** it runs (host vs container) and which flags matter (`--use_primitive_urdfs`,
  `--headless`).
- If it's a long job (full training, full-dataset retarget), say so and suggest a smoke variant first
  (`--max_iterations 1`, `--max-sequences 2`, or a `dummy` run).
- If the command fails, hand off to the **`robotic_grounding_doctor`** skill.
