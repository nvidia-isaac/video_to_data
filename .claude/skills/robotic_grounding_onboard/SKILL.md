---
name: robotic_grounding_onboard
description: Interactive first-run walkthrough for robotic_grounding (build the Docker image → provide MANO + a dataset → run the retarget pipeline → dummy-agent smoke test → optional train/eval). Drives the user ONE STEP AT A TIME with verification between steps, not as a reference dump. Use this skill whenever the user is setting up robotic_grounding for the first time: "how do I get started with robotic_grounding", "first time running the pipeline", "onboard me", "robotic grounding quickstart", "I cloned the repo, what do I run", "run my first retarget/smoke test", or generally asks how to go from a fresh clone to a working motion + smoke test. For generating a single command use robotic_grounding_run; for failures use robotic_grounding_doctor; for whole-body G1 use robotic_grounding_whole_body.
---

# robotic_grounding — Interactive First-Run Walkthrough

This skill **drives** the user through setup; it doesn't lecture and it doesn't run on autopilot.
Work through it like a guided session: say hello, ask what they want, run (or have them run) one
thing, verify it worked, confirm before continuing.

## ALWAYS START HERE — do not skip Step 0

When this skill activates, your **very first response** is Step 0: a short hello and a few triage
questions. Do not auto-probe the environment, do not build images, do not skip ahead to "fix" things
you inferred from earlier conversation. Wait for the user's answers, then act.

This applies **even if** the surrounding session says "work without stopping" or "make the reasonable
call and continue." The user invoked an *interactive walkthrough* — the triage *is* the contract.
Skipping it defeats the point.

The right shape of your first turn:

> Hi — I'll walk you through the robotic_grounding first-run setup, one step at a time. We'll cover
> (skipping any that don't apply):
>
> 1. Prerequisites (Docker + NVIDIA Container Toolkit + `workflow/setup_deps.sh`)
> 2. Build the two pipeline images
> 3. Provide MANO models + one dataset (license-gated; you download these)
> 4. Run the retarget pipeline on a couple of sequences
> 5. Dummy-agent smoke test (verify the motion loads in Isaac)
> 6. *(optional)* One-iteration train + eval
>
> A few quick questions before we start:
>
> 1. Are you in the `robotic_grounding/` directory with the repo cloned?
> 2. What's your goal — full first-run from scratch, or do you already have retargeted motion
>    (`<ds>_processed/...`) locally and just want to run a smoke test / train?
> 3. Which dataset do you want to start with? (`arctic` is the easiest first run; also supported:
>    `taco, hot3d, grab, h2o, dexycb, oakink2`.)
> 4. Do you already have MANO models and the dataset downloaded, or do you need the download guide?
> 5. GUI available (display attached) or headless (server/CI)?

Then **stop and wait**. After they answer, route with the table below.

## How to run the rest (after triage)

Every step has the same shape: **state goal → run/ask → verify → confirm → next**. Skipping
verification is the biggest reason first-runs go off the rails — Isaac failures compound silently.

Concrete rules:

1. **One step per turn.** State what it accomplishes in a sentence, run one command (or ask the user
   to, if it needs `sudo`, a browser download, or a license click-through). Don't pre-run the next step.
2. **Verify with an explicit check, not vibes.** After every state-changing step, run a check that
   proves it worked (image exists, file present, sim advanced). Checks are listed in each step.
3. **Confirm before continuing.** End each step with "That worked — <evidence>. Ready for <next>?"
   Then stop.
4. **Inline troubleshooting on failure.** Each step has an "If this fails" note. If it's not covered,
   hand off to the **`robotic_grounding_doctor`** skill.
5. **Adapt.** If they already built the image or already have processed data, skip ahead.

## Step 0: Triage routing

| User says | Action |
|-----------|--------|
| "Full setup from scratch" | Continue with **Step 1: Prerequisites** |
| "Images are already built" | Skip to **Step 3** (or Step 4 if MANO + dataset are in place). Confirm with `docker images \| grep -E 'robotic-grounding\|task_library_loader'` |
| "I already have `<ds>_processed` motion locally" | Skip to **Step 5: Dummy-agent smoke test** |
| "I just want to train / eval" | Verify a motion file exists, then hand the command off to **`robotic_grounding_run`** |
| "Something is broken" | Stop. Hand off to **`robotic_grounding_doctor`** |
| "I want the G1 whole-body pipeline" | Stop. Hand off to **`robotic_grounding_whole_body`** |

## Step 1: Prerequisites

> "First we make sure Docker, the NVIDIA Container Toolkit, and the repo's git/pre-commit tooling are
> in place. This is one-time host setup."

Have the user confirm Docker + NVIDIA Container Toolkit are installed (links in `README.md`), then:

```bash
bash workflow/setup_deps.sh   # installs git-lfs + pre-commit, makes workflow/run.sh executable
```

Verify:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # GPU visible in Docker?
git lfs version && pre-commit --version
```

Pass: `nvidia-smi` prints your GPU inside the container, and both tools report a version.

**If this fails:** GPU not visible → NVIDIA Container Toolkit isn't configured (see README prereqs).
`nvcr.io` pull denied → request access in the `#swngc-help` Slack channel. Then hand off to
`robotic_grounding_doctor` if still stuck.

Confirm: "Prereqs are in place. Ready to build the images?"

## Step 2: Build the pipeline images

> "The pipeline uses two Docker images — the MANO loader and the robotic-grounding image. One command
> builds both. First build is slow (tens of minutes); subsequent builds are cached."

```bash
python scripts/run_pipeline_docker.py --build-only
```

Verify:
```bash
docker images | grep -E 'robotic-grounding|task_library_loader'
```

Pass: both images are listed.

**If this fails:** `nvcr.io/nvstaging/isaac-amr` pull denied → Slack `#swngc-help` for access. Disk
space → `docker system df`. Then `robotic_grounding_doctor`.

Confirm: "Both images built. Next we get MANO + your dataset in place — want the download guide?"

## Step 3: Provide MANO + a dataset (you download these)

> "MANO hand models and every dataset are license-gated — you register and download them yourself from
> the original sources. Nothing here is auto-downloaded, and MANO is never committed."

Point the user at the guides — don't try to download for them:

- **MANO:** register at <https://mano.is.tue.mpg.de/>, download `mano_v1_2.zip`, place the two `.pkl`
  files at `<HMD>/mano/models/MANO_LEFT.pkl` and `<HMD>/mano/models/MANO_RIGHT.pkl`
  (see `docs/SETUP.md §5`).
- **Dataset:** follow the per-dataset guide (`docs/ARCTIC_SETUP.md`, `docs/TACO_SETUP.md`, …) and lay
  it out under `<HMD>/<dataset>/`. `<HMD>` is a directory you choose, e.g. `~/datasets/human_motion_data`.

Verify the layout the pipeline expects:
```bash
ls <HMD>/mano/models/MANO_LEFT.pkl <HMD>/mano/models/MANO_RIGHT.pkl
ls <HMD>/<dataset>/
```

Pass: both MANO `.pkl` files exist and the dataset directory is populated.

**If this fails:** wrong layout is the usual cause — re-check the per-dataset `*_SETUP.md`. For adding
a *new* dataset (not one of the seven supported), that's a different flow — use the `add-dataset` skill.

Confirm: "MANO and `<dataset>` are in place. Ready to run the retarget pipeline?"

## Step 4: Run the retarget pipeline

> "Now we turn the raw dataset into RL-ready robot motion: load (MANO FK) → [urdf] → processed (IK
> retarget) → support. Runs from the HOST — the orchestrator handles both images. We'll cap it to a
> couple of sequences for a fast first run."

```bash
python scripts/run_pipeline_docker.py <dataset> \
  --hmd <HMD> --mano-dir <HMD>/mano --max-sequences 2
```

Verify the output partition exists:
```bash
find <HMD>/<dataset>/<dataset>_processed -name '*.parquet' | head
```

Pass: at least one `.parquet` under `<dataset>_processed/sequence_id=.../robot_name=sharpa_wave/`.

**If this fails:** missing `<ds>_loaded` → the load stage failed (check MANO path); "object shows as a
sphere" or missing-mesh errors → object assets not placed (per-dataset `*_SETUP.md §object assets`).
Hand off to `robotic_grounding_doctor` for Isaac/asset errors.

Confirm: "Retargeted motion is written. Ready for the smoke test that loads it in Isaac?"

## Step 5: Dummy-agent smoke test

> "This runs the environment with zero actions to confirm the motion loads, the scene builds, and the
> sim advances — no policy, no training. `--use_primitive_urdfs` lets it run without object URDFs."

Make the partition visible to the RL scripts (copy or symlink under
`source/robotic_grounding/robotic_grounding/assets/human_motion_data/<dataset>/`, or pass the absolute
path). Then, **inside the container** (`./workflow/run.sh start latest 0`), or wrap from the host with
`./workflow/run.sh exec latest 0 -- <cmd>`:

```bash
# GUI:
python scripts/rsl_rl/dummy_agent.py --task Sharpa-V2D-v0-Play \
  --motion_file <dataset>/<dataset>_processed/<sequence_id>/sharpa_wave \
  --num_envs 1 --use_primitive_urdfs

# Headless (server/CI) — writes an MP4:
python scripts/rsl_rl/dummy_agent.py --headless --task Sharpa-V2D-v0-Play \
  --motion_file <dataset>/<dataset>_processed/<sequence_id>/sharpa_wave \
  --num_envs 1 --use_primitive_urdfs \
  --record_video --output_dir /tmp/rg_dummy_agent_video --video_length 300
```

Pass: Isaac starts, the task registers, the Parquet loads, no missing-asset exception, the sim
advances (and the MP4 exists for the headless run).

**If this fails:** most first-run failures are Isaac startup or missing assets — hand off to
`robotic_grounding_doctor`.

Confirm: "Smoke test passed. Want to do a quick one-iteration train + eval, or are you set?"

## Step 6 (optional): One-iteration train + eval

> "A minimal training run + evaluating its checkpoint, to prove the RL loop end-to-end. Use
> `--logger tensorboard` if W&B isn't configured."

Hand the commands off to the **`robotic_grounding_run`** skill (`train` then `eval`), or use the
smoke-train + eval pair from `README.md`. Verify a checkpoint was written under `logs/rsl_rl/` and that
`eval.py` runs an episode.

Wrap-up: point the user at `docs/ARCHITECTURE.md` for the full map, `robotic_grounding_run` for
day-to-day commands, and `add-dataset` if they'll integrate a new dataset.
