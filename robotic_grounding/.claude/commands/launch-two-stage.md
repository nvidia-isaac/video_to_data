---
name: launch-two-stage
description: Launch and monitor two-stage training experiments on OSMO with automatic crash recovery
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
---

# /launch-two-stage — Launch and Monitor Two-Stage Experiments

Build the Docker image, submit stage1 workflows for each experiment, then set
up periodic monitoring that auto-relaunches crashed stage1 tasks and triggers
stage2 once all stage1 runs complete.

Arguments: `$ARGUMENTS`
Expected format: `<exp_id1> [exp_id2 ...] [--image TAG] [--pool POOL] [--no-build] [--dry-run] [--state-file PATH]`

Examples:
- `/launch-two-stage exp52 exp53 exp54 exp55 exp56`
- `/launch-two-stage exp57 exp58 --no-build`
- `/launch-two-stage exp52 --dry-run`

---

## Step 1 — Parse arguments

Parse from: `$ARGUMENTS`

- `exp_ids`: all positional arguments (e.g. `exp52 exp53 exp54 exp55 exp56`)
- `image_tag`: from `--image <TAG>`, default `v2d`  → full image: `nvcr.io/nvstaging/isaac-amr/robotic-grounding:<TAG>`
- `pool`: from `--pool <POOL>`, default `isaac-dev-l40s-04`
- `no_build`: flag `--no-build`, default false
- `dry_run`: flag `--dry-run`, default false
- `state_file`: from `--state-file <PATH>`, default `scripts/monitor_state.json`

Validate: each exp_id must exist in `experiments/registry.local.yaml` or `experiments/registry.yaml`.
If an id is missing, report an error and stop.

---

## Step 2 — Confirm experiments exist

For each exp_id in exp_ids, verify its pipeline config has `pipeline: true`
and the expected `stage1_exp_id` / `stage2_config_id` fields.

Report a brief summary: experiment IDs and their stage1/stage2 config names.

---

## Step 3 — Build and push Docker image (unless --no-build)

The image must be rebuilt to pick up any local code changes before submitting.

Run from the `robotic_grounding/` directory:

```bash
./workflow/run.sh build <image_tag>
./workflow/run.sh push <image_tag>
```

Report progress. If build fails, stop and report the error.

If `--no-build` was passed, skip this step and confirm the existing image will
be used.

---

## Step 4 — Initialize state file and submit stage1 workflows

Run the monitoring script in init mode. This creates the state file and
submits stage1 OSMO workflows for all experiments in one shot:

```bash
python scripts/monitor_two_stage.py --init \
  --exp-ids <exp_ids...> \
  --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<image_tag> \
  --pool <pool> \
  --state-file <state_file> \
  [--dry-run]
```

This submits a multi-task OSMO workflow per experiment (6 parallel tasks each),
with each task named `{exp_id}_stage1_{seq_key}`.

The state file tracks:
- Which sequences are running/complete
- Which crashes have been relaunched (and how many times)
- Whether stage2 has been launched

---

## Step 5 — Set up monitoring cron

Use CronCreate to run the monitoring script every 10 minutes:

```
*/10 * * * * cd /home/zeol/Documents/gitlab/video_to_data/robotic_grounding && python scripts/monitor_two_stage.py --state-file <state_file> >> logs/monitor_two_stage.log 2>&1
```

The monitoring script on each run:
1. Queries W&B for stage1 runs matching `{exp_id}_stage1_{seq_key}`
2. For any run that crashed with 50 ≤ steps < 1000: downloads last checkpoint and submits a single-task resume OSMO workflow
3. When all 6 stage1 sequences for an experiment are finished: calls `launch_stage2.py`
4. After stage2 is launched: monitors stage2 runs; if any crash with 50 ≤ steps < 10000: calls `launch_stage2.py --resume-crashed`

---

## Step 6 — Wait for stage1 runs in the background

Launch a background Agent (run_in_background=True) to watch for stage1 runs
appearing on W&B. Do NOT poll inline — this must be entirely silent until
stage1 runs are confirmed running.

The background agent should:
1. Poll W&B every 2 minutes (up to 30 minutes total) using:
   ```python
   import wandb, time
   api = wandb.Api()
   for exp_id in ["exp52", ...]:
       runs = api.runs("nvidia-isaac/v2p_hands",
                       filters={"display_name": {"$regex": f"{exp_id}_stage1_"}})
       running = [r for r in runs if r.state == "running"]
   ```
2. When all experiments have at least one stage1 run showing "running", call
   PushNotification with a message like:
   "stage1 runs are live on W&B: exp52 (3/6), exp53 (6/6), ..."
3. If no runs appear after 30 minutes, send a PushNotification warning.

After launching the background agent, return control to the user immediately
with a brief summary: which experiments were submitted, cron job ID, state
file path. Nothing else.

---

## Monitoring script reference

**State file location**: `scripts/monitor_state.json` (default)

**Manual monitor run**: `python scripts/monitor_two_stage.py`

**Dry run**: `python scripts/monitor_two_stage.py --dry-run`

**View logs**: `tail -f logs/monitor_two_stage.log`

**Key thresholds** (can be overridden at --init time):
- `--stage1-threshold 1000`: relaunch stage1 if crashed before 1000 steps
- `--stage2-threshold 10000`: relaunch stage2 if crashed before 10000 steps  
- `--min-steps 50`: treat crashes before 50 steps as system errors (skip)
- Max 3 reruns per stage1 seq_key, 3 reruns per stage2 experiment

**To stop monitoring**: delete the cron job and optionally the state file.
