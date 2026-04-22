---
name: eval-record
description: Download wandb run model(s) (if not already present) then run one full-trajectory recording eval in Docker for each, saving videos to experiments/eval_recordings/<run_name>/. Also supports --stats N mode for aggregate episode statistics.
user-invocable: true
allowed-tools:
  - Bash
---

# /eval-record — Get Run(s) + Record One Full Trajectory in Docker

Given one or more run names, for each: ensure the model checkpoint is downloaded, compute the
trajectory length from the motion file, then launch eval.py with `--video --video_length <N>
--num_envs 7` inside the Docker container. Videos auto-copy to `experiments/eval_recordings/<run_name>/`
named `<run_name>_model<N>.mp4`. When multiple runs are given, they execute **sequentially**.

**Stats mode** (`--stats N`): instead of recording video, runs N episodes across 32 envs (headless)
and prints aggregate stats (mean episode length, completion ratio, full completion rate). No video
is saved in this mode.

Arguments: `$ARGUMENTS`

**Single run:**
```
/eval-record <run_name> [--project <project>] [--model <model_name>] [voc=<value>] [debug_vis | no debug_vis] [--stats N]
```

**Multiple runs (shared options apply to all):**
```
/eval-record <run1> <run2> <run3> ... [shared options]
```

Examples:
- `/eval-record 2026-04-13_10-14-06_exp54_v3_stage2_20k_mixer_grab_01`
- `/eval-record 2026-04-13_10-14-06_exp54_v3_stage2_20k_mixer_grab_01 voc=1.0`
- `/eval-record 2026-04-13_10-14-06_exp54_v3_stage2_20k_mixer_grab_01 debug_viz`
- `/eval-record 2026-04-15_12-02-40_exp67_stage2_espressomachine_grab_01 model_9000.pt loose termination`
- `/eval-record 2026-04-15_12-02-40_exp67_stage2_espressomachine_grab_01 --stats 100`
- `/eval-record 2026-04-13_10-14-06_exp54_v3_stage2_20k_mixer_grab_01 2026-04-15_12-02-40_exp67_stage2_espressomachine_grab_01`
- `/eval-record 2026-04-13_10-14-06_exp54_v3_stage2_20k_mixer_grab_01 2026-04-15_12-02-40_exp67_stage2_espressomachine_grab_01 2026-04-15_14-06-44_exp65_stage2_microwave_grab_01`

---

## Step 1 — Parse arguments

Parse from: `$ARGUMENTS`

**Identify run names** by matching tokens against the pattern `\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\S+`.
All matching tokens are run names; everything else is a shared option.

**Shared options** (applied to every run in the batch):
- `project`: from `--project <value>`, default `v2p_hands`
- `model`: from `--model <value>`, default `None` (most recent). Also accept bare `model <N>` or `model_<N>` → `model_<N>.pt`. Applies to all runs if specified (unusual for batches — warn if used with multiple runs).
- `voc`: from `voc=<value>`, default `0.0`. Hydra key: `env.commands.dual_hands_object_tracking_command.initial_virtual_object_control_curriculum_scale=<value>`
- `debug_vis`: `debug_vis` → `true`; `no debug_vis` → `false`; default `false`. Always emit `env.commands.dual_hands_object_tracking_command.debug_vis=<true|false>`.
- `loose_termination`: any of `loose`, `loose termination`, `loose_termination`, `loose term` → sets all early-termination thresholds to 100.0 (effectively disabling them, only timeout terminates). Default: `false` (use training defaults). When enabled, adds these three Hydra overrides:
  - `env.terminations.hand_wrist_away_from_trajectory.params.threshold=100.0`
  - `env.terminations.object_away_from_trajectory.params.position_threshold=100.0`
  - `env.terminations.object_away_from_trajectory.params.orientation_threshold=100.0`
- `stats`: from `--stats <N>`, default `None`. When set, runs N episodes with `--eval_episodes N --num_envs 32 --headless` (no video). Reports the printed summary from eval.py. Mutually exclusive with video mode.

After parsing, print the job list before starting:
```
[eval-record] Batch: N run(s) to process
  [1/N] <run_name_1>
  [2/N] <run_name_2>
  ...
Shared options: project=v2p_hands, voc=0.0, debug_vis=false, model=most recent, loose_termination=false, stats=N or off
```

---

## Steps 2–8 loop — For each run in the list

Execute steps 2 through 8 for each run in order. Print a header before each:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[eval-record] Run [i/N]: <run_name>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

If a run fails (download error, eval crash, missing video), log the error, mark it as FAILED in the
results table, and continue with the next run. Do not abort the batch.

---

## Step 2 — Determine checkpoint path

The local checkpoint folder is:
```
logs/rsl_rl/sharpa_v2p/<run_name>/
```

**If `--model` was specified**: check if that exact file exists locally. If yes, skip to Step 3.5. If no, proceed to Step 3 to download it.

**If no model specified**: always query W&B first (Step 3) to find the latest checkpoint iteration, then check if that specific file is already local. This ensures we use the newest W&B checkpoint even if a (older) local file exists.

## Step 3 — Resolve latest checkpoint from W&B

```python
import wandb, os, re
api = wandb.Api()
runs = api.runs(f"nvidia-isaac/{project}", filters={"display_name": {"$regex": run_name}})
runs = sorted(runs, key=lambda r: r.created_at, reverse=True)
run = runs[0]

files = [f for f in run.files() if f.name.endswith(".pt")]
def iteration(f):
    m = re.search(r'model_(\d+)\.pt', f.name)
    return int(m.group(1)) if m else -1
target = max(files, key=iteration)
model_filename = os.path.basename(target.name)

dest = f"logs/rsl_rl/sharpa_v2p/{run_name}"
local_path = os.path.join(dest, model_filename)

if os.path.exists(local_path):
    print(f"Already have latest: {model_filename} — skipping download")
else:
    os.makedirs(dest, exist_ok=True)
    run.file(target.name).download(root=dest, replace=True)
    print(f"Downloaded: {model_filename}")
```

## Step 3.5 — Read network architecture from checkpoint

```bash
docker exec robotic-grounding-v2d-gpu1 bash -c "cat > /tmp/get_arch.py << 'PYEOF'
import torch
model_path = 'logs/rsl_rl/sharpa_v2p/<run_name>/<model_filename>'
checkpoint = torch.load(model_path, map_location='cpu')
sd = checkpoint.get('model_state_dict', checkpoint)
def dims(sd, prefix):
    keys = sorted([k for k in sd if k.startswith(prefix+'.') and k.endswith('.weight')], key=lambda k: int(k.split('.')[1]))
    return [sd[k].shape[0] for k in keys[:-1]] if len(keys)>1 else None
print('actor='+str(dims(sd,'actor')))
print('critic='+str(dims(sd,'critic')))
PYEOF
cd /workspace/video_to_data/robotic_grounding && /workspace/isaaclab/_isaac_sim/python.sh /tmp/get_arch.py"
```

Parse `actor=[...]` and `critic=[...]`. Format as `[512,256,128]` (no spaces). Always add to eval command:
```
agent.policy.actor_hidden_dims=<actor_hidden_dims>
agent.policy.critic_hidden_dims=<critic_hidden_dims>
```
If extraction fails, omit and warn.

## Step 4 — Derive the motion_file from the run name

Look for `(\w+)_grab_(\d+)` at the end of the run name. Examples:
- `..._mixer_grab_01` → `arctic_s01_mixer_grab_01`
- `..._espressomachine_grab_01` → `arctic_s01_espressomachine_grab_01`
- `..._microwave_grab_01` → `arctic_s01_microwave_grab_01`

If not found via that pattern, use the last `_`-separated token as the object name and construct
`arctic_s01_<object>_grab_01`. If the last token is `grab`, use the second-to-last as the object.

Motion file path: `arctic/arctic_processed/<sequence_id>/sharpa_wave`

## Step 5 — Compute trajectory length from motion file

**Important**: Use pyarrow directly (not SceneConfig — it can't be imported without the full omni.* stack):

```bash
docker exec robotic-grounding-v2d-gpu1 bash -c "cat > /tmp/get_traj_len.py << 'PYEOF'
import pyarrow.parquet as pq, glob, os
HUMAN_MOTION_DATA_DIR = '/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data'
seq_id = '<sequence_id>'
parquet_dir = os.path.join(HUMAN_MOTION_DATA_DIR, 'arctic', 'arctic_processed', f'sequence_id={seq_id}', 'robot_name=sharpa_wave')
files = glob.glob(os.path.join(parquet_dir, '*.parquet'))
data = pq.read_table(files[0]).to_pydict()
timesteps = len(data['object_articulation'][0])
fps = float(data.get('fps', [[30.0]])[0])
episode_length_s = timesteps / fps
video_length = int(round(episode_length_s * 20))
print(f'episode_length_s={episode_length_s:.3f}')
print(f'video_length={video_length}')
PYEOF
/workspace/isaaclab/_isaac_sim/python.sh /tmp/get_traj_len.py"
```

Fallback: `video_length=200` if script fails.

## Step 6 — Ensure the Docker container is running

Container: `robotic-grounding-v2d-gpu1`. Check once before the first run; skip check for subsequent runs.

```bash
docker ps --format '{{.Names}}' | grep -q "^robotic-grounding-v2d-gpu1$"
```

If NOT running, start it:
```bash
WANDB_API_KEY_VALUE="${WANDB_API_KEY}" docker run --rm --runtime=nvidia --gpus device=1 --network host --name robotic-grounding-v2d-gpu1 -v $(pwd):/workspace/video_to_data/robotic_grounding -v ~/.ssh:/root/.ssh:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY="${DISPLAY}" -e WANDB_API_KEY="${WANDB_API_KEY_VALUE}" -e "ACCEPT_EULA=Y" -d --entrypoint /bin/bash robotic-grounding:v2d
```

## Step 7 — Write and launch the eval script

**IMPORTANT**: Command must be a single line (no `\` continuation).

### 7a — Stats mode (`--stats N` was specified)

Use `--eval_episodes N --num_envs 32 --headless` (no `--video`). No video is saved.

```bash
cat > /tmp/run_eval_record.sh << 'EOF'
#!/bin/bash
set -e
cd /workspace/video_to_data/robotic_grounding
python scripts/rsl_rl/eval.py --task Sharpa-V2P-v0-Play --num_envs 32 --headless --eval_episodes <N> --checkpoint logs/rsl_rl/sharpa_v2p/<run_name>/<model_filename> --motion_file arctic/arctic_processed/<sequence_id>/sharpa_wave 'env.commands.dual_hands_object_tracking_command.initial_virtual_object_control_curriculum_scale=<voc_value>' 'env.commands.dual_hands_object_tracking_command.debug_vis=false' 'env.commands.dual_hands_object_tracking_command.always_reset_to_first_frame=true' 'agent.policy.actor_hidden_dims=<actor_hidden_dims>' 'agent.policy.critic_hidden_dims=<critic_hidden_dims>' [IF loose_termination: 'env.terminations.hand_wrist_away_from_trajectory.params.threshold=100.0' 'env.terminations.object_away_from_trajectory.params.position_threshold=100.0' 'env.terminations.object_away_from_trajectory.params.orientation_threshold=100.0']
EOF
docker cp /tmp/run_eval_record.sh robotic-grounding-v2d-gpu1:/tmp/run_eval_record.sh
docker exec robotic-grounding-v2d-gpu1 bash -c "nohup bash /tmp/run_eval_record.sh > /tmp/eval_record_output.log 2>&1 &"
```

After completion, grep the log for the summary block and print it verbatim:
```bash
docker exec robotic-grounding-v2d-gpu1 grep -A6 'Eval Summary' /tmp/eval_record_output.log || docker exec robotic-grounding-v2d-gpu1 tail -20 /tmp/eval_record_output.log
```

### 7b — Video mode (default, no `--stats`)

**IMPORTANT**: Use `--num_envs 7` — Play config has `viewer.env_index=6` (not Hydra-overridable); needs ≥7 envs.

Output video name: `<run_name>_model<model_number>.mp4`

Note: gymnasium names the recorded file `rl-video-step-0.mp4` (uses `step_trigger`). The copy step renames it.

```bash
cat > /tmp/run_eval_record.sh << 'EOF'
#!/bin/bash
set -e
cd /workspace/video_to_data/robotic_grounding
python scripts/rsl_rl/eval.py --task Sharpa-V2P-v0-Play --num_envs 7 --video --video_length <video_length> --checkpoint logs/rsl_rl/sharpa_v2p/<run_name>/<model_filename> --motion_file arctic/arctic_processed/<sequence_id>/sharpa_wave 'env.commands.dual_hands_object_tracking_command.initial_virtual_object_control_curriculum_scale=<voc_value>' 'env.commands.dual_hands_object_tracking_command.debug_vis=<true|false>' 'agent.policy.actor_hidden_dims=<actor_hidden_dims>' 'agent.policy.critic_hidden_dims=<critic_hidden_dims>' [IF loose_termination: 'env.terminations.hand_wrist_away_from_trajectory.params.threshold=100.0' 'env.terminations.object_away_from_trajectory.params.position_threshold=100.0' 'env.terminations.object_away_from_trajectory.params.orientation_threshold=100.0']
VIDEO_DIR="logs/rsl_rl/sharpa_v2p/<run_name>/videos/play"
OUT_DIR="experiments/eval_recordings/<run_name>"
OUT_NAME="<run_name>_model<model_number>.mp4"
mkdir -p "${OUT_DIR}"
VIDEO_FILE=$(ls "${VIDEO_DIR}"/*.mp4 2>/dev/null | head -1)
if [ -n "${VIDEO_FILE}" ]; then
    cp "${VIDEO_FILE}" "${OUT_DIR}/${OUT_NAME}"
    echo "[eval-record] Video saved: ${OUT_DIR}/${OUT_NAME}"
else
    echo "[eval-record] WARNING: no .mp4 found in ${VIDEO_DIR}"
    ls "${VIDEO_DIR}" 2>/dev/null || true
fi
EOF
docker cp /tmp/run_eval_record.sh robotic-grounding-v2d-gpu1:/tmp/run_eval_record.sh
docker exec robotic-grounding-v2d-gpu1 bash -c "nohup bash /tmp/run_eval_record.sh > /tmp/eval_record_output.log 2>&1 &"
```

Confirm process started:
```bash
docker exec robotic-grounding-v2d-gpu1 pgrep -a python | grep eval.py
```

## Step 8 — Monitor to completion

Poll every 60 seconds until eval exits. Stats mode with 32 envs typically takes 3–8 minutes; video mode 1–3 minutes.

```bash
while docker exec robotic-grounding-v2d-gpu1 pgrep -f "eval.py" > /dev/null 2>&1; do
    echo "  [i/N] Still running... ($(date '+%H:%M:%S'))"
    sleep 60
done
```

Once exited:

**Stats mode**: extract and print the summary block from the log:
```bash
docker exec robotic-grounding-v2d-gpu1 grep -A6 'Eval Summary' /tmp/eval_record_output.log || docker exec robotic-grounding-v2d-gpu1 tail -20 /tmp/eval_record_output.log
```

**Video mode**: tail the log and verify file:
```bash
docker exec robotic-grounding-v2d-gpu1 tail -5 /tmp/eval_record_output.log
ls -lh experiments/eval_recordings/<run_name>/
```

Record the outcome, then proceed to the next run.

---

## Step 9 — Final summary table

After all runs complete, print a summary table:

**Video mode:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[eval-record] Batch complete — N run(s)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓  <run_name_1>  model_XXXXX.pt  <episode_length_s>s  <file_size>  experiments/eval_recordings/...
  ✓  <run_name_2>  model_XXXXX.pt  <episode_length_s>s  <file_size>  experiments/eval_recordings/...
  ✗  <run_name_3>  FAILED: <reason>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Stats mode:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[eval-record] Batch complete — N run(s)  [stats mode: <M> episodes each]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓  <run_name_1>  model_XXXXX.pt  ratio=0.XXX±0.XXX  completions=XX/M (XX%)
  ✓  <run_name_2>  model_XXXXX.pt  ratio=0.XXX±0.XXX  completions=XX/M (XX%)
  ✗  <run_name_3>  FAILED: <reason>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
