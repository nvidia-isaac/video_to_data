---
name: getneval
description: Download a wandb run model (if not already present) then run eval.py inside Docker on it.
user-invocable: true
allowed-tools:
  - Bash
---

# /getneval — Get Run + Evaluate in Docker

Given a run name, ensure the model checkpoint is downloaded, then launch eval.py inside the Docker container.

Arguments: `$ARGUMENTS`
Expected format: `<run_name> [--project <project>] [--model <model_name>] [voc=<value>]`
Examples:
- `/getneval 2026-03-31_11-33-13_exp37_cont5p0_cum0p5_with_contact_tracking_capsulemachine`
- `/getneval 2026-03-31_11-33-13_exp37_cont5p0_cum0p5_with_contact_tracking_capsulemachine --model model_2000.pt`
- `/getneval 2026-04-11_16-25-54_exp58_v3_stage2_microwave_grab_01 model 2000 voc=1.0`

---

## Step 1 — Parse arguments

Parse from: `$ARGUMENTS`

- `run_name`: first positional argument (full run name string)
- `project`: from `--project <value>`, default `v2p_hands`
- `model`: from `--model <value>`, default `None` (means use most recent). Also accept bare `model <N>` or `model_<N>` positional shorthand → `model_<N>.pt`
- `voc`: from `voc=<value>` anywhere in the args, default `0.0`. Expands to the full Hydra key `env.commands.dual_hands_object_tracking_command.initial_virtual_object_control_curriculum_scale=<value>`

## Step 2 — Determine checkpoint path

The local checkpoint folder is:
```
logs/rsl_rl/sharpa_v2p/<run_name>/
```

Check if any `model_*.pt` file already exists in that folder.

- If `--model` was specified: check if that exact file exists.
- Otherwise: look for all `model_*.pt` files, extract iteration numbers, and identify the highest one as the "most recent".

If the required model is already present, skip to Step 4 (do not re-download).
If it is missing (or the folder doesn't exist), proceed to Step 3.

## Step 3 — Download the model via /getrun logic

Use the wandb Python API to find the run and download the model exactly as `/getrun` does:

```python
import wandb, os, re
api = wandb.Api()
runs = api.runs(f"nvidia-isaac/{project}", filters={"display_name": {"$regex": run_name}})
runs = sorted(runs, key=lambda r: r.created_at, reverse=True)
run = runs[0]

files = [f for f in run.files() if f.name.endswith(".pt")]
# find most recent by extracting N from model_N.pt
def iteration(f):
    m = re.search(r'model_(\d+)\.pt', f.name)
    return int(m.group(1)) if m else -1
target = max(files, key=iteration)

dest = f"logs/rsl_rl/sharpa_v2p/{run_name}"
os.makedirs(dest, exist_ok=True)
target_file = run.file(target.name)
target_file.download(root=dest, replace=True)
model_filename = target.name
```

Report which model was downloaded.

## Step 3.5 — Read network architecture from checkpoint

After downloading (or if the model was already cached), read the actor/critic hidden dims directly from the `.pt` checkpoint's state dict. This avoids `size mismatch` errors. Run this inside the container using `python.sh` (torch is only available via that wrapper):

```bash
# Write script to extract architecture
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

Parse `actor=[...]` and `critic=[...]` from the output. Linear layers are at even indices (0, 2, 4, ...); weight shape is `[out, in]`. All but the last are hidden layers.

Always add the architecture overrides to the eval command in Step 6:
```
agent.policy.actor_hidden_dims=<actor_hidden_dims>
agent.policy.critic_hidden_dims=<critic_hidden_dims>
```
Format lists as `[512,256,128]` (no spaces). If extraction fails, omit and warn the user.

## Step 4 — Derive the motion_file from the run name

`eval.py` uses `--motion_file`, not `--sequence_filter`.

**First**, check if the run name already contains a full `arctic_s01_..._grab_01` pattern.
If so, extract the sequence id directly — e.g.:
- `..._stage2_microwave_grab_01` → sequence id `arctic_s01_microwave_grab_01`

Use a regex like `(arctic_s01_\w+_grab_\d+)` to detect this case.

**Otherwise**, the run name ends with just the object name as the last `_`-separated token.
Construct the sequence id as `arctic_s01_<object>_grab_01`:
- `..._capsulemachine`     → `arctic_s01_capsulemachine_grab_01`
- `..._espressomachine`    → `arctic_s01_espressomachine_grab_01`
- `..._waffleiron`         → `arctic_s01_waffleiron_grab_01`
- `..._microwave`          → `arctic_s01_microwave_grab_01`
- `..._box`                → `arctic_s01_box_grab_01`

If the last token is literally `grab`, use the second-to-last token as the object name.

Then form the full motion_file path: `arctic/arctic_processed/<sequence_id>/sharpa_wave`

## Step 5 — Ensure the Docker container is running

Container name: `robotic-grounding-v2d-gpu1`

Check if it is already running:
```bash
docker ps --format '{{.Names}}' | grep -q "^robotic-grounding-v2d-gpu1$"
```

If NOT running, start it (replicating what `./workflow/run.sh start v2d 1` does, but detached without entering an interactive shell):
```bash
cd /path/to/repo && \
WANDB_API_KEY_VALUE="${WANDB_API_KEY}" && \
docker run --rm \
    --runtime=nvidia \
    --gpus device=1 \
    --network host \
    --name robotic-grounding-v2d-gpu1 \
    -v $(pwd):/workspace/video_to_data/robotic_grounding \
    -v ~/.ssh:/root/.ssh:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -e DISPLAY="${DISPLAY}" \
    -e WANDB_API_KEY="${WANDB_API_KEY_VALUE}" \
    -e "ACCEPT_EULA=Y" \
    -d \
    --entrypoint /bin/bash \
    robotic-grounding:v2d
```

Wait briefly and confirm the container started successfully.

## Step 6 — Run eval.py inside the container

The checkpoint path inside the container mirrors the host mount:
```
/workspace/video_to_data/robotic_grounding/logs/rsl_rl/sharpa_v2p/<run_name>/<model_filename>
```

**IMPORTANT**: The eval command MUST be issued as a single line — multiline `\` continuation
commands break when pasted into a terminal and cause Isaac Sim to launch with no arguments
(`task_name=None`). Always build the command as one line.

Write the command to a script file on the host, then launch it detached inside the container
so Isaac Sim opens without blocking Claude:

```bash
# 1. Write script (on host via Bash tool)
cat > /tmp/run_eval.sh << 'EOF'
#!/bin/bash
cd /workspace/video_to_data/robotic_grounding
python scripts/rsl_rl/eval.py --task Sharpa-V2P-v0-Play --checkpoint logs/rsl_rl/sharpa_v2p/<run_name>/<model_filename> --motion_file arctic/arctic_processed/<sequence_id>/sharpa_wave 'env.commands.dual_hands_object_tracking_command.initial_virtual_object_control_curriculum_scale=<voc_value>' <'agent.policy.actor_hidden_dims=[...]' if found> <'agent.policy.critic_hidden_dims=[...]' if found>
EOF

# 2. Copy script into container (/tmp is not shared between host and container)
docker cp /tmp/run_eval.sh robotic-grounding-v2d-gpu1:/tmp/run_eval.sh

# 3. Launch detached inside container (output goes to /tmp/eval_output.log)
docker exec robotic-grounding-v2d-gpu1 bash -c "nohup bash /tmp/run_eval.sh > /tmp/eval_output.log 2>&1 &"
```

After launching, confirm the process started:
```bash
docker exec robotic-grounding-v2d-gpu1 pgrep -a python | grep eval.py
```

If architecture overrides are not found in W&B, omit those args and warn the user that the eval task's default architecture will be used.

## Step 7 — Report

Print a summary:
- Run name and project
- Model checkpoint used (and whether it was freshly downloaded or already cached)
- Motion file derived
- Container used
- Full eval command executed
