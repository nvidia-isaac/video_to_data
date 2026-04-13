"""OSMO workflow generator for exp_stage1_nocoll.

For each of the 6 objects:
  1. Launches a training task with --disable_robot_to_object_collisions so the
     hands can freely learn to track keypoints before contact rewards are introduced.
  2. After training, uploads the final checkpoint as a W&B artifact so that
     launch_stage2.py can retrieve it for the second training stage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from experiments.utils import (  # noqa: E402
    overrides_to_cli,
    sequence_to_object,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WANDB_PROJECT = "v2p_hands"


# ---------------------------------------------------------------------------
# Local --variant support
# ---------------------------------------------------------------------------


def _sequence_to_seq_key(seq_id: str) -> str:
    parts = seq_id.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else seq_id


def get_variant_overrides(variant_name: str, config: dict) -> dict:
    """Return overrides for --local --variant <seq_key>.

    seq_key is the sequence id with the arctic_sXX_ prefix stripped,
    e.g. 'espressomachine_grab_01' for 'arctic_s01_espressomachine_grab_01'.
    """
    mt = config["osmo_multi_task"]
    seq_keys = [_sequence_to_seq_key(s) for s in mt["sequence_ids"]]
    if variant_name not in seq_keys:
        raise ValueError(f"Unknown variant '{variant_name}'. Available: {seq_keys}")

    overrides = dict(config["train_overrides"])
    return overrides


# ---------------------------------------------------------------------------
# Checkpoint upload snippet (embedded in entry.sh)
# ---------------------------------------------------------------------------


def _checkpoint_upload_snippet(seq_key: str, run_name: str) -> str:
    """Bash snippet that uploads the final checkpoint as a W&B artifact."""
    artifact_name = f"checkpoint_stage1_nocoll_{seq_key}"
    return f"""\

# --- upload final checkpoint to W&B artifact ---
export CHECKPOINT=$(find logs/rsl_rl -name "model_*.pt" | grep "{run_name}" | sort -t_ -k2 -rn | head -1)
if [ -z "$CHECKPOINT" ]; then
  echo "[WARN] No checkpoint found for {run_name}, skipping artifact upload."
else
  echo "Uploading checkpoint: $CHECKPOINT"
  python - <<'PYEOF'
import glob, os, sys, wandb

artifact_name = "{artifact_name}"
project = "{_WANDB_PROJECT}"
checkpoint = os.environ.get("CHECKPOINT", "")
if not checkpoint:
    checkpoints = sorted(
        glob.glob("logs/rsl_rl/**/*{run_name}*/model_*.pt", recursive=True),
        key=lambda p: int(p.rsplit("_", 1)[-1].replace(".pt", ""))
    )
    if not checkpoints:
        print("[WARN] No checkpoint found, skipping upload.")
        sys.exit(0)
    checkpoint = checkpoints[-1]

print(f"Uploading {{checkpoint}} as artifact {{artifact_name}}")
artifact = wandb.Artifact(artifact_name, type="model")
artifact.add_file(checkpoint, name="model_final.pt")
# Log the artifact to the existing training run to avoid cluttering W&B with
# separate upload runs. Fall back to a new run if the training run isn't found.
api = wandb.Api()
training_runs = api.runs(
    f"nvidia-isaac/{{project}}",
    filters={{"display_name": {{"$regex": "{run_name}"}}}},
)
training_runs = sorted(training_runs, key=lambda r: r.created_at, reverse=True)
if training_runs:
    run = wandb.init(id=training_runs[0].id, project=project, resume="must")
else:
    print(f"[WARN] Training run '{run_name}' not found, creating upload run.")
    run = wandb.init(project=project, name=f"upload_{{artifact_name}}", job_type="artifact-upload")
logged_artifact = run.log_artifact(artifact)
logged_artifact.wait()  # block until artifact is fully committed to W&B server
run.finish()
print("Upload complete.")
PYEOF
fi"""


# ---------------------------------------------------------------------------
# OSMO workflow generation
# ---------------------------------------------------------------------------


def generate_workflow(exp_id: str, config: dict) -> str:
    """Generate OSMO multi-task workflow YAML for stage1: one task per object."""
    mt = config["osmo_multi_task"]
    sequence_ids = mt["sequence_ids"]
    base_overrides = dict(config["train_overrides"])
    max_iterations = config.get("max_iterations", 1000)
    _video = config.get("video", False)
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_api_key:
        print("[WARNING] WANDB_API_KEY not set — wandb will fail in the container")

    tasks_yaml = []
    for seq_id in sequence_ids:
        _obj = sequence_to_object(seq_id)
        seq_key = _sequence_to_seq_key(seq_id)

        prefix = config.get("run_name_prefix", "")
        run_name = f"{prefix}stage1_{seq_key}"

        # Build overrides: base + dynamic asset_path resolved at container runtime
        overrides_cli = " \\\n  ".join(overrides_to_cli(base_overrides))

        entry = f"""set -ex

python scripts/rsl_rl/train.py \\
  --headless \\
  --task Sharpa-V2P-v0 \\
  --run_name {run_name} \\
  --motion_file arctic_processed/{seq_id}/sharpa_wave \\
  --disable_robot_to_object_collisions \\
  --max_iterations {max_iterations} \\
  --logger wandb \\
  --log_project_name {_WANDB_PROJECT} \\
  {overrides_cli}
""" + _checkpoint_upload_snippet(
            seq_key, run_name
        )

        entry_indent = "\n".join("        " + line for line in entry.split("\n"))
        tasks_yaml.append(
            f"""  - name: train-{seq_key.replace("_", "-")}
    image: {{{{image}}}}
    command: [/bin/bash]
    args: [/tmp/entry.sh]
    environment:
      ACCEPT_EULA: Y
      OMNI_SERVER: omniverse://isaac-dev.ov.nvidia.com
      WANDB_API_KEY: {wandb_api_key}
    files:
    - path: /tmp/entry.sh
      contents: |-
{entry_indent}"""
        )

    return f"""# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Generated from experiments/exp_stage1_nocoll/workflow.py

workflow:
  name: {{{{workflow_name}}}}
  resources:
    default:
      cpu: 6
      gpu: 1
      memory: 120Gi
      storage: 200Gi

  tasks:
{chr(10).join(tasks_yaml)}

default-values:
  workflow_name: robotic_grounding_{exp_id}
  image: nvcr.io/nvstaging/isaac-amr/robotic-grounding:latest
"""
