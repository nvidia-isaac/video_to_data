#!/usr/bin/env python3
r"""Monitor two-stage training experiments and auto-resume crashed runs.

Watches W&B for stage1/stage2 runs and relaunches any that crash early.
Designed to be run periodically (e.g., via cron every 10 minutes).

State is persisted between runs at --state-file.

First run (--init): submits stage1 workflows and creates state file.
Subsequent runs: checks W&B, relaunches crashed stage1 tasks, detects
stage1 completion to trigger stage2, and monitors stage2 for crashes.

Usage (first time):
    python scripts/monitor_two_stage.py --init \
        --exp-ids exp52 exp53 exp54 exp55 exp56 \
        --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:v2d

Usage (subsequent runs via cron):
    python scripts/monitor_two_stage.py

SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import wandb

_RG_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _RG_ROOT / "experiments"
_WANDB_PROJECT = "v2p_hands"
_DEFAULT_STATE_FILE = _RG_ROOT / "scripts" / "monitor_state.json"
_DEFAULT_IMAGE = "nvcr.io/nvstaging/isaac-amr/robotic-grounding:v2d"
_DEFAULT_POOL = "isaac-dev-l40s-04"

if str(_RG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RG_ROOT))

from experiments.utils import overrides_to_cli  # noqa: E402

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[{_now_str()}] {msg}", flush=True)


def _load_registry() -> dict[str, str]:
    registry: dict[str, str] = {}
    for path in [
        _EXPERIMENTS_DIR / "registry.yaml",
        _EXPERIMENTS_DIR / "registry.local.yaml",
    ]:
        if path.exists():
            with open(path) as f:
                registry.update(yaml.safe_load(f) or {})
    return registry


def _load_config(exp_id: str) -> dict:
    registry = _load_registry()
    if exp_id not in registry:
        raise ValueError(f"Unknown experiment id: '{exp_id}'")
    config_path = _EXPERIMENTS_DIR / registry[exp_id] / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _sequence_to_seq_key(seq_id: str) -> str:
    parts = seq_id.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else seq_id


def _sequence_to_object(seq_id: str) -> str:
    parts = seq_id.split("_")
    return parts[2] if len(parts) >= 3 else seq_id


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def load_state(state_file: Path) -> dict:
    """Load persisted monitoring state from disk."""
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {}


def save_state(state: dict, state_file: Path) -> None:
    """Persist monitoring state to disk."""
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    _log(f"State saved to {state_file}")


# ---------------------------------------------------------------------------
# Stage 1 relaunch helpers
# ---------------------------------------------------------------------------


def _checkpoint_upload_snippet(seq_key: str, run_name: str) -> str:
    """Bash snippet that uploads the final checkpoint as a W&B artifact.

    Mirrors example_stage1_nocoll/workflow.py::_checkpoint_upload_snippet.
    Keep in sync if that function changes.
    """
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
project = "v2p_hands"
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
logged_artifact.wait()
run.finish()
print("Upload complete.")
PYEOF
fi"""


def _generate_stage1_resume_yaml(
    exp_id: str,
    seq_id: str,
    ckpt_path: Path,
    stage1_config: dict,
    image: str,
) -> str:
    """Generate OSMO workflow YAML for resuming a single stage1 task from a checkpoint.

    ckpt_path is the local .pt file (downloaded from W&B artifact into a per-seq directory).
    OSMO receives the parent directory as a dataset input and the training script reads
    the checkpoint from the container mount point {{{input:0}}}/ckpt_{seq_key}/{filename}.
    """
    seq_key = _sequence_to_seq_key(seq_id)
    run_name = f"{exp_id}_stage1_{seq_key}"
    max_iterations = stage1_config.get("max_iterations", 1000)
    overrides = dict(stage1_config.get("train_overrides", {}))
    overrides_cli = " \\\n  ".join(overrides_to_cli(overrides))
    ckpt_filename = ckpt_path.name
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    upload_snippet = _checkpoint_upload_snippet(seq_key, run_name)

    entry = f"""set -ex

python scripts/rsl_rl/train.py \\
  --headless \\
  --task Sharpa-V2P-v0 \\
  --run_name {run_name} \\
  --motion_file arctic/arctic_processed/{seq_id}/sharpa_wave \\
  --max_iterations {max_iterations} \\
  --disable_robot_to_object_collisions \\
  --logger wandb \\
  --log_project_name {_WANDB_PROJECT} \\
  --resume --checkpoint "{{{{input:0}}}}/ckpt_{seq_key}/{ckpt_filename}" \\
  {overrides_cli}
{upload_snippet}"""

    entry_indent = "\n".join("        " + line for line in entry.split("\n"))

    return f"""# Generated by monitor_two_stage.py — stage1 resume for {exp_id} / {seq_key}
workflow:
  name: {{{{workflow_name}}}}
  resources:
    default:
      cpu: 6
      gpu: 1
      memory: 120Gi
      storage: 200Gi

  tasks:
  - name: train-{seq_key.replace("_", "-")}
    image: {{{{image}}}}
    command: [/bin/bash]
    args: [/tmp/entry.sh]
    environment:
      ACCEPT_EULA: Y
      OMNI_SERVER: omniverse://isaac-dev.ov.nvidia.com
      WANDB_API_KEY: {wandb_api_key}
    inputs:
    - dataset:
        name: ckpt_{seq_key}
        localpath: '{ckpt_path}'
    files:
    - path: /tmp/entry.sh
      contents: |-
{entry_indent}

default-values:
  workflow_name: robotic_grounding_{exp_id}_stage1_rerun_{seq_key}
  image: {image}
"""


def _generate_stage1_fresh_yaml(
    exp_id: str,
    seq_id: str,
    stage1_config: dict,
    image: str,
) -> str:
    """Generate OSMO workflow YAML for a fresh (no checkpoint) stage1 restart."""
    seq_key = _sequence_to_seq_key(seq_id)
    run_name = f"{exp_id}_stage1_{seq_key}"
    max_iterations = stage1_config.get("max_iterations", 1000)
    overrides = dict(stage1_config.get("train_overrides", {}))
    overrides_cli = " \\\n  ".join(overrides_to_cli(overrides))
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    upload_snippet = _checkpoint_upload_snippet(seq_key, run_name)

    entry = f"""set -ex

python scripts/rsl_rl/train.py \\
  --headless \\
  --task Sharpa-V2P-v0 \\
  --run_name {run_name} \\
  --motion_file arctic/arctic_processed/{seq_id}/sharpa_wave \\
  --max_iterations {max_iterations} \\
  --disable_robot_to_object_collisions \\
  --logger wandb \\
  --log_project_name {_WANDB_PROJECT} \\
  {overrides_cli}
{upload_snippet}"""

    entry_indent = "\n".join("        " + line for line in entry.split("\n"))

    return f"""# Generated by monitor_two_stage.py — stage1 fresh restart for {exp_id} / {seq_key}
workflow:
  name: {{{{workflow_name}}}}
  resources:
    default:
      cpu: 6
      gpu: 1
      memory: 120Gi
      storage: 200Gi

  tasks:
  - name: train-{seq_key.replace("_", "-")}
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
{entry_indent}

default-values:
  workflow_name: robotic_grounding_{exp_id}_stage1_fresh_{seq_key}
  image: {image}
"""


def _download_stage1_checkpoint(seq_key: str, outdir: Path) -> Path | None:
    """Download the stage1 checkpoint for seq_key from W&B artifacts.

    Returns the local path to the .pt file, or None if no artifact exists.
    RSL-RL does not upload model files to W&B run files automatically; the only
    checkpoint in W&B is the artifact uploaded by the _checkpoint_upload_snippet
    at the end of a successful run. For crashed mid-training runs there will
    typically be no artifact available, so callers should fall back to a fresh start.
    """
    artifact_name = f"checkpoint_stage1_nocoll_{seq_key}"
    candidates = [artifact_name, f"checkpoint_stage1_nocoll_{seq_key.split('_')[0]}"]
    api = wandb.Api()
    for name in candidates:
        full_name = f"nvidia-isaac/{_WANDB_PROJECT}/{name}:latest"
        print(f"  Fetching artifact {full_name} ...", end=" ", flush=True)
        try:
            artifact = api.artifact(full_name)
            dest_dir = outdir / seq_key
            dest_dir.mkdir(parents=True, exist_ok=True)
            artifact.download(root=str(dest_dir))
            pts = list(dest_dir.glob("*.pt"))
            if not pts:
                print("no .pt in artifact")
                return None
            print("done")
            return pts[0]
        except Exception:
            print("not found")
    return None


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def submit_stage1(
    exp_id: str,
    stage1_exp_id: str,
    image: str,
    pool: str,
    priority: str,
    dry_run: bool,
    build_image: bool = False,
) -> bool:
    """Submit stage1 OSMO multi-task workflow for one experiment."""
    cmd = [
        sys.executable,
        str(_RG_ROOT / "experiments" / "run_experiment.py"),
        stage1_exp_id,
        "--osmo",
        "--run-name-prefix",
        f"{exp_id}_",
        "--image",
        image,
        "--pool",
        pool,
        "--priority",
        priority,
    ]
    if build_image:
        cmd.append("--build-image")
    if dry_run:
        cmd.append("--dry-run")
    _log(f"Submitting stage1 for {exp_id} (config: {stage1_exp_id})")
    if build_image:
        _log(f"Building image {image} ...")
    result = subprocess.run(cmd, cwd=_RG_ROOT, check=False)
    if result.returncode != 0:
        _log(f"WARNING: Stage1 submission for {exp_id} exited {result.returncode}")
        return False
    if build_image:
        registry = image.split("/")[0]
        _log(f"Done: built image {image} and pushed to {registry}")
    return True


def relaunch_stage1_task(
    exp_id: str,
    seq_id: str,
    crashed_run: Any,
    stage1_config: dict,
    image: str,
    pool: str,
    priority: str,
    dry_run: bool,
) -> bool:
    """Resubmit a crashed stage1 task, resuming from checkpoint if one exists in W&B.

    RSL-RL does not upload model files to W&B run files automatically; checkpoints
    only exist in W&B artifacts after a run completes. For mid-training crashes there
    will typically be no artifact, so we fall back to a fresh restart (1000 iterations
    is short enough that starting over is acceptable).
    """
    seq_key = _sequence_to_seq_key(seq_id)
    steps = crashed_run.summary.get("_step", 0) or 0
    _log(f"Relaunching stage1 {exp_id}/{seq_key} (crashed at step {steps})")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = _download_stage1_checkpoint(seq_key, Path(tmpdir))

        if ckpt_path is not None:
            _log(f"  Found checkpoint {ckpt_path.name} — resuming from it")
            workflow_name = f"{exp_id}_stage1_rerun_{seq_key}"
            try:
                workflow_yaml = _generate_stage1_resume_yaml(
                    exp_id, seq_id, ckpt_path, stage1_config, image
                )
            except Exception as e:
                _log(f"  ERROR generating resume workflow: {e}")
                return False
        else:
            _log(
                "  No checkpoint found — relaunching fresh (stage1 is short, ~1k iters)"
            )
            workflow_name = f"{exp_id}_stage1_fresh_{seq_key}"
            try:
                workflow_yaml = _generate_stage1_fresh_yaml(
                    exp_id, seq_id, stage1_config, image
                )
            except Exception as e:
                _log(f"  ERROR generating fresh workflow: {e}")
                return False

        workflow_file = _RG_ROOT / f"tmp_monitor_{workflow_name}.yaml"
        workflow_file.write_text(workflow_yaml)

        try:
            run_osmo_py = _RG_ROOT / "scripts" / "run_osmo.py"
            cmd = [
                sys.executable,
                str(run_osmo_py),
                "--experiment-name",
                workflow_name,
                "--workflow-yaml",
                str(workflow_file),
                "--pool",
                pool,
                "--priority",
                priority,
                "--image",
                image,
            ]
            if dry_run:
                _log(f"  [DRY RUN] {' '.join(cmd)}")
                return True
            _log(f"  Submitting {'resume' if ckpt_path else 'fresh'} workflow...")
            result = subprocess.run(cmd, cwd=_RG_ROOT, check=False)
            if result.returncode != 0:
                _log(f"  WARNING: OSMO submission failed (exit {result.returncode})")
                return False
            return True
        finally:
            workflow_file.unlink(missing_ok=True)


def launch_stage2(
    exp_id: str,
    stage2_config_id: str,
    image: str,
    pool: str,
    priority: str,
    dry_run: bool,
    build_image: bool = False,
) -> bool:
    """Launch stage2 using stage1 W&B artifacts."""
    _log(f"Launching stage2 for {exp_id} (config: {stage2_config_id})")
    cmd = [
        sys.executable,
        str(_RG_ROOT / "scripts" / "launch_stage2.py"),
        "--config-id",
        stage2_config_id,
        "--exp-id",
        f"{exp_id}_stage2",
        "--pool",
        pool,
        "--priority",
        priority,
        "--image",
        image,
    ]
    if build_image:
        cmd.append("--build-image")
    if dry_run:
        cmd.append("--dry-run")
    if build_image:
        _log(f"Building image {image} ...")
    result = subprocess.run(cmd, cwd=_RG_ROOT, check=False)
    if result.returncode != 0:
        _log(f"  WARNING: Stage2 launch failed (exit {result.returncode})")
        return False
    if build_image:
        registry = image.split("/")[0]
        _log(f"Done: built image {image} and pushed to {registry}")
    return True


def resume_crashed_stage2(
    exp_id: str,
    stage2_config_id: str,
    image: str,
    pool: str,
    priority: str,
    dry_run: bool,
) -> bool:
    """Resume crashed stage2 runs from their latest W&B checkpoints."""
    _log(f"Resuming crashed stage2 for {exp_id} (config: {stage2_config_id})")
    cmd = [
        sys.executable,
        str(_RG_ROOT / "scripts" / "launch_stage2.py"),
        "--config-id",
        stage2_config_id,
        "--exp-id",
        f"{exp_id}_stage2_rerun",
        "--pool",
        pool,
        "--priority",
        priority,
        "--image",
        image,
        "--resume-crashed",
    ]
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(cmd, cwd=_RG_ROOT, check=False)
    if result.returncode != 0:
        _log(f"  WARNING: Stage2 resume failed (exit {result.returncode})")
        return False
    return True


# ---------------------------------------------------------------------------
# Core monitoring logic
# ---------------------------------------------------------------------------


def monitor_once(state: dict, dry_run: bool = False) -> dict:
    """Run one monitoring pass. Mutates and returns state."""
    api = wandb.Api()
    submitted_after = state["submitted_after"]
    stage1_threshold = state.get("stage1_threshold", 1000)
    stage2_threshold = state.get("stage2_threshold", 10000)
    min_steps = state.get("min_steps", 50)
    image = state.get("image", _DEFAULT_IMAGE)
    pool = state.get("pool", _DEFAULT_POOL)
    priority = state.get("priority", "NORMAL")
    build_image = state.get("build_image", False)
    max_reruns_per_seq = state.get("max_reruns_per_seq", 3)
    max_stage2_reruns = state.get("max_stage2_reruns", 3)

    _log(f"Monitoring {state['exp_ids']} (submitted_after={submitted_after})")

    for exp_id in state["exp_ids"]:
        print(f"\n--- {exp_id} ---")
        exp_state = state["experiments"].setdefault(
            exp_id,
            {
                "stage1_submitted": False,
                "stage1_completed_seqs": [],
                "stage2_launched": False,
                "stage2_rerun_count": 0,
                "stage1_reruns": {},
            },
        )

        # Load pipeline config
        try:
            pipeline_config = _load_config(exp_id)
        except Exception as e:
            _log(f"  ERROR loading {exp_id} config: {e}")
            continue

        stage1_exp_id = pipeline_config.get("stage1_exp_id", f"{exp_id}_stage1")
        stage2_config_id = pipeline_config.get("stage2_config_id", f"{exp_id}_stage2")

        # Load stage1 config + sequences
        try:
            stage1_config = _load_config(stage1_exp_id)
            sequences = stage1_config["osmo_multi_task"]["sequence_ids"]
        except Exception as e:
            _log(f"  ERROR loading stage1 config for {exp_id}: {e}")
            continue

        # ── Submit stage1 if not done yet ──
        if not exp_state.get("stage1_submitted"):
            ok = submit_stage1(
                exp_id,
                stage1_exp_id,
                image,
                pool,
                priority,
                dry_run,
                build_image=build_image,
            )
            if ok or dry_run:
                exp_state["stage1_submitted"] = True
                exp_state["stage1_submit_time"] = _now_str()
            continue  # let W&B catch up before checking run states

        # ── Stage 1 monitoring ──
        if not exp_state.get("stage2_launched"):
            try:
                stage1_runs_all = list(
                    api.runs(
                        f"nvidia-isaac/{_WANDB_PROJECT}",
                        filters={
                            "display_name": {"$regex": f"{re.escape(exp_id)}_stage1_"}
                        },
                    )
                )
                stage1_runs = [
                    r for r in stage1_runs_all if r.created_at > submitted_after
                ]
            except Exception as e:
                _log(f"  WARNING: W&B query failed for {exp_id} stage1: {e}")
                stage1_runs = []

            # Group by seq_key; most recent run first
            runs_by_seq: dict[str, list] = {}
            for run in stage1_runs:
                m = re.search(rf"{re.escape(exp_id)}_stage1_(.+)$", run.display_name)
                if m:
                    sk = m.group(1)
                    runs_by_seq.setdefault(sk, []).append(run)
            for sk in runs_by_seq:  # noqa: PLC0206
                runs_by_seq[sk].sort(key=lambda r: r.created_at, reverse=True)

            completed_seqs = set(exp_state.get("stage1_completed_seqs", []))

            for seq_id in sequences:
                seq_key = _sequence_to_seq_key(seq_id)
                if seq_key in completed_seqs:
                    print(f"  stage1/{seq_key}: done (cached)")
                    continue

                runs = runs_by_seq.get(seq_key, [])
                if not runs:
                    print(f"  stage1/{seq_key}: no W&B runs found yet")
                    continue

                latest = runs[0]
                steps = latest.summary.get("_step", 0) or 0
                state_str = latest.state
                print(
                    f"  stage1/{seq_key}: {state_str} @ step {steps} ({latest.display_name})"
                )

                if state_str == "finished":
                    completed_seqs.add(seq_key)

                elif state_str in ("crashed", "failed"):
                    if steps < min_steps:
                        _log(
                            f"    → step {steps} < {min_steps}: likely system error, not relaunching"
                        )
                    elif steps >= stage1_threshold:
                        _log(
                            f"    → step {steps} >= {stage1_threshold}: treating as complete"
                        )
                        completed_seqs.add(seq_key)
                    else:
                        reruns = exp_state["stage1_reruns"].setdefault(seq_key, [])
                        already = any(r["run_id"] == latest.id for r in reruns)
                        if already:
                            _log(
                                f"    → already relaunched this crash (run_id={latest.id})"
                            )
                        elif len(reruns) >= max_reruns_per_seq:
                            _log(
                                f"    → max reruns ({max_reruns_per_seq}) reached for {seq_key}, skipping"
                            )
                        else:
                            ok = relaunch_stage1_task(
                                exp_id,
                                seq_id,
                                latest,
                                stage1_config,
                                image,
                                pool,
                                priority,
                                dry_run,
                            )
                            if ok or dry_run:
                                reruns.append(
                                    {
                                        "run_id": latest.id,
                                        "run_name": latest.display_name,
                                        "steps": steps,
                                        "relaunched_at": _now_str(),
                                    }
                                )

            exp_state["stage1_completed_seqs"] = sorted(completed_seqs)

            # Check if all sequences complete → trigger stage2
            all_done = len(completed_seqs) == len(sequences)
            print(
                f"  stage1 progress: {len(completed_seqs)}/{len(sequences)} sequences complete"
            )
            if all_done:
                _log(f"  All stage1 tasks complete for {exp_id}! Launching stage2...")
                ok = launch_stage2(
                    exp_id,
                    stage2_config_id,
                    image,
                    pool,
                    priority,
                    dry_run,
                    build_image=build_image,
                )
                if ok or dry_run:
                    exp_state["stage2_launched"] = True
                    exp_state["stage2_launch_time"] = _now_str()

        # ── Stage 2 monitoring ──
        else:
            # Find stage2 run name suffix from stage2 config
            try:
                stage2_config = _load_config(stage2_config_id)
                stage2_suffix = stage2_config.get("run_name_suffix", f"{exp_id}_stage2")
            except Exception:
                stage2_suffix = f"{exp_id}_stage2"

            try:
                stage2_runs_all = list(
                    api.runs(
                        f"nvidia-isaac/{_WANDB_PROJECT}",
                        filters={
                            "display_name": {"$regex": f"{re.escape(stage2_suffix)}_"}
                        },
                    )
                )
                stage2_runs = [
                    r for r in stage2_runs_all if r.created_at > submitted_after
                ]
            except Exception as e:
                _log(f"  WARNING: W&B query failed for {exp_id} stage2: {e}")
                stage2_runs = []

            for r in stage2_runs:
                steps = r.summary.get("_step", 0) or 0
                print(f"  stage2/{r.display_name}: {r.state} @ step {steps}")

            crashed = [
                r
                for r in stage2_runs
                if r.state in ("crashed", "failed")
                and min_steps <= (r.summary.get("_step", 0) or 0) < stage2_threshold
            ]

            stage2_processed_crash_ids: set[str] = set(
                exp_state.get("stage2_processed_crash_ids", [])
            )
            new_crashes = [r for r in crashed if r.id not in stage2_processed_crash_ids]

            if new_crashes:
                rerun_count = exp_state.get("stage2_rerun_count", 0)
                if rerun_count >= max_stage2_reruns:
                    _log(
                        f"  Stage2 max reruns ({max_stage2_reruns}) reached for {exp_id}, skipping"
                    )
                else:
                    ok = resume_crashed_stage2(
                        exp_id, stage2_config_id, image, pool, priority, dry_run
                    )
                    if ok or dry_run:
                        exp_state["stage2_rerun_count"] = rerun_count + 1
                        exp_state["stage2_last_rerun"] = _now_str()
                        for r in new_crashes:
                            stage2_processed_crash_ids.add(r.id)
                        exp_state["stage2_processed_crash_ids"] = sorted(
                            stage2_processed_crash_ids
                        )

    return state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the two-stage training monitor."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=_DEFAULT_STATE_FILE,
        help=f"State file path (default: {_DEFAULT_STATE_FILE})",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize state file and submit stage1 workflows",
    )
    parser.add_argument(
        "--exp-ids",
        nargs="+",
        help="Experiment IDs to monitor (required with --init)",
    )
    parser.add_argument(
        "--image", default=_DEFAULT_IMAGE, help="Docker image for OSMO tasks"
    )
    parser.add_argument("--pool", default=_DEFAULT_POOL, help="OSMO pool")
    parser.add_argument("--priority", default="NORMAL", help="OSMO priority")
    parser.add_argument(
        "--stage1-threshold",
        type=int,
        default=1000,
        help="Relaunch stage1 if it crashed before this many steps (default: 1000)",
    )
    parser.add_argument(
        "--stage2-threshold",
        type=int,
        default=10000,
        help="Relaunch stage2 if it crashed before this many steps (default: 10000)",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=50,
        help="Ignore crashes before this many steps (system error) (default: 50)",
    )
    parser.add_argument(
        "--build-image",
        action="store_true",
        help="Build and push Docker image before submitting",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without executing"
    )
    args = parser.parse_args()

    if args.init:
        if not args.exp_ids:
            parser.error("--exp-ids required with --init")
        if args.state_file.exists():
            _log(f"WARNING: State file {args.state_file} already exists. Overwriting.")
        state = {
            "exp_ids": args.exp_ids,
            "submitted_after": _now_str(),
            "stage1_threshold": args.stage1_threshold,
            "stage2_threshold": args.stage2_threshold,
            "min_steps": args.min_steps,
            "image": args.image,
            "pool": args.pool,
            "priority": args.priority,
            "build_image": args.build_image,
            "max_reruns_per_seq": 3,
            "max_stage2_reruns": 3,
            "experiments": {},
        }
        _log(f"Initialized state for experiments: {args.exp_ids}")
    else:
        state = load_state(args.state_file)
        if not state:
            print(
                f"Error: state file {args.state_file} not found. "
                "Run with --init first, or specify --state-file."
            )
            sys.exit(1)
        # Allow CLI overrides of image/pool/build-image
        if args.image != _DEFAULT_IMAGE:
            state["image"] = args.image
        if args.pool != _DEFAULT_POOL:
            state["pool"] = args.pool
        if args.build_image:
            state["build_image"] = True

    state = monitor_once(state, dry_run=args.dry_run)

    if not args.dry_run:
        save_state(state, args.state_file)
    else:
        _log("[DRY RUN] State not written.")


if __name__ == "__main__":
    main()
