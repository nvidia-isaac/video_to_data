#!/usr/bin/env python3
"""Launch stage 2 training by fetching stage 1 checkpoints from W&B.

Stage 1 trains each object collision-free and uploads the final checkpoint as a
W&B artifact. This script:
  1. Loads stage 2 config from the experiments registry.
  2. Downloads per-object checkpoint artifacts from W&B.
  3. Generates an OSMO multi-task workflow and submits it via run_osmo.py.

Usage:
    python robotic_grounding/scripts/launch_stage2.py                          # uses 'stage2' from registry
    python robotic_grounding/scripts/launch_stage2.py --config-id my_stage2   # use a different config
    python robotic_grounding/scripts/launch_stage2.py --dry-run                # preview without submitting
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import wandb
import yaml

_ROBOTIC_GROUNDING_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _ROBOTIC_GROUNDING_DIR / "experiments"


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


def _load_stage2_config(config_id: str) -> dict:
    registry = _load_registry()
    if config_id not in registry:
        raise SystemExit(f"Unknown config id '{config_id}'. Run --list to see options.")
    config_path = _EXPERIMENTS_DIR / registry[config_id] / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _sequence_to_object(seq_id: str) -> str:
    parts = seq_id.split("_")
    return parts[2] if len(parts) >= 3 else seq_id


def _sequence_to_seq_key(seq_id: str) -> str:
    parts = seq_id.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else seq_id


def fetch_crashed_checkpoints(outdir: Path, config: dict) -> dict[str, Path]:
    """Download latest checkpoints from crashed stage2 W&B runs. Returns {seq_key: local_path}."""
    api = wandb.Api()
    project = config.get("wandb_project", "v2p_hands")
    run_name_suffix = config.get("run_name_suffix", "stage2")
    sequences = config["sequences"]
    checkpoints: dict[str, Path] = {}

    for seq_id in sequences:
        seq_key = _sequence_to_seq_key(seq_id)
        run_name = f"{run_name_suffix}_{seq_key}"
        runs = api.runs(
            f"nvidia-isaac/{project}",
            filters={"display_name": {"$regex": run_name}},
        )
        runs = sorted(runs, key=lambda r: r.created_at, reverse=True)
        crashed = [r for r in runs if r.state == "crashed"]
        if not crashed:
            print(f"  No crashed run found for {run_name}, skipping...")
            continue
        run = crashed[0]
        print(
            f"  Found crashed run: {run.display_name} (step={run.summary.get('_step', 'N/A')})",
            end=" ",
            flush=True,
        )

        files = [f for f in run.files() if f.name.endswith(".pt")]
        if not files:
            print(f"\n  WARNING: No .pt checkpoints in {run.display_name} — skipping")
            continue

        def _iteration(f: Any) -> int:
            m = re.search(r"model_(\d+)\.pt", f.name)
            return int(m.group(1)) if m else -1

        target = max(files, key=_iteration)
        seq_dir = outdir / seq_key
        seq_dir.mkdir(parents=True, exist_ok=True)
        run.file(target.name).download(root=str(seq_dir), replace=True)
        checkpoints[seq_key] = seq_dir / target.name
        print(f"→ {target.name}")

    return checkpoints


def fetch_checkpoints(outdir: Path, config: dict) -> dict[str, Path]:
    """Download stage1 checkpoint artifacts from W&B. Returns {seq_key: local_path}."""
    api = wandb.Api()
    # stage1_wandb_project allows stage2 to look for artifacts in a different project than
    # the one stage2 training runs are logged to (e.g. stage1 uploads to v2p_hands but
    # stage2 trains in v2p_hands).
    artifact_project = config.get("stage1_wandb_project", "v2p_hands")
    artifact_prefix = config.get("stage1_artifact_prefix", "checkpoint_stage1_nocoll_")
    sequences = config["sequences"]
    checkpoints: dict[str, Path] = {}

    for seq_id in sequences:
        seq_key = _sequence_to_seq_key(seq_id)
        obj = _sequence_to_object(seq_id)
        # Try per-sequence artifact first, fall back to per-object for compatibility
        # with stage1 runs that predate per-sequence artifact uploads.
        candidates = [
            f"{artifact_prefix}{seq_key}",
            f"{artifact_prefix}{obj}",
        ]
        artifact = None
        for artifact_name in candidates:
            full_name = f"{artifact_project}/{artifact_name}:latest"
            print(f"  Fetching {full_name} ...", end=" ", flush=True)
            try:
                artifact = api.artifact(full_name)
                print("ok")
                break
            except Exception:
                print("not found, trying fallback...")
        if artifact is None:
            print(f"FAILED: no artifact found for {seq_id} (tried {candidates})")
            sys.exit(1)
        try:
            seq_dir = outdir / seq_key
            seq_dir.mkdir(parents=True, exist_ok=True)
            artifact.download(root=str(seq_dir))
            pts = list(seq_dir.glob("*.pt"))
            if not pts:
                raise FileNotFoundError(f"No .pt file in {seq_dir}")
            checkpoints[seq_key] = pts[0]
            print(f"  → {pts[0].name}")
        except Exception as exc:
            print(f"FAILED: {exc}")
            sys.exit(1)
    return checkpoints


def _overrides_cli(overrides: dict) -> str:
    return " \\\n  ".join(f"{k}={v}" for k, v in overrides.items())


def _make_task_yaml(
    obj: str,
    seq_id: str,
    ckpt_path: Path,
    exp_id: str,
    config: dict,
    wandb_api_key: str,
) -> str:
    project = config.get("wandb_project", "v2p_hands")
    overrides = dict(config.get("train_overrides", {}))
    run_name_suffix = config.get("run_name_suffix", "stage2")
    seq_key = _sequence_to_seq_key(seq_id)
    run_name = f"{run_name_suffix}_{seq_key}"
    ckpt_filename = ckpt_path.name
    overrides_str = _overrides_cli(overrides)

    entry = f"""set -ex

python scripts/rsl_rl/train.py \\
  --headless \\
  --task Sharpa-V2P-v0 \\
  --run_name {run_name} \\
  --motion_file arctic_processed/{seq_id}/sharpa_wave \\
  --logger wandb \\
  --log_project_name {project} \\
  --resume --checkpoint "{{{{input:0}}}}/ckpt_{obj}/{ckpt_filename}" \\
  {overrides_str}"""

    entry_indent = "\n".join("        " + line for line in entry.split("\n"))
    return f"""  - name: train-{seq_key.replace("_", "-")}
    image: {{{{image}}}}
    command: [/bin/bash]
    args: [/tmp/entry.sh]
    environment:
      ACCEPT_EULA: Y
      OMNI_SERVER: omniverse://isaac-dev.ov.nvidia.com
      WANDB_API_KEY: {wandb_api_key}
    inputs:
    - dataset:
        name: ckpt_{obj}
        localpath: '{ckpt_path}'
    files:
    - path: /tmp/entry.sh
      contents: |-
{entry_indent}"""


def generate_workflow(exp_id: str, config: dict, checkpoints: dict[str, Path]) -> str:
    """Generate OSMO workflow YAML for stage 2 training tasks."""
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_api_key:
        print("[WARNING] WANDB_API_KEY not set — wandb will fail in the container")
    sequences = config["sequences"]
    tasks = []
    for seq_id in sequences:
        seq_key = _sequence_to_seq_key(seq_id)
        if seq_key not in checkpoints:
            continue
        tasks.append(
            _make_task_yaml(
                _sequence_to_object(seq_id),
                seq_id,
                checkpoints[seq_key],
                exp_id,
                config,
                wandb_api_key,
            )
        )
    return f"""# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Generated by scripts/launch_stage2.py from config id: {config.get('id', exp_id)}

workflow:
  name: {{{{workflow_name}}}}
  resources:
    default:
      cpu: 6
      gpu: 1
      memory: 120Gi
      storage: 200Gi

  tasks:
{chr(10).join(tasks)}

default-values:
  workflow_name: robotic_grounding_{exp_id}
  image: nvcr.io/nvstaging/isaac-amr/robotic-grounding:latest
"""


def main() -> None:
    """Parse arguments and launch stage 2 training."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config-id",
        default="stage2",
        help="Stage 2 experiment config id from registry (default: stage2)",
    )
    parser.add_argument(
        "--exp-id", default=None, help="Override OSMO job name (defaults to config id)"
    )
    parser.add_argument(
        "--run-name-suffix", default=None, help="Override run_name_suffix from config"
    )
    parser.add_argument("--pool", default="isaac-dev-l40s-04")
    parser.add_argument("--priority", default="NORMAL")
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument(
        "--image", default=None, help="Use specific Docker image for OSMO"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-crashed",
        action="store_true",
        help="Resume crashed stage2 runs from their latest W&B checkpoint instead of fetching stage1 artifacts",
    )
    args = parser.parse_args()

    config = _load_stage2_config(args.config_id)
    if args.run_name_suffix is not None:
        config["run_name_suffix"] = args.run_name_suffix
    exp_id = args.exp_id or args.config_id

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.resume_crashed:
            print("Fetching latest checkpoints from crashed stage2 runs...")
            checkpoints = fetch_crashed_checkpoints(Path(tmpdir), config)
        else:
            print("Fetching stage1 checkpoints from W&B...")
            checkpoints = fetch_checkpoints(Path(tmpdir), config)

        print("Generating stage2 workflow...")
        workflow_yaml = generate_workflow(exp_id, config, checkpoints)

        if args.dry_run:
            print("\n--- Workflow YAML ---")
            print(workflow_yaml)
            return

        workflow_path = _ROBOTIC_GROUNDING_DIR / f"tmp_{exp_id}.yaml"
        workflow_path.write_text(workflow_yaml)
        try:
            run_osmo_py = _ROBOTIC_GROUNDING_DIR / "scripts" / "run_osmo.py"
            cmd = [
                sys.executable,
                str(run_osmo_py),
                "--experiment-name",
                exp_id,
                "--workflow-yaml",
                str(workflow_path),
                "--pool",
                args.pool,
                "--priority",
                args.priority,
            ]
            if args.build_image or config.get("osmo", {}).get("build_image"):
                cmd.append("--build-image")
            image = args.image or config.get("osmo", {}).get("image")
            if image:
                cmd.extend(["--image", image])
            print(f"Submitting {exp_id} to OSMO...")
            result = subprocess.run(cmd, cwd=_ROBOTIC_GROUNDING_DIR, check=False)
            sys.exit(result.returncode)
        finally:
            workflow_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
