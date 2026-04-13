"""Pipeline orchestrator for example_two_stage.

Runs the full two-stage training pipeline:
  Stage 1 — submit collision-free warm-up OSMO job (exp stage1, collision disabled via Hydra override)
  Wait    — poll OSMO until all stage1 tasks finish
  Stage 2 — fetch W&B artifacts, submit exp44-style OSMO job (launch_stage2.py)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

_RG_ROOT = Path(__file__).resolve().parent.parent.parent  # robotic_grounding/
_EXPERIMENTS_DIR = _RG_ROOT / "experiments"
_SCRIPTS_DIR = _RG_ROOT / "scripts"
_RG_SCRIPTS_DIR = _RG_ROOT / "scripts"

if str(_RG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RG_ROOT))


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


# ---------------------------------------------------------------------------
# OSMO helpers
# ---------------------------------------------------------------------------


def _submit_stage1(
    stage1_exp_id: str, pool: str, priority: str, build_image: bool, dry_run: bool
) -> str | None:
    """Submit stage1 via run_experiment.py. Returns the OSMO workflow ID."""
    cmd = [
        sys.executable,
        str(_RG_SCRIPTS_DIR / "run_experiment.py"),
        stage1_exp_id,
        "--osmo",
        "--pool",
        pool,
        "--priority",
        priority,
    ]
    if build_image:
        cmd.append("--build-image")
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n[pipeline] Submitting stage 1 ({stage1_exp_id})...")
    result = subprocess.run(
        cmd, cwd=_RG_ROOT, capture_output=True, text=True, check=False
    )
    print(result.stdout, end="")
    print(result.stderr, end="")
    if result.returncode != 0:
        raise SystemExit(
            f"[pipeline] Stage 1 submission failed (exit {result.returncode})"
        )

    if dry_run:
        print("[pipeline] Dry-run: skipping stage1 poll and stage2 launch.")
        return None

    # Extract workflow ID from the submission output
    match = re.search(r"Workflow ID\s+-\s+(\S+)", result.stdout + result.stderr)
    if match:
        return match.group(1)
    raise SystemExit(
        "[pipeline] Could not determine stage1 workflow ID from OSMO output."
    )


def _poll_until_done(workflow_id: str, poll_interval: int) -> None:
    """Block until all tasks in the workflow reach a terminal state."""
    terminal = {"SUCCEEDED", "FAILED", "FAILED_CANCELED", "CANCELED"}
    print(f"\n[pipeline] Polling workflow {workflow_id} every {poll_interval}s...")

    while True:
        result = subprocess.run(
            ["osmo", "workflow", "query", workflow_id],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr

        # Overall workflow status
        wf_match = re.search(r"Status\s*:\s*(\S+)", output)
        wf_status = wf_match.group(1) if wf_match else "UNKNOWN"

        # Per-task statuses
        task_statuses = re.findall(r"\S+\s+\S+\s+(\S+)\s*$", output, re.MULTILINE)

        print(f"[pipeline]   {workflow_id}: {wf_status}  tasks: {task_statuses}")

        if wf_status in terminal:
            if wf_status != "SUCCEEDED":
                raise SystemExit(f"[pipeline] Stage 1 {wf_status} — aborting pipeline.")
            print("[pipeline] Stage 1 SUCCEEDED.")
            return

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Entry point called by run_experiment.py
# ---------------------------------------------------------------------------


def run_pipeline(_exp_id: str, config: dict, args: argparse.Namespace) -> None:
    """Orchestrate the full two-stage pipeline."""
    stage1_exp_id = config.get("stage1_exp_id", "stage1")
    stage2_exp_id = config.get("stage2_exp_id", "stage2_from_stage1")
    osmo_cfg = config.get("osmo", {})
    pool = getattr(args, "pool", None) or osmo_cfg.get("pool", "isaac-dev-l40s-04")
    priority = getattr(args, "priority", None) or osmo_cfg.get("priority", "NORMAL")
    build_image = getattr(args, "build_image", False) or osmo_cfg.get(
        "build_image", False
    )
    dry_run = getattr(args, "dry_run", False)
    poll_interval = config.get("poll_interval_seconds", 60)

    # Stage 1
    workflow_id = _submit_stage1(stage1_exp_id, pool, priority, build_image, dry_run)
    if dry_run or workflow_id is None:
        return

    # Wait
    _poll_until_done(workflow_id, poll_interval)

    # Stage 2
    stage2_config_id = config.get("stage2_config_id", stage2_exp_id)
    print(
        f"\n[pipeline] Launching stage 2 (config: {stage2_config_id}, job: {stage2_exp_id})..."
    )
    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "launch_stage2.py"),
        "--config-id",
        stage2_config_id,
        "--exp-id",
        stage2_exp_id,
        "--pool",
        pool,
        "--priority",
        priority,
    ]
    if build_image:
        cmd.append("--build-image")
    result = subprocess.run(cmd, cwd=_RG_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"[pipeline] Stage 2 launch failed (exit {result.returncode})")
    print("[pipeline] Done.")
