#!/usr/bin/env python3
"""Unified experiment runner - run experiments locally or submit to OSMO.

SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Usage:
  python scripts/run_experiment.py exp5              # defaults to --osmo
  python scripts/run_experiment.py exp5 --local
  python scripts/run_experiment.py exp5 --osmo --pool isaac-dev-l40s-04 --build-image
  python scripts/run_experiment.py exp6 --wandb-sweep-create   # Create wandb sweep, print agent command
  python scripts/run_experiment.py exp7 --osmo                # Submit OSMO curriculum sweep
  python scripts/run_experiment.py --list                    # List all experiments
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RG_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = RG_ROOT / "experiments"
WORKFLOW_DIR = REPO_ROOT / "workflow"

if str(RG_ROOT) not in sys.path:
    sys.path.insert(0, str(RG_ROOT))
from experiments.utils import (  # noqa: E402
    build_train_command,
    make_entry_script,
    overrides_to_cli,
)


def load_registry() -> dict[str, str]:
    """Load experiment id -> directory mapping.

    Merges registry.yaml (committed) with registry.local.yaml (gitignored).
    Local entries take precedence, allowing private experiments to coexist with
    the committed examples without modifying tracked files.
    """
    reg_path = EXPERIMENTS_DIR / "registry.yaml"
    local_path = EXPERIMENTS_DIR / "registry.local.yaml"
    registry: dict[str, str] = {}
    if reg_path.exists():
        with open(reg_path, encoding="utf-8") as f:
            registry.update(yaml.safe_load(f) or {})
    if local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            registry.update(yaml.safe_load(f) or {})
    return registry


def load_experiment_config(exp_id: str) -> tuple[Path, dict]:
    """Load config for experiment. Returns (config_path, config dict)."""
    registry = load_registry()
    if exp_id not in registry:
        raise SystemExit(
            f"Unknown experiment: {exp_id}. Use --list to see available experiments."
        )
    exp_dir = EXPERIMENTS_DIR / registry[exp_id]
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config:
        raise SystemExit(f"Empty or invalid config: {config_path}")
    return config_path, config


def get_effective_overrides(
    config: dict, osmo: bool = False
) -> tuple[str, dict[str, str]]:
    """Get run_name and overrides for the experiment. Handles osmo overrides."""
    run_name = config.get("run_name", "experiment")
    overrides = dict(config.get("train_overrides", {}))
    if osmo and "osmo" in config:
        osmo_cfg = config["osmo"]
        if "run_name_overrides" in osmo_cfg:
            overrides.update(osmo_cfg["run_name_overrides"])
        if "run_name_suffix" in osmo_cfg:
            run_name = "${RUN_TS}" + osmo_cfg["run_name_suffix"]
    return run_name, overrides


def run_local(
    exp_id: str, config: dict, variant: str | None = None, dry_run: bool = False
) -> None:
    """Run experiment locally via train.py."""
    run_name = config.get("run_name", f"{exp_id}_run")
    overrides = dict(config.get("train_overrides", {}))
    resume_from = config.get("resume_from")
    seed = config.get("seed")
    motion_file = config.get("motion_file")
    max_iterations = config.get("max_iterations")
    no_collision = False

    # Stage 1 (osmo_multi_task): pick first sequence, derive motion_file, enable no-collision.
    if "osmo_multi_task" in config:
        seq_ids = config["osmo_multi_task"].get("sequence_ids", [])
        if seq_ids:
            motion_file = motion_file or f"arctic_processed/{seq_ids[0]}/sharpa_wave"
            no_collision = True

    # Stage 2 (sequences): pick first sequence, derive motion_file.
    elif "sequences" in config:
        seqs = config["sequences"]
        if seqs:
            motion_file = motion_file or f"arctic_processed/{seqs[0]}/sharpa_wave"

    # Apply variant-specific overrides from workflow.py if --variant is given.
    if variant is not None:
        workflow_path = EXPERIMENTS_DIR / exp_id / "workflow.py"
        if not workflow_path.exists():
            # Resolve experiment dir via registry (exp_id may differ from folder name)
            config_path, _ = load_experiment_config(exp_id)
            workflow_path = config_path.parent / "workflow.py"
        if workflow_path.exists():
            spec = importlib.util.spec_from_file_location("_workflow", workflow_path)
            if spec is None or spec.loader is None:
                print(
                    f"[WARNING] Could not load workflow module from {workflow_path}; "
                    "variant flag ignored."
                )
            else:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "get_variant_overrides"):
                    overrides.update(mod.get_variant_overrides(variant, config))
                else:
                    print(
                        f"[WARNING] workflow.py for {exp_id} has no get_variant_overrides(); "
                        "variant flag ignored."
                    )
        else:
            print(f"[WARNING] No workflow.py found for {exp_id}; variant flag ignored.")

    video = config.get("video", True)
    num_envs = config.get("num_envs")
    cmd = build_train_command(
        run_name,
        overrides,
        resume_from=resume_from,
        seed=seed,
        motion_file=motion_file,
        num_envs=num_envs,
        max_iterations=max_iterations,
        no_collision=no_collision,
        video=video,
    )
    cmd_str = " ".join(cmd)
    print(f"Running: {cmd_str}")
    if dry_run:
        return 0
    result = subprocess.run(cmd_str, shell=True, check=False)
    return result.returncode


def _is_local_path(path: str) -> bool:
    """True if path looks like a local file (not s3/http/gs)."""
    return not (
        path.startswith("s3://")
        or path.startswith("http://")
        or path.startswith("https://")
        or path.startswith("gs://")
    )


def generate_single_task_workflow(
    exp_id: str, config: dict, run_name: str, overrides: dict[str, str]
) -> str:
    """Generate OSMO workflow YAML for a single training task."""
    osmo_cfg = config.get("osmo", {})
    # OSMO runs in a fresh container; local checkpoint paths don't exist.
    # - osmo.resume_from: false -> train from scratch
    # - osmo.resume_from: "<local path>" -> upload via dataset input (localpath) at submit time
    # - osmo.resume_from: "<s3/http url>" -> use as-is (must be accessible from container)
    resume_from = osmo_cfg.get("resume_from")
    if resume_from is None:
        resume_from = config.get("resume_from")
    if resume_from is False or resume_from == "false":
        resume_from = None
    # Track if we're using localpath upload (add inputs block)
    checkpoint_localpath = None
    checkpoint_in_container = None
    if resume_from and _is_local_path(resume_from):
        checkpoint_localpath = resume_from
        # OSMO mounts dataset at {{input:0}}/dataset_name/filename for a single file
        dataset_name = "resume_checkpoint"
        filename = Path(resume_from).name
        checkpoint_in_container = f"{{{{input:0}}}}/{dataset_name}/{filename}"
    if "run_name_suffix" in osmo_cfg:
        run_name = osmo_cfg["run_name_suffix"].lstrip("_")
    if "run_name_overrides" in osmo_cfg:
        overrides = {**overrides, **osmo_cfg["run_name_overrides"]}
    seed = config.get("seed")
    video = config.get("video", True)
    motion_file = config.get("motion_file")
    num_envs = config.get("num_envs")
    entry = make_entry_script(
        run_name,
        overrides,
        resume_from=checkpoint_in_container if checkpoint_in_container else resume_from,
        seed=seed,
        motion_file=motion_file,
        num_envs=num_envs,
        video=video,
        use_timestamp=True,
    )
    # Escape for YAML literal block
    entry_indent = "\n".join("        " + line for line in entry.split("\n"))
    inputs_block = ""
    if checkpoint_localpath:
        inputs_block = f"""
    inputs:
    - dataset:
        name: resume_checkpoint
        localpath: {checkpoint_localpath!r}
"""
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_api_key:
        print(
            "[WARNING] WANDB_API_KEY not set in local environment — wandb will fail in the container"
        )
    return f"""# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Generated from experiments/ for {exp_id}
# When resume_from is a local path, OSMO uploads it at submit time via dataset localpath.

workflow:
  name: {{{{workflow_name}}}}
  resources:
    default:
      cpu: 6
      gpu: 1
      memory: 120Gi
      storage: 200Gi

  tasks:
  - name: train
    image: {{{{image}}}}
    command: [/bin/bash]
    args: [/tmp/entry.sh]
    environment:
      ACCEPT_EULA: Y
      OMNI_SERVER: omniverse://isaac-dev.ov.nvidia.com
      WANDB_API_KEY: {wandb_api_key}
{inputs_block}    files:
    - path: /tmp/entry.sh
      contents: |-
{entry_indent}

default-values:
  workflow_name: robotic_grounding_{exp_id}
  image: nvcr.io/nvstaging/isaac-amr/robotic-grounding:latest
"""


def _load_workflow_generator(exp_dir: Path) -> Callable[[str, dict], str] | None:
    """Load generate_workflow from experiment's workflow.py if it exists."""
    workflow_py = exp_dir / "workflow.py"
    if not workflow_py.exists():
        return None
    spec = importlib.util.spec_from_file_location("workflow", workflow_py)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "generate_workflow", None)


def _print_workflow(exp_id: str, config: dict) -> None:
    """Print the OSMO entry script (train command) for verification."""
    run_name, overrides = get_effective_overrides(config, osmo=True)
    if "osmo" in config and "run_name_suffix" in config["osmo"]:
        run_name = config["osmo"]["run_name_suffix"].lstrip("_")
    else:
        run_name = config.get("run_name", exp_id)
    osmo_cfg = config.get("osmo", {})
    resume_from = osmo_cfg.get("resume_from")
    if resume_from is None:
        resume_from = config.get("resume_from")
    if resume_from is False or resume_from == "false":
        resume_from = None
    seed = config.get("seed")
    video = config.get("video", True)
    motion_file = config.get("motion_file")
    num_envs = config.get("num_envs")
    if "run_name_overrides" in osmo_cfg:
        overrides = {**overrides, **osmo_cfg["run_name_overrides"]}
    entry = make_entry_script(
        run_name,
        overrides,
        resume_from=resume_from,
        seed=seed,
        motion_file=motion_file,
        num_envs=num_envs,
        video=video,
        use_timestamp=True,
    )
    print("# OSMO entry script (train command) for", exp_id)
    print("# Run: python scripts/run_experiment.py", exp_id, "--osmo")
    print()
    print(entry)


def _check_osmo_login() -> None:
    """Abort early if the OSMO CLI is not authenticated."""
    result = subprocess.run(
        ["osmo", "profile", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("[ERROR] OSMO is not logged in. Run: osmo login")
        print(result.stderr.strip())
        sys.exit(1)


def run_osmo(
    exp_id: str,
    config: dict,
    pool: str = "isaac-dev-l40s-04",
    build_image: bool = False,
    image: str | None = None,
    priority: str = "NORMAL",
    dry_run: bool = False,
) -> None:
    """Generate workflow YAML and submit to OSMO via run_osmo.py."""
    if not dry_run:
        _check_osmo_login()

    if "osmo_multi_task" in config:
        exp_dir = EXPERIMENTS_DIR / load_registry()[exp_id]
        generator = _load_workflow_generator(exp_dir)
        if generator is None:
            raise SystemExit(
                f"Experiment {exp_id} has osmo_multi_task but no workflow.py found in {exp_dir}"
            )
        workflow_content = generator(exp_id, config)
    else:
        run_name, overrides = get_effective_overrides(config, osmo=True)
        if "osmo" in config and "run_name_suffix" in config["osmo"]:
            run_name = config["osmo"]["run_name_suffix"].lstrip("_")
        else:
            run_name = config.get("run_name", exp_id)
        workflow_content = generate_single_task_workflow(
            exp_id, config, run_name, overrides
        )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        dir=REPO_ROOT,
    ) as f:
        f.write(workflow_content)
        workflow_path = Path(f.name)

    try:
        run_osmo_py = Path(__file__).resolve().parent / "run_osmo.py"
        exp_name = exp_id
        if exp_id == "exp6":
            exp_name = "exp6_contact_sweep"
        elif exp_id == "exp7":
            exp_name = "exp7_curriculum_sweep"
        elif exp_id == "exp10":
            exp_name = "exp10_sequence_parallel"
        elif exp_id == "exp11":
            exp_name = "exp11_zeroinit"
        cmd = [
            sys.executable,
            str(run_osmo_py),
            "--experiment-name",
            exp_name,
            "--workflow-yaml",
            str(workflow_path),
            "--pool",
            pool,
        ]
        if build_image:
            cmd.append("--build-image")
        if image:
            cmd.extend(["--image", image])
        cmd.extend(["--priority", priority])
        if dry_run:
            cmd.append("--dry-run")
        result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        sys.exit(result.returncode)
    finally:
        workflow_path.unlink(missing_ok=True)


def create_wandb_sweep(exp_id: str, config: dict) -> None:
    """Create wandb sweep for exp6 and print agent command."""
    if "wandb_sweep" not in config:
        raise SystemExit(f"Experiment {exp_id} does not support wandb sweep")
    ws = config["wandb_sweep"]
    param = ws["sweep_param"]
    values = ws["values"]
    project = ws.get("project", "v2p_hands")
    base = dict(config["train_overrides"])
    sweep_config = {
        "program": "scripts/rsl_rl/train.py",
        "method": "grid",
        "project": project,
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--headless",
            "--video",
            "--task",
            "Sharpa-V2P-v0",
            "--run_name",
            config.get("run_name", "experiment_6_contactTrackingSweep"),
            *overrides_to_cli(base),
        ],
        "parameters": {param: {"values": values}},
    }
    exp_dir = EXPERIMENTS_DIR / load_registry()[exp_id]
    sweep_path = exp_dir / "sweep_wandb.yaml"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_path, "w", encoding="utf-8") as f:
        yaml.dump(sweep_config, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {sweep_path}")
    print("Run: wandb sweep", sweep_path)
    print("Then: wandb agent <entity>/" + project + "/<sweep_id>")


# ---------------------------------------------------------------------------
# Two-stage pipeline orchestration (generic, driven entirely by config.yaml)
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(run_name: str) -> Path | None:
    """Find the highest-iteration checkpoint for a given RSL-RL run_name, or None.

    RSL-RL saves to logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/model_<N>.pt.
    We search recursively so we're not coupled to the experiment_name sub-dir.
    """
    import glob as _glob
    pattern = str(RG_ROOT / "logs" / "rsl_rl" / "**" / f"*_{run_name}" / "model_*.pt")
    checkpoints = _glob.glob(pattern, recursive=True)
    if not checkpoints:
        return None

    def _iteration(p: str) -> int:
        m = re.search(r"model_(\d+)\.pt", p)
        return int(m.group(1)) if m else -1

    return Path(max(checkpoints, key=_iteration))


def _fetch_stage1_artifact(stage2_config: dict) -> Path | None:
    """Download the stage1 W&B artifact for stage2 and return the local .pt path, or None."""
    import shutil
    import sys as _sys
    import tempfile
    _scripts = str(Path(__file__).resolve().parent)
    if _scripts not in _sys.path:
        _sys.path.insert(0, _scripts)
    try:
        from launch_stage2 import fetch_checkpoints as _fetch
    except ImportError:
        return None

    tmpdir_obj = tempfile.TemporaryDirectory()
    try:
        checkpoints = _fetch(Path(tmpdir_obj.name), stage2_config)
    except SystemExit:
        tmpdir_obj.cleanup()
        return None
    if not checkpoints:
        tmpdir_obj.cleanup()
        return None
    # Copy to a stable location under logs/ before tmpdir is cleaned up
    seq_key, tmp_path = next(iter(checkpoints.items()))
    stable_dir = RG_ROOT / "logs" / "rsl_rl" / "sharpa_v2p" / f"stage1_ckpt_{seq_key}"
    stable_dir.mkdir(parents=True, exist_ok=True)
    stable_path = stable_dir / tmp_path.name
    shutil.copy2(tmp_path, stable_path)
    tmpdir_obj.cleanup()
    return stable_path


def _run_pipeline_local(exp_id: str, config: dict, args) -> None:
    """Run the two-stage pipeline locally: stage1 → checkpoint → stage2.

    Resume logic (skips completed stages):
    - If --fresh: ignore existing stage2 checkpoint; start stage2 from stage1 checkpoint
      (downloads W&B artifact if no local stage1 checkpoint found)
    - If a stage2 checkpoint exists → resume stage2 from it (skip stage1 entirely)
    - Elif a stage1 checkpoint exists → skip stage1, run stage2 from stage1 checkpoint
    - Else → run stage1 from scratch, then stage2
    """
    stage1_exp_id = config.get("stage1_exp_id", "stage1")
    stage2_config_id = config.get("stage2_config_id")
    dry_run = getattr(args, "dry_run", False)
    fresh = getattr(args, "fresh", False)

    if not stage2_config_id:
        raise SystemExit("[pipeline:local] No stage2_config_id in pipeline config.")

    _, stage1_config = load_experiment_config(stage1_exp_id)
    _, stage2_config = load_experiment_config(stage2_config_id)
    stage2_config = dict(stage2_config)

    stage1_run_name = stage1_config.get("run_name", "stage1_nocoll")
    stage2_run_name = stage2_config.get("run_name", f"{stage2_config_id}_run")

    # --- Check for existing checkpoints ---
    stage2_ckpt = None if fresh else _find_latest_checkpoint(stage2_run_name)
    stage1_ckpt = _find_latest_checkpoint(stage1_run_name)

    if stage2_ckpt:
        print(f"\n[pipeline:local] Resuming stage 2 from existing checkpoint: {stage2_ckpt}")
        stage2_config["resume_from"] = str(stage2_ckpt)
    elif stage1_ckpt:
        print(f"\n[pipeline:local] Stage 1 checkpoint found, skipping stage 1: {stage1_ckpt}")
        stage2_config["resume_from"] = str(stage1_ckpt)
    elif fresh:
        # --fresh with no local stage1 checkpoint: download W&B artifact
        print(f"\n[pipeline:local] --fresh: fetching stage1 artifact from W&B...")
        artifact_ckpt = _fetch_stage1_artifact(stage2_config)
        if artifact_ckpt:
            print(f"[pipeline:local] Using stage1 artifact: {artifact_ckpt}")
            stage2_config["resume_from"] = str(artifact_ckpt)
        else:
            raise SystemExit(
                "[pipeline:local] --fresh: no local stage1 checkpoint and W&B artifact download failed."
            )
    else:
        # --- Run stage 1 from scratch ---
        print(f"\n[pipeline:local] Running stage 1 ({stage1_exp_id})...")
        exit_code = run_local(stage1_exp_id, stage1_config, dry_run=dry_run)
        if dry_run:
            print("[pipeline:local] Dry-run: skipping stage 2.")
            return
        if exit_code != 0:
            raise SystemExit(f"[pipeline:local] Stage 1 failed (exit {exit_code})")
        stage1_ckpt = _find_latest_checkpoint(stage1_run_name)
        if not stage1_ckpt:
            raise SystemExit("[pipeline:local] Stage 1 completed but no checkpoint found.")
        print(f"\n[pipeline:local] Stage 1 checkpoint: {stage1_ckpt}")
        stage2_config["resume_from"] = str(stage1_ckpt)

    # --- Stage 2 ---
    print(f"\n[pipeline:local] Running stage 2 ({stage2_config_id})...")
    exit_code = run_local(stage2_config_id, stage2_config, dry_run=dry_run)
    if exit_code != 0:
        raise SystemExit(f"[pipeline:local] Stage 2 failed (exit {exit_code})")
    print("[pipeline:local] Done.")


def _pipeline_submit_stage1(stage1_exp_id: str, pool: str, priority: str, build_image: bool, dry_run: bool, run_name_prefix: str = "", image: str | None = None) -> str | None:
    """Submit stage1 via run_experiment.py. Returns the OSMO workflow ID."""
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        stage1_exp_id, "--osmo",
        "--pool", pool,
        "--priority", priority,
    ]
    if build_image:
        cmd.append("--build-image")
    if image:
        cmd.extend(["--image", image])
    if dry_run:
        cmd.append("--dry-run")
    if run_name_prefix:
        cmd.extend(["--run-name-prefix", run_name_prefix])

    print(f"\n[pipeline] Submitting stage 1 ({stage1_exp_id})...")
    result = subprocess.run(cmd, cwd=RG_ROOT, capture_output=True, text=True)
    print(result.stdout, end="")
    print(result.stderr, end="")
    if result.returncode != 0:
        raise SystemExit(f"[pipeline] Stage 1 submission failed (exit {result.returncode})")

    if dry_run:
        print("[pipeline] Dry-run: skipping stage1 poll and stage2 launch.")
        return None

    match = re.search(r"Workflow ID\s+-\s+(\S+)", result.stdout + result.stderr)
    if match:
        return match.group(1)
    raise SystemExit("[pipeline] Could not determine stage1 workflow ID from OSMO output.")


def _pipeline_poll_until_done(workflow_id: str, poll_interval: int) -> None:
    """Block until the workflow reaches a terminal state."""
    terminal = {"COMPLETED", "SUCCEEDED", "FAILED", "FAILED_CANCELED", "CANCELED"}
    print(f"\n[pipeline] Polling workflow {workflow_id} every {poll_interval}s...")
    while True:
        result = subprocess.run(
            ["osmo", "workflow", "query", workflow_id],
            capture_output=True, text=True,
        )
        output = result.stdout + result.stderr
        wf_match = re.search(r"Status\s*:\s*(\S+)", output)
        wf_status = wf_match.group(1) if wf_match else "UNKNOWN"
        task_statuses = re.findall(r"\S+\s+\S+\s+(\S+)\s*$", output, re.MULTILINE)
        print(f"[pipeline]   {workflow_id}: {wf_status}  tasks: {task_statuses}")
        if wf_status in terminal:
            if wf_status not in {"SUCCEEDED", "COMPLETED"}:
                raise SystemExit(f"[pipeline] Stage 1 {wf_status} — aborting pipeline.")
            print("[pipeline] Stage 1 SUCCEEDED.")
            return
        time.sleep(poll_interval)


def _pipeline_generate_urdfs(stage1_exp_id: str) -> None:
    """Generate *_no_collision.urdf for all objects in the stage1 config."""
    from experiments.utils import generate_no_collision_urdfs  # noqa: E402
    registry = load_registry()
    if stage1_exp_id not in registry:
        raise SystemExit(f"[pipeline] Unknown stage1 exp id '{stage1_exp_id}'")
    config_path = EXPERIMENTS_DIR / registry[stage1_exp_id] / "config.yaml"
    with open(config_path) as f:
        stage1_config = yaml.safe_load(f)
    sequence_ids = stage1_config["osmo_multi_task"]["sequence_ids"]
    print("\n[pipeline] Step 0: generating no-collision URDFs...")
    generate_no_collision_urdfs(sequence_ids)


def run_pipeline(exp_id: str, config: dict, args) -> None:
    """Orchestrate the full two-stage pipeline (generic, config-driven)."""
    if getattr(args, "local", False):
        _run_pipeline_local(exp_id, config, args)
        return

    stage1_exp_id = config.get("stage1_exp_id", "stage1")
    stage2_exp_id = config.get("stage2_exp_id", "stage2_from_stage1")
    osmo_cfg = config.get("osmo", {})
    pool = getattr(args, "pool", None) or osmo_cfg.get("pool", "isaac-dev-l40s-04")
    priority = getattr(args, "priority", None) or osmo_cfg.get("priority", "NORMAL")
    build_image = getattr(args, "build_image", False) or osmo_cfg.get("build_image", False)
    image = getattr(args, "image", None) or osmo_cfg.get("image")
    # stage1_image / stage2_image allow separate overrides; fall back to shared image
    stage1_image = osmo_cfg.get("stage1_image") or image
    stage2_image = osmo_cfg.get("stage2_image") or image
    dry_run = getattr(args, "dry_run", False)
    poll_interval = config.get("poll_interval_seconds", 60)

    _pipeline_generate_urdfs(stage1_exp_id)

    # Build and push the image once before any submissions so both stages use
    # the same freshly built image and we don't redundantly rebuild mid-pipeline.
    if build_image and not dry_run:
        build_target = stage1_image or stage2_image or "nvcr.io/nvstaging/isaac-amr/robotic-grounding:v2d"
        image_version = build_target.split(":")[-1]
        print(f"\n[pipeline] Building and pushing Docker image: {build_target} ...")
        result = subprocess.run(f"./workflow/run.sh build {image_version}", shell=True, cwd=RG_ROOT)
        if result.returncode != 0:
            raise SystemExit("[pipeline] Docker build failed.")
        result = subprocess.run(f"./workflow/run.sh push {image_version}", shell=True, cwd=RG_ROOT)
        if result.returncode != 0:
            raise SystemExit("[pipeline] Docker push failed.")
        build_image = False  # skip rebuild in stage1/stage2 individual submissions

    workflow_id = _pipeline_submit_stage1(
        stage1_exp_id, pool, priority, build_image, dry_run,
        run_name_prefix=f"{exp_id}_",
        image=stage1_image,
    )
    if dry_run or workflow_id is None:
        return

    _pipeline_poll_until_done(workflow_id, poll_interval)

    stage2_config_id = config.get("stage2_config_id", stage2_exp_id)
    stage2_run_name_suffix = config.get("stage2_run_name_suffix")
    print(f"\n[pipeline] Launching stage 2 (config: {stage2_config_id}, job: {stage2_exp_id})...")
    cmd = [
        sys.executable, str(RG_ROOT / "scripts" / "launch_stage2.py"),
        "--config-id", stage2_config_id,
        "--exp-id", stage2_exp_id,
        "--pool", pool,
        "--priority", priority,
    ]
    if stage2_run_name_suffix:
        cmd.extend(["--run-name-suffix", stage2_run_name_suffix])
    if build_image:
        cmd.append("--build-image")
    if stage2_image:
        cmd.extend(["--image", stage2_image])
    result = subprocess.run(cmd, cwd=RG_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"[pipeline] Stage 2 launch failed (exit {result.returncode})")
    print("[pipeline] Done.")


def main() -> None:
    """Parse arguments and run experiment locally or on OSMO."""
    parser = argparse.ArgumentParser(description="Run experiments locally or on OSMO")
    parser.add_argument("exp_id", nargs="?", help="Experiment id (e.g. exp0, exp5)")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--local", action="store_true", help="Run locally via train.py")
    parser.add_argument("--osmo", action="store_true", help="Submit to OSMO")
    parser.add_argument(
        "--wandb-sweep-create",
        action="store_true",
        help="Create wandb sweep config (exp6 only)",
    )
    parser.add_argument(
        "--pool",
        default="isaac-dev-l40s-04",
        help="OSMO pool (default: isaac-dev-l40s-04)",
    )
    parser.add_argument(
        "--build-image", action="store_true", help="Build and push image before OSMO"
    )
    parser.add_argument("--image", help="Use specific Docker image for OSMO")
    parser.add_argument(
        "--priority",
        default="NORMAL",
        help="OSMO job priority (default: NORMAL)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print without executing"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="(pipeline --local only) Ignore existing stage2 checkpoint; start stage2 fresh from stage1 checkpoint (downloads W&B artifact if no local stage1 checkpoint found)",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="For multi-task experiments: name of workflow variant to run locally (e.g. hand_kp_w2)",
    )
    parser.add_argument(
        "--print-workflow",
        action="store_true",
        help="Print generated OSMO entry script (train command) without submitting",
    )
    parser.add_argument(
        "--run-name-prefix",
        default=None,
        help="Prefix to prepend to W&B run names (e.g. 'exp48_')",
    )
    args = parser.parse_args()

    if args.list:
        registry = load_registry()
        for eid, dirname in sorted(registry.items()):
            try:
                _, cfg = load_experiment_config(eid)
                desc = cfg.get("description", "")
                print(f"  {eid}: {desc} ({dirname})")
            except Exception:
                print(f"  {eid}: {dirname}")
        return

    if not args.exp_id:
        parser.error("exp_id required unless --list")
    if (
        not args.local
        and not args.osmo
        and not args.wandb_sweep_create
        and not args.print_workflow
    ):
        args.osmo = True

    _, config = load_experiment_config(args.exp_id)

    if args.run_name_prefix is not None:
        config["run_name_prefix"] = args.run_name_prefix

    if args.print_workflow:
        _print_workflow(args.exp_id, config)
        return

    # Pipeline experiments delegate all orchestration to run_pipeline().
    if config.get("pipeline"):
        run_pipeline(args.exp_id, config, args)
        return

    if args.local:
        sys.exit(run_local(args.exp_id, config, variant=args.variant, dry_run=args.dry_run))
    elif args.osmo:
        # Use --build-image from CLI, or osmo.build_image from config (e.g. exp9 needs it for contact_force)
        build_image = args.build_image or config.get("osmo", {}).get(
            "build_image", False
        )
        run_osmo(
            args.exp_id,
            config,
            pool=args.pool,
            build_image=build_image,
            image=args.image,
            priority=args.priority,
            dry_run=args.dry_run,
        )
    elif args.wandb_sweep_create:
        create_wandb_sweep(args.exp_id, config)


if __name__ == "__main__":
    main()
