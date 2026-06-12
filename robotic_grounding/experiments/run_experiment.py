#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import subprocess
import sys
import tempfile
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
    DEFAULT_OSMO_IMAGE_LATEST,
    DEFAULT_OSMO_IMAGE_REPO,
    DEFAULT_WANDB_ENTITY,
    build_eval_command,
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
    exp_id: str,
    config: dict,
    variant: str | None = None,
    dry_run: bool = False,
    *,
    num_envs_override: int | None = None,
    max_iterations_override: int | None = None,
    logger_override: str | None = None,
    extra_overrides: dict[str, str] | None = None,
) -> int:
    """Run experiment locally via train.py."""
    run_name = config.get("run_name", f"{exp_id}_run")
    overrides = dict(config.get("train_overrides", {}))
    if extra_overrides:
        overrides.update(extra_overrides)
    resume_from = config.get("resume_from")
    seed = config.get("seed")
    motion_file = config.get("motion_file")
    max_iterations = (
        max_iterations_override
        if max_iterations_override is not None
        else config.get("max_iterations")
    )
    # Stage 1 (osmo_multi_task): pick first sequence, derive motion_file.
    if "osmo_multi_task" in config:
        seq_ids = config["osmo_multi_task"].get("sequence_ids", [])
        if seq_ids:
            motion_file = (
                motion_file or f"arctic/arctic_processed/{seq_ids[0]}/sharpa_wave"
            )

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
    num_envs = (
        num_envs_override if num_envs_override is not None else config.get("num_envs")
    )
    logger = logger_override if logger_override is not None else config.get("logger")
    cmd = build_train_command(
        run_name,
        overrides,
        resume_from=resume_from,
        seed=seed,
        motion_file=motion_file,
        num_envs=num_envs,
        max_iterations=max_iterations,
        video=video,
        task=config.get("task", "Sharpa-V2P-v0"),
        logger=logger,
        log_project_name=config.get("log_project_name"),
        zero_actor=config.get("zero_actor", False),
    )
    cmd_str = " ".join(cmd)
    print(f"Running: {cmd_str}")
    if dry_run:
        return 0
    result = subprocess.run(cmd_str, shell=True, check=False)
    return result.returncode


def run_eval(
    exp_id: str,
    config: dict,
    *,
    checkpoint: str | None = None,
    num_envs: int | None = None,
    video: bool = False,
    video_length: int | None = None,
    eval_episodes: int | None = None,
    real_time: bool = False,
    extra_overrides: dict[str, str] | None = None,
    dry_run: bool = False,
) -> int:
    """Run eval.py for the given experiment using its config.yaml as source of truth.

    Pulls motion_file, task, and the same train_overrides block used by training
    so train + eval stay in sync (frame range, freeze steps, etc.). CLI flags
    layered on top via run_experiment.py override config values.
    """
    motion_file = config.get("motion_file")
    overrides = dict(config.get("train_overrides", {}))
    if extra_overrides:
        overrides.update(extra_overrides)

    if num_envs is None:
        num_envs = config.get("num_envs")
    seed = config.get("seed")

    cmd = build_eval_command(
        overrides,
        checkpoint=checkpoint,
        seed=seed,
        motion_file=motion_file,
        num_envs=num_envs,
        video=video,
        video_length=video_length,
        eval_episodes=eval_episodes,
        task=config.get("task", "Sharpa-V2P-v0"),
        logger=config.get("logger"),
        log_project_name=config.get("log_project_name"),
        use_primitive_urdfs=config.get("use_primitive_urdfs", False),
        real_time=real_time,
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
    storage_gi = osmo_cfg.get("storage_gi", 200)
    # OSMO runs in a fresh container; local checkpoint paths don't exist.
    # - osmo.resume_from: false -> train from scratch
    # - osmo.resume_from: "<local path>" -> upload via dataset input (localpath) at submit time
    # - osmo.resume_from: "<s3/http url>" -> use as-is (must be accessible from container)
    motion_data_url = osmo_cfg.get("motion_data_url")
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
    eval_video_only = config.get("eval_video_only", False)
    video_length = config.get("video_length")
    video_interval = config.get("video_interval")
    eval_episodes_per_save = config.get("eval_episodes_per_save", 0)
    motion_file = config.get("motion_file")
    num_envs = config.get("num_envs")
    max_iterations = config.get("max_iterations")
    task = config.get("task", "Sharpa-V2P-v0")
    zero_actor = config.get("zero_actor", False)
    use_primitive_urdfs = config.get("use_primitive_urdfs", False)
    logger = config.get("logger", "wandb")
    log_project_name = config.get("wandb_project") or config.get(
        "log_project_name", "v2p_hands"
    )
    # Derive motion_file from OSMO dataset path when motion_data_url is set, matching
    # the behaviour of generate_multi_sequence_workflow so single-sequence relaunch
    # batches use the same path format as the original multi-sequence workflow.
    urdfs_src_path = None
    if (
        motion_file is None
        and motion_data_url
        and "sequences" in config
        and config["sequences"]
    ):
        seq_id = config["sequences"][0]
        dataset_name = motion_data_url.rstrip("/").split("/")[-1]
        sequences_subfolder = osmo_cfg.get("sequences_subfolder", "arctic_processed")
        if sequences_subfolder == "arctic_processed":
            dataset_seq_id = seq_id.replace("arctic_", "dataset_", 1)
        else:
            dataset_seq_id = seq_id
        motion_file = f"{{{{input:0}}}}/{dataset_name}/{sequences_subfolder}/sequence_id={dataset_seq_id}/robot_name=sharpa_wave"
        urdfs_subfolder = osmo_cfg.get("urdfs_subfolder")
        if urdfs_subfolder:
            urdfs_src_path = f"{{{{input:0}}}}/{dataset_name}/{urdfs_subfolder}"

    entry = make_entry_script(
        run_name,
        overrides,
        resume_from=checkpoint_in_container if checkpoint_in_container else resume_from,
        seed=seed,
        motion_file=motion_file,
        num_envs=num_envs,
        max_iterations=max_iterations,
        video=video,
        eval_video_only=eval_video_only,
        video_length=video_length,
        video_interval=video_interval,
        eval_episodes_per_save=eval_episodes_per_save,
        task=task,
        logger=logger,
        log_project_name=log_project_name,
        zero_actor=zero_actor,
        use_primitive_urdfs=use_primitive_urdfs,
        use_timestamp=True,
        urdfs_src_path=urdfs_src_path,
    )
    # Escape for YAML literal block
    entry_indent = "\n".join("        " + line for line in entry.split("\n"))
    inputs_entries = []
    if checkpoint_localpath:
        inputs_entries.append(
            f"    - dataset:\n        name: resume_checkpoint\n        localpath: {checkpoint_localpath!r}"
        )
    if motion_data_url:
        dataset_name = motion_data_url.rstrip("/").split("/")[-1]
        inputs_entries.append(f"    - dataset:\n        name: {dataset_name}")
    inputs_block = (
        ("\n    inputs:\n" + "\n".join(inputs_entries) + "\n") if inputs_entries else ""
    )
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_api_key:
        print(
            "[WARNING] WANDB_API_KEY not set in local environment — wandb will fail in the container"
        )
    # Default to the shared NVIDIA team entity so OSMO runs don't land in a submitter's
    # personal workspace. Individual experiments can override via config `wandb_entity`.
    wandb_entity = config.get("wandb_entity", DEFAULT_WANDB_ENTITY)
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
      storage: {storage_gi}Gi

  tasks:
  - name: train
    image: {{{{image}}}}
    command: [/bin/bash]
    args: [/tmp/entry.sh]
    environment:
      ACCEPT_EULA: Y
      OMNI_SERVER: omniverse://isaac-dev.ov.nvidia.com
      WANDB_API_KEY: {wandb_api_key}
      WANDB_ENTITY: {wandb_entity}
{inputs_block}    files:
    - path: /tmp/entry.sh
      contents: |-
{entry_indent}

default-values:
  workflow_name: robotic_grounding_{exp_id}
  image: {DEFAULT_OSMO_IMAGE_LATEST}
"""


def _seq_to_key(seq_id: str) -> str:
    """Strip the 'arctic_' dataset prefix, keeping subject + action for uniqueness.

    E.g. 'arctic_s01_capsulemachine_grab_01' -> 's01_capsulemachine_grab_01'.
    Including the subject avoids collisions when the same object/action appears
    under multiple subjects (e.g. s02_waffleiron_use_01 vs s07_waffleiron_use_01).
    """
    prefix = "arctic_"
    return seq_id[len(prefix) :] if seq_id.startswith(prefix) else seq_id


def generate_multi_sequence_workflow(
    exp_id: str, config: dict, overrides: dict[str, str], workflow_label: str = ""
) -> str:
    """Generate a multi-task OSMO workflow for single-stage configs with multiple sequences.

    Creates one task per sequence — the same pattern stage2 uses, but without checkpoint inputs.
    """
    sequences = config.get("sequences", [])
    osmo_cfg = config.get("osmo", {})
    storage_gi = osmo_cfg.get("storage_gi", 200)
    motion_data_url = osmo_cfg.get("motion_data_url")
    run_name_suffix = config.get("run_name_suffix", exp_id)
    project = config.get("wandb_project", "v2p_hands")
    eval_video_only = config.get("eval_video_only", False)
    video = config.get("video", True)
    video_length = config.get("video_length")
    video_interval = config.get("video_interval")
    eval_episodes_per_save = config.get("eval_episodes_per_save", 0)
    seed = config.get("seed")
    num_envs = config.get("num_envs")
    task = config.get("task", "Sharpa-V2P-v0")
    use_primitive_urdfs = config.get("use_primitive_urdfs", False)
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_api_key:
        print(
            "[WARNING] WANDB_API_KEY not set in local environment — wandb will fail in the container"
        )

    tasks = []
    for seq_id in sequences:
        seq_key = _seq_to_key(seq_id)
        run_name = f"{run_name_suffix}_{seq_key}"

        if motion_data_url:
            dataset_name = motion_data_url.rstrip("/").split("/")[-1]
            sequences_subfolder = osmo_cfg.get(
                "sequences_subfolder", "arctic_processed"
            )
            if sequences_subfolder == "arctic_processed":
                dataset_seq_id = seq_id.replace("arctic_", "dataset_", 1)
            else:
                dataset_seq_id = seq_id
            motion_file = f"{{{{input:0}}}}/{dataset_name}/{sequences_subfolder}/sequence_id={dataset_seq_id}/robot_name=sharpa_wave"
            inputs_block = (
                "\n    inputs:\n" "    - dataset:\n" f"        name: {dataset_name}\n"
            )
            urdfs_subfolder = osmo_cfg.get("urdfs_subfolder")
            urdfs_src_path = (
                f"{{{{input:0}}}}/{dataset_name}/{urdfs_subfolder}"
                if urdfs_subfolder
                else None
            )
        else:
            motion_file = f"arctic/arctic_processed/{seq_id}/sharpa_wave"
            inputs_block = ""
            urdfs_src_path = None

        entry = make_entry_script(
            run_name,
            overrides,
            seed=seed,
            motion_file=motion_file,
            num_envs=num_envs,
            video=video,
            eval_video_only=eval_video_only,
            video_length=video_length,
            video_interval=video_interval,
            eval_episodes_per_save=eval_episodes_per_save,
            task=task,
            logger="wandb",
            log_project_name=project,
            use_primitive_urdfs=use_primitive_urdfs,
            use_timestamp=True,
            urdfs_src_path=urdfs_src_path,
        )
        entry_indent = "\n".join("        " + line for line in entry.split("\n"))
        task_name = f"train-{seq_key.replace('_', '-')}"
        tasks.append(
            f"  - name: {task_name}\n"
            f"    image: {{{{image}}}}\n"
            f"    command: [/bin/bash]\n"
            f"    args: [/tmp/entry.sh]\n"
            f"    environment:\n"
            f"      ACCEPT_EULA: Y\n"
            f"      OMNI_SERVER: omniverse://isaac-dev.ov.nvidia.com\n"
            f"      WANDB_API_KEY: {wandb_api_key}\n"
            f"{inputs_block}"
            f"    files:\n"
            f"    - path: /tmp/entry.sh\n"
            f"      contents: |-\n"
            f"{entry_indent}"
        )

    tasks_str = "\n".join(tasks)
    return (
        f"# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        f"# SPDX-License-Identifier: Apache-2.0\n"
        f"# Generated from experiments/ for {exp_id}\n"
        f"\n"
        f"workflow:\n"
        f"  name: {{{{workflow_name}}}}\n"
        f"  resources:\n"
        f"    default:\n"
        f"      cpu: 6\n"
        f"      gpu: 1\n"
        f"      memory: 120Gi\n"
        f"      storage: {storage_gi}Gi\n"
        f"\n"
        f"  tasks:\n"
        f"{tasks_str}\n"
        f"\n"
        f"default-values:\n"
        f"  workflow_name: robotic_grounding_{exp_id}{'_' + workflow_label if workflow_label else ''}\n"
        f"  image: nvcr.io/nvstaging/isaac-amr/robotic-grounding:latest\n"
    )


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
    eval_video_only = config.get("eval_video_only", False)
    video_length = config.get("video_length")
    video_interval = config.get("video_interval")
    eval_episodes_per_save = config.get("eval_episodes_per_save", 0)
    motion_file = config.get("motion_file")
    num_envs = config.get("num_envs")
    max_iterations = config.get("max_iterations")
    task = config.get("task", "Sharpa-V2P-v0")
    zero_actor = config.get("zero_actor", False)
    use_primitive_urdfs = config.get("use_primitive_urdfs", False)
    logger = config.get("logger", "wandb")
    log_project_name = config.get("log_project_name", "v2p_hands")
    if "run_name_overrides" in osmo_cfg:
        overrides = {**overrides, **osmo_cfg["run_name_overrides"]}
    entry = make_entry_script(
        run_name,
        overrides,
        resume_from=resume_from,
        seed=seed,
        motion_file=motion_file,
        num_envs=num_envs,
        max_iterations=max_iterations,
        video=video,
        eval_video_only=eval_video_only,
        video_length=video_length,
        video_interval=video_interval,
        eval_episodes_per_save=eval_episodes_per_save,
        task=task,
        logger=logger,
        log_project_name=log_project_name,
        zero_actor=zero_actor,
        use_primitive_urdfs=use_primitive_urdfs,
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
    workflow_label: str = "",
) -> None:
    """Generate workflow YAML and submit to OSMO via run_osmo.py."""
    if not dry_run:
        _check_osmo_login()

    # When the user asks to build but doesn't pin an explicit image tag, derive one from
    # exp_id so the tag produced by run_osmo.py matches the tag baked into the workflow
    # YAML below. Without this, --build-image silently builds and pushes a new tag while
    # the submitted workflow still references :latest, which causes OSMO to run a stale
    # image and e.g. miss newly-registered gym task IDs (see SonicG1-ReconBody-v0).
    if build_image and image is None:
        image = f"{DEFAULT_OSMO_IMAGE_REPO}:{exp_id}"
        print(f"[INFO] --build-image without --image: pinning to {image}")

    if "osmo_multi_task" in config:
        exp_dir = EXPERIMENTS_DIR / load_registry()[exp_id]
        generator = _load_workflow_generator(exp_dir)
        if generator is None:
            raise SystemExit(
                f"Experiment {exp_id} has osmo_multi_task but no workflow.py found in {exp_dir}"
            )
        workflow_content = generator(exp_id, config)
    elif "sequences" in config and len(config["sequences"]) > 1:
        _, overrides = get_effective_overrides(config, osmo=True)
        workflow_content = generate_multi_sequence_workflow(
            exp_id, config, overrides, workflow_label
        )
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
        run_osmo_py = RG_ROOT / "scripts" / "run_osmo.py"
        exp_name = exp_id
        if exp_id == "exp6":
            exp_name = "exp6_contact_sweep"
        elif exp_id == "exp7":
            exp_name = "exp7_curriculum_sweep"
        elif exp_id == "exp10":
            exp_name = "exp10_sequence_parallel"
        elif exp_id == "exp11":
            exp_name = "exp11_zeroinit"
        if workflow_label:
            exp_name = f"{exp_name}_{workflow_label}"
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


def main() -> None:
    """Parse arguments and run experiment locally or on OSMO."""
    parser = argparse.ArgumentParser(description="Run experiments locally or on OSMO")
    parser.add_argument("exp_id", nargs="?", help="Experiment id (e.g. exp0, exp5)")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--local", action="store_true", help="Run locally via train.py")
    parser.add_argument("--osmo", action="store_true", help="Submit to OSMO")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run eval.py locally with the experiment's motion_file + train_overrides.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="(--eval) Path to the policy checkpoint to evaluate.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="(--eval) Override num_envs from config.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="(--eval) Record an evaluation video.",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=None,
        help="(--eval) Number of steps in the recorded video.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="(--eval) Stop after this many completed episodes and print stats.",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="(--eval) Run eval at real time, sleeping between sim steps.",
    )
    parser.add_argument(
        "--wandb-sweep-create",
        action="store_true",
        help="Create wandb sweep config (exp6 only)",
    )
    parser.add_argument(
        "--pool",
        default=None,
        help="OSMO pool (default: from config osmo.pool, or isaac-dev-l40s-04)",
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
        "--run-name",
        default=None,
        help="Override the config's run_name (the W&B run name will be {timestamp}_{run_name}).",
    )
    parser.add_argument(
        "--run-name-prefix",
        default=None,
        help="Prefix to prepend to W&B run names (e.g. 'exp48_')",
    )
    parser.add_argument(
        "--workflow-label",
        default="",
        help="Label appended to the OSMO workflow name for descriptive identification (e.g. 'init', 'rerun_2')",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="(--local) Override max_iterations from config (useful for smoke tests).",
    )
    parser.add_argument(
        "--logger",
        default=None,
        help="(--local) Override logger from config, e.g. tensorboard for offline smoke tests.",
    )
    parser.add_argument(
        "-O",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Hydra-style override appended to train_overrides / eval overrides. Repeatable. "
        "Example: -O env.episode_length_s=2.0 -O env.commands.motion.voc_decay_steps=2",
    )
    args = parser.parse_args()

    if args.list:
        registry = load_registry()
        for eid, dirname in sorted(registry.items()):
            config_path = EXPERIMENTS_DIR / dirname / "config.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                desc = cfg.get("description", "")
                print(f"  {eid}: {desc} ({dirname})")
            else:
                print(f"  {eid}: missing config ({dirname})")
        return

    if not args.exp_id:
        parser.error("exp_id required unless --list")
    if (
        not args.local
        and not args.osmo
        and not args.eval
        and not args.wandb_sweep_create
        and not args.print_workflow
    ):
        args.osmo = True

    _, config = load_experiment_config(args.exp_id)

    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.run_name_prefix is not None:
        config["run_name_prefix"] = args.run_name_prefix

    extra_overrides: dict[str, str] = {}
    for item in args.override:
        if "=" not in item:
            parser.error(f"--override expects KEY=VAL, got: {item!r}")
        k, v = item.split("=", 1)
        extra_overrides[k.strip()] = v.strip()

    if args.print_workflow:
        _print_workflow(args.exp_id, config)
        return

    if args.eval:
        sys.exit(
            run_eval(
                args.exp_id,
                config,
                checkpoint=args.checkpoint,
                num_envs=args.num_envs,
                video=args.video,
                video_length=args.video_length,
                eval_episodes=args.eval_episodes,
                real_time=args.real_time,
                extra_overrides=extra_overrides or None,
                dry_run=args.dry_run,
            )
        )
    if args.local:
        sys.exit(
            run_local(
                args.exp_id,
                config,
                variant=args.variant,
                dry_run=args.dry_run,
                num_envs_override=args.num_envs,
                max_iterations_override=args.max_iterations,
                logger_override=args.logger,
                extra_overrides=extra_overrides or None,
            )
        )
    elif args.osmo:
        # Use --build-image from CLI, or osmo.build_image from config (e.g. exp9 needs it for contact_force)
        osmo_cfg = config.get("osmo", {})
        build_image = args.build_image or osmo_cfg.get("build_image", False)
        # Image precedence: CLI --image > config osmo.image > (auto-derived from exp_id
        # when --build-image and nothing else is set, see run_osmo) > workflow YAML default.
        image = args.image or osmo_cfg.get("image")
        pool = args.pool or osmo_cfg.get("pool", "isaac-dev-l40s-04")
        priority = args.priority or osmo_cfg.get("priority", "NORMAL")
        run_osmo(
            args.exp_id,
            config,
            pool=pool,
            build_image=build_image,
            image=image,
            priority=priority,
            dry_run=args.dry_run,
            workflow_label=args.workflow_label,
        )
    elif args.wandb_sweep_create:
        create_wandb_sweep(args.exp_id, config)


if __name__ == "__main__":
    main()
