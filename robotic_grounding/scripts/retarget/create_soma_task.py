#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scaffold a ReconBody experiment for a retargeted SOMA sequence.

Given a SOMA ``sequence_id`` (and optionally a robot name + soma subdir),
this script:

1. Writes ``experiments/<exp_id>/config.yaml`` from the same template used
   by the existing ``recon_body_*`` whole-body experiments (e.g.
   ``experiments/recon_body_snack_box_pick_and_place_01/config.yaml``).
2. Idempotently registers ``<exp_id>`` in ``experiments/registry.yaml``,
   inserting it under the ``# Whole-body experiments:`` section and
   preserving every existing comment / blank line.

The script is invoked by ``scripts/retarget/process_soma_sequence.sh``
(stage 6) but is also runnable standalone:

    python scripts/retarget/create_soma_task.py 2026-03-06_10-24-18_snack_box_pick_and_place_01

By default the experiment id is derived from the sequence id by stripping
the leading ``YYYY-MM-DD_HH-MM-SS_`` timestamp and prepending
``recon_body_`` (so the example above becomes
``recon_body_snack_box_pick_and_place_01``). Use ``--experiment-id`` to
override.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
REGISTRY_PATH = EXPERIMENTS_DIR / "registry.yaml"
DEFAULT_MOTION_ROOT_REL = (
    "source/robotic_grounding/robotic_grounding/assets/human_motion_data"
)
DEFAULT_SOMA_SUBDIR = "soma"
DEFAULT_ROBOT_NAME = "g1"

# Matches the leading "YYYY-MM-DD_HH-MM-SS_" timestamp on every known SOMA
# sequence id (e.g. "2026-03-06_10-24-18_snack_box_pick_and_place_01").
_TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_")

# Matches the "# Whole-body experiments" section header in registry.yaml.
_WHOLE_BODY_HEADER = re.compile(r"^\s*#\s*Whole-body experiments")


def default_experiment_id(sequence_id: str) -> str:
    """Derive ``recon_body_<short_name>`` from a SOMA sequence id.

    Strips the leading ``YYYY-MM-DD_HH-MM-SS_`` timestamp prefix when
    present; otherwise falls back to using the full sequence id as the
    short name.
    """
    short = _TIMESTAMP_PREFIX.sub("", sequence_id)
    return f"recon_body_{short}"


def motion_file_relpath(
    sequence_id: str,
    *,
    robot_name: str = DEFAULT_ROBOT_NAME,
    soma_subdir: str = DEFAULT_SOMA_SUBDIR,
    motion_root_rel: str = DEFAULT_MOTION_ROOT_REL,
) -> str:
    """Build the repo-relative motion-file partition path used by experiments."""
    return (
        f"{motion_root_rel}/whole_body/{soma_subdir}/"
        f"sequence_id={sequence_id}/robot_name={robot_name}"
    )


def render_config_yaml(
    *,
    exp_id: str,
    sequence_id: str,
    robot_name: str,
    soma_subdir: str,
    motion_root_rel: str,
) -> str:
    """Render the YAML body for ``experiments/<exp_id>/config.yaml``.

    Mirrors the format of the existing ``recon_body_*`` configs so newly
    scaffolded experiments are visually indistinguishable from
    hand-authored ones.
    """
    motion_file = motion_file_relpath(
        sequence_id,
        robot_name=robot_name,
        soma_subdir=soma_subdir,
        motion_root_rel=motion_root_rel,
    )
    return (
        f"# ReconBody: {exp_id} (auto-scaffolded from create_soma_task.py)\n"
        "#\n"
        "# Usage:\n"
        f"#   python robotic_grounding/experiments/run_experiment.py {exp_id} --local\n"
        f"id: {exp_id}\n"
        f'description: "ReconBody {exp_id} with SONIC JOINT_RESIDUAL"\n'
        f"run_name: {exp_id}\n"
        "task: SonicG1-ReconBody-v0\n"
        f"# Use the explicit repo-relative partition path because whole_body/{soma_subdir} does not\n"
        "# follow the 4-part dataset/dataset_retargeted/sequence_id/robot shorthand.\n"
        f"motion_file: {motion_file}\n"
        "video: true\n"
        "num_envs: 4096\n"
        "max_iterations: 20000\n"
        "zero_actor: true\n"
        "logger: wandb\n"
        "log_project_name: v2d-whole-body-tpv\n"
        "\n"
        "train_overrides:\n"
        "  env.commands.motion.reset_freeze_steps: 50\n"
        "  env.commands.motion.voc_decay_steps: 10\n"
        "  env.commands.motion.voc_reset_scale: 1.0\n"
        "  env.commands.motion.initial_virtual_object_control_curriculum_scale: 1.0\n"
        "  # Frame range of the source motion to train on. -1 for motion_end_frame\n"
        "  # means run to the end of the sequence.\n"
        "  # Use replay_motion_viser.py to visualize the motion and get the frame range.\n"
        "  env.commands.motion.motion_start_frame: 0\n"
        "  env.commands.motion.motion_end_frame: -1\n"
        "  env.episode_length_s: 5.0\n"
        "  # Reset the shoulder spread to 0.2\n"
        "  env.commands.motion.reset_shoulder_spread: 0.2\n"
        "\n"
        "osmo:\n"
        "  build_image: false\n"
    )


@dataclass(frozen=True)
class ConfigWriteResult:
    """Outcome of writing an experiment config.yaml."""

    path: Path
    # One of "wrote", "overwrote", "kept-existing".
    action: str


def write_experiment_config(
    *,
    exp_id: str,
    sequence_id: str,
    robot_name: str,
    soma_subdir: str,
    motion_root_rel: str,
    overwrite: bool,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> ConfigWriteResult:
    """Materialize ``experiments/<exp_id>/config.yaml``.

    Creates the parent directory if needed. When the config already
    exists, only overwrites it when ``overwrite=True`` so callers can
    keep the operation safe-by-default.
    """
    exp_dir = experiments_dir / exp_id
    exp_config = exp_dir / "config.yaml"
    if exp_config.exists() and not overwrite:
        return ConfigWriteResult(path=exp_config, action="kept-existing")

    exp_dir.mkdir(parents=True, exist_ok=True)
    body = render_config_yaml(
        exp_id=exp_id,
        sequence_id=sequence_id,
        robot_name=robot_name,
        soma_subdir=soma_subdir,
        motion_root_rel=motion_root_rel,
    )
    pre_existed = exp_config.exists()
    exp_config.write_text(body, encoding="utf-8")
    return ConfigWriteResult(
        path=exp_config,
        action="overwrote" if pre_existed else "wrote",
    )


@dataclass(frozen=True)
class RegistryUpdateResult:
    """Outcome of updating experiments/registry.yaml for a sequence."""

    path: Path
    # One of "noop", "updated-in-place", "inserted-in-section",
    # "appended-at-end".
    action: str
    previous_value: str | None = None


def _parse_top_level_value(line: str) -> str | None:
    """Return the right-hand value of a ``key: value`` line, ignoring comments.

    Returns ``None`` if the line is not a top-level mapping entry.
    """
    if not line or line.startswith((" ", "\t", "#")):
        return None
    if ":" not in line:
        return None
    rhs = line.split(":", 1)[1].strip()
    if "#" in rhs:
        rhs = rhs.split("#", 1)[0].strip()
    return rhs


def update_registry(
    *,
    exp_id: str,
    exp_dir_name: str | None = None,
    registry_path: Path = REGISTRY_PATH,
) -> RegistryUpdateResult:
    """Idempotently add / update ``exp_id`` in ``registry.yaml``.

    Behavior:
    - If ``registry.yaml`` already maps ``exp_id`` to ``exp_dir_name``,
      this is a no-op.
    - If ``exp_id`` exists with a different value, the line is rewritten
      in place (preserving every other line).
    - Otherwise the new entry is inserted at the end of the contiguous
      block of entries that follows the ``# Whole-body experiments:``
      section header. If that header is missing the entry is appended at
      EOF instead.
    """
    if exp_dir_name is None:
        exp_dir_name = exp_id

    text = registry_path.read_text(encoding="utf-8")
    # Preserve the trailing-newline policy of whatever was on disk so the
    # rewrite is minimally disruptive (matches the rest of the file's
    # formatting conventions).
    trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    key_prefix = f"{exp_id}:"

    found_idx = None
    found_value = None
    for i, line in enumerate(lines):
        if line.startswith(key_prefix):
            value = _parse_top_level_value(line)
            if value is not None:
                found_idx = i
                found_value = value
                break

    if found_idx is not None:
        if found_value == exp_dir_name:
            return RegistryUpdateResult(
                path=registry_path,
                action="noop",
                previous_value=found_value,
            )
        lines[found_idx] = f"{exp_id}: {exp_dir_name}"
        _write_lines(registry_path, lines, trailing_newline)
        return RegistryUpdateResult(
            path=registry_path,
            action="updated-in-place",
            previous_value=found_value,
        )

    new_line = f"{exp_id}: {exp_dir_name}"
    header_idx = next(
        (i for i, line in enumerate(lines) if _WHOLE_BODY_HEADER.match(line)),
        None,
    )

    if header_idx is not None:
        # Append at the end of the contiguous block of top-level entries
        # that follows the header (stop at first blank line or comment).
        insert_idx = header_idx + 1
        while insert_idx < len(lines):
            stripped = lines[insert_idx].strip()
            if stripped == "" or stripped.startswith("#"):
                break
            insert_idx += 1
        lines.insert(insert_idx, new_line)
        _write_lines(registry_path, lines, trailing_newline)
        return RegistryUpdateResult(path=registry_path, action="inserted-in-section")

    if lines and lines[-1].strip() != "":
        lines.append("")
    lines.append(new_line)
    _write_lines(registry_path, lines, trailing_newline)
    return RegistryUpdateResult(path=registry_path, action="appended-at-end")


def _write_lines(path: Path, lines: list[str], trailing_newline: bool) -> None:
    """Write ``lines`` back to ``path`` with a controlled trailing newline."""
    text = "\n".join(lines) + ("\n" if trailing_newline else "")
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Scaffold experiments/<exp_id>/config.yaml for a retargeted "
            "SOMA sequence and register it in experiments/registry.yaml."
        ),
    )
    parser.add_argument(
        "sequence_id",
        type=str,
        help=(
            "SOMA sequence id, e.g. "
            "'2026-03-06_10-24-18_snack_box_pick_and_place_01'."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help=(
            "Override the experiment id (also used as the directory name "
            "and run_name). Default: 'recon_body_<short_name>' where "
            "<short_name> is <sequence_id> with the leading "
            "YYYY-MM-DD_HH-MM-SS_ timestamp stripped."
        ),
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default=DEFAULT_ROBOT_NAME,
        help=f"Robot config name (default: {DEFAULT_ROBOT_NAME}).",
    )
    parser.add_argument(
        "--soma-subdir",
        type=str,
        default=DEFAULT_SOMA_SUBDIR,
        help=(
            "Schema subfolder under <motion-root>/whole_body "
            f"(default: {DEFAULT_SOMA_SUBDIR})."
        ),
    )
    parser.add_argument(
        "--motion-root-rel",
        type=str,
        default=DEFAULT_MOTION_ROOT_REL,
        help=(
            "Repo-relative motion-data root encoded into config.motion_file "
            f"(default: {DEFAULT_MOTION_ROOT_REL})."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "If the experiment config already exists, overwrite it. "
            "Without this flag, an existing config is kept and only the "
            "registry entry is reconciled."
        ),
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=REGISTRY_PATH,
        help=f"Path to registry.yaml (default: {REGISTRY_PATH}).",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=EXPERIMENTS_DIR,
        help=f"Path to experiments/ directory (default: {EXPERIMENTS_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)

    exp_id = args.experiment_id or default_experiment_id(args.sequence_id)

    print(f"[create_soma_task] sequence_id   : {args.sequence_id}")
    print(f"[create_soma_task] experiment_id : {exp_id}")
    print(f"[create_soma_task] robot_name    : {args.robot_name}")
    print(f"[create_soma_task] soma_subdir   : {args.soma_subdir}")

    cfg_result = write_experiment_config(
        exp_id=exp_id,
        sequence_id=args.sequence_id,
        robot_name=args.robot_name,
        soma_subdir=args.soma_subdir,
        motion_root_rel=args.motion_root_rel,
        overwrite=args.overwrite,
        experiments_dir=args.experiments_dir,
    )
    if cfg_result.action == "wrote":
        print(f"[create_soma_task] wrote {cfg_result.path}")
    elif cfg_result.action == "overwrote":
        print(f"[create_soma_task] overwrote {cfg_result.path}")
    else:
        print(
            f"[create_soma_task] kept existing {cfg_result.path} "
            "(pass --overwrite to replace)"
        )

    reg_result = update_registry(
        exp_id=exp_id,
        registry_path=args.registry_path,
    )
    if reg_result.action == "noop":
        print(
            f"[create_soma_task] registry already has '{exp_id}: {exp_id}'; "
            "nothing to do."
        )
    elif reg_result.action == "updated-in-place":
        print(
            f"[create_soma_task] updated existing entry to '{exp_id}: {exp_id}' "
            f"(was '{reg_result.previous_value}')."
        )
    elif reg_result.action == "inserted-in-section":
        print(
            f"[create_soma_task] inserted '{exp_id}: {exp_id}' under the "
            "Whole-body experiments block in "
            f"{reg_result.path}."
        )
    else:
        print(
            f"[create_soma_task] appended '{exp_id}: {exp_id}' at end of "
            f"{reg_result.path} (no Whole-body header found)."
        )

    print(
        "[create_soma_task] launch with: "
        f"python robotic_grounding/experiments/run_experiment.py {exp_id} --local"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
