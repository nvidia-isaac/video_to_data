#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Generic retarget (IK) entry point.

Dispatches to the correct dataset-specific retarget script using the
dataset registry. Replaces the per-dataset if-elif blocks in retarget.yaml.

Usage::

    python scripts/retarget/run_retarget.py --dataset taco --robot sharpa_wave \
        --input_dir /data/loaded --output_dir /data/processed --device cuda:0 --save

    # Same dataset, different robot (per-robot retargeter):
    python scripts/retarget/run_retarget.py --dataset arctic --robot dex3 ...
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Repo root is two levels up from scripts/retarget/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_module_from_path(script_path: str) -> ModuleType:
    """Dynamically import a Python module from a file path.

    Registers the module in ``sys.modules`` *before* executing it so that
    features like ``@dataclass`` (which look up ``sys.modules[cls.__module__]``
    during class construction) work correctly.
    """
    full_path = REPO_ROOT / script_path
    if not full_path.exists():
        raise FileNotFoundError(f"Retarget script not found: {full_path}")
    module_name = full_path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(full_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {full_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Parse --dataset, then delegate to the dataset's retarget script."""
    if "--dataset" not in sys.argv:
        print("Error: --dataset is required", file=sys.stderr)
        print(
            "Usage: python scripts/retarget/run_retarget.py --dataset <name> "
            "[--robot <robot>] [retarget args...]",
            file=sys.stderr,
        )
        sys.exit(1)

    idx = sys.argv.index("--dataset")
    if idx + 1 >= len(sys.argv):
        print("Error: --dataset requires a value", file=sys.stderr)
        sys.exit(1)

    dataset_name = sys.argv[idx + 1]
    remaining_argv = sys.argv[:idx] + sys.argv[idx + 2 :]

    robot_name = "sharpa_wave"
    if "--robot" in remaining_argv:
        ridx = remaining_argv.index("--robot")
        if ridx + 1 >= len(remaining_argv):
            print("Error: --robot requires a value", file=sys.stderr)
            sys.exit(1)
        robot_name = remaining_argv[ridx + 1]
        remaining_argv = remaining_argv[:ridx] + remaining_argv[ridx + 2 :]

    # Import registry
    source_dir = str(REPO_ROOT / "source" / "robotic_grounding")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    from robotic_grounding.retarget.dataset_registry import (  # noqa: PLC0415
        get_dataset_config,
    )

    config = get_dataset_config(dataset_name)
    if robot_name not in config.retarget_scripts:
        available = ", ".join(sorted(config.retarget_scripts)) or "<none>"
        print(
            f"Error: dataset '{dataset_name}' has no retargeter for robot "
            f"'{robot_name}'. Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    module = _load_module_from_path(config.retarget_scripts[robot_name])

    sys.argv = remaining_argv
    args = module.parse_args()
    module.main(args)


if __name__ == "__main__":
    main()
