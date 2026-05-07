#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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


def _consume_named_arg(name: str, default: str | None = None) -> str | None:
    """Pop ``--name <value>`` from ``sys.argv`` and return the value (or default)."""
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        print(f"Error: {name} requires a value", file=sys.stderr)
        sys.exit(1)
    value = sys.argv[idx + 1]
    del sys.argv[idx : idx + 2]
    return value


def main() -> None:
    """Parse --dataset/--robot, then delegate to the dataset's retarget script."""
    dataset_name = _consume_named_arg("--dataset")
    if dataset_name is None:
        print("Error: --dataset is required", file=sys.stderr)
        print(
            "Usage: python scripts/retarget/run_retarget.py --dataset <name> "
            "[--robot <robot>] [retarget args...]",
            file=sys.stderr,
        )
        sys.exit(1)

    robot_name = _consume_named_arg("--robot", default="sharpa_wave")

    # Import registry
    source_dir = str(REPO_ROOT / "source" / "robotic_grounding")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    from robotic_grounding.retarget.dataset_registry import (  # noqa: PLC0415
        get_dataset_config,
    )

    config = get_dataset_config(dataset_name)
    if not config.retarget_scripts:
        print(
            f"Error: dataset '{dataset_name}' has no retarget_scripts configured",
            file=sys.stderr,
        )
        sys.exit(1)
    if robot_name not in config.retarget_scripts:
        available = ", ".join(sorted(config.retarget_scripts))
        print(
            f"Error: dataset '{dataset_name}' has no retargeter for robot "
            f"'{robot_name}'. Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    module = _load_module_from_path(config.retarget_scripts[robot_name])

    args = module.parse_args()
    module.main(args)


if __name__ == "__main__":
    main()
