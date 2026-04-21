#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generic retarget (IK) entry point.

Dispatches to the correct dataset-specific retarget script using the
dataset registry. Replaces the per-dataset if-elif blocks in retarget.yaml.

Usage::

    python scripts/retarget/run_retarget.py --dataset taco --input_dir /data/loaded --output_dir /data/processed --device cuda:0 --save
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
            "Usage: python scripts/retarget/run_retarget.py --dataset <name> [retarget args...]",
            file=sys.stderr,
        )
        sys.exit(1)

    idx = sys.argv.index("--dataset")
    if idx + 1 >= len(sys.argv):
        print("Error: --dataset requires a value", file=sys.stderr)
        sys.exit(1)

    dataset_name = sys.argv[idx + 1]
    remaining_argv = sys.argv[:idx] + sys.argv[idx + 2 :]

    # Import registry
    source_dir = str(REPO_ROOT / "source" / "robotic_grounding")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    from robotic_grounding.retarget.dataset_registry import (  # noqa: PLC0415
        get_dataset_config,
    )

    config = get_dataset_config(dataset_name)
    if not config.retarget_script:
        print(
            f"Error: dataset '{dataset_name}' has no retarget_script configured",
            file=sys.stderr,
        )
        sys.exit(1)

    module = _load_module_from_path(config.retarget_script)

    sys.argv = remaining_argv
    args = module.parse_args()
    module.main(args)


if __name__ == "__main__":
    main()
