#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic dataset loader entry point (module-based dispatch).

Extracts ``--dataset`` from argv and delegates the remaining args to the mapped
loader module's ``parse_args()`` / ``main(args)``. Ported from robotic_grounding's
``scripts/retarget/run_loader.py``; the only change is dispatching by module name
(``importlib.import_module``) instead of by file path, since the loaders are now
package modules.

Usage::

    python -m v2d.task_library_loader.lib.run_loader \\
        --dataset taco --output_dir /data/out --mano_model_dir /data/mano \\
        --device cuda:0 --save
"""

from __future__ import annotations

import importlib
import sys

from v2d.task_library_loader.lib.loader_registry import LOADER_MODULES


def main() -> None:
    """Parse --dataset, then delegate to the dataset's loader module."""
    if "--dataset" not in sys.argv:
        print("Error: --dataset is required", file=sys.stderr)
        print(
            "Usage: python -m v2d.task_library_loader.lib.run_loader "
            "--dataset <name> [loader args...]",
            file=sys.stderr,
        )
        sys.exit(1)

    idx = sys.argv.index("--dataset")
    if idx + 1 >= len(sys.argv):
        print("Error: --dataset requires a value", file=sys.stderr)
        sys.exit(1)

    dataset_name = sys.argv[idx + 1]

    # Remove --dataset <name> from argv so the loader's parse_args() doesn't see it.
    remaining_argv = sys.argv[:idx] + sys.argv[idx + 2 :]

    module_name = LOADER_MODULES.get(dataset_name)
    if module_name is None:
        print(
            f"Error: unknown dataset '{dataset_name}'. "
            f"Known: {sorted(LOADER_MODULES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    module = importlib.import_module(module_name)

    # Replace sys.argv so the loader's argparse sees the right args.
    sys.argv = remaining_argv
    args = module.parse_args()
    module.main(args)


if __name__ == "__main__":
    main()
