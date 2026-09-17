# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start the upstream GR00T server with deterministic action-head sampling."""

from __future__ import annotations

import argparse
import random
import runpy
import sys


def _parse_args(argv: list[str] | None = None) -> tuple[int, list[str]]:
    parser = argparse.ArgumentParser(
        description="Seed inference RNGs, then run gr00t.eval.run_gr00t_server.",
    )
    parser.add_argument("--seed", type=int, required=True)
    args, server_args = parser.parse_known_args(argv)
    return args.seed, server_args


def main(argv: list[str] | None = None) -> None:
    """Seed Python, NumPy, and PyTorch, then delegate to the upstream server."""
    seed, server_args = _parse_args(argv)

    import numpy as np  # noqa: PLC0415 - resolved from the selected Isaac-GR00T uv environment
    import torch  # noqa: PLC0415 - resolved from the selected Isaac-GR00T uv environment

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Inference seed: {seed} (deterministic action sampling)", flush=True)

    sys.argv = ["gr00t.eval.run_gr00t_server", *server_args]
    runpy.run_module("gr00t.eval.run_gr00t_server", run_name="__main__")


if __name__ == "__main__":
    main()
