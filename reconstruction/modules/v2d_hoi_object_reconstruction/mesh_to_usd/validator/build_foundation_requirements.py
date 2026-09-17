#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the requirement package expected by simready-foundation-core."""

from __future__ import annotations

import argparse
from pathlib import Path

from omni.usd_profiles.codegen._py_generate import PythonGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    PythonGenerator(
        capabilities_root=args.capabilities,
        destination_dir=args.output,
        namespace="omni.capabilities",
    ).generate()


if __name__ == "__main__":
    main()
