# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Naming helpers shared between dataset loading and URDF/asset generation.

Kept dependency-free (pure string ops) so consumers like
``scripts/generate_rigid_urdfs.py`` can import it without pulling in the
hand-FK / dataset-loading stack (which now lives in reconstruction).
"""


def make_usd_safe(name: str) -> str:
    """Make a name safe for USD prim paths (no leading digits, no @ etc.)."""
    safe = name.replace("@", "_")
    if safe and (safe[0].isdigit() or not (safe[0].isalpha() or safe[0] == "_")):
        safe = f"obj_{safe}"
    return safe
