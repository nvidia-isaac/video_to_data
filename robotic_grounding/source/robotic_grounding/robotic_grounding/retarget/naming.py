# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
