# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Curated per-object facts the motion parquet cannot express.

Arctic objects are articulated (a revolute lid/joint). The parquet's
``object_articulation`` trajectory is zero when the joint does not move in a clip,
so it cannot be used to tell an articulable object apart from a rigid one — hence
this small curated set. Object URDFs and meshes are resolved from the motion
file's own dataset assets (``object_assets/``), not from here.
"""
from __future__ import annotations

ARTICULATED_OBJECTS: set[str] = {
    "box",
    "capsulemachine",
    "espressomachine",
    "ketchup",
    "laptop",
    "microwave",
    "mixer",
    "notebook",
    "phone",
    "waffleiron",
}


def is_articulated(object_name: str | None) -> bool:
    """Whether the named object is articulated (Arctic objects)."""
    return bool(object_name) and object_name in ARTICULATED_OBJECTS
