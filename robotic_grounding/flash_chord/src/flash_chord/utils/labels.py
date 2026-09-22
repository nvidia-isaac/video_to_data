# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""URDF joint/link label helpers: strip the namespace/side prefix and infer a label's side."""

from __future__ import annotations

_SIDE_PREFIXES = ("left_", "right_")


def clean_label(label: str) -> str:
    """Strip the URDF namespace and side prefix for a compact link/joint name."""
    name = label.split("/")[-1]
    for prefix in _SIDE_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def side_of(label: str, *, unknown: str | None = "left") -> str | None:
    """Infer ``'left'`` / ``'right'`` from a label's namespace or name prefix (``left_``/``right_`` or
    ``L_``/``R_``); returns ``unknown`` when neither matches."""
    head, name = label.split("/")[0], label.split("/")[-1]
    if head.startswith("right") or name.startswith("right") or name.startswith("R_"):
        return "right"
    if head.startswith("left") or name.startswith("left") or name.startswith("L_"):
        return "left"
    return unknown
