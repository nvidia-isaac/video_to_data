# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-aware task-reference ingestion.

Use :func:`load_reference` to dispatch a declared schema while preserving its narrow
robot-motion capability.
"""

from __future__ import annotations

from pathlib import Path

from flash_chord.data.reference import (
    HandPoseReference,
    NamedFrameReference,
    NamedRobotReference,
    Reference,
    ReferenceMetadata,
)


def load_reference(
    parquet_path: str | Path,
    control_fps: float | None = None,
    motion_speed: float = 1.0,
    source_frame_playback: bool = False,
) -> Reference:
    """Load a task reference without importing concrete schema readers at package import time."""
    from flash_chord.data.loader import load_reference as load

    return load(
        parquet_path,
        control_fps=control_fps,
        motion_speed=motion_speed,
        source_frame_playback=source_frame_playback,
    )


def frame_window(reference: Reference, start_frame: int = 0, end_frame: int = -1) -> Reference:
    """Restrict a loaded reference before embodiment binding and scene construction."""
    if start_frame == 0 and end_frame < 0:
        return reference
    return reference.frame_window(start_frame, end_frame)


__all__ = [
    "HandPoseReference",
    "NamedFrameReference",
    "NamedRobotReference",
    "Reference",
    "ReferenceMetadata",
    "frame_window",
    "load_reference",
]
