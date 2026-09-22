# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve named semantic frames across Newton fixed-joint collapse."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from flash_chord.embodiments.base import BodyFrame


@dataclass(frozen=True)
class DeviceBodyFrameMap:
    """Ordered semantic frames encoded as immutable device arrays.

    Multiple semantic frames may share one retained body after fixed-joint collapse.
    Quaternion normalization happens once at this setup boundary so every consuming
    kernel receives the same valid body-local transform representation.
    """

    count: int
    body_ids: wp.array
    body_to_frame_pos: wp.array
    body_to_frame_quat: wp.array

    @classmethod
    def build(cls, frames: Iterable[BodyFrame], device=None) -> "DeviceBodyFrameMap":
        ordered = tuple(frames)
        if not ordered:
            raise ValueError("at least one semantic body frame is required")
        names = tuple(frame.name for frame in ordered)
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError(f"semantic body frame names must be unique and nonempty, got {names}")
        body_ids = np.asarray([frame.body_id for frame in ordered], dtype=np.int32)
        if np.any(body_ids < 0):
            raise ValueError(f"semantic body frame IDs must be nonnegative, got {body_ids.tolist()}")
        positions = np.asarray([frame.body_to_frame_pos for frame in ordered], dtype=np.float32)
        quaternions = np.asarray([frame.body_to_frame_quat_xyzw for frame in ordered], dtype=np.float32)
        if positions.shape != (len(ordered), 3) or not np.all(np.isfinite(positions)):
            raise ValueError(f"semantic body-frame positions must be finite vec3 values, got shape {positions.shape}")
        if quaternions.shape != (len(ordered), 4) or not np.all(np.isfinite(quaternions)):
            raise ValueError(
                f"semantic body-frame quaternions must be finite xyzw values, got shape {quaternions.shape}"
            )
        quaternion_norm = np.linalg.norm(quaternions, axis=1)
        if np.any(quaternion_norm <= 1.0e-8):
            raise ValueError("semantic body-frame quaternions must have nonzero norm")
        quaternions = quaternions / quaternion_norm[:, None]
        return cls(
            count=len(ordered),
            body_ids=wp.array(body_ids, dtype=wp.int32, device=device),
            body_to_frame_pos=wp.array(positions, dtype=wp.vec3, device=device),
            body_to_frame_quat=wp.array(quaternions, dtype=wp.quat, device=device),
        )


def capture_body_ids(builder: Any, names: Iterable[str]) -> dict[str, int]:
    """Resolve exact terminal body labels before collapse, preserving requested order."""
    requested = tuple(names)
    if len(requested) != len(set(requested)):
        raise ValueError(f"semantic frame names must be unique, got {requested}")

    matches: dict[str, list[int]] = {name: [] for name in requested}
    for body_id, label in enumerate(builder.body_label):
        terminal = str(label).rsplit("/", 1)[-1]
        if terminal in matches:
            matches[terminal].append(body_id)

    missing = [name for name, body_ids in matches.items() if not body_ids]
    ambiguous = {name: body_ids for name, body_ids in matches.items() if len(body_ids) > 1}
    if missing or ambiguous:
        raise ValueError(f"unable to resolve semantic bodies: missing={missing}, ambiguous={ambiguous}")
    return {name: matches[name][0] for name in requested}


def resolve_body_frames(
    body_ids_before_collapse: Mapping[str, int],
    collapse_result: Mapping[str, Any],
) -> tuple[BodyFrame, ...]:
    """Map pre-collapse named bodies to retained bodies plus constant local transforms."""
    body_remap = collapse_result["body_remap"]
    merged_transform = collapse_result["body_merged_transform"]
    frames: list[BodyFrame] = []
    for name, old_body_id in body_ids_before_collapse.items():
        if old_body_id in body_remap:
            frames.append(BodyFrame(name=name, body_id=_resolve_retained_body_id(old_body_id, collapse_result)))
            continue
        if old_body_id not in merged_transform:
            raise ValueError(f"semantic body {name!r} ({old_body_id}) was not retained or merged")
        body_to_frame = merged_transform[old_body_id]
        frames.append(
            BodyFrame(
                name=name,
                body_id=_resolve_retained_body_id(old_body_id, collapse_result),
                body_to_frame_pos=tuple(float(value) for value in body_to_frame.p),
                body_to_frame_quat_xyzw=tuple(float(value) for value in body_to_frame.q),
            )
        )
    return tuple(frames)


def resolve_retained_body_ids(
    body_ids_before_collapse: Iterable[int],
    collapse_result: Mapping[str, Any],
) -> tuple[int, ...]:
    """Map pre-collapse bodies to the retained bodies that carry their geometry."""
    return tuple(_resolve_retained_body_id(body_id, collapse_result) for body_id in body_ids_before_collapse)


def _resolve_retained_body_id(old_body_id: int, collapse_result: Mapping[str, Any]) -> int:
    if old_body_id < 0:
        raise ValueError(f"pre-collapse body IDs must be nonnegative, got {old_body_id}")
    body_remap = collapse_result["body_remap"]
    if old_body_id in body_remap:
        return int(body_remap[old_body_id])
    merged_parent = collapse_result["body_merged_parent"]
    if old_body_id not in merged_parent:
        raise ValueError(f"body {old_body_id} was not retained or merged")
    anchor_old_id = merged_parent[old_body_id]
    if anchor_old_id == -1:
        raise ValueError(f"body {old_body_id} collapsed into the static world")
    if anchor_old_id not in body_remap:
        raise ValueError(f"body {old_body_id} has unresolved retained parent {anchor_old_id}")
    return int(body_remap[anchor_old_id])
