# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upload a host :class:`Reference` into GPU-resident Warp buffers.

The runtime indexes these by timestep (the reset reads frame 0; the replay reads frame ``t``).
Per-side robot fields are absent (length-0) for un-retargeted parquets. Quaternions are wxyz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from flash_chord.data.reference import Reference


@dataclass
class ReferenceBuffers:
    """Per-frame reference on device (float32 Warp arrays), indexed by timestep."""

    num_frames: int
    fps: float
    sides: tuple[str, ...]
    wrist_pos_w: dict[str, wp.array]  # side -> (T, 3)
    wrist_quat_w: dict[str, wp.array]  # side -> (T, 4) wxyz
    finger_joint_pos: dict[str, wp.array]  # side -> (T, Nf)
    object_body_pos_w: wp.array  # (T, B, 3)
    object_body_quat_w: wp.array  # (T, B, 4) wxyz
    object_articulation: wp.array  # (T, A)


def _f32(a, device) -> wp.array:
    return wp.array(np.ascontiguousarray(np.asarray(a, dtype=np.float32)), dtype=wp.float32, device=device)


def upload_reference(reference: Reference, device=None) -> ReferenceBuffers:
    """Copy ``reference``'s per-frame arrays to ``device`` (default: current Warp device)."""
    sides = reference.sides
    return ReferenceBuffers(
        num_frames=reference.num_frames,
        fps=reference.fps,
        sides=sides,
        wrist_pos_w={s: _f32(reference.wrist_pos_w(s), device) for s in sides},
        wrist_quat_w={s: _f32(reference.wrist_quat_w(s), device) for s in sides},
        finger_joint_pos={s: _f32(reference.finger_joint_pos(s), device) for s in sides},
        object_body_pos_w=_f32(reference.object_body_pos_w(), device),
        object_body_quat_w=_f32(reference.object_body_quat_w(), device),
        object_articulation=_f32(reference.object_articulation(), device),
    )
