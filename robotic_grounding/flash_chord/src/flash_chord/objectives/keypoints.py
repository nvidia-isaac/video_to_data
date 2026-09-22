# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keypoint tracking objectives (motion imitation).

Object keypoints = the 6 object-frame axis unit vectors (±X,±Y,±Z) transformed to world by the object
pose; hand keypoints = the wrist + fingertip bodies. Each keypoint's squared error feeds the shared
:func:`~flash_chord.objectives.shaping.shaped_objective`; the per-world objective is the mean over keypoints.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from flash_chord.objectives.shaping import shaped_objective

# 6 principal axis unit vectors in the object frame.
KEYPOINT_VECS_NP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float32,
)


@wp.kernel
def object_keypoints_objective(
    body_q: wp.array(dtype=wp.transform),
    object_body_ids: wp.array(dtype=wp.int32),  # [n_obj] per-world object body ids
    target_pos: wp.array(dtype=wp.vec3),  # [world_count * n_obj] command object positions
    target_quat: wp.array(dtype=wp.quat),  # [world_count * n_obj] command object quats (xyzw)
    keypoint_vecs: wp.array(dtype=wp.vec3),  # [6] axis unit vectors
    n_obj: int,
    bodies_per_world: int,
    var: float,
    kind: int,
    objective_sum: wp.array(dtype=wp.float32),  # [world_count] atomic accumulator; caller divides by n_obj*6
) -> None:
    """Per (world, object-body, keypoint) shaped objective, accumulated per world; launch over W*n_obj*6."""
    tid = wp.tid()
    kp = tid % 6
    body_local = (tid // 6) % n_obj
    world = tid // (6 * n_obj)
    xf = body_q[world * bodies_per_world + object_body_ids[body_local]]
    idx = world * n_obj + body_local
    v = keypoint_vecs[kp]
    kp_sim = wp.transform_get_translation(xf) + wp.quat_rotate(wp.transform_get_rotation(xf), v)
    kp_cmd = target_pos[idx] + wp.quat_rotate(target_quat[idx], v)
    d = kp_sim - kp_cmd
    wp.atomic_add(objective_sum, world, shaped_objective(wp.dot(d, d), var, kind, 0.0))


@wp.kernel
def hand_keypoints_objective(
    body_q: wp.array(dtype=wp.transform),
    keypoint_body_ids: wp.array(dtype=wp.int32),  # [n_hands * n_kp] retained per-world bodies
    keypoint_local_pos: wp.array(dtype=wp.vec3),  # [n_hands * n_kp] body-frame semantic offsets
    target_pos: wp.array(dtype=wp.vec3),  # [world_count * n_hands * n_kp] command keypoint positions
    n_kp: int,  # keypoints per hand (wrist + fingertips)
    n_hands: int,
    bodies_per_world: int,
    var: float,
    kind: int,
    threshold: float,
    objective_sum: wp.array(dtype=wp.float32),  # [world_count * n_hands] atomic accumulator; caller divides by n_kp
) -> None:
    """Accumulate semantic-frame position tracking per hand; launch over W*n_hands*n_kp."""
    tid = wp.tid()
    kp = tid % n_kp
    hand = (tid // n_kp) % n_hands
    world = tid // (n_kp * n_hands)
    keypoint = hand * n_kp + kp
    body = body_q[world * bodies_per_world + keypoint_body_ids[keypoint]]
    sim = wp.transform_get_translation(body) + wp.quat_rotate(
        wp.transform_get_rotation(body),
        keypoint_local_pos[keypoint],
    )
    d = sim - target_pos[tid]
    wp.atomic_add(objective_sum, world * n_hands + hand, shaped_objective(wp.dot(d, d), var, kind, threshold))
