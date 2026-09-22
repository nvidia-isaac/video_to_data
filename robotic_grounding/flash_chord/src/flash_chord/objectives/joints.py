# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hand finger-joint tracking objective.

Per hand, the sum of squared finger-joint errors feeds the shared shaping primitive. The reference
finger joints must be supplied in the sim joint order (reorder the parquet's
``{side}_robot_finger_joint_names`` to the robot's joint order by name).
"""

from __future__ import annotations

import warp as wp

from flash_chord.objectives.shaping import shaped_objective


@wp.kernel
def hand_joint_objective(
    joint_q: wp.array(dtype=wp.float32),
    finger_q_ids: wp.array(dtype=wp.int32),  # [n_hands * n_finger] per-world finger coord ids (sim order)
    target_q: wp.array(dtype=wp.float32),  # [world_count * n_hands * n_finger] command finger joint positions
    n_finger: int,
    n_hands: int,
    num_joint_q: int,  # coords per world
    var: float,
    kind: int,
    threshold: float,
    objective: wp.array(dtype=wp.float32),  # [world_count * n_hands] out, in (0, 1]
) -> None:
    """Per (world, hand): shaped objective of the summed squared finger-joint error; launch over W*n_hands.
    Caller averages the hands."""
    tid = wp.tid()
    hand = tid % n_hands
    world = tid // n_hands
    sse = float(0.0)
    for f in range(n_finger):
        cur = joint_q[world * num_joint_q + finger_q_ids[hand * n_finger + f]]
        d = cur - target_q[world * n_hands * n_finger + hand * n_finger + f]
        sse += d * d
    objective[tid] = shaped_objective(sse, var, kind, threshold)
