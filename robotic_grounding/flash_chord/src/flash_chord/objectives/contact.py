# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact-wrench objectives on the per-(hand, body, basis-direction) wrench supports.

Each reads the reference and current (sim) support vectors, shaped ``[W·H·B·K]`` (K basis directions per
body). A basis direction is "active" where its support ``> 1e-3``.
"""

from __future__ import annotations

import warp as wp

_ACTIVE = wp.constant(1.0e-3)
_FORCE_SQUARED_NUMERIC_CEILING = wp.constant(1.0e30)


@wp.func
def _idx(w: int, h: int, b: int, k: int, n_hands: int, n_bodies: int, n_basis: int) -> int:
    return ((w * n_hands + h) * n_bodies + b) * n_basis + k


@wp.kernel
def contact_force_l2(
    contact_force_w: wp.array(dtype=wp.vec3),
    slots_per_world: int,
    threshold: float,
    objective: wp.array(dtype=wp.float32),
) -> None:
    """Sum squared per-slot force exceeding ``threshold`` per world [N²]."""
    world = wp.tid()
    total = float(0.0)
    for local_slot in range(slots_per_world):
        force = contact_force_w[world * slots_per_world + local_slot]
        if threshold == 0.0:
            total += wp.dot(force, force)
        else:
            excess = wp.length(force) - threshold
            if excess > 0.0:
                total += excess * excess
    objective[world] = total


@wp.kernel
def contact_force_per_hand_log(
    contact_force_w: wp.array(dtype=wp.vec3),
    hand_slot_starts: wp.array(dtype=wp.int32),
    link_counts: wp.array(dtype=wp.int32),
    num_hands: int,
    num_objects: int,
    slots_per_world: int,
    world_count: int,
    history_length: int,
    log_force_floor_squared: float,
    log_force_reference_value: float,
    log_force_inverse_span: float,
    history_cursor: wp.array(dtype=wp.int32),
    force_squared_history: wp.array(dtype=wp.float32),
    objective: wp.array(dtype=wp.float32),
) -> None:
    """Reward the log-scaled RMS total contact force per hand."""
    world = wp.tid()
    cursor = history_cursor[0]
    for local_slot in range(slots_per_world):
        force = contact_force_w[world * slots_per_world + local_slot]
        history_index = (cursor * world_count + world) * slots_per_world + local_slot
        force_squared_history[history_index] = wp.dot(force, force)

    total_reward = float(0.0)
    for hand in range(num_hands):
        link_count = link_counts[hand]
        hand_start = hand_slot_starts[hand]
        total_force_squared = float(0.0)
        for history_slot in range(history_length):
            hand_force = float(0.0)
            for object_id in range(num_objects):
                for link in range(link_count):
                    local_slot = hand_start + object_id * link_count + link
                    history_index = (history_slot * world_count + world) * slots_per_world + local_slot
                    hand_force += wp.sqrt(wp.max(force_squared_history[history_index], 0.0))
            total_force_squared += hand_force * hand_force
        total_force_squared /= float(history_length)

        if total_force_squared > 1.0e-3:
            force_squared_clamped = wp.min(
                wp.max(total_force_squared, log_force_floor_squared),
                _FORCE_SQUARED_NUMERIC_CEILING,
            )
            force_log = 0.5 * wp.log(force_squared_clamped)
            total_reward += (log_force_reference_value - force_log) * log_force_inverse_span

    objective[world] = total_reward / float(num_hands)


@wp.kernel
def advance_contact_force_history(history_cursor: wp.array(dtype=wp.int32), history_length: int) -> None:
    """Advance the shared contact-force history cursor once per control step."""
    history_cursor[0] = (history_cursor[0] + 1) % history_length


@wp.kernel
def reset_contact_force_history(
    reset_mask: wp.array(dtype=wp.int32),
    slots_per_world: int,
    world_count: int,
    history_length: int,
    force_squared_history: wp.array(dtype=wp.float32),
) -> None:
    """Clear contact-force history for selected worlds."""
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    for history_slot in range(history_length):
        base = (history_slot * world_count + world) * slots_per_world
        for local_slot in range(slots_per_world):
            force_squared_history[base + local_slot] = 0.0


@wp.kernel
def contact_wrench_support_objective(
    cmd_sup: wp.array(dtype=wp.float32),  # [W*H*B*K] command (demo) supports
    cur_sup: wp.array(dtype=wp.float32),  # [W*H*B*K] current (sim) supports
    n_hands: int,
    n_bodies: int,
    n_basis: int,
    tolerance: float,
    var: float,
    objective: wp.array(dtype=wp.float32),  # [W] out
) -> None:
    """Tolerance-band match of cur vs cmd supports, gated by both active; mean over basis dirs / bodies /
    active hands."""
    w = wp.tid()
    hand_sum = float(0.0)
    n_active_hands = float(0.0)
    for h in range(n_hands):
        body_sum = float(0.0)
        n_active_bodies = float(0.0)
        for b in range(n_bodies):
            k_sum = float(0.0)
            cmd_num = float(0.0)
            body_active = int(0)
            for k in range(n_basis):
                i = _idx(w, h, b, k, n_hands, n_bodies, n_basis)
                cs = cmd_sup[i]
                us = cur_sup[i]
                if cs > _ACTIVE:
                    cmd_num += 1.0
                    body_active = 1
                    if us > _ACTIVE:
                        better = wp.max((1.0 - tolerance) * cs - us, 0.0)
                        too_large = wp.max(us - (1.0 + tolerance) * cs, 0.0)
                        loss = better * better + too_large * too_large
                        k_sum += wp.exp(-loss / var)
            if cmd_num > 1.0e-6:
                body_sum += k_sum / cmd_num
            if body_active == 1:
                n_active_bodies += 1.0
        if n_active_bodies > 1.0e-6:
            hand_sum += body_sum / n_active_bodies
            n_active_hands += 1.0
    if n_active_hands > 1.0e-6:
        objective[w] = hand_sum / n_active_hands
    else:
        objective[w] = 0.0


@wp.kernel
def missed_contact_penalty(
    cmd_sup: wp.array(dtype=wp.float32),
    cur_sup: wp.array(dtype=wp.float32),
    n_hands: int,
    n_bodies: int,
    n_basis: int,
    penalty: wp.array(dtype=wp.float32),  # [W] out
) -> None:
    """Per active hand: fraction of commanded basis directions per body that the agent failed to cover,
    averaged over commanded-active bodies; summed over hands / active hands."""
    w = wp.tid()
    hand_sum = float(0.0)
    n_active_hands = float(0.0)
    for h in range(n_hands):
        body_sum = float(0.0)
        n_active_bodies = float(0.0)
        for b in range(n_bodies):
            n_expected = float(0.0)
            n_missed = float(0.0)
            for k in range(n_basis):
                i = _idx(w, h, b, k, n_hands, n_bodies, n_basis)
                if cmd_sup[i] > _ACTIVE:
                    n_expected += 1.0
                    if cur_sup[i] <= _ACTIVE:
                        n_missed += 1.0
            if n_expected > 0.0:
                body_sum += n_missed / n_expected
                n_active_bodies += 1.0
        if n_active_bodies > 1.0e-6:
            hand_sum += body_sum / n_active_bodies
            n_active_hands += 1.0
    if n_active_hands > 1.0e-6:
        penalty[w] = hand_sum / n_active_hands
    else:
        penalty[w] = 0.0


@wp.kernel
def unintended_contact_penalty(
    cmd_sup: wp.array(dtype=wp.float32),
    cur_sup: wp.array(dtype=wp.float32),
    n_hands: int,
    n_bodies: int,
    n_basis: int,
    penalty: wp.array(dtype=wp.float32),  # [W] out
) -> None:
    """Per hand: fraction of bodies with contact the reference doesn't ask for (binary) + mean squared
    support on those bodies (continuous); summed over hands (no hand normalization)."""
    w = wp.tid()
    total = float(0.0)
    for h in range(n_hands):
        unintended_count = float(0.0)
        support_sum = float(0.0)
        n_inactive = float(0.0)
        for b in range(n_bodies):
            cmd_active = int(0)
            cur_active = int(0)
            sup_sq = float(0.0)
            for k in range(n_basis):
                i = _idx(w, h, b, k, n_hands, n_bodies, n_basis)
                if cmd_sup[i] > _ACTIVE:
                    cmd_active = 1
                if cur_sup[i] > _ACTIVE:
                    cur_active = 1
                u = wp.max(cur_sup[i], 0.0)
                sup_sq += u * u
            if cmd_active == 0:  # demo asks for no contact on this body
                n_inactive += 1.0
                support_sum += sup_sq / float(n_basis)  # mean over K
                if cur_active == 1:
                    unintended_count += 1.0
        binary = unintended_count / float(n_bodies)  # mean over all bodies
        cont = support_sum / wp.max(n_inactive, 1.0e-6)
        total += binary + cont
    penalty[w] = total


@wp.kernel
def force_closure_objective(
    cmd_sup: wp.array(dtype=wp.float32),
    cur_sup: wp.array(dtype=wp.float32),
    n_hands: int,
    n_bodies: int,
    n_basis: int,
    min_support: float,
    objective: wp.array(dtype=wp.float32),
) -> None:
    """Reward per-body wrench-direction coverage on commanded-active hands."""
    w = wp.tid()
    hand_sum = float(0.0)
    n_active_hands = float(0.0)
    for h in range(n_hands):
        body_sum = float(0.0)
        n_active_bodies = float(0.0)
        for b in range(n_bodies):
            n_closed = float(0.0)
            body_active = int(0)
            for k in range(n_basis):
                i = _idx(w, h, b, k, n_hands, n_bodies, n_basis)
                if cmd_sup[i] > _ACTIVE:
                    body_active = 1
                if cur_sup[i] > min_support:
                    n_closed += 1.0
            if body_active == 1:
                body_sum += n_closed / float(n_basis)
                n_active_bodies += 1.0
        if n_active_bodies > 1.0e-6:
            hand_sum += body_sum / n_active_bodies
            n_active_hands += 1.0
    if n_active_hands > 1.0e-6:
        objective[w] = hand_sum / n_active_hands
    else:
        objective[w] = 0.0
