# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-world physics-step delay for device control targets."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp


@dataclass(frozen=True)
class TargetDelayConfig:
    """Uniform reset-time delay range in physics steps."""

    min_steps: int = 0
    max_steps: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.min_steps < 0:
            raise ValueError(f"min_steps must be non-negative, got {self.min_steps}")
        if self.max_steps < self.min_steps:
            raise ValueError(f"max_steps ({self.max_steps}) must be >= min_steps ({self.min_steps})")


@wp.kernel
def reset_target_delay(
    reset_mask: wp.array(dtype=wp.int32),
    min_steps: int,
    max_steps: int,
    seed: int,
    delay_steps: wp.array(dtype=wp.int32),
    num_pushes: wp.array(dtype=wp.int32),
    reset_count: wp.array(dtype=wp.int32),
    delayed_target: wp.array(dtype=wp.float32),
    num_joint_dof: int,
) -> None:
    """Resample lag and clear stream state for selected worlds."""
    world = wp.tid()
    if reset_mask[world] == 0:
        return

    span = max_steps - min_steps + 1
    sampled = min_steps
    if span > 1:
        offset = world + reset_count[world] * 104729
        rng = wp.rand_init(seed, offset)
        sampled += int(wp.randf(rng) * float(span))
    delay_steps[world] = sampled
    num_pushes[world] = 0
    reset_count[world] += 1

    target_base = world * num_joint_dof
    for dof in range(num_joint_dof):
        delayed_target[target_base + dof] = 0.0


@wp.kernel
def advance_target_delay(
    target: wp.array(dtype=wp.float32),
    world_count: int,
    num_joint_dof: int,
    num_delayed_dof: int,
    delayed_slot_by_dof: wp.array(dtype=wp.int32),
    capacity: int,
    cursor: wp.array(dtype=wp.int32),
    delay_steps: wp.array(dtype=wp.int32),
    num_pushes: wp.array(dtype=wp.int32),
    history: wp.array(dtype=wp.float32),
    delayed_target: wp.array(dtype=wp.float32),
) -> None:
    """Append current targets and emit each world's selected history entry."""
    world = wp.tid()
    write_slot = cursor[0]
    pushes = num_pushes[world]
    target_base = world * num_joint_dof

    for dof in range(num_joint_dof):
        value = target[target_base + dof]
        delayed_dof = delayed_slot_by_dof[dof]
        if delayed_dof >= 0:
            if pushes == 0:
                for slot in range(capacity):
                    history[(slot * world_count + world) * num_delayed_dof + delayed_dof] = value
            else:
                history[(write_slot * world_count + world) * num_delayed_dof + delayed_dof] = value

            lag = delay_steps[world]
            if lag > pushes:
                lag = pushes
            read_slot = (write_slot - lag + capacity) % capacity
            value = history[(read_slot * world_count + world) * num_delayed_dof + delayed_dof]
        delayed_target[target_base + dof] = value

    if pushes < capacity:
        num_pushes[world] = pushes + 1


@wp.kernel
def advance_target_delay_cursor(cursor: wp.array(dtype=wp.int32), capacity: int) -> None:
    """Advance the shared history slot after all worlds have read it."""
    cursor[0] = (cursor[0] + 1) % capacity


@wp.kernel
def copy_delayed_target_dofs(
    delayed_target: wp.array(dtype=wp.float32),
    delayed_dof_ids: wp.array(dtype=wp.int32),
    num_joint_dof: int,
    output: wp.array(dtype=wp.float32),
) -> None:
    """Copy only this buffer's delayed DOFs into a complete control target."""
    world, slot = wp.tid()
    dof = delayed_dof_ids[slot]
    index = world * num_joint_dof + dof
    output[index] = delayed_target[index]


@dataclass
class TargetDelayBuffer:
    """Device ring buffer with independently sampled per-world target lag."""

    config: TargetDelayConfig
    world_count: int
    num_joint_dof: int
    num_delayed_dof: int
    delayed_dof_ids: tuple[int, ...]
    delayed_dof_ids_device: wp.array
    delayed_slot_by_dof: wp.array
    capacity: int
    cursor: wp.array
    delay_steps: wp.array
    num_pushes: wp.array
    reset_count: wp.array
    history: wp.array
    delayed_target: wp.array

    @classmethod
    def build(
        cls,
        world_count: int,
        num_joint_dof: int,
        delayed_dof_ids: tuple[int, ...],
        config: TargetDelayConfig,
        device=None,
    ) -> "TargetDelayBuffer":
        """Allocate a delay stream for an explicit ordered set of target DOFs."""
        delayed_dof_ids = tuple(delayed_dof_ids)
        if not delayed_dof_ids:
            raise ValueError("at least one delayed DOF is required")
        if len(set(delayed_dof_ids)) != len(delayed_dof_ids):
            raise ValueError(f"delayed DOF IDs must be unique, got {delayed_dof_ids}")
        invalid = [dof for dof in delayed_dof_ids if dof < 0 or dof >= num_joint_dof]
        if invalid:
            raise ValueError(f"delayed DOF IDs must be in [0, {num_joint_dof}), got {invalid}")
        num_delayed_dof = len(delayed_dof_ids)
        delayed_slot_by_dof = [-1] * num_joint_dof
        for slot, dof in enumerate(delayed_dof_ids):
            delayed_slot_by_dof[dof] = slot
        capacity = config.max_steps + 1
        return cls(
            config=config,
            world_count=world_count,
            num_joint_dof=num_joint_dof,
            num_delayed_dof=num_delayed_dof,
            delayed_dof_ids=delayed_dof_ids,
            delayed_dof_ids_device=wp.array(delayed_dof_ids, dtype=wp.int32, device=device),
            delayed_slot_by_dof=wp.array(delayed_slot_by_dof, dtype=wp.int32, device=device),
            capacity=capacity,
            cursor=wp.zeros(1, dtype=wp.int32, device=device),
            delay_steps=wp.zeros(world_count, dtype=wp.int32, device=device),
            num_pushes=wp.zeros(world_count, dtype=wp.int32, device=device),
            reset_count=wp.zeros(world_count, dtype=wp.int32, device=device),
            history=wp.zeros(
                capacity * world_count * num_delayed_dof,
                dtype=wp.float32,
                device=device,
            ),
            delayed_target=wp.zeros(
                world_count * num_joint_dof,
                dtype=wp.float32,
                device=device,
            ),
        )

    def reset(self, reset_mask: wp.array) -> None:
        """Resample and clear delay state for selected worlds."""
        wp.launch(
            reset_target_delay,
            dim=self.world_count,
            inputs=[
                reset_mask,
                self.config.min_steps,
                self.config.max_steps,
                self.config.seed,
                self.delay_steps,
                self.num_pushes,
                self.reset_count,
                self.delayed_target,
                self.num_joint_dof,
            ],
        )

    def advance(self, target: wp.array) -> wp.array:
        """Append ``target`` and return the persistent delayed target buffer."""
        wp.launch(
            advance_target_delay,
            dim=self.world_count,
            inputs=[
                target,
                self.world_count,
                self.num_joint_dof,
                self.num_delayed_dof,
                self.delayed_slot_by_dof,
                self.capacity,
                self.cursor,
                self.delay_steps,
                self.num_pushes,
                self.history,
                self.delayed_target,
            ],
        )
        wp.launch(
            advance_target_delay_cursor,
            dim=1,
            inputs=[self.cursor, self.capacity],
        )
        return self.delayed_target

    def advance_into(self, target: wp.array, output: wp.array) -> None:
        """Advance the stream and overwrite only its selected DOFs in ``output``."""
        delayed_target = self.advance(target)
        wp.launch(
            copy_delayed_target_dofs,
            dim=(self.world_count, self.num_delayed_dof),
            inputs=[
                delayed_target,
                self.delayed_dof_ids_device,
                self.num_joint_dof,
            ],
            outputs=[output],
        )
