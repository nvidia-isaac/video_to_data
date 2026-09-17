# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-level closed-loop evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from groot_finetune.task_profile import LiftHoldEvaluator


@dataclass(frozen=True)
class LiftHoldResult:
    """Completed-episode lift-and-hold measurements."""

    success: bool
    max_lift_m: float
    max_hold_steps: int
    sample_count: int


def evaluate_lift_hold(
    object_z: np.ndarray,
    *,
    config: LiftHoldEvaluator,
) -> LiftHoldResult:
    """Evaluate one object-height trace relative to its first sample."""
    trace = np.asarray(object_z, dtype=np.float64)
    if trace.ndim != 1 or trace.size == 0:
        raise ValueError(f"object_z must be a non-empty 1-D array, got {trace.shape}")
    if not np.isfinite(trace).all():
        raise ValueError("object_z contains non-finite values")
    tracker = LiftHoldTracker(np.asarray([trace[0]]), config=config)
    for value in trace[1:]:
        tracker.update(np.asarray([value]))
    return tracker.complete(0, next_initial_z=float(trace[-1]))


class LiftHoldTracker:
    """Track lift-and-hold metrics independently for parallel environments."""

    def __init__(self, initial_z: np.ndarray, *, config: LiftHoldEvaluator) -> None:
        """Initialize independent accumulators from each environment's baseline."""
        initial = np.asarray(initial_z, dtype=np.float64)
        if initial.ndim != 1 or initial.size == 0:
            raise ValueError(
                f"initial_z must be a non-empty 1-D array, got {initial.shape}"
            )
        if not np.isfinite(initial).all():
            raise ValueError("initial_z contains non-finite values")
        self.config = config
        self._initial_z = initial.copy()
        self._max_lift_m = np.zeros_like(initial)
        self._consecutive_hold_steps = np.zeros(initial.shape, dtype=np.int64)
        self._max_hold_steps = np.zeros(initial.shape, dtype=np.int64)
        self._sample_count = np.zeros(initial.shape, dtype=np.int64)

    @property
    def num_envs(self) -> int:
        """Return the number of tracked parallel environments."""
        return int(self._initial_z.size)

    def update(self, object_z: np.ndarray) -> None:
        """Accumulate one pre-action object-height sample for every environment."""
        current = np.asarray(object_z, dtype=np.float64)
        if current.shape != self._initial_z.shape:
            raise ValueError(
                f"object_z must have shape {self._initial_z.shape}, got {current.shape}"
            )
        if not np.isfinite(current).all():
            raise ValueError("object_z contains non-finite values")
        lift_m = current - self._initial_z
        self._max_lift_m = np.maximum(self._max_lift_m, lift_m)
        self._consecutive_hold_steps = np.where(
            lift_m >= self.config.hold_threshold_m,
            self._consecutive_hold_steps + 1,
            0,
        )
        self._max_hold_steps = np.maximum(
            self._max_hold_steps, self._consecutive_hold_steps
        )
        self._sample_count += 1

    def complete(self, env_index: int, *, next_initial_z: float) -> LiftHoldResult:
        """Finish one episode and reset only that environment's accumulator."""
        if not 0 <= env_index < self.num_envs:
            raise IndexError(
                f"env_index must be in [0, {self.num_envs}), got {env_index}"
            )
        if not np.isfinite(next_initial_z):
            raise ValueError("next_initial_z must be finite")
        max_lift = float(self._max_lift_m[env_index])
        max_hold = int(self._max_hold_steps[env_index])
        result = LiftHoldResult(
            success=(
                max_lift >= self.config.lift_threshold_m
                and max_hold >= self.config.min_hold_steps
            ),
            max_lift_m=max_lift,
            max_hold_steps=max_hold,
            sample_count=int(self._sample_count[env_index]),
        )
        self._initial_z[env_index] = float(next_initial_z)
        self._max_lift_m[env_index] = 0.0
        self._consecutive_hold_steps[env_index] = 0
        self._max_hold_steps[env_index] = 0
        self._sample_count[env_index] = 0
        return result
