# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact terminal progress for local training runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol


class TrainingProgress(Protocol):
    """Progress reporter driven by completed environment steps."""

    def start(self, completed_environment_steps: int) -> None:
        """Start reporting from a new or resumed training state."""
        ...

    def update(self, completed_environment_steps: int) -> None:
        """Report a later training state when its display boundary is crossed."""
        ...


class ConsoleTrainingProgress:
    """Print bounded progress at one-percent environment-step intervals."""

    def __init__(self, total_environment_steps: int, checkpoint_dir: str | Path) -> None:
        if total_environment_steps <= 0:
            raise ValueError(f"total_environment_steps must be positive, got {total_environment_steps}")
        self.total_environment_steps = total_environment_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self._start_environment_steps = 0
        self._last_environment_steps = -1
        self._next_percent = 0
        self._start_time: float | None = None

    def start(self, completed_environment_steps: int) -> None:
        """Print the initial state and initialize throughput/ETA timing."""
        self._validate_steps(completed_environment_steps)
        self._start_environment_steps = completed_environment_steps
        self._last_environment_steps = completed_environment_steps
        self._next_percent = min(100, completed_environment_steps * 100 // self.total_environment_steps + 1)
        self._start_time = time.perf_counter()
        print(
            f"training progress: {completed_environment_steps:,}/{self.total_environment_steps:,} "
            f"environment steps ({self._percent(completed_environment_steps):.1f}%); "
            f"checkpoints: {self.checkpoint_dir}",
            flush=True,
        )

    def update(self, completed_environment_steps: int) -> None:
        """Print after crossing the next whole percent, and always at completion."""
        if self._start_time is None:
            raise RuntimeError("training progress must be started before it is updated")
        self._validate_steps(completed_environment_steps)
        if completed_environment_steps < self._last_environment_steps:
            raise ValueError(
                "completed_environment_steps must not decrease: "
                f"{completed_environment_steps} < {self._last_environment_steps}"
            )
        percent = self._percent(completed_environment_steps)
        terminal = completed_environment_steps == self.total_environment_steps
        if not terminal and percent < self._next_percent:
            return
        if completed_environment_steps == self._last_environment_steps:
            return

        elapsed_seconds = time.perf_counter() - self._start_time
        completed_since_start = completed_environment_steps - self._start_environment_steps
        rate = completed_since_start / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        remaining = self.total_environment_steps - completed_environment_steps
        eta = _format_duration(remaining / rate) if rate > 0.0 else "calculating"
        print(
            f"training progress: {completed_environment_steps:,}/{self.total_environment_steps:,} "
            f"environment steps ({percent:.1f}%); {rate:,.0f} env steps/s; "
            f"elapsed {_format_duration(elapsed_seconds)}; ETA {eta}",
            flush=True,
        )
        self._last_environment_steps = completed_environment_steps
        self._next_percent = min(100, int(percent) + 1)

    def _percent(self, completed_environment_steps: int) -> float:
        return 100.0 * completed_environment_steps / self.total_environment_steps

    def _validate_steps(self, completed_environment_steps: int) -> None:
        if not 0 <= completed_environment_steps <= self.total_environment_steps:
            raise ValueError(
                f"completed_environment_steps must be in [0, {self.total_environment_steps}], "
                f"got {completed_environment_steps}"
            )


def _format_duration(seconds: float) -> str:
    """Format a non-negative duration without subsecond terminal churn."""
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
