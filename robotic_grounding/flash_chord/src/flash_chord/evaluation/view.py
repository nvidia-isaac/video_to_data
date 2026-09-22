# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Algorithm-neutral configuration for interactive policy visualization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyViewConfig:
    """Checkpoint reconstruction and policy-sampling settings for a policy viewer."""

    checkpoint: str
    deterministic: bool = True
    use_checkpoint_config: bool = True
    validate_policy_schema: bool = True
    start_frame: int = 0
    reset_mode: str = "explicit"
    motion_start_frame: int | None = None
    motion_end_frame: int | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("evaluation checkpoint must be specified")
        if self.start_frame < 0:
            raise ValueError(f"evaluation start_frame must be non-negative, got {self.start_frame}")
        if self.reset_mode not in ("explicit", "sampled_settled"):
            raise ValueError(f"evaluation reset_mode must be 'explicit' or 'sampled_settled', got {self.reset_mode!r}")
        if self.motion_start_frame is not None and self.motion_start_frame < 0:
            raise ValueError(
                f"evaluation motion_start_frame must be non-negative or None, got {self.motion_start_frame}"
            )
        if self.motion_end_frame is not None and self.motion_end_frame < -1:
            raise ValueError(
                f"evaluation motion_end_frame must be -1, non-negative, or None, got {self.motion_end_frame}"
            )
