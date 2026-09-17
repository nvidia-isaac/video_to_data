# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Playback timing utilities shared by motion consumers."""

from __future__ import annotations

import math


def resolve_playback_timing(
    motion_dt: float, env_step_dt: float, motion_speed: float
) -> tuple[float, float]:
    """Validate timing and return ``(playback_dt, target_fps)``.

    Motion consumers advance by one reference frame per environment step, so
    the reference must be sampled at ``1 / (env_step_dt * motion_speed)``.
    """
    if not math.isfinite(env_step_dt) or env_step_dt <= 0.0:
        raise ValueError(
            f"Environment step_dt must be positive and finite, got {env_step_dt}."
        )
    if not math.isfinite(motion_dt) or motion_dt <= 0.0:
        raise ValueError(f"motion.dt must be positive and finite, got {motion_dt}.")
    if not math.isclose(motion_dt, env_step_dt, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(
            f"motion.dt ({motion_dt}) must equal env.step_dt ({env_step_dt}). "
            "Set motion.dt to sim.dt * decimation; source motion FPS is handled "
            "by automatic resampling."
        )
    if not math.isfinite(motion_speed) or motion_speed <= 0.0:
        raise ValueError(
            f"motion.motion_speed must be positive and finite, got {motion_speed}."
        )
    return env_step_dt, 1.0 / (env_step_dt * motion_speed)
