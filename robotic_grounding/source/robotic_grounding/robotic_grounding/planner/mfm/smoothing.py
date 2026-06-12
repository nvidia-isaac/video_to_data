# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporal smoothing for planner qpos output.

Hamming-window convolution and heading-safe qpos smoother.
Operates on torch tensors (matching the stitching pipeline).
"""

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, D417
# Planner is still in active development and this file is likely to change
# significantly with the new groot planner. Suppress annotation/docstring
# lint for now; real code issues are fixed individually.

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def hamming_smooth(signal: torch.Tensor, width: int) -> torch.Tensor:
    """Apply Hamming-window convolution to a [T, C] tensor along time.

    Pads edges by repeating the first/last frame.

    Args:
        signal: [T, C] tensor.
        width: Half-width of the Hamming window. Kernel size = 2*width + 1.

    Returns:
        Smoothed [T, C] tensor.
    """
    T, C = signal.shape

    padded = torch.cat(
        [
            signal[[0]].expand(width, -1),
            signal,
            signal[[-1]].expand(width, -1),
        ],
        dim=0,
    ).T.unsqueeze(
        0
    )  # [1, C, T+2*width]

    kernel_size = 2 * width + 1
    kernel = torch.hamming_window(kernel_size, device=signal.device)
    kernel = kernel / kernel.sum()
    kernel = kernel[None, None].expand(C, -1, -1)

    return F.conv1d(padded, kernel, groups=C)[0].T  # [T, C]


def smooth_qpos(
    qpos: torch.Tensor,
    pos_width: int = 6,
    joint_width: int = 4,
    heading_width: int = 8,
) -> torch.Tensor:
    """Smooth qpos trajectory with heading-safe quaternion handling.

    Layout: [pos(3), xyzw_quat(4), joints(29)] = 36 dims.

    Args:
        qpos: [T, 36] tensor.
        pos_width: Hamming half-width for root position.
        joint_width: Hamming half-width for joint angles.
        heading_width: Hamming half-width for yaw heading.

    Returns:
        [T, 36] smoothed qpos tensor.
    """
    qpos = qpos.clone()

    qpos[:, :3] = hamming_smooth(qpos[:, :3], pos_width)
    qpos[:, 7:] = hamming_smooth(qpos[:, 7:], joint_width)

    # Extract yaw from xyzw quaternion
    qx, qy, qz, qw = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    # Unwrap yaw to prevent 2π discontinuities
    dyaw = torch.diff(yaw)
    dyaw = dyaw - 2 * np.pi * torch.round(dyaw / (2 * np.pi))
    yaw_unwrapped = torch.cat([yaw[:1], yaw[:1] + torch.cumsum(dyaw, dim=0)])

    yaw_smooth = hamming_smooth(yaw_unwrapped[:, None], heading_width)[:, 0]

    # Rebuild pure-yaw quaternion (xyzw convention)
    half = yaw_smooth / 2
    qpos[:, 3] = 0.0  # qx
    qpos[:, 4] = 0.0  # qy
    qpos[:, 5] = torch.sin(half)  # qz
    qpos[:, 6] = torch.cos(half)  # qw

    return qpos
