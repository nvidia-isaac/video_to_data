# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Objective shaping: Gaussian or Laplacian decay of a squared error.

Gaussian is ``exp(-SSE/var)``; Laplacian (the default) is ``exp(-sqrt(SSE)/var)``. Each routes through
:func:`shaped_objective`, selected by an int ``kind`` (:class:`ObjectiveShape`) — a config flag, not a
code fork.
"""

from __future__ import annotations

from enum import IntEnum

import warp as wp


class ObjectiveShape(IntEnum):
    LAPLACIAN = 0  # exp(-sqrt(sse)/var) — default
    GAUSSIAN = 1  # exp(-sse/var)


@wp.func
def shaped_objective(sse: float, var: float, kind: int, threshold: float) -> float:
    """Shaped value in ``(0, 1]`` from a squared error ``sse``. ``metric = sqrt(sse)`` (Laplacian,
    ``kind=0``) or ``sse`` (Gaussian, ``kind=1``); value ``= exp(-max(metric - threshold, 0) / var)``."""
    if kind == 0:
        metric = wp.sqrt(sse)
    else:
        metric = sse
    return wp.exp(-wp.max(metric - threshold, 0.0) / var)
