# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native JAX implementation of the FlashSAC reinforcement-learning algorithm.

The algorithm is ported from Holiday Robotics FlashSAC at the pinned commit below. This package intentionally
keeps its public initializer lightweight so replay and configuration workflows do not import learner frameworks.
"""

UPSTREAM_REPOSITORY = "https://github.com/Holiday-Robot/FlashSAC"
UPSTREAM_COMMIT = "87edc9061150ae9e962dd84e6544e27a1554b3ab"

__all__ = ["UPSTREAM_COMMIT", "UPSTREAM_REPOSITORY"]
