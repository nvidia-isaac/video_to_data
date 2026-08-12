# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject sequences where the retargeted robot hand penetrates an object or itself by > 2 cm.

Wraps the capsule-based penetration check from ``scripts/filter_penetrations.py``.
Checks both hand-object penetration (convex hull signed-distance) and hand-hand
penetration (analytic capsule-capsule distance).  Hollow objects (AR glasses,
mugs, bowls) are skipped via hull_volume_ratio filtering to avoid false positives.

score = max penetration depth in centimetres across all sampled frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

# filter_penetrations.py lives one directory up (scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filter_penetrations import check  # noqa: F401, E402
