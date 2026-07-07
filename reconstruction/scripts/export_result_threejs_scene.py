#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper for the pipeline-owned Three.js exporter."""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from v2d.pipelines.export_result_threejs_scene import main
except ModuleNotFoundError as exc:
    if exc.name not in {"v2d", "v2d.pipelines", "v2d.pipelines.export_result_threejs_scene"}:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules" / "v2d_pipelines"))
    from export_result_threejs_scene import main


if __name__ == "__main__":
    main()
