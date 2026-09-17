#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Infer one strict drop orientation from a recorded HOI sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR / "runtime"))

from recorded_support import (  # noqa: E402
    DEFAULT_INITIAL_SEARCH_FRAMES,
    DEFAULT_STABLE_WINDOW_FRAMES,
    recorded_support_pose_save,
    recording_to_support_pose,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument(
        "--frame-start",
        type=int,
        help="Inclusive manual frame override; requires --frame-end",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        help="Exclusive manual frame override; requires --frame-start",
    )
    parser.add_argument(
        "--stable-window-frames",
        type=int,
        default=DEFAULT_STABLE_WINDOW_FRAMES,
        help="Window length for automatic initial-stability search (default: 30)",
    )
    parser.add_argument(
        "--initial-search-frames",
        type=int,
        default=DEFAULT_INITIAL_SEARCH_FRAMES,
        help="Initial prefix searched for a stable window (default: 90)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-up-deviation-degrees", type=float, default=2.0)
    parser.add_argument("--max-rotation-deviation-degrees", type=float, default=3.0)
    parser.add_argument("--max-translation-deviation-m", type=float, default=0.02)
    args = parser.parse_args()
    if (args.frame_start is None) != (args.frame_end is None):
        parser.error("--frame-start and --frame-end must be provided together")

    pose = recording_to_support_pose(
        args.sequence_dir,
        frame_start=args.frame_start,
        frame_end_exclusive=args.frame_end,
        stable_window_frames=args.stable_window_frames,
        initial_search_frames=args.initial_search_frames,
        max_up_deviation_degrees=args.max_up_deviation_degrees,
        max_rotation_deviation_degrees=args.max_rotation_deviation_degrees,
        max_translation_deviation_m=args.max_translation_deviation_m,
    )
    output = recorded_support_pose_save(pose, args.output)
    result = pose.to_dict()
    result["output"] = str(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
