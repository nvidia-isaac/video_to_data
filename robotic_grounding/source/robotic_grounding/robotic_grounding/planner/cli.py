# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI surface for the G1 whole-body planner.

`parse_args` builds the argparse namespace consumed by `g1_planner.main()`.
It lives in its own module so the orchestration script stays focused on
the pipeline rather than CLI plumbing.
"""

from __future__ import annotations

import argparse

# CLI defaults for the planner's nominal-hold + interp + hold-start
# approach segment. Mirror the constants `g1_planner.main` uses at runtime;
# duplicated here as literals so this module has no dependency on
# `g1_planner.py` and stays free of circular-import gymnastics.
_HOLD_START_S = 5.0
_INTERP_DURATION_S = 5.0
_HOLD_END_S = 5.0
_ROOT_FIX_COMPONENTS = ("x", "y", "z", "roll", "pitch", "yaw")


def parse_args() -> argparse.Namespace:
    """Build the planner CLI parser and return the parsed namespace."""
    parser = argparse.ArgumentParser(description="G1 whole-body planner")
    parser.add_argument("--robot", choices=["sharpa", "dex3"], default="sharpa")
    parser.add_argument("--v2p_parquet", required=True)
    parser.add_argument("--v2p_robot_name", default="sharpa_wave")
    parser.add_argument("--v2p_sequence", default="box_grab")
    parser.add_argument("--v2p_trajectory_id", type=int, default=0)
    parser.add_argument(
        "--v2p_start_frame",
        type=int,
        default=0,
        help=(
            "Drop this many frames from the interpolated V2P reference before "
            "building the planner warmup/interp trajectory. Useful for skipping "
            "dataset-specific T-pose/approach lead-ins."
        ),
    )
    parser.add_argument(
        "--v2p_start_at_first_contact",
        action="store_true",
        help=(
            "Start the reference at the first detected hand-object contact "
            "minus --v2p_pre_contact_frames."
        ),
    )
    parser.add_argument(
        "--v2p_pre_contact_frames",
        type=int,
        default=10,
        help="Number of interpolated V2P frames to keep before first contact.",
    )
    parser.add_argument(
        "--v2p_end_after_last_contact_frames",
        type=int,
        default=-1,
        help=(
            "If >= 0, truncate the interpolated V2P reference after the last "
            "detected hand-object contact plus this many frames. A value of 0 "
            "keeps through the last contact frame."
        ),
    )
    parser.add_argument("--target_fps", type=float, default=150.0)
    parser.add_argument("--hold_start_s", type=float, default=_HOLD_START_S)
    parser.add_argument("--interp_s", type=float, default=_INTERP_DURATION_S)
    parser.add_argument("--hold_end_s", type=float, default=_HOLD_END_S)
    parser.add_argument(
        "--no_approach",
        action="store_true",
        help=(
            "Disable the planner's nominal hold/interp/hold approach segment. "
            "The generated trajectory starts directly at the V2P reference."
        ),
    )
    parser.add_argument(
        "--workspace_offset", type=float, nargs=3, default=[-0.10, 0.0, -0.15]
    )
    parser.add_argument("--ref_seconds", type=float, default=-1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--ik_verify", action="store_true")
    parser.add_argument("--ik_plan", action="store_true")
    parser.add_argument(
        "--fix_lower_body",
        action="store_true",
        help=(
            "Override the model's lower-body (hip/knee/ankle) predictions "
            "with a static crouch and run the AR-aware loop that pins those "
            "bodies in the model's chunk seeds."
        ),
    )
    parser.add_argument(
        "--fix_root",
        nargs="+",
        choices=_ROOT_FIX_COMPONENTS,
        default=(),
        help=(
            "Pin selected root components. Components are x y z roll pitch yaw; "
            "e.g. '--fix_root z roll pitch' clamps height and roll/pitch while "
            "leaving root XY translation and yaw free."
        ),
    )
    parser.add_argument(
        "--fix_root_pos",
        action="store_true",
        help="Legacy alias for '--fix_root x y z'.",
    )
    parser.add_argument(
        "--fix_root_z",
        action="store_true",
        help="Legacy alias for '--fix_root z'.",
    )
    parser.add_argument(
        "--fix_root_rot",
        action="store_true",
        help="Legacy alias for '--fix_root roll pitch yaw'.",
    )
    parser.add_argument(
        "--fix_root_rp",
        action="store_true",
        help="Legacy alias for '--fix_root roll pitch'.",
    )
    parser.add_argument(
        "--no_smooth_qpos",
        action="store_true",
        help="Disable post-inference qpos smoothing (global Hamming + boundary blend).",
    )
    parser.add_argument(
        "--search_heading_deg",
        type=float,
        default=0.0,
        help=(
            "If > 0, run inference at heading offsets [-N, -N/2, 0, +N/2, +N] "
            "degrees around the heading-toward-object correction and pick the "
            "candidate with the lowest mean wrist tracking error."
        ),
    )
    parser.add_argument(
        "--heading_align_frame",
        choices=("start", "first_contact"),
        default="start",
        help=(
            "Frame used for the heading-toward-object correction. 'start' "
            "keeps the legacy behavior; 'first_contact' uses the detected "
            "first contact frame within the trimmed reference."
        ),
    )
    return parser.parse_args()
