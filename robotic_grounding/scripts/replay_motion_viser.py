# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Viser-based replay for motion_v1 parquets.

Loads a partition and opens a browser-accessible viser scene with a Frame
slider and Play/FPS/Loop/Step controls for data-quality inspection.

Shows robot + object + per-side contact markers. The parquet carries no
source body mesh, so no human-body overlay is available in replay; that
overlay is a retarget-time layer provided by ``nvhuman_to_g1.py`` when it
streams frames through the same ``ViserPlayback`` module.

Usage:
    python scripts/replay_viser.py --motion_file <partition_dir_or_file>
    python scripts/replay_viser.py --motion_file <path> --port 8080 --start-paused
    python scripts/replay_viser.py --motion_file <path> --start-frame 120
"""

from __future__ import annotations

import argparse

from robotic_grounding.retarget.viser_playback import ViserPlayback


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion_file",
        type=str,
        required=True,
        help="motion_v1 parquet partition directory or parquet file path.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser HTTP port (default: 8080).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for MotionData tensors (default: cpu).",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Open viewer paused so the user can scrub.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Seek to this frame on open (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    """Open viser on the configured port and enter the tick loop."""
    args = _parse_args()
    playback = ViserPlayback(
        motion_file=args.motion_file,
        port=args.port,
        device=args.device,
        start_paused=args.start_paused,
        start_frame=args.start_frame,
    )
    print(
        f"[replay_viser] serving on http://localhost:{args.port}  "
        f"(motion_file={args.motion_file})"
    )
    playback.run()


if __name__ == "__main__":
    main()
