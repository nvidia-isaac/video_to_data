# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Run the WiLoR ego pipeline with a caller-provided object OBJ.

This entry point is the object-mesh variant of run_ego_wilor.py. It still uses
--object_prompt for GroundingDINO/SAM2 object masks, but it skips SAM3D mesh
generation and FoundationPose scale estimation. The provided OBJ is tracked
directly and its scale is treated as authoritative.

Run from reconstruction/:

    python modules/v2d_pipelines/run_ego_wilor_with_obj.py \\
        --video_path data/clip.mp4 \\
        --output_dir data/outputs/clip_obj \\
        --object_prompt "red mug" \\
        --object_mesh_path data/assets/red_mug.obj
"""

from __future__ import annotations

try:
    from v2d.pipelines.run_ego_wilor import parse_args, run_from_args
except ModuleNotFoundError as exc:  # Direct execution before editable install.
    if exc.name not in {"v2d", "v2d.pipelines", "v2d.pipelines.run_ego_wilor"}:
        raise
    from run_ego_wilor import parse_args, run_from_args


def main() -> None:
    args = parse_args()
    if args.object_prompt is None:
        raise SystemExit("run_ego_wilor_with_obj.py requires --object_prompt for object masks")
    if args.object_mesh_path is None:
        raise SystemExit("run_ego_wilor_with_obj.py requires --object_mesh_path")
    run_from_args(args)


if __name__ == "__main__":
    main()
