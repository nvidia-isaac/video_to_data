# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full ego hand + object reconstruction pipeline, fed by ingestion segments.

For each segment in `runs/<run>/clips_final.jsonl` (or `outputs/graph.db`):
  1. ffmpeg-cut the source video to [start_t, end_t] -> seg_dir/clip.mp4
  2. Hand off to reconstruction's `run_v2d_ego_e2e.py` orchestrator (16 steps;
     each shells out to a v2d_*:latest container internally) by subprocessing
     into reconstruction's own `.venv`.

Why this shape:
- The orchestrator imports `v2d.*` Python directly; running it requires the
  lightweight orchestration packages (v2d_docker, v2d_common, v2d_depth, the
  10 docker wrappers, v2d_pipelines, trimesh) installed in some Python env.
- We keep ingestion's `.venv` clear of those packages by using a dedicated
  `reconstruction/.venv` and crossing the boundary as a subprocess.
- Stdout/stderr inherit so the orchestrator's `[run ] <label>` / `[skip] <label>`
  markers from `_step()` flow straight through to whoever invoked us — the
  webapp's reconstruction service parses those to drive the 16-stage status bar.

See `reconstruction/docs/ego_e2e_setup.md` for the upstream container-build +
MANO + BMC weight-staging steps. Run from `video_ingestion_agent/.venv`:

    python -m video_ingestion_agent.reconstruction_interface.ego_e2e.run_ego_e2e \\
        --segments runs/<run>/clips_final.jsonl \\
        --reconstruction-python ../reconstruction/.venv/bin/python \\
        --reconstruction-root ../reconstruction \\
        --out outputs/ego_e2e_demo \\
        --moge-weights /tmp/moge_weights \\
        --grounding-dino-weights /tmp/gd_weights \\
        --sam2-weights /tmp/sam2_weights \\
        --sam3d-weights /tmp/sam3d_weights \\
        --foundation-pose-weights /tmp/fp_weights \\
        --hand-reconstruction-weights /tmp/hand_weights \\
        --depth-source moge --limit 1
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from video_ingestion_agent.reconstruction_interface._common.ingestion_io import (
    IngestedSegment,
    read_segments_from_clips_jsonl,
    read_segments_from_graph_db,
)

log = logging.getLogger("ego_e2e")


def slice_video_if_missing(src: Path, start_t: float, end_t: float, dst: Path) -> Path:
    """ffmpeg cut [start_t, end_t]. -c copy is fast (cuts on keyframes). Idempotent."""
    if dst.is_file():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start_t:.3f}",
            "-to",
            f"{end_t:.3f}",
            "-i",
            str(src),
            "-c",
            "copy",
            str(dst),
        ],
        check=True,
    )
    return dst


def run_orchestrator(
    seg: IngestedSegment,
    out_root: Path,
    reconstruction_python: Path,
    reconstruction_root: Path,
    moge_weights: Path,
    grounding_dino_weights: Path,
    sam2_weights: Path,
    sam3d_weights: Path,
    foundation_pose_weights: Path,
    hand_reconstruction_weights: Path,
    depth_source: str,
    reference_frame: int,
    reregister_iou_thresh: float,
    smooth_sigma: float,
) -> None:
    """Slice the segment + invoke `run_v2d_ego_e2e.py` once. Stdout inherits
    so the orchestrator's `_step()` markers reach the caller's terminal/pipe.
    """
    seg_dir = (out_root / seg.segment_id).resolve()
    seg_dir.mkdir(parents=True, exist_ok=True)
    clip = slice_video_if_missing(seg.video_path, seg.start_t, seg.end_t, seg_dir / "clip.mp4")

    e2e_script = reconstruction_root / "modules" / "v2d_pipelines" / "run_v2d_ego_e2e.py"
    if not e2e_script.is_file():
        raise FileNotFoundError(
            f"orchestrator not found at {e2e_script} — check --reconstruction-root"
        )

    cmd = [
        str(reconstruction_python),
        str(e2e_script),
        "--video_path",
        str(clip),
        "--prompt",
        seg.object_label or "",
        "--output_dir",
        str(seg_dir),
        "--depth_source",
        depth_source,
        "--reference_frame",
        str(reference_frame),
        "--reregister_iou_thresh",
        f"{reregister_iou_thresh}",
        "--smooth_sigma",
        f"{smooth_sigma}",
        "--moge_weights",
        str(moge_weights),
        "--grounding_dino_weights",
        str(grounding_dino_weights),
        "--sam2_weights",
        str(sam2_weights),
        "--sam3d_weights",
        str(sam3d_weights),
        "--foundation_pose_weights",
        str(foundation_pose_weights),
        "--hand_reconstruction_weights",
        str(hand_reconstruction_weights),
    ]
    log.info("segment %s: handing off to orchestrator", seg.segment_id)
    log.debug("  cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(reconstruction_root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--segments", type=Path, help="Path to clips_final.jsonl.")
    src.add_argument("--graph-db", type=Path, help="Path to graph.db.")
    parser.add_argument("--out", type=Path, required=True, help="Per-segment subdirs root.")
    parser.add_argument(
        "--reconstruction-python",
        type=Path,
        required=True,
        help="Path to reconstruction's .venv/bin/python (the orchestrator's interpreter).",
    )
    parser.add_argument(
        "--reconstruction-root",
        type=Path,
        required=True,
        help="Path to reconstruction package root (containing modules/v2d_pipelines/).",
    )
    parser.add_argument("--moge-weights", type=Path, required=True)
    parser.add_argument("--grounding-dino-weights", type=Path, required=True)
    parser.add_argument("--sam2-weights", type=Path, required=True)
    parser.add_argument("--sam3d-weights", type=Path, required=True)
    parser.add_argument("--foundation-pose-weights", type=Path, required=True)
    parser.add_argument("--hand-reconstruction-weights", type=Path, required=True)
    parser.add_argument(
        "--depth-source",
        choices=["moge", "vipe"],
        default="moge",
        help="Depth source for SAM3D/FP/alignment (default: moge).",
    )
    parser.add_argument(
        "--ref-frame",
        type=int,
        default=0,
        help="Reference frame for DINO + SAM3D + FP registration (default: 0).",
    )
    parser.add_argument(
        "--reregister-iou-thresh",
        type=float,
        default=0.3,
        help="FP re-registration IoU threshold (default: 0.3).",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=5.0,
        help="Hand-translation Gaussian sigma in frames (default: 5.0).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N segments.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    segments = (
        read_segments_from_clips_jsonl(args.segments)
        if args.segments
        else read_segments_from_graph_db(args.graph_db)
    )
    if not segments:
        log.error("no segments read from input")
        sys.exit(1)
    if args.limit is not None:
        segments = segments[: args.limit]

    log.info("processing %d segments -> %s", len(segments), args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    for seg in segments:
        if not seg.video_path.is_file():
            log.warning("skipping %s — video not found at %s", seg.segment_id, seg.video_path)
            n_fail += 1
            continue
        try:
            run_orchestrator(
                seg,
                out_root=args.out,
                reconstruction_python=args.reconstruction_python,
                reconstruction_root=args.reconstruction_root,
                moge_weights=args.moge_weights,
                grounding_dino_weights=args.grounding_dino_weights,
                sam2_weights=args.sam2_weights,
                sam3d_weights=args.sam3d_weights,
                foundation_pose_weights=args.foundation_pose_weights,
                hand_reconstruction_weights=args.hand_reconstruction_weights,
                depth_source=args.depth_source,
                reference_frame=args.ref_frame,
                reregister_iou_thresh=args.reregister_iou_thresh,
                smooth_sigma=args.smooth_sigma,
            )
            n_ok += 1
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.error("segment %s failed: %s", seg.segment_id, e)
            n_fail += 1

    log.info("done: %d ok, %d failed", n_ok, n_fail)


if __name__ == "__main__":
    main()
