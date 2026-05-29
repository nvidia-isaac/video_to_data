# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Stereo image preprocessing: rectification, rescaling, and cropping.

Processes a single stereo image pair and returns updated camera parameters.
"""

import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from v2d.common.video import FrameSource, FrameWriter
from v2d.mv.rig import CameraParam, RigConfig

from v2d.mv.preprocess.lib.image_proc import (
    ImagePipeline,
    image_proc_build_rectify,
    image_proc_build_rescale,
    image_proc_build_center_crop,
)


logger = logging.getLogger(__name__)


def _rectify_frame(
    order: int,
    left_idx: int,
    right_idx: int,
    left_source: FrameSource,
    right_source: FrameSource,
    left_pipeline: ImagePipeline,
    right_pipeline: ImagePipeline,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Read a matched pair from both sources and run rectification pipelines.

    Returns (order, left_rectified, right_rectified).  No I/O writes.
    """
    img1 = left_pipeline(left_source[left_idx])
    img2 = right_pipeline(right_source[right_idx])
    return order, img1, img2


def _scale_focal(param: CameraParam, scale: float) -> CameraParam:
    """Scale focal length and projection matrix by a factor."""
    param.K[:2, :2] *= scale
    if param.P is not None:
        param.P[:2, :2] *= scale
        param.P[:2, 3] *= scale
    return param


def preprocess_stereo(
    left_path: Path,
    right_path: Path,
    left_output_image_dir: Path,
    right_output_image_dir: Path,
    left_param: CameraParam,
    right_param: CameraParam,
    scale: float = 1.0,
    output_resolution: tuple[int, int] | None = None,
    correction_focal: dict[int, float] | None = None,
    left_cam_id: int | None = None,
    right_cam_id: int | None = None,
    num_workers: int | None = None,
    frames_slice: slice | None = None,
    left_output_video_path: Path | None = None,
    right_output_video_path: Path | None = None,
) -> tuple[tuple[ImagePipeline, ImagePipeline], tuple[CameraParam, CameraParam]]:
    """Preprocess a stereo pair: rectify, rescale, and crop.

    Args:
        left_path: Path to left camera frames (image dir, .h5, or video file).
        right_path: Path to right camera frames (image dir, .h5, or video file).
        left_output_image_dir: Output directory for processed left frames.
        right_output_image_dir: Output directory for processed right frames.
        left_param: Camera parameters for the left camera.
        right_param: Camera parameters for the right camera.
        scale: Scale factor for images (e.g. 0.8 for 80%).
        output_resolution: (width, height) target after center cropping.
        correction_focal: Per-camera focal length correction factors (keyed by cam_id).
        left_cam_id: Camera ID for the left camera (used for correction_focal lookup).
        right_cam_id: Camera ID for the right camera (used for correction_focal lookup).
        num_workers: Number of parallel workers (defaults to CPU count).
        frames_slice: Optional slice to limit frame range.
        left_output_video_path: Optional path to write left camera preview video.
        right_output_video_path: Optional path to write right camera preview video.

    Returns:
        ((left_pipeline, right_pipeline), (left_param, right_param)).
    """
    if num_workers is None:
        num_workers = os.cpu_count()
    if correction_focal is None:
        correction_focal = {}

    left_source = FrameSource.from_path(left_path, frames_slice=frames_slice)
    right_source = FrameSource.from_path(right_path, frames_slice=frames_slice)

    logger.info(
        f"Processing stereo pair (cam {left_cam_id} / {right_cam_id})"
        f"\n\t- Scale: {scale}, Output resolution: {output_resolution}"
    )

    # Build image processing pipeline
    left_pipeline = ImagePipeline()
    right_pipeline = ImagePipeline()

    (left_rect, right_rect), (left_param, right_param) = image_proc_build_rectify(left_param, right_param)
    left_pipeline.add_processor(left_rect)
    right_pipeline.add_processor(right_rect)

    if scale != 1.0:
        left_rescale, left_param = image_proc_build_rescale(left_param, scale)
        right_rescale, right_param = image_proc_build_rescale(right_param, scale)
        left_pipeline.add_processor(left_rescale)
        right_pipeline.add_processor(right_rescale)

    if output_resolution is not None:
        w_target, h_target = output_resolution
        left_crop, left_param = image_proc_build_center_crop(left_param, w_target, h_target)
        right_crop, right_param = image_proc_build_center_crop(right_param, w_target, h_target)
        left_pipeline.add_processor(left_crop)
        right_pipeline.add_processor(right_crop)

    if left_cam_id is not None and left_cam_id in correction_focal:
        logger.warning(f"Applying focal correction {correction_focal[left_cam_id]} to camera {left_cam_id}")
        left_param = _scale_focal(left_param, correction_focal[left_cam_id])
    if right_cam_id is not None and right_cam_id in correction_focal:
        logger.warning(f"Applying focal correction {correction_focal[right_cam_id]} to camera {right_cam_id}")
        right_param = _scale_focal(right_param, correction_focal[right_cam_id])

    # Stem-matching: match left/right frames by stem name
    if left_source.n_frames != right_source.n_frames:
        logger.warning(
            f"Frame count mismatch: left={left_source.n_frames}, right={right_source.n_frames}"
        )
    right_stem_to_idx = {s: i for i, s in enumerate(right_source.stems)}
    matched_pairs: list[tuple[int, int, str]] = []
    for left_idx, stem in enumerate(left_source.stems):
        if stem in right_stem_to_idx:
            matched_pairs.append((left_idx, right_stem_to_idx[stem], stem))
        else:
            logger.warning(f"[skip] no matching right frame for stem: {stem}")

    n_matched = len(matched_pairs)
    assert n_matched > 0, "No matched frames to process"
    logger.info(f"Processing {n_matched} matched frames with {num_workers} workers...")

    # Streaming pipeline: rectify in thread pool, write in-order via bounded queue.
    # Caps peak memory to ~(maxsize * 2 frames * frame_size) instead of all frames.
    frame_queue: queue.Queue = queue.Queue(maxsize=num_workers * 2)
    writer_error: list[BaseException] = []

    left_writer = FrameWriter.from_path(left_output_image_dir)
    right_writer = FrameWriter.from_path(right_output_image_dir)
    left_vid_writer = FrameWriter.from_path(left_output_video_path) if left_output_video_path else None
    right_vid_writer = FrameWriter.from_path(right_output_video_path) if right_output_video_path else None

    pbar = tqdm(total=n_matched, desc="Rectify + write")

    def _writer_loop():
        """Drain the queue in order and write to all FrameWriters."""
        next_idx = 0
        pending: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        try:
            while True:
                item = frame_queue.get()
                if item is None:
                    break
                idx, img1, img2 = item
                pending[idx] = (img1, img2)
                while next_idx in pending:
                    im1, im2 = pending.pop(next_idx)
                    stem = matched_pairs[next_idx][2]
                    left_writer.write_frame(im1, stem=stem)
                    right_writer.write_frame(im2, stem=stem)
                    if left_vid_writer is not None:
                        left_vid_writer.write_frame(im1)
                    if right_vid_writer is not None:
                        right_vid_writer.write_frame(im2)
                    next_idx += 1
                    pbar.update(1)
        except BaseException as exc:
            writer_error.append(exc)

    writer_thread = threading.Thread(target=_writer_loop, daemon=True)
    writer_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _rectify_frame, order,
                    left_idx, right_idx,
                    left_source, right_source,
                    left_pipeline, right_pipeline,
                )
                for order, (left_idx, right_idx, _stem) in enumerate(matched_pairs)
            ]
            for fut in futures:
                frame_queue.put(fut.result())
        frame_queue.put(None)
        writer_thread.join()
    finally:
        pbar.close()
        left_writer.close()
        right_writer.close()
        if left_vid_writer is not None:
            left_vid_writer.close()
        if right_vid_writer is not None:
            right_vid_writer.close()

    if writer_error:
        raise writer_error[0]

    logger.info(f"{n_matched} frames processed")

    return (left_pipeline, right_pipeline), (left_param, right_param)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Single stereo pair preprocessing")
    parser.add_argument("--left_image_dir", type=Path, required=True)
    parser.add_argument("--right_image_dir", type=Path, required=True)
    parser.add_argument("--left_output_image_dir", type=Path, required=True)
    parser.add_argument("--right_output_image_dir", type=Path, required=True)
    parser.add_argument("--camera_params_path", type=Path, required=True)
    parser.add_argument("--rig_name", type=str, required=True)
    parser.add_argument("--left_cam_id", type=int, required=True)
    parser.add_argument("--right_cam_id", type=int, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output_resolution", type=int, nargs=2, default=None,
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_camera_params_path", type=Path, required=True)
    parser.add_argument("--left_output_video_path", type=Path, default=None)
    parser.add_argument("--right_output_video_path", type=Path, default=None)
    args = parser.parse_args()

    rig = RigConfig(args.rig_name, camera_params_path=args.camera_params_path)
    frames_slice = slice(args.start, args.stop, args.step)

    (left_pipeline, right_pipeline), (left_param, right_param) = preprocess_stereo(
        left_path=args.left_image_dir,
        right_path=args.right_image_dir,
        left_output_image_dir=args.left_output_image_dir,
        right_output_image_dir=args.right_output_image_dir,
        left_param=rig.get_camera(args.left_cam_id).param,
        right_param=rig.get_camera(args.right_cam_id).param,
        scale=args.scale,
        output_resolution=tuple(args.output_resolution) if args.output_resolution else None,
        left_cam_id=args.left_cam_id,
        right_cam_id=args.right_cam_id,
        num_workers=args.num_workers,
        frames_slice=frames_slice,
        left_output_video_path=args.left_output_video_path,
        right_output_video_path=args.right_output_video_path,
    )

    rig.cameras[args.left_cam_id].param = left_param
    rig.cameras[args.right_cam_id].param = right_param
    rig.save_camera_params(
        source_path=args.camera_params_path,
        output_path=args.output_camera_params_path,
    )
