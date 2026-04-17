"""Stereo image preprocessing: rectification, rescaling, and cropping.

Processes a single stereo image pair and returns updated camera parameters.
"""

import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

from v2d.mv.rig import CameraParam, RigConfig
from v2d.mv.io.video import FrameSource

from v2d.mv.preprocess.lib.image_proc import (
    ImagePipeline,
    image_proc_build_rectify,
    image_proc_build_rescale,
    image_proc_build_center_crop,
)


logger = logging.getLogger(__name__)


def _process_frame(
    i: int,
    left_files: list[Path],
    right_files: list[Path],
    left_output_image_dir: Path,
    right_output_image_dir: Path,
    left_pipeline: ImagePipeline,
    right_pipeline: ImagePipeline,
):
    img1 = iio.imread(left_files[i], plugin="pillow")
    img2 = iio.imread(right_files[i], plugin="pillow")

    img1 = left_pipeline(img1)
    img2 = right_pipeline(img2)

    iio.imwrite(left_output_image_dir / left_files[i].name, img1)
    iio.imwrite(right_output_image_dir / right_files[i].name, img2)


def _generate_video(image_dir: Path, output_path: Path, fps: int = 30, crf: int = 17):
    """Generate an MP4 video from a directory of PNG images using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", str(image_dir / "*.png"),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ], check=True, capture_output=True)


def _scale_focal(param: CameraParam, scale: float) -> CameraParam:
    """Scale focal length and projection matrix by a factor."""
    param.K[:2, :2] *= scale
    if param.P is not None:
        param.P[:2, :2] *= scale
        param.P[:2, 3] *= scale
    return param


def preprocess_stereo(
    left_source: FrameSource,
    right_source: FrameSource,
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
    left_output_video_path: Path | None = None,
    right_output_video_path: Path | None = None,
) -> tuple[tuple[ImagePipeline, ImagePipeline], tuple[CameraParam, CameraParam]]:
    """Preprocess a stereo pair: rectify, rescale, and crop.

    Args:
        left_source: FrameSource for left camera images.
        right_source: FrameSource for right camera images.
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
        left_output_video_path: Optional path to write left camera preview video.
        right_output_video_path: Optional path to write right camera preview video.

    Returns:
        ((left_pipeline, right_pipeline), (left_param, right_param)).
    """
    if num_workers is None:
        num_workers = os.cpu_count()
    if correction_focal is None:
        correction_focal = {}

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

    # Process frames
    os.makedirs(left_output_image_dir, exist_ok=True)
    os.makedirs(right_output_image_dir, exist_ok=True)

    left_files = left_source.image_paths
    right_files = right_source.image_paths

    if len(left_files) != len(right_files):
        left_names = {p.name for p in left_files}
        right_names = {p.name for p in right_files}
        only_left = sorted(left_names - right_names)
        only_right = sorted(right_names - left_names)
        logger.error(
            f"Frame count mismatch: left={len(left_files)}, right={len(right_files)}. "
            f"Only in left ({len(only_left)}): {only_left[:10]}. "
            f"Only in right ({len(only_right)}): {only_right[:10]}."
        )
        raise AssertionError(
            f"Mismatch in number of left ({len(left_files)}) and right ({len(right_files)}) images"
        )
    assert len(left_files) > 0, "No frames to process"

    logger.info(f"Processing {len(left_files)} frames with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _process_frame, i,
                left_files, right_files,
                left_output_image_dir, right_output_image_dir,
                left_pipeline, right_pipeline,
            )
            for i in range(len(left_files))
        ]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            fut.result()

    logger.info(f"{len(left_files)} frames processed")

    # Generate preview videos
    if left_output_video_path is not None:
        _generate_video(left_output_image_dir, left_output_video_path)
    if right_output_video_path is not None:
        _generate_video(right_output_image_dir, right_output_video_path)

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
    left_source = FrameSource(image_dir=args.left_image_dir, frames_slice=frames_slice)
    right_source = FrameSource(image_dir=args.right_image_dir, frames_slice=frames_slice)

    (left_pipeline, right_pipeline), (left_param, right_param) = preprocess_stereo(
        left_source=left_source,
        right_source=right_source,
        left_output_image_dir=args.left_output_image_dir,
        right_output_image_dir=args.right_output_image_dir,
        left_param=rig.get_camera(args.left_cam_id).param,
        right_param=rig.get_camera(args.right_cam_id).param,
        scale=args.scale,
        output_resolution=tuple(args.output_resolution) if args.output_resolution else None,
        left_cam_id=args.left_cam_id,
        right_cam_id=args.right_cam_id,
        num_workers=args.num_workers,
        left_output_video_path=args.left_output_video_path,
        right_output_video_path=args.right_output_video_path,
    )

    rig.cameras[args.left_cam_id].param = left_param
    rig.cameras[args.right_cam_id].param = right_param
    rig.save_camera_params(
        source_path=args.camera_params_path,
        output_path=args.output_camera_params_path,
    )
