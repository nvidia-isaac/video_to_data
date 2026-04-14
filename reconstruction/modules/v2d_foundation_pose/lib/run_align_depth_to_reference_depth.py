"""Align a folder of raw depth images to a metric reference depth via ICP + affine solve.

For each frame in depth_folder, finds (scale, shift) such that
scale * D_raw + shift best matches the reference depth on the static background,
accounting for small camera motion via ICP. The foreground mask excludes the
object and hands from fitting.
"""
import argparse
import logging
import os

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask
from v2d.foundation_pose.lib.depth_alignment import align_depth_to_reference_depth

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_align_depth_to_reference_depth(
    depth_folder: str,
    depth_reference_path: str,
    intrinsics_path: str,
    output_folder: str,
    masks_folder: str = None,
    reference_mask_path: str = None,
    n_iterations: int = 3,
    outlier_trim_ratio: float = 0.2,
    max_points: int = 20000,
) -> None:
    """Align all depth frames in a folder to a metric reference depth.

    Args:
        depth_folder:          Directory of raw monocular depth PNGs to align.
        depth_reference_path:  Metric reference depth PNG (from align_depth_to_object).
        intrinsics_path:       Camera intrinsics JSON.
        output_folder:         Destination for aligned depth PNGs.
        masks_folder:          Optional per-frame foreground mask folder. Masks exclude
                               the object and hands from ICP fitting.
        reference_mask_path:   Optional foreground mask for the reference frame.
                               If None, per-frame mask is reused as a proxy.
        n_iterations:          Alternating ICP/affine iterations per frame. Default 3.
        outlier_trim_ratio:    Fraction of worst-fitting points discarded per iter.
                               Default 0.2.
        max_points:            Max background points per frame (random subsample).
                               Default 20000.
    """
    intrinsics = CameraIntrinsics.load(intrinsics_path)
    depth_ref = DepthImage.load(depth_reference_path)
    fg_mask_ref = Mask.load(reference_mask_path) if reference_mask_path else None

    depth_files = sorted(f for f in os.listdir(depth_folder) if f.endswith('.png'))
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Aligning {len(depth_files)} depth frames to reference")

    for fname in depth_files:
        frame_idx = int(os.path.splitext(fname)[0])
        depth_raw = DepthImage.load(os.path.join(depth_folder, fname))

        fg_mask = None
        if masks_folder is not None:
            mask_path = os.path.join(masks_folder, fname)
            if os.path.exists(mask_path):
                fg_mask = Mask.load(mask_path)

        logger.info(f"Frame {frame_idx}")
        corrected = align_depth_to_reference_depth(
            depth_raw, depth_ref, intrinsics,
            fg_mask=fg_mask,
            fg_mask_ref=fg_mask_ref,
            n_iterations=n_iterations,
            outlier_trim_ratio=outlier_trim_ratio,
            max_points=max_points,
        )
        corrected.save(os.path.join(output_folder, fname))

    logger.info(f"Aligned depth written to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align depth folder to reference depth via ICP + affine")
    parser.add_argument("--depth_folder", required=True)
    parser.add_argument("--depth_reference_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--masks_folder", default=None)
    parser.add_argument("--reference_mask_path", default=None)
    parser.add_argument("--n_iterations", type=int, default=3)
    parser.add_argument("--outlier_trim_ratio", type=float, default=0.2)
    parser.add_argument("--max_points", type=int, default=20000)
    args = parser.parse_args()
    run_align_depth_to_reference_depth(
        args.depth_folder, args.depth_reference_path, args.intrinsics_path,
        args.output_folder,
        masks_folder=args.masks_folder,
        reference_mask_path=args.reference_mask_path,
        n_iterations=args.n_iterations,
        outlier_trim_ratio=args.outlier_trim_ratio,
        max_points=args.max_points,
    )
