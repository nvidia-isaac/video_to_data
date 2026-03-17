"""
Estimate the scale factor between a mesh and a scene's metric depth.

Uses a coarse-to-fine grid search: samples candidate scales in log-space,
registers the mesh with FoundationPose at each scale, renders the result,
and scores it against the observed depth and mask. The search range is halved
around the winner each level.
"""
import argparse
import json
import logging
import os

import cv2

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask
from v2d.common.datatypes import Image as V2dImage
from v2d.mesh.lib.mesh import Mesh
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_estimate_mesh_scale(
    mesh_path: str,
    rgb_path: str,
    depth_path: str,
    mask_path: str,
    intrinsics_path: str,
    weights_dir: str,
    scale_path: str,
    rescaled_mesh_path: str = None,
    lo: float = 0.5,
    hi: float = 2.0,
    n_samples: int = 7,
    n_levels: int = 3,
    iou_weight: float = 1.0,
    depth_weight: float = 1.0,
    registration_iterations: int = 5,
) -> float:
    """Estimate mesh scale via coarse-to-fine grid search.

    Args:
        mesh_path:               Input mesh file.
        rgb_path:                Single reference frame RGB image (PNG).
        depth_path:              Corresponding depth PNG (uint16 inverse-depth encoding).
        mask_path:               Corresponding segmentation mask PNG.
        intrinsics_path:         Camera intrinsics JSON.
        weights_dir:             FoundationPose weights directory.
        scale_path:              Output JSON path for {"scale": <best scale factor>}.
        rescaled_mesh_path:      If provided, saves the rescaled mesh here.
        lo:                      Lower bound of initial scale search range. Default 0.5.
        hi:                      Upper bound of initial scale search range. Default 2.0.
        n_samples:               Scales to evaluate per level. Default 7.
        n_levels:                Refinement levels. Default 3.
        iou_weight:              Weight for mask IoU score component. Default 1.0.
        depth_weight:            Weight for depth consistency score component. Default 1.0.
        registration_iterations: FP register() iterations per candidate. Default 5.

    Returns:
        Best scale factor relative to the original mesh.
    """
    mesh = Mesh.load(mesh_path)
    tracker = FoundationPoseTracker(mesh, weights_dir)

    intrinsics = CameraIntrinsics.load(intrinsics_path)
    depth = DepthImage.load(depth_path)
    mask = Mask.load(mask_path)

    frame = cv2.imread(rgb_path)
    if frame is None:
        raise RuntimeError(f"Failed to load RGB image: {rgb_path}")
    rgb = V2dImage(data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    scale = tracker.estimate_scale_grid_search(
        rgb, depth, mask, intrinsics,
        lo=lo, hi=hi,
        n_samples=n_samples, n_levels=n_levels,
        iou_weight=iou_weight, depth_weight=depth_weight,
        registration_iterations=registration_iterations,
    )
    logger.info(f"Best scale: {scale:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(scale_path)), exist_ok=True)
    with open(scale_path, "w") as f:
        json.dump({"scale": scale}, f)
    logger.info(f"Saved scale to {scale_path}")

    if rescaled_mesh_path:
        tracker._mesh.save(rescaled_mesh_path)
        logger.info(f"Saved rescaled mesh to {rescaled_mesh_path}")

    return scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate mesh scale via coarse-to-fine grid search")
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--rgb_path", required=True)
    parser.add_argument("--depth_path", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--scale_path", required=True)
    parser.add_argument("--rescaled_mesh_path", default=None)
    parser.add_argument("--lo", type=float, default=0.5)
    parser.add_argument("--hi", type=float, default=2.0)
    parser.add_argument("--n_samples", type=int, default=7)
    parser.add_argument("--n_levels", type=int, default=3)
    parser.add_argument("--iou_weight", type=float, default=1.0)
    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--registration_iterations", type=int, default=5)

    args = parser.parse_args()
    run_estimate_mesh_scale(
        args.mesh_path,
        args.rgb_path,
        args.depth_path,
        args.mask_path,
        args.intrinsics_path,
        args.weights_dir,
        args.scale_path,
        rescaled_mesh_path=args.rescaled_mesh_path,
        lo=args.lo,
        hi=args.hi,
        n_samples=args.n_samples,
        n_levels=args.n_levels,
        iou_weight=args.iou_weight,
        depth_weight=args.depth_weight,
        registration_iterations=args.registration_iterations,
    )
