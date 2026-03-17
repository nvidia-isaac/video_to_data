"""
FoundationPose video-to-poses processing function.
Can be called directly from command line or imported as a function.
"""
import os
import argparse
import logging

import numpy as np
import cv2

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Transform3d
from v2d.common.datatypes import Image as V2dImage
from v2d.mesh.lib.mesh import Mesh
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_video_to_poses(
    video_path: str,
    depth_folder: str,
    masks_folder: str,
    camera_intrinsics_path: str,
    mesh_path: str,
    poses_dir: str,
    weights_dir: str,
    reference_frame: int = 0,
    target_width: int = None,
    target_height: int = None,
    reregister_iou_thresh: float = None,
) -> None:
    """Process a video to track object poses and save per-frame Transform3d JSON files."""
    mesh = Mesh.load(mesh_path)
    tracker = FoundationPoseTracker(mesh, weights_dir)

    camera_intrinsics = CameraIntrinsics.load(camera_intrinsics_path)
    K = camera_intrinsics.to_matrix()

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if target_width and target_height:
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale_x
        K_scaled[1, 1] *= scale_y
        K_scaled[0, 2] *= scale_x
        K_scaled[1, 2] *= scale_y
        logger.info(f"Scaling {orig_width}x{orig_height} → {target_width}x{target_height}")
    else:
        target_width, target_height = orig_width, orig_height
        K_scaled = K

    scaled_intrinsics = CameraIntrinsics(
        fx=float(K_scaled[0, 0]), fy=float(K_scaled[1, 1]),
        cx=float(K_scaled[0, 2]), cy=float(K_scaled[1, 2]),
        width=target_width, height=target_height,
    )

    def _load_frame(idx: int) -> tuple:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            return None, None, None
        rgb_arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (target_width, target_height) != (orig_width, orig_height):
            rgb_arr = cv2.resize(rgb_arr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        depth_p = os.path.join(depth_folder, f"{idx:06d}.png")
        depth = None
        if os.path.exists(depth_p):
            d = DepthImage.load(depth_p)
            if (target_width, target_height) != (orig_width, orig_height):
                d = DepthImage(depth=cv2.resize(d.depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST))
            depth = d

        mask_p = os.path.join(masks_folder, f"{idx:06d}.png")
        mask = None
        if os.path.exists(mask_p):
            m = Mask.load(mask_p)
            if (target_width, target_height) != (orig_width, orig_height):
                m = Mask(mask=cv2.resize(m.mask.astype(np.float32), (target_width, target_height), interpolation=cv2.INTER_NEAREST))
            mask = m

        return V2dImage(data=rgb_arr), depth, mask

    def _save_pose(frame_idx: int, pose: Transform3d) -> None:
        os.makedirs(poses_dir, exist_ok=True)
        pose.save(os.path.join(poses_dir, f"{frame_idx:06d}.json"))

    def _apply_mask(depth: DepthImage, mask: Mask | None) -> DepthImage:
        """Zero out depth outside the object mask."""
        if mask is None:
            return depth
        masked = depth.depth.copy()
        masked[~mask.mask.astype(bool)] = 0.0
        return DepthImage(depth=masked)

    # Register at reference frame
    logger.info(f"Registering at reference frame {reference_frame}")
    rgb, depth, mask = _load_frame(reference_frame)
    if rgb is None or depth is None or mask is None:
        raise RuntimeError(f"Failed to load data for reference frame {reference_frame}")

    initial_pose = tracker.register(rgb, _apply_mask(depth, mask), mask, scaled_intrinsics)
    _save_pose(reference_frame, initial_pose)
    logger.info("Registered at reference frame")

    # Forward tracking
    logger.info(f"Tracking forward: {reference_frame + 1} → {num_frames - 1}")
    tracker.reset_to_pose(initial_pose)
    for frame_idx in range(reference_frame + 1, num_frames):
        rgb, depth, mask = _load_frame(frame_idx)
        if rgb is None or depth is None:
            break
        logger.info(f"Forward frame {frame_idx}/{num_frames}")
        masked_depth = _apply_mask(depth, mask)
        if reregister_iou_thresh is not None and mask is not None:
            pose, recovered = tracker.track_one_with_recovery(
                rgb, masked_depth, mask, scaled_intrinsics, iou_thresh=reregister_iou_thresh,
            )
            if recovered:
                logger.info(f"  Re-registered at frame {frame_idx}")
        else:
            pose = tracker.track_one(rgb, masked_depth, scaled_intrinsics)
        _save_pose(frame_idx, pose)

    # Backward tracking
    if reference_frame > 0:
        logger.info(f"Tracking backward: {reference_frame - 1} → 0")
        tracker.reset_to_pose(initial_pose)
        for frame_idx in range(reference_frame - 1, -1, -1):
            rgb, depth, mask = _load_frame(frame_idx)
            if rgb is None or depth is None:
                break
            logger.info(f"Backward frame {frame_idx}/{num_frames}")
            masked_depth = _apply_mask(depth, mask)
            if reregister_iou_thresh is not None and mask is not None:
                pose, recovered = tracker.track_one_with_recovery(
                    rgb, masked_depth, mask, scaled_intrinsics,
                    iteration=2, iou_thresh=reregister_iou_thresh,
                )
                if recovered:
                    logger.info(f"  Re-registered at frame {frame_idx}")
            else:
                pose = tracker.track_one(rgb, masked_depth, scaled_intrinsics, iteration=2)
            _save_pose(frame_idx, pose)

    cap.release()
    logger.info(f"Completed {num_frames} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to poses using FoundationPose")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--depth_folder", required=True)
    parser.add_argument("--masks_folder", required=True)
    parser.add_argument("--camera_intrinsics_path", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--poses_dir", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--reference_frame", type=int, default=0)
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--reregister_iou_thresh", type=float, default=None)

    args = parser.parse_args()
    run_video_to_poses(
        args.video_path,
        args.depth_folder,
        args.masks_folder,
        args.camera_intrinsics_path,
        args.mesh_path,
        args.poses_dir,
        args.weights_dir,
        reference_frame=args.reference_frame,
        target_width=args.target_width,
        target_height=args.target_height,
        reregister_iou_thresh=args.reregister_iou_thresh,
    )
