"""Stateful FoundationPose tracker.

Lifecycle:
    tracker = FoundationPoseTracker(mesh, weights_dir)
    pose0   = tracker.register(rgb, depth, mask, intrinsics)
    pose1   = tracker.track_one(rgb1, depth1, intrinsics)
    pose2   = tracker.track_one(rgb2, depth2, intrinsics)
    tracker.reset_to_pose(pose0)   # rewind for backward pass
    pose_bk = tracker.track_one(rgb_prev, depth_prev, intrinsics, iteration=2)
"""
import os
import sys

import numpy as np
import torch

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Transform3d
from v2d.common.datatypes import Image as V2dImage
from v2d.mesh.lib.mesh import Mesh

_FP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FoundationPose')
sys.path.insert(0, _FP_DIR)

from estimater import FoundationPose  # noqa: E402
from learning.training.predict_score import ScorePredictor  # noqa: E402
from learning.training.predict_pose_refine import PoseRefinePredictor  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402


class FoundationPoseTracker:
    """Wraps FoundationPose for stateful 6-DoF object tracking."""

    def __init__(self, mesh: Mesh, weights_dir: str) -> None:
        if weights_dir:
            os.environ.setdefault("FOUNDATIONPOSE_WEIGHTS_DIR", weights_dir)

        tm = mesh.to_trimesh()
        self._est = FoundationPose(
            model_pts=tm.vertices,
            model_normals=tm.vertex_normals,
            mesh=tm,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(),
            debug=0,
            debug_dir='/tmp/foundationpose_debug',
        )

    def register(
        self,
        rgb: V2dImage,
        depth: DepthImage,
        mask: Mask,
        intrinsics: CameraIntrinsics,
        iteration: int = 10,
    ) -> Transform3d:
        """Register initial pose from a reference frame. Returns object-to-camera Transform3d."""
        with torch.no_grad():
            pose = self._est.register(
                K=intrinsics.to_matrix(),
                rgb=rgb.data,
                depth=depth.depth,
                ob_mask=mask.mask.astype(bool),
                iteration=iteration,
            )
        return Transform3d.from_matrix(pose)

    def track_one(
        self,
        rgb: V2dImage,
        depth: DepthImage,
        intrinsics: CameraIntrinsics,
        iteration: int = 5,
    ) -> Transform3d:
        """Track pose for the next frame. Returns object-to-camera Transform3d."""
        with torch.no_grad():
            pose = self._est.track_one(
                rgb=rgb.data,
                depth=depth.depth,
                K=intrinsics.to_matrix(),
                iteration=iteration,
            )
        return Transform3d.from_matrix(pose)

    def reset_to_pose(self, pose: Transform3d) -> None:
        """Reset internal pose state. Use before a backward tracking pass."""
        torch.cuda.empty_cache()
        self._est.pose_last = torch.as_tensor(pose.to_matrix(), device='cuda', dtype=torch.float)
