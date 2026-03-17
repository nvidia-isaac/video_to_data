"""Stateful FoundationPose tracker.

Lifecycle:
    tracker = FoundationPoseTracker(mesh, weights_dir)
    pose0   = tracker.register(rgb, depth, mask, intrinsics)
    pose1   = tracker.track_one(rgb1, depth1, intrinsics)
    pose2   = tracker.track_one(rgb2, depth2, intrinsics)
    tracker.reset_to_pose(pose0)   # rewind for backward pass
    pose_bk = tracker.track_one(rgb_prev, depth_prev, intrinsics, iteration=2)

Scale estimation:
    scale = tracker.estimate_scale_grid_search(rgb, depth, mask, intrinsics)
    # tracker is now in the best-scale state; use it directly for tracking
"""
import logging
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
from Utils import nvdiffrast_render  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402

logger = logging.getLogger(__name__)


class FoundationPoseTracker:
    """Wraps FoundationPose for stateful 6-DoF object tracking."""

    def __init__(self, mesh: Mesh, weights_dir: str) -> None:
        if weights_dir:
            os.environ.setdefault("FOUNDATIONPOSE_WEIGHTS_DIR", weights_dir)

        self._scorer = ScorePredictor()
        self._refiner = PoseRefinePredictor()
        self._glctx = dr.RasterizeCudaContext()
        self._original_mesh = mesh
        self._mesh = mesh
        self._current_scale = 1.0
        self._init_fp(mesh)

    def _init_fp(self, mesh: Mesh) -> None:
        tm = mesh.to_trimesh()
        self._est = FoundationPose(
            model_pts=tm.vertices,
            model_normals=tm.vertex_normals,
            mesh=tm,
            scorer=self._scorer,
            refiner=self._refiner,
            glctx=self._glctx,
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

    def rescale_mesh(self, scale: float) -> None:
        """Rescale the current mesh by `scale` relative to its current size."""
        self.rescale_to(self._current_scale * scale)

    def rescale_to(self, scale: float) -> None:
        """Rescale to `scale` times the *original* mesh size and reinitialize FoundationPose."""
        scaled = Mesh(
            vertices=self._original_mesh.vertices * scale,
            faces=self._original_mesh.faces,
            vertex_colors=self._original_mesh.vertex_colors,
            uv=self._original_mesh.uv,
            texture=self._original_mesh.texture,
        )
        self._mesh = scaled
        self._current_scale = scale
        self._init_fp(scaled)

    def _score_scale(
        self,
        depth: DepthImage,
        mask: Mask,
        intrinsics: CameraIntrinsics,
        pose: Transform3d,
        iou_weight: float = 1.0,
        depth_weight: float = 1.0,
    ) -> float:
        """Score how well the current mesh scale fits the observed depth and mask.

        Renders the mesh at `pose` and computes a weighted combination of:
          - Mask IoU: overlap between rendered and observed masks.
          - Depth consistency: exp(-MARE) where MARE = median |obs/rendered - 1|
            over valid pixels. Equals 1.0 at perfect match.

        Returns a higher value for a better fit.
        """
        K = intrinsics.to_matrix()
        H, W = intrinsics.height, intrinsics.width
        pose_mat = torch.as_tensor(pose.to_matrix()[None], device='cuda', dtype=torch.float)

        with torch.no_grad():
            _, rendered_depth, _ = nvdiffrast_render(
                K, H, W, pose_mat,
                glctx=self._glctx,
                mesh_tensors=self._est.mesh_tensors,
                get_normal=False,
            )

        rendered = rendered_depth[0].cpu().numpy()
        obs_depth = depth.depth
        obs_mask = mask.mask.astype(bool)
        rendered_mask = rendered > 0.001

        score = 0.0

        if iou_weight > 0.0:
            intersection = float((obs_mask & rendered_mask).sum())
            union = float((obs_mask | rendered_mask).sum())
            score += iou_weight * (intersection / (union + 1e-6))

        if depth_weight > 0.0:
            valid = obs_mask & rendered_mask & (obs_depth > 0.001)
            if valid.sum() >= 10:
                ratios = obs_depth[valid] / rendered[valid]
                mare = float(np.median(np.abs(ratios - 1.0)))
                score += depth_weight * float(np.exp(-mare))

        return score

    def estimate_scale_grid_search(
        self,
        rgb: V2dImage,
        depth: DepthImage,
        mask: Mask,
        intrinsics: CameraIntrinsics,
        lo: float = 0.5,
        hi: float = 2.0,
        n_samples: int = 7,
        n_levels: int = 3,
        iou_weight: float = 1.0,
        depth_weight: float = 1.0,
        registration_iterations: int = 5,
    ) -> float:
        """Coarse-to-fine grid search for the optimal mesh scale.

        Samples `n_samples` scales evenly in log-space across [lo, hi] (relative
        to the original mesh). For each candidate, registers the mesh with FP,
        renders it, and scores the result. The search range is halved around the
        winner each level, giving `n_levels` rounds of refinement.

        Leaves the tracker in the best-scale state so it can be used directly
        for subsequent tracking.

        Args:
            rgb:                    Reference frame image.
            depth:                  Corresponding observed depth.
            mask:                   Corresponding segmentation mask.
            intrinsics:             Camera intrinsics.
            lo:                     Lower bound of initial scale search range
                                    (relative to original mesh). Default 0.5.
            hi:                     Upper bound of initial scale search range.
                                    Default 2.0.
            n_samples:              Scales to evaluate per level. Default 7.
            n_levels:               Refinement levels. Default 3.
            iou_weight:             Weight for mask IoU in the score. Default 1.0.
            depth_weight:           Weight for depth consistency in the score. Default 1.0.
            registration_iterations: FP register() iterations per candidate. Default 5.

        Returns:
            Best scale factor relative to the original mesh.
        """
        log_lo = np.log(lo)
        log_hi = np.log(hi)
        best_scale = float(np.sqrt(lo * hi))  # geometric midpoint as initial guess

        for level in range(n_levels):
            scales = np.exp(np.linspace(log_lo, log_hi, n_samples))
            best_score = -np.inf

            for scale in scales:
                self.rescale_to(scale)
                pose = self.register(rgb, depth, mask, intrinsics, iteration=registration_iterations)
                score = self._score_scale(depth, mask, intrinsics, pose, iou_weight, depth_weight)
                logger.debug(f"  scale={scale:.4f}  score={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_scale = float(scale)

            logger.info(f"Level {level + 1}/{n_levels}: best_scale={best_scale:.4f}  score={best_score:.4f}")

            # Halve the log-space search range around the winner
            log_radius = (log_hi - log_lo) / 4
            log_lo = np.log(best_scale) - log_radius
            log_hi = np.log(best_scale) + log_radius

        # Leave tracker in the winning state
        self.rescale_to(best_scale)
        return best_scale
