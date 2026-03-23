"""Stateful FoundationPose tracker.

Lifecycle:
    tracker = FoundationPoseTracker(mesh, weights_dir)
    pose0   = tracker.register(rgb, depth, mask, intrinsics)
    pose1   = tracker.track_one(rgb1, depth1, intrinsics)
    pose2   = tracker.track_one(rgb2, depth2, intrinsics)
    tracker.reset_to_pose(pose0)   # rewind for backward pass
    pose_bk = tracker.track_one(rgb_prev, depth_prev, intrinsics, iteration=2)

Particle filter tracking (multi-hypothesis):
    pose1   = tracker.track_one_particles(rgb1, depth1, intrinsics, n_particles=20)
    pose2   = tracker.track_one_particles(rgb2, depth2, intrinsics, n_particles=20)

Scale estimation:
    scale = tracker.estimate_scale_grid_search(rgb, depth, mask, intrinsics)
    # tracker is now in the best-scale state; use it directly for tracking
"""
import logging
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _Rotation

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Transform3d
from v2d.common.datatypes import Image as V2dImage
from v2d.mesh.lib.mesh import Mesh

_FP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FoundationPose')
sys.path.insert(0, _FP_DIR)

from estimater import FoundationPose  # noqa: E402
from learning.training.predict_score import ScorePredictor  # noqa: E402
from learning.training.predict_pose_refine import PoseRefinePredictor  # noqa: E402
from Utils import nvdiffrast_render, erode_depth, bilateral_filter_depth, depth2xyzmap_batch  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SE(3) particle filter helpers
# ---------------------------------------------------------------------------

def _perturb_se3(poses: np.ndarray, sigma_t: float, sigma_r: float) -> np.ndarray:
    """Add independent SE(3) noise to each pose.

    Translation noise is additive. Rotation noise is a left-perturbation in
    so(3): R_new = exp(δr) @ R_old, where δr ~ N(0, σ_r² I).

    Args:
        poses:   (N, 4, 4) float32 poses.
        sigma_t: Translation std (metres).
        sigma_r: Rotation std (radians).

    Returns:
        (N, 4, 4) float32 perturbed poses.
    """
    N = len(poses)
    out = poses.copy()
    out[:, :3, 3] += np.random.randn(N, 3).astype(np.float32) * sigma_t
    delta_r = _Rotation.from_rotvec(
        np.random.randn(N, 3).astype(np.float32) * sigma_r
    ).as_matrix()                                  # (N, 3, 3)
    out[:, :3, :3] = (delta_r @ poses[:, :3, :3])
    return out


def _ess(weights: np.ndarray) -> float:
    """Effective sample size 1 / Σw²."""
    return 1.0 / float(np.sum(weights ** 2))


def _resample(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Multinomial resampling."""
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[indices].copy()


def _weighted_mean_se3(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted Fréchet mean on SE(3).

    Translation is linearly averaged. Rotation uses iterative geodesic mean
    on SO(3) (Riemannian gradient descent), converging in ~3 iterations.

    Args:
        particles: (N, 4, 4) poses.
        weights:   (N,) normalised weights summing to 1.

    Returns:
        (4, 4) float32 mean pose.
    """
    t_mean = (weights[:, None] * particles[:, :3, 3]).sum(0)

    # Initialise at the highest-weight particle
    R_mean = _Rotation.from_matrix(particles[int(weights.argmax()), :3, :3])
    for _ in range(5):
        # Tangent vectors from R_mean to each particle (left formulation)
        tangents = (_Rotation.from_matrix(particles[:, :3, :3]) * R_mean.inv()).as_rotvec()
        mean_tangent = (weights[:, None] * tangents).sum(0)
        R_mean = _Rotation.from_rotvec(mean_tangent) * R_mean
        if np.linalg.norm(mean_tangent) < 1e-7:
            break

    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R_mean.as_matrix().astype(np.float32)
    M[:3, 3] = t_mean.astype(np.float32)
    return M


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
        self._particles: np.ndarray | None = None  # (N, 4, 4) poses in pose_last frame
        self._weights: np.ndarray | None = None    # (N,) normalised
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

    def track_one_with_recovery(
        self,
        rgb: V2dImage,
        depth: DepthImage,
        mask: Mask,
        intrinsics: CameraIntrinsics,
        iteration: int = 5,
        iou_thresh: float = 0.3,
        recovery_iteration: int = 10,
    ) -> tuple[Transform3d, bool]:
        """Track one frame, re-registering if the tracked pose has low mask IoU.

        After tracking, renders the mesh at the tracked pose and computes IoU
        against the observed mask. If IoU < iou_thresh, re-registers from
        scratch using the current frame's mask.

        Args:
            rgb:                Current frame image.
            depth:              Current frame depth.
            mask:               Current frame segmentation mask.
            intrinsics:         Camera intrinsics.
            iteration:          Tracking refinement iterations.
            iou_thresh:         IoU below this triggers re-registration. Default 0.3.
            recovery_iteration: Registration iterations on recovery. Default 10.

        Returns:
            (pose, recovered) — recovered is True if re-registration was triggered.
        """
        pose = self.track_one(rgb, depth, intrinsics, iteration=iteration)
        iou = self._mask_iou(mask, intrinsics, pose)
        if iou < iou_thresh:
            logger.info(f"Tracking loss detected (IoU={iou:.3f} < {iou_thresh}) — re-registering")
            pose = self.register(rgb, depth, mask, intrinsics, iteration=recovery_iteration)
            return pose, True
        return pose, False

    def track_one_particles(
        self,
        rgb: V2dImage,
        depth: DepthImage,
        intrinsics: CameraIntrinsics,
        mask: Mask | None = None,
        n_particles: int = 20,
        process_noise_t: float = 0.005,
        process_noise_r: float = 0.02,
        iteration: int = 3,
        mask_iou_weight: float = 1.0,
    ) -> Transform3d:
        """Track using a particle filter with FP's refiner and scorer as measurement model.

        Each frame: perturb particles → refine (single batched GPU call) →
        score (single batched GPU call) → optionally weight by mask IoU
        (single batched render) → softmax weight → conditional resample.

        The scorer captures appearance quality (RGB+depth match); mask IoU adds
        an independent geometric silhouette constraint that is robust to lateral
        drift. When mask is provided, the combined log-weight is:
            log_w_i = scorer_logit_i + mask_iou_weight * log(iou_i + ε)

        Particles are lazily initialised on the first call from pose_last, and
        cleared by reset_to_pose() so forward/backward passes are independent.

        Args:
            rgb:              Current frame image.
            depth:            Current frame depth.
            intrinsics:       Camera intrinsics.
            mask:             Optional segmentation mask for IoU weighting.
                              When provided, particles are additionally weighted
                              by their silhouette overlap with this mask.
            n_particles:      Number of particles. Default 20.
            process_noise_t:  Per-frame translation perturbation std (metres).
                              Default 0.005.
            process_noise_r:  Per-frame rotation perturbation std (radians).
                              Default 0.02.
            iteration:        Refiner iterations per particle per frame. Default 3.
            mask_iou_weight:  Log-space weight for the mask IoU term. Default 1.0.
                              Set to 0.0 to disable mask weighting.

        Returns:
            Weighted mean pose as Transform3d.
        """
        if self._est.pose_last is None:
            raise RuntimeError("Call register() before track_one_particles()")

        # Lazy-init or reinit if n_particles changed
        if self._particles is None or len(self._particles) != n_particles:
            seed = np.tile(
                self._est.pose_last.cpu().numpy().astype(np.float32),
                (n_particles, 1, 1),
            )
            self._particles = _perturb_se3(seed, process_noise_t, process_noise_r)
            self._weights = np.full(n_particles, 1.0 / n_particles, dtype=np.float32)

        K = intrinsics.to_matrix()

        # Depth preprocessing — mirrors _est.track_one internals
        depth_t = torch.as_tensor(depth.depth, device='cuda', dtype=torch.float)
        depth_t = erode_depth(depth_t, radius=2, device='cuda')
        depth_t = bilateral_filter_depth(depth_t, radius=2, device='cuda')
        K_t = torch.as_tensor(K, device='cuda', dtype=torch.float)
        xyz_map = depth2xyzmap_batch(depth_t[None], K_t[None], zfar=np.inf)[0]

        # 1. Propagate: perturb each particle
        candidates = _perturb_se3(self._particles, process_noise_t, process_noise_r)

        # 2. Refine: single batched call over all particles
        with torch.no_grad():
            refined, _ = self._est.refiner.predict(
                mesh=self._est.mesh,
                mesh_tensors=self._est.mesh_tensors,
                rgb=rgb.data,
                depth=depth_t,
                K=K,
                ob_in_cams=candidates,
                xyz_map=xyz_map,
                mesh_diameter=self._est.diameter,
                glctx=self._glctx,
                iteration=iteration,
                get_vis=False,
            )
        refined_np = refined.cpu().numpy()

        # 3. Score: single batched call over all particles
        with torch.no_grad():
            scores, _ = self._est.scorer.predict(
                mesh=self._est.mesh,
                mesh_tensors=self._est.mesh_tensors,
                rgb=rgb.data,
                depth=depth_t.cpu().numpy(),
                K=K,
                ob_in_cams=refined_np,
                glctx=self._glctx,
                mesh_diameter=self._est.diameter,
                get_vis=False,
            )

        # 4. Combine scorer logits with optional mask IoU in log-space
        log_w = scores.cpu().numpy().astype(np.float64)

        if mask is not None and mask_iou_weight > 0.0:
            # Batch render all N refined particles — one GPU call
            pose_batch = torch.as_tensor(refined_np, device='cuda', dtype=torch.float)
            with torch.no_grad():
                _, rendered_depths, _ = nvdiffrast_render(
                    K, intrinsics.height, intrinsics.width, pose_batch,
                    glctx=self._glctx,
                    mesh_tensors=self._est.mesh_tensors,
                    get_normal=False,
                )
            rendered_masks = rendered_depths.cpu().numpy() > 0.001   # (N, H, W)
            obs_mask = mask.mask.astype(bool)
            intersection = (obs_mask[None] & rendered_masks).sum(axis=(1, 2))
            union       = (obs_mask[None] | rendered_masks).sum(axis=(1, 2))
            iou_scores  = intersection / (union + 1e-6)               # (N,)
            log_w += mask_iou_weight * np.log(np.maximum(iou_scores, 1e-6))
            logger.debug(f"  IoU scores — mean={iou_scores.mean():.3f}  min={iou_scores.min():.3f}")

        log_w -= log_w.max()   # numerical stability
        w = np.exp(log_w)
        self._weights = (w / w.sum()).astype(np.float32)
        self._particles = refined_np

        logger.debug(
            f"Particle filter: ESS={_ess(self._weights):.1f}/{n_particles}  "
            f"best_score={log_w.max():.3f}"
        )

        # 5. Conditional resample when diversity collapses
        if _ess(self._weights) < n_particles / 2.0:
            logger.debug("Resampling particles")
            self._particles = _resample(self._particles, self._weights)
            self._weights = np.full(n_particles, 1.0 / n_particles, dtype=np.float32)

        # 6. Keep pose_last = best particle (maintains reset_to_pose compatibility)
        best_idx = int(self._weights.argmax())
        self._est.pose_last = torch.as_tensor(
            self._particles[best_idx], device='cuda', dtype=torch.float
        )

        # 7. Return weighted mean as Transform3d
        mean_centered = _weighted_mean_se3(self._particles, self._weights)
        mean_pose = mean_centered @ self._est.get_tf_to_centered_mesh().cpu().numpy()
        return Transform3d.from_matrix(mean_pose)

    def _mask_iou(
        self,
        mask: Mask,
        intrinsics: CameraIntrinsics,
        pose: Transform3d,
    ) -> float:
        """Render the mesh at `pose` and return IoU with the observed mask."""
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

        rendered_mask = rendered_depth[0].cpu().numpy() > 0.001
        obs_mask = mask.mask.astype(bool)

        intersection = float((obs_mask & rendered_mask).sum())
        union = float((obs_mask | rendered_mask).sum())
        return intersection / (union + 1e-6)

    def reset_to_pose(self, pose: Transform3d) -> None:
        """Reset internal pose state. Use before a backward tracking pass.

        pose is in the caller's frame (with get_tf_to_centered_mesh applied), but
        pose_last must be stored in the pre-centering frame that track_one expects.
        """
        torch.cuda.empty_cache()
        matrix = torch.as_tensor(pose.to_matrix(), device='cuda', dtype=torch.float)
        tf_to_center = self._est.get_tf_to_centered_mesh()
        self._est.pose_last = matrix @ torch.linalg.inv(tf_to_center)
        self._particles = None
        self._weights = None

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
        self._particles = None
        self._weights = None
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

    def align_depth_to_object(
        self,
        rgb: V2dImage,
        depth_raw: DepthImage,
        mask: Mask,
        intrinsics: CameraIntrinsics,
        scale_lo: float = 0.5,
        scale_hi: float = 2.0,
        shift_lo: float = -0.5,
        shift_hi: float = 0.5,
        n_scale_samples: int = 7,
        n_shift_samples: int = 5,
        n_levels: int = 3,
        iou_weight: float = 1.0,
        depth_weight: float = 1.0,
        registration_iterations: int = 5,
    ) -> DepthImage:
        """Find the depth affine (scale, shift) that best aligns raw monocular depth to the mesh.

        Searches over D_aligned = scale * D_raw + shift via coarse-to-fine 2D grid search.
        For each candidate, registers with FP and scores via mask IoU + depth MARE.
        Leaves the tracker registered at the best-fit depth, ready for tracking.

        Args:
            rgb:                    Reference frame image.
            depth_raw:              Raw (uncalibrated) monocular depth.
            mask:                   Object segmentation mask.
            intrinsics:             Camera intrinsics.
            scale_lo:               Lower bound of scale search range. Default 0.5.
            scale_hi:               Upper bound of scale search range. Default 2.0.
            shift_lo:               Lower bound of shift search range (metres). Default -0.5.
            shift_hi:               Upper bound of shift search range (metres). Default 0.5.
            n_scale_samples:        Scale candidates per level. Default 7.
            n_shift_samples:        Shift candidates per level. Default 5.
            n_levels:               Refinement levels. Default 3.
            iou_weight:             Weight for mask IoU in score. Default 1.0.
            depth_weight:           Weight for depth MARE in score. Default 1.0.
            registration_iterations: FP register() iterations per candidate. Default 5.

        Returns:
            Corrected DepthImage with the best-fitting affine applied.
        """
        best_score = -np.inf
        best_scale = 1.0
        best_shift = 0.0
        log_scale_lo = np.log(scale_lo)
        log_scale_hi = np.log(scale_hi)

        for level in range(n_levels):
            scales = np.exp(np.linspace(log_scale_lo, log_scale_hi, n_scale_samples))
            shifts = np.linspace(shift_lo, shift_hi, n_shift_samples)

            for scale in scales:
                for shift in shifts:
                    depth_candidate = DepthImage(
                        depth=np.clip(scale * depth_raw.depth + shift, 0.0, None).astype(np.float32)
                    )
                    pose = self.register(rgb, depth_candidate, mask, intrinsics, iteration=registration_iterations)
                    score = self._score_scale(depth_candidate, mask, intrinsics, pose, iou_weight, depth_weight)
                    logger.debug(f"  scale={scale:.4f}  shift={shift:.4f}  score={score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_scale = float(scale)
                        best_shift = float(shift)

            logger.info(
                f"Level {level + 1}/{n_levels}: "
                f"best_scale={best_scale:.4f}  best_shift={best_shift:.4f}  score={best_score:.4f}"
            )
            log_scale_radius = (log_scale_hi - log_scale_lo) / 4
            log_scale_lo = np.log(best_scale) - log_scale_radius
            log_scale_hi = np.log(best_scale) + log_scale_radius
            shift_radius = (shift_hi - shift_lo) / 4
            shift_lo = best_shift - shift_radius
            shift_hi = best_shift + shift_radius

        logger.info(f"Best depth affine: scale={best_scale:.4f}  shift={best_shift:.4f}")
        corrected = np.clip(best_scale * depth_raw.depth + best_shift, 0.0, None).astype(np.float32)
        best_depth = DepthImage(depth=corrected)
        # Leave tracker registered at best depth
        self.register(rgb, best_depth, mask, intrinsics, iteration=registration_iterations)
        return best_depth

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
