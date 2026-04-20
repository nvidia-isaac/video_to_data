"""Multi-view 6-DoF object tracker using FoundationPose.

Creates N FoundationPose estimators with shared weights (scorer, refiner,
glctx) and fuses per-camera poses each frame via visibility-based selection
and anisotropic pose averaging.
"""

import logging
import os
import sys
from contextlib import contextmanager
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh

from v2d.mesh.lib.mesh import Mesh
from v2d.mv.math.numpy_fn import xyz_to_uv, se3_from_rot_trans


@contextmanager
def _suppress_fp_logging():
    """Temporarily raise root log level to suppress FoundationPose info spam."""
    prev = logging.root.level
    logging.root.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logging.root.setLevel(prev)

_FP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FoundationPose")
if _FP_DIR not in sys.path:
    sys.path.insert(0, _FP_DIR)

from estimater import FoundationPose  # noqa: E402
from learning.training.predict_score import ScorePredictor  # noqa: E402
from learning.training.predict_pose_refine import PoseRefinePredictor  # noqa: E402
from Utils import set_seed  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402


class MultiViewTracker:
    """Coordinate N FoundationPose estimators with shared weights."""

    def __init__(
        self,
        mesh: Mesh,
        weights_dir: str,
        num_cameras: int,
        depth_direction_trust: float = 0.5,
        visible_ratio_cutoff_high: float = 0.3,
        visible_ratio_cutoff_low: float = 0.3,
        precision_high: float = 1.0,
        precision_low: float = 0.01,
    ):
        if weights_dir:
            os.environ.setdefault("FOUNDATIONPOSE_WEIGHTS_DIR", weights_dir)

        set_seed(0)

        with _suppress_fp_logging():
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            glctx = dr.RasterizeCudaContext()

            tm = mesh.to_trimesh()
            to_origin, extents = trimesh.bounds.oriented_bounds(tm)
            self.to_origin = to_origin
            self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

            self._estimators = [
                FoundationPose(
                    model_pts=tm.vertices,
                    model_normals=tm.vertex_normals,
                    mesh=tm,
                    scorer=scorer,
                    refiner=refiner,
                    glctx=glctx,
                    debug=0,
                    debug_dir=f"/tmp/fp_debug_{i}",
                )
                for i in range(num_cameras)
            ]
            
        self.mesh = mesh
        self.pose_last = None
        self.depth_direction_trust = depth_direction_trust
        self.visible_ratio_cutoff_high = visible_ratio_cutoff_high
        self.visible_ratio_cutoff_low = visible_ratio_cutoff_low
        self.precision_high = precision_high
        self.precision_low = precision_low

    @property
    def num_cameras(self) -> int:
        return len(self._estimators)

    def register(
        self,
        rgbs: list[np.ndarray],
        depths: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
        iteration: int = 5,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """Register frame 0 across all cameras.

        Returns:
            avg_pose: (4,4) best world-frame pose
            world_poses: list of (4,4) per-camera world poses
            visible_ratios: (D,) per-camera visibility ratios
            select_idx: boolean mask of cameras where object is visible
        """
        world_poses = []
        with _suppress_fp_logging():
            for j, est in enumerate(self._estimators):
                cam_pose = est.register(
                    K=Ks[j], rgb=rgbs[j], depth=depths[j],
                    ob_mask=masks[j], iteration=iteration,
                )
                world_poses.append(Ts[j] @ cam_pose)
                torch.cuda.empty_cache()

        avg_pose, visible_ratios, select_idx = self._avg_poses(world_poses, masks, Ks, Ts)
        assert np.sum(select_idx) > 0, "Object not visible from any camera in first frame"
        self._sync_to_avg(avg_pose, Ts)
        return avg_pose, world_poses, visible_ratios, select_idx

    def track(
        self,
        rgbs: list[np.ndarray],
        depths: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
        iteration: int = 2,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """Track next frame across all cameras.

        Returns:
            avg_pose: (4,4) best world-frame pose
            world_poses: list of (4,4) per-camera world poses
            visible_ratios: (D,) per-camera visibility ratios
            select_idx: boolean mask of cameras where object is visible
        """
        world_poses = []
        with _suppress_fp_logging():
            for j, est in enumerate(self._estimators):
                cam_pose = est.track_one(
                    rgb=rgbs[j], depth=depths[j],
                    K=Ks[j], iteration=iteration,
                )
                world_poses.append(Ts[j] @ cam_pose)

        avg_pose, visible_ratios, select_idx = self._avg_poses(world_poses, masks, Ks, Ts)
        self._sync_to_avg(avg_pose, Ts)
        return avg_pose, world_poses, visible_ratios, select_idx

    def _visible_ratio_to_precision(self, visible_ratio: np.ndarray) -> np.ndarray:
        if self.visible_ratio_cutoff_high <= self.visible_ratio_cutoff_low:
            return np.full_like(visible_ratio, self.precision_high)
        slope = (
            (self.precision_high - self.precision_low)
            / (self.visible_ratio_cutoff_high - self.visible_ratio_cutoff_low)
        )
        p = self.precision_low + slope * (visible_ratio - self.visible_ratio_cutoff_low)
        return np.clip(p, self.precision_low, self.precision_high)

    @staticmethod
    def _se3_split_mean_weighted(
        poses: np.ndarray,
        frame_rotations: np.ndarray,
        W_trans: np.ndarray,
        w_pose: np.ndarray,
    ) -> np.ndarray:
        """
        Mean of SE(3) poses with anisotropic translation precision and per-pose
        confidence weights.

        Args:
            poses: (D, 4, 4) world-frame poses
            frame_rotations: (D, 3, 3) rotations defining each pose's local frame
            W_trans: (3, 3) diagonal precision matrix of translations in the local frame
            w_pose: (D,) per-pose confidence weights. Scales the rotation mean and
                the translation precision so low-confidence views contribute less.
        Returns:
            mean_pose: (4, 4)
        """
        rot = poses[:, :3, :3]
        trans = poses[:, :3, 3]
        mean_rot = Rotation.from_matrix(rot).mean(weights=w_pose).as_matrix()

        P_sum = np.zeros((3, 3))
        Pt_sum = np.zeros(3)
        for j in range(len(poses)):
            R_j = frame_rotations[j]
            P_j = w_pose[j] * (R_j @ W_trans @ R_j.T)
            P_sum += P_j
            Pt_sum += P_j @ trans[j]
        mean_trans = np.linalg.solve(P_sum, Pt_sum)

        return se3_from_rot_trans(mean_rot, mean_trans)

    def _avg_poses(
        self,
        world_poses: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
    ) -> np.ndarray:
        D = len(world_poses)
        visible_ratios = np.zeros(D)
        verts_hom = np.hstack([self.mesh.vertices,
                               np.ones((len(self.mesh.vertices), 1))])
        for j, world_pose in enumerate(world_poses):
            world_verts = (verts_hom @ world_pose.T)[:, :3]
            H, W = masks[j].shape[:2]
            uv, in_bounds = xyz_to_uv(
                world_verts, Ks[j], Ts[j], image_size=(W, H),
            )
            if in_bounds.sum() == 0:
                continue
            in_mask = masks[j][uv[in_bounds, 1], uv[in_bounds, 0]].sum()
            visible_ratios[j] = in_mask / in_bounds.sum()

        select_idx = visible_ratios > self.visible_ratio_cutoff_low
        if not select_idx.any():
            print(f"Object not visible from any camera, using last pose")
            return self.pose_last, visible_ratios, select_idx
        selected = np.where(select_idx)[0]
        cam_rotations = np.array([Ts[j][:3, :3] for j in selected])
        W_trans = np.diag([1.0, 1.0, self.depth_direction_trust])
        w_pose = self._visible_ratio_to_precision(visible_ratios[selected])
        avg_pose = self._se3_split_mean_weighted(
            np.array(world_poses)[select_idx],
            cam_rotations,
            W_trans,
            w_pose,
        )
        return avg_pose, visible_ratios, select_idx

    def _sync_to_avg(self, avg_pose: np.ndarray, Ts: list[np.ndarray]):
        """Project the fused world pose back into each camera and set pose_last."""
        for j, est in enumerate(self._estimators):
            pose_last = (
                np.linalg.inv(Ts[j])
                @ avg_pose
                @ np.linalg.inv(est.get_tf_to_centered_mesh().cpu().numpy())
            )
            est.pose_last = torch.as_tensor(
                pose_last, device="cuda", dtype=torch.float
            ).reshape(1, 4, 4)
        self.pose_last = avg_pose
