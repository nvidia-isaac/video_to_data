"""Multi-view 6-DoF object tracker using FoundationPose.

Creates N FoundationPose estimators with shared weights (scorer, refiner,
glctx) and fuses per-camera poses each frame via se3_pose_select.
"""

import logging
import os
import sys
from contextlib import contextmanager
import numpy as np
import torch
import trimesh

from v2d.mesh.lib.mesh import Mesh
from v2d.mv.math.numpy_fn import xyz_to_uv, se3_split_mean


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

    def __init__(self, mesh: Mesh, weights_dir: str, num_cameras: int):
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
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Register frame 0 across all cameras.

        Returns:
            avg_pose: (4,4) best world-frame pose
            world_poses: list of (4,4) per-camera world poses
            select_idx: indices selected by se3_pose_select
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

        select_idx, avg_pose = self._avg_poses(world_poses, masks, Ks, Ts)
        assert np.sum(select_idx) > 0, "Object not visible from any camera in first frame"
        self._sync_to_avg(avg_pose, Ts)
        return avg_pose, world_poses, select_idx

    def track(
        self,
        rgbs: list[np.ndarray],
        depths: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
        iteration: int = 2,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Track next frame across all cameras.

        Returns:
            avg_pose: (4,4) best world-frame pose
            world_poses: list of (4,4) per-camera world poses
            select_idx: indices selected by se3_pose_select
        """
        world_poses = []
        with _suppress_fp_logging():
            for j, est in enumerate(self._estimators):
                cam_pose = est.track_one(
                    rgb=rgbs[j], depth=depths[j],
                    K=Ks[j], iteration=iteration,
                )
                world_poses.append(Ts[j] @ cam_pose)

        select_idx, avg_pose = self._avg_poses(world_poses, masks, Ks, Ts)
        self._sync_to_avg(avg_pose, Ts)
        return avg_pose, world_poses, select_idx

    def _avg_poses(
        self,
        world_poses: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
        visible_ratio: float = 0.3,
    ) -> np.ndarray:
        select_idx = np.zeros(len(world_poses), dtype=bool)
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
            if in_mask / in_bounds.sum() > visible_ratio:
                select_idx[j] = True
        if not select_idx.any():
            print(f"Object not visible from any camera, using last pose")
            return select_idx, self.pose_last
        avg_pose = se3_split_mean(np.array(world_poses)[select_idx])
        return select_idx, avg_pose

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
