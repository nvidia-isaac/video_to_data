"""Multi-view 6-DoF object tracker using FoundationPose.

Creates N FoundationPose estimators with shared weights (scorer, refiner,
glctx) and fuses per-camera poses each frame via visibility-based selection
and anisotropic pose averaging.
"""

import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh

from v2d.mesh.lib.mesh import Mesh
from v2d.mv.math.numpy_fn import xyz_to_uv, se3_from_rot_trans


def _canonicalize_poses_with_indices(
    poses: list[np.ndarray],
    group: list[np.ndarray],
    reference: np.ndarray,
) -> tuple[list[np.ndarray], list[tuple[int, float]]]:
    """Canonicalize each pose against `reference`; also return (chosen_index, geodesic_dist_rad)."""
    ref_R = np.asarray(reference, dtype=float)[:3, :3]
    canonicals: list[np.ndarray] = []
    indices: list[tuple[int, float]] = []
    for pose in poses:
        pose = np.asarray(pose, dtype=float)
        best_idx = -1
        best_dist = float("inf")
        best_pose = None
        for k, R_s in enumerate(group):
            cand = pose @ R_s
            cos_t = (np.trace(ref_R.T @ cand[:3, :3]) - 1.0) / 2.0
            d = float(np.arccos(np.clip(cos_t, -1.0, 1.0)))
            if d < best_dist:
                best_dist = d
                best_idx = k
                best_pose = cand
        canonicals.append(best_pose)
        indices.append((best_idx, best_dist))
    return canonicals, indices


def _dump_register_debug(
    path: str,
    symmetry_group_size: int,
    fp_scores: list[float],
    visible_ratios: np.ndarray,
    select_idx: np.ndarray,
    ref_idx: int,
    raw_world_poses: list[np.ndarray],
    canonical_world_poses: list[np.ndarray],
    snap_indices_per_iter: list[list[tuple[int, float]]],
    avg_pose: np.ndarray,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "symmetry_group_size": int(symmetry_group_size),
        "fp_scores": [float(s) for s in fp_scores],
        "visible_ratios": [float(v) for v in visible_ratios],
        "select_idx": [bool(v) for v in select_idx],
        "ref_idx": int(ref_idx),
        "n_iter": len(snap_indices_per_iter),
        "raw_world_poses": [np.asarray(p).tolist() for p in raw_world_poses],
        "canonical_world_poses": [np.asarray(p).tolist() for p in canonical_world_poses],
        "snap_indices_per_iter": [
            [[int(idx), float(dist)] for idx, dist in iter_indices]
            for iter_indices in snap_indices_per_iter
        ],
        "avg_pose": np.asarray(avg_pose).tolist(),
    }
    with open(p, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[MultiViewTracker] register debug dumped to {p}")


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


@dataclass
class MultiViewTrackingResult:
    avg_pose: np.ndarray
    world_poses: list[np.ndarray]
    visible_ratios: np.ndarray
    select_idx: np.ndarray
    status: dict


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
        symmetry_group: list[np.ndarray] | None = None,
        recovery_enabled: bool = True,
        recovery_refine_iter: int = 5,
        recovery_min_views: int = 1,
        recovery_min_valid_depth_pixels: int = 25,
        recovery_visible_ratio_cutoff: float = 0.3,
        recovery_attempt_stride: int = 1,
        under_supported_recovery_enabled: bool = True,
        mask_pose_iou_cutoff: float = 0.05,
        mask_explained_ratio_cutoff: float = 0.10,
        register_debug_path: str | None = None,
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
        self.symmetry_group = symmetry_group
        self.recovery_enabled = recovery_enabled
        self.recovery_refine_iter = recovery_refine_iter
        self.recovery_min_views = max(1, int(recovery_min_views))
        self.recovery_min_valid_depth_pixels = max(1, int(recovery_min_valid_depth_pixels))
        self.recovery_visible_ratio_cutoff = recovery_visible_ratio_cutoff
        self.recovery_attempt_stride = max(1, int(recovery_attempt_stride))
        self.under_supported_recovery_enabled = under_supported_recovery_enabled
        self.mask_pose_iou_cutoff = mask_pose_iou_cutoff
        self.mask_explained_ratio_cutoff = mask_explained_ratio_cutoff
        self.register_debug_path = register_debug_path
        self._lost_frame_count = 0
        self._under_supported_frame_count = 0
        self.last_status: dict = {}
        if symmetry_group is None:
            print("[MultiViewTracker] symmetry canonicalization disabled (no symmetry annotation)")
        else:
            print(f"[MultiViewTracker] symmetry canonicalization enabled (group size {len(symmetry_group)})")

    @property
    def num_cameras(self) -> int:
        return len(self._estimators)

    def seed_pose(self, avg_pose: np.ndarray, Ts: list[np.ndarray]) -> None:
        """Seed every single-view estimator from a known fused world pose."""
        self._sync_to_avg(np.asarray(avg_pose, dtype=float).reshape(4, 4), Ts)
        self._lost_frame_count = 0
        self._under_supported_frame_count = 0

    def register(
        self,
        rgbs: list[np.ndarray],
        depths: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
        iteration: int = 5,
        debug_dump_path: str | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """Register frame 0 across all cameras.

        Returns:
            avg_pose: (4,4) best world-frame pose
            world_poses: list of (4,4) per-camera canonicalized world poses
            visible_ratios: (D,) per-camera visibility ratios
            select_idx: boolean mask of cameras where object is visible
        """
        raw_world_poses = []
        fp_scores = []
        with _suppress_fp_logging():
            for j, est in enumerate(self._estimators):
                cam_pose = est.register(
                    K=Ks[j], rgb=rgbs[j], depth=depths[j],
                    ob_mask=masks[j], iteration=iteration,
                )
                raw_world_poses.append(Ts[j] @ cam_pose)
                # FoundationPose.register() returns early without setting
                # self.scores when the mask has <4 valid depth pixels.
                scores = getattr(est, "scores", None)
                fp_scores.append(float(scores[0].item()) if scores is not None else float("nan"))
                torch.cuda.empty_cache()

        visible_ratios = self._compute_visibility(raw_world_poses, masks, Ks, Ts)
        select_idx = visible_ratios > self.visible_ratio_cutoff_low
        assert np.sum(select_idx) > 0, "Object not visible from any camera in first frame"

        avg_pose, world_poses, ref_idx, snap_indices_per_iter = self._fuse_registered_poses(
            raw_world_poses, Ts, visible_ratios, select_idx,
        )

        if debug_dump_path is not None:
            _dump_register_debug(
                path=debug_dump_path,
                symmetry_group_size=len(self.symmetry_group) if self.symmetry_group is not None else 1,
                fp_scores=fp_scores,
                visible_ratios=visible_ratios,
                select_idx=select_idx,
                ref_idx=ref_idx,
                raw_world_poses=raw_world_poses,
                canonical_world_poses=world_poses,
                snap_indices_per_iter=snap_indices_per_iter,
                avg_pose=avg_pose,
            )

        self._sync_to_avg(avg_pose, Ts)
        self._lost_frame_count = 0
        self._under_supported_frame_count = 0
        status_extra = {}
        if 0 < int(np.count_nonzero(select_idx)) < self.recovery_min_views:
            status_extra["under_supported"] = True
            status_extra["under_supported_initialization"] = True
        self.last_status = self._make_status(
            status="registered",
            pose_valid=True,
            visible_ratios=visible_ratios,
            select_idx=select_idx,
            **status_extra,
        )
        return avg_pose, world_poses, visible_ratios, select_idx

    def track(
        self,
        rgbs: list[np.ndarray],
        depths: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
        frame_index: int | None = None,
        register_iteration: int = 5,
        track_iteration: int = 2,
    ) -> MultiViewTrackingResult:
        """Process one frame across all cameras.

        The first call registers the object pose. Later calls track from the
        previous fused pose and may enter held/recovery states.
        """
        if self.pose_last is None:
            result = self.register(
                rgbs, depths, masks, Ks, Ts,
                iteration=register_iteration,
                debug_dump_path=self.register_debug_path,
            )
        else:
            result = self._track_registered_frame(
                rgbs, depths, masks, Ks, Ts,
                iteration=track_iteration,
                frame_index=frame_index,
            )
        return self._make_result(*result, frame_index=frame_index)

    def _track_registered_frame(
        self,
        rgbs: list[np.ndarray],
        depths: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
        iteration: int = 2,
        frame_index: int | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """Track next frame after initial registration."""
        world_poses = []
        with _suppress_fp_logging():
            for j, est in enumerate(self._estimators):
                cam_pose = est.track_one(
                    rgb=rgbs[j], depth=depths[j],
                    K=Ks[j], iteration=iteration,
                )
                world_poses.append(Ts[j] @ cam_pose)

        visible_ratios = self._compute_visibility(world_poses, masks, Ks, Ts)
        select_idx = visible_ratios > self.visible_ratio_cutoff_low
        if not select_idx.any():
            self._under_supported_frame_count = 0
            return self._handle_all_view_loss(
                rgbs=rgbs,
                depths=depths,
                masks=masks,
                Ks=Ks,
                Ts=Ts,
                tracked_visible_ratios=visible_ratios,
                tracked_select_idx=select_idx,
                frame_index=frame_index,
            )

        if int(select_idx.sum()) < self.recovery_min_views:
            return self._handle_under_supported_tracking(
                rgbs=rgbs,
                depths=depths,
                masks=masks,
                Ks=Ks,
                Ts=Ts,
                tracked_world_poses=world_poses,
                tracked_visible_ratios=visible_ratios,
                tracked_select_idx=select_idx,
                frame_index=frame_index,
            )

        avg_pose = self._weighted_se3_mean_from_poses(
            world_poses, Ts, visible_ratios, select_idx,
        )
        self._sync_to_avg(avg_pose, Ts)
        self._lost_frame_count = 0
        self._under_supported_frame_count = 0
        self.last_status = self._make_status(
            status="tracked",
            pose_valid=True,
            visible_ratios=visible_ratios,
            select_idx=select_idx,
        )
        return avg_pose, world_poses, visible_ratios, select_idx

    def _make_result(
        self,
        avg_pose: np.ndarray,
        world_poses: list[np.ndarray],
        visible_ratios: np.ndarray,
        select_idx: np.ndarray,
        frame_index: int | None,
    ) -> MultiViewTrackingResult:
        status = dict(self.last_status)
        status["frame_index"] = int(frame_index) if frame_index is not None else None
        self.last_status = status
        return MultiViewTrackingResult(
            avg_pose=avg_pose,
            world_poses=world_poses,
            visible_ratios=visible_ratios,
            select_idx=select_idx,
            status=status,
        )

    def _fuse_registered_poses(
        self,
        raw_world_poses: list[np.ndarray],
        Ts: list[np.ndarray],
        visible_ratios: np.ndarray,
        select_idx: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray], int, list[list[tuple[int, float]]]]:
        """Fuse freshly registered poses, including optional symmetry snapping."""
        snap_indices_per_iter: list[list[tuple[int, float]]] = []
        ref_idx = -1
        if self.symmetry_group is not None and len(self.symmetry_group) > 1:
            scores_for_ref = visible_ratios.copy()
            scores_for_ref[~select_idx] = -np.inf
            ref_idx = int(np.argmax(scores_for_ref))
            ref = raw_world_poses[ref_idx]

            n_iter_max = 3
            world_poses = list(raw_world_poses)
            for _ in range(n_iter_max):
                world_poses, snap_indices = _canonicalize_poses_with_indices(
                    raw_world_poses, self.symmetry_group, ref,
                )
                snap_indices_per_iter.append(snap_indices)
                new_ref = self._weighted_se3_mean_from_poses(
                    world_poses, Ts, visible_ratios, select_idx,
                )
                cos_t = (np.trace(ref[:3, :3].T @ new_ref[:3, :3]) - 1.0) / 2.0
                angle_diff = float(np.arccos(np.clip(cos_t, -1.0, 1.0)))
                ref = new_ref
                if angle_diff < 1e-4:
                    break
            avg_pose = ref
        else:
            world_poses = list(raw_world_poses)
            avg_pose = self._weighted_se3_mean_from_poses(
                world_poses, Ts, visible_ratios, select_idx,
            )
        return avg_pose, world_poses, ref_idx, snap_indices_per_iter

    def _valid_depth_pixels(self, depth: np.ndarray, mask: np.ndarray) -> int:
        valid = mask.astype(bool) & np.isfinite(depth) & (depth >= 0.001)
        return int(valid.sum())

    def _should_attempt_recovery(self) -> bool:
        if not self.recovery_enabled:
            return False
        return (self._lost_frame_count - 1) % self.recovery_attempt_stride == 0

    def _should_attempt_under_supported_recovery(self) -> bool:
        if not self.recovery_enabled or not self.under_supported_recovery_enabled:
            return False
        return (self._under_supported_frame_count - 1) % self.recovery_attempt_stride == 0

    def _handle_all_view_loss(
        self,
        rgbs: list[np.ndarray],
        depths: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
        tracked_visible_ratios: np.ndarray,
        tracked_select_idx: np.ndarray,
        frame_index: int | None,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        held_pose = np.asarray(self.pose_last, dtype=float).reshape(4, 4)
        self._lost_frame_count += 1

        valid_depth_pixels = np.array([
            self._valid_depth_pixels(depths[j], masks[j])
            for j in range(self.num_cameras)
        ])
        candidate_idx = valid_depth_pixels >= self.recovery_min_valid_depth_pixels
        attempt_recovery = self._should_attempt_recovery()

        status = self._make_status(
            status="held",
            pose_valid=False,
            visible_ratios=tracked_visible_ratios,
            select_idx=tracked_select_idx,
            recovery_attempted=bool(attempt_recovery),
            recovery_candidate_idx=candidate_idx,
            recovery_valid_depth_pixels=valid_depth_pixels,
            frame_index=frame_index,
        )

        if attempt_recovery and candidate_idx.any():
            recovered = self._attempt_recovery(
                rgbs=rgbs,
                depths=depths,
                masks=masks,
                Ks=Ks,
                Ts=Ts,
                candidate_idx=candidate_idx,
                tracked_visible_ratios=tracked_visible_ratios,
                tracked_select_idx=tracked_select_idx,
                valid_depth_pixels=valid_depth_pixels,
                frame_index=frame_index,
            )
            if recovered is not None:
                avg_pose, world_poses, visible_ratios, select_idx, status = recovered
                self._sync_to_avg(avg_pose, Ts)
                self._lost_frame_count = 0
                self.last_status = status
                return avg_pose, world_poses, visible_ratios, select_idx
            status = self.last_status

        if attempt_recovery and not candidate_idx.any():
            status["recovery_failure_reason"] = "no_candidate_views"

        print("Object not visible from any camera, using last pose")
        self._sync_to_avg(held_pose, Ts)
        self.last_status = status
        held_world_poses = [held_pose.copy() for _ in range(self.num_cameras)]
        return held_pose, held_world_poses, tracked_visible_ratios, tracked_select_idx

    def _handle_under_supported_tracking(
        self,
        rgbs: list[np.ndarray],
        depths: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
        tracked_world_poses: list[np.ndarray],
        tracked_visible_ratios: np.ndarray,
        tracked_select_idx: np.ndarray,
        frame_index: int | None,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        self._under_supported_frame_count += 1
        (
            mask_valid_depth_pixels,
            mask_pose_iou,
            mask_explained_ratio,
            candidate_idx,
        ) = self._under_supported_recovery_candidates(
            tracked_world_poses, depths, masks, Ks, Ts, tracked_select_idx,
        )
        if not self.recovery_enabled or not self.under_supported_recovery_enabled:
            candidate_idx = np.zeros_like(candidate_idx, dtype=bool)
        attempt_recovery = self._should_attempt_under_supported_recovery()
        recovery_mode = "under_supported_mask_evidence"

        if attempt_recovery and candidate_idx.any():
            recovered = self._attempt_under_supported_recovery(
                rgbs=rgbs,
                depths=depths,
                masks=masks,
                Ks=Ks,
                Ts=Ts,
                tracked_world_poses=tracked_world_poses,
                tracked_visible_ratios=tracked_visible_ratios,
                tracked_select_idx=tracked_select_idx,
                candidate_idx=candidate_idx,
                mask_valid_depth_pixels=mask_valid_depth_pixels,
                mask_pose_iou=mask_pose_iou,
                mask_explained_ratio=mask_explained_ratio,
                frame_index=frame_index,
            )
            if recovered is not None:
                avg_pose, world_poses, visible_ratios, select_idx, status = recovered
                self._sync_to_avg(avg_pose, Ts)
                self._lost_frame_count = 0
                self._under_supported_frame_count = 0
                self.last_status = status
                return avg_pose, world_poses, visible_ratios, select_idx

            status = self.last_status
            held_pose = np.asarray(self.pose_last, dtype=float).reshape(4, 4)
            self._sync_to_avg(held_pose, Ts)
            held_world_poses = [held_pose.copy() for _ in range(self.num_cameras)]
            return held_pose, held_world_poses, tracked_visible_ratios, tracked_select_idx

        if candidate_idx.any():
            status = self._make_status(
                status="held",
                pose_valid=False,
                visible_ratios=tracked_visible_ratios,
                select_idx=tracked_select_idx,
                recovery_attempted=False,
                under_supported=True,
                recovery_mode=recovery_mode,
                mask_pose_iou=mask_pose_iou,
                mask_explained_ratio=mask_explained_ratio,
                mask_valid_depth_pixels=mask_valid_depth_pixels,
                under_supported_recovery_candidate_idx=candidate_idx,
                under_supported_recovery_visible_ratios=tracked_visible_ratios,
                under_supported_recovery_select_idx=tracked_select_idx,
                recovery_failure_reason="recovery_stride_skip",
                frame_index=frame_index,
            )
            held_pose = np.asarray(self.pose_last, dtype=float).reshape(4, 4)
            self._sync_to_avg(held_pose, Ts)
            self.last_status = status
            held_world_poses = [held_pose.copy() for _ in range(self.num_cameras)]
            return held_pose, held_world_poses, tracked_visible_ratios, tracked_select_idx

        avg_pose = self._weighted_se3_mean_from_poses(
            tracked_world_poses, Ts, tracked_visible_ratios, tracked_select_idx,
        )
        self._sync_to_avg(avg_pose, Ts)
        self._lost_frame_count = 0
        self.last_status = self._make_status(
            status="tracked",
            pose_valid=True,
            visible_ratios=tracked_visible_ratios,
            select_idx=tracked_select_idx,
            recovery_attempted=False,
            under_supported=True,
            recovery_mode=recovery_mode,
            mask_pose_iou=mask_pose_iou,
            mask_explained_ratio=mask_explained_ratio,
            mask_valid_depth_pixels=mask_valid_depth_pixels,
            under_supported_recovery_candidate_idx=candidate_idx,
            under_supported_recovery_visible_ratios=tracked_visible_ratios,
            under_supported_recovery_select_idx=tracked_select_idx,
            frame_index=frame_index,
        )
        return avg_pose, tracked_world_poses, tracked_visible_ratios, tracked_select_idx

    def _under_supported_recovery_candidates(
        self,
        world_poses: list[np.ndarray],
        depths: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
        select_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (
            mask_valid_depth_pixels,
            mask_pose_iou,
            mask_explained_ratio,
        ) = self._mask_pose_overlap_metrics(world_poses, depths, masks, Ks, Ts)
        has_mask_evidence = mask_valid_depth_pixels >= self.recovery_min_valid_depth_pixels
        candidate_idx = (
            (~np.asarray(select_idx, dtype=bool))
            & has_mask_evidence
            & (mask_pose_iou < self.mask_pose_iou_cutoff)
            & (mask_explained_ratio < self.mask_explained_ratio_cutoff)
        )
        return mask_valid_depth_pixels, mask_pose_iou, mask_explained_ratio, candidate_idx

    def _mask_pose_overlap_metrics(
        self,
        world_poses: list[np.ndarray],
        depths: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        D = len(world_poses)
        mask_valid_depth_pixels = np.zeros(D, dtype=int)
        mask_pose_iou = np.zeros(D, dtype=float)
        mask_explained_ratio = np.zeros(D, dtype=float)

        for j, world_pose in enumerate(world_poses):
            mask = np.asarray(masks[j], dtype=bool)
            mask_valid_depth_pixels[j] = self._valid_depth_pixels(depths[j], mask)
            mask_area = int(mask.sum())
            if mask_area == 0:
                continue

            silhouette = self._mesh_silhouette_mask(
                world_pose, Ks[j], Ts[j], mask.shape[:2],
            )
            intersection = int(np.logical_and(silhouette, mask).sum())
            union = int(np.logical_or(silhouette, mask).sum())
            if union > 0:
                mask_pose_iou[j] = intersection / union
            mask_explained_ratio[j] = intersection / mask_area

        return mask_valid_depth_pixels, mask_pose_iou, mask_explained_ratio

    def _mesh_silhouette_mask(
        self,
        world_pose: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        H, W = image_shape
        silhouette = np.zeros((H, W), dtype=np.uint8)
        verts_hom = np.hstack([
            self.mesh.vertices,
            np.ones((len(self.mesh.vertices), 1)),
        ])
        world_verts = (verts_hom @ np.asarray(world_pose).reshape(4, 4).T)[:, :3]
        uv, in_bounds = xyz_to_uv(world_verts, K, T, image_size=(W, H))
        if int(np.asarray(in_bounds, dtype=bool).sum()) < 3:
            return silhouette.astype(bool)

        pts = np.asarray(uv[np.asarray(in_bounds, dtype=bool)], dtype=np.float32)
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        if len(np.unique(pts, axis=0)) < 3:
            return silhouette.astype(bool)

        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        cv2.fillConvexPoly(silhouette, hull, 1)
        return silhouette.astype(bool)

    def _attempt_under_supported_recovery(
        self,
        rgbs: list[np.ndarray],
        depths: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
        tracked_world_poses: list[np.ndarray],
        tracked_visible_ratios: np.ndarray,
        tracked_select_idx: np.ndarray,
        candidate_idx: np.ndarray,
        mask_valid_depth_pixels: np.ndarray,
        mask_pose_iou: np.ndarray,
        mask_explained_ratio: np.ndarray,
        frame_index: int | None,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, dict] | None:
        recovery_world_poses = [
            np.asarray(pose, dtype=float).reshape(4, 4).copy()
            for pose in tracked_world_poses
        ]
        fp_scores = np.full(self.num_cameras, np.nan, dtype=float)

        with _suppress_fp_logging():
            for j in np.where(candidate_idx)[0]:
                cam_pose = self._estimators[j].register(
                    K=Ks[j], rgb=rgbs[j], depth=depths[j],
                    ob_mask=masks[j], iteration=self.recovery_refine_iter,
                )
                recovery_world_poses[j] = Ts[j] @ cam_pose
                scores = getattr(self._estimators[j], "scores", None)
                fp_scores[j] = float(scores[0].item()) if scores is not None else float("nan")
                torch.cuda.empty_cache()

        visible_ratios = self._compute_visibility(recovery_world_poses, masks, Ks, Ts)
        select_idx = visible_ratios > self.recovery_visible_ratio_cutoff
        recovery_mode = "under_supported_mask_evidence"
        if int(select_idx.sum()) < self.recovery_min_views:
            status = self._make_status(
                status="held",
                pose_valid=False,
                visible_ratios=tracked_visible_ratios,
                select_idx=tracked_select_idx,
                recovery_attempted=True,
                under_supported=True,
                recovery_mode=recovery_mode,
                mask_pose_iou=mask_pose_iou,
                mask_explained_ratio=mask_explained_ratio,
                mask_valid_depth_pixels=mask_valid_depth_pixels,
                under_supported_recovery_candidate_idx=candidate_idx,
                under_supported_recovery_visible_ratios=visible_ratios,
                under_supported_recovery_select_idx=select_idx,
                recovery_fp_scores=fp_scores,
                recovery_failure_reason="not_enough_recovered_views",
                frame_index=frame_index,
            )
            self.last_status = status
            return None

        avg_pose, world_poses, _, _ = self._fuse_registered_poses(
            recovery_world_poses, Ts, visible_ratios, select_idx,
        )
        status = self._make_status(
            status="partially_recovered",
            pose_valid=True,
            visible_ratios=visible_ratios,
            select_idx=select_idx,
            recovery_attempted=True,
            under_supported=True,
            recovery_mode=recovery_mode,
            mask_pose_iou=mask_pose_iou,
            mask_explained_ratio=mask_explained_ratio,
            mask_valid_depth_pixels=mask_valid_depth_pixels,
            under_supported_recovery_candidate_idx=candidate_idx,
            under_supported_recovery_visible_ratios=visible_ratios,
            under_supported_recovery_select_idx=select_idx,
            recovery_fp_scores=fp_scores,
            tracked_visible_ratios=tracked_visible_ratios,
            tracked_select_idx=tracked_select_idx,
            frame_index=frame_index,
        )
        return avg_pose, world_poses, visible_ratios, select_idx, status

    def _attempt_recovery(
        self,
        rgbs: list[np.ndarray],
        depths: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
        candidate_idx: np.ndarray,
        tracked_visible_ratios: np.ndarray,
        tracked_select_idx: np.ndarray,
        valid_depth_pixels: np.ndarray,
        frame_index: int | None,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, dict] | None:
        recovery_world_poses = [np.asarray(self.pose_last, dtype=float).reshape(4, 4).copy()
                                for _ in range(self.num_cameras)]
        fp_scores = np.full(self.num_cameras, np.nan, dtype=float)

        with _suppress_fp_logging():
            for j in np.where(candidate_idx)[0]:
                cam_pose = self._estimators[j].register(
                    K=Ks[j], rgb=rgbs[j], depth=depths[j],
                    ob_mask=masks[j], iteration=self.recovery_refine_iter,
                )
                recovery_world_poses[j] = Ts[j] @ cam_pose
                scores = getattr(self._estimators[j], "scores", None)
                fp_scores[j] = float(scores[0].item()) if scores is not None else float("nan")
                torch.cuda.empty_cache()

        visible_ratios = self._compute_visibility(recovery_world_poses, masks, Ks, Ts)
        select_idx = (visible_ratios > self.recovery_visible_ratio_cutoff) & candidate_idx
        if int(select_idx.sum()) < self.recovery_min_views:
            status = self._make_status(
                status="held",
                pose_valid=False,
                visible_ratios=tracked_visible_ratios,
                select_idx=tracked_select_idx,
                recovery_attempted=True,
                recovery_candidate_idx=candidate_idx,
                recovery_valid_depth_pixels=valid_depth_pixels,
                recovery_visible_ratios=visible_ratios,
                recovery_select_idx=select_idx,
                recovery_fp_scores=fp_scores,
                recovery_failure_reason="not_enough_recovered_views",
                frame_index=frame_index,
            )
            self.last_status = status
            return None

        avg_pose, world_poses, _, _ = self._fuse_registered_poses(
            recovery_world_poses, Ts, visible_ratios, select_idx,
        )
        status = self._make_status(
            status="recovered",
            pose_valid=True,
            visible_ratios=visible_ratios,
            select_idx=select_idx,
            recovery_attempted=True,
            recovery_candidate_idx=candidate_idx,
            recovery_valid_depth_pixels=valid_depth_pixels,
            recovery_visible_ratios=visible_ratios,
            recovery_select_idx=select_idx,
            recovery_fp_scores=fp_scores,
            tracked_visible_ratios=tracked_visible_ratios,
            tracked_select_idx=tracked_select_idx,
            frame_index=frame_index,
        )
        return avg_pose, world_poses, visible_ratios, select_idx, status

    @staticmethod
    def _make_status(
        status: str,
        pose_valid: bool,
        visible_ratios: np.ndarray,
        select_idx: np.ndarray,
        recovery_attempted: bool = False,
        frame_index: int | None = None,
        **extra,
    ) -> dict:
        record = {
            "status": status,
            "pose_valid": bool(pose_valid),
            "visible_ratios": np.asarray(visible_ratios, dtype=float).tolist(),
            "select_idx": np.asarray(select_idx, dtype=bool).tolist(),
            "recovery_attempted": bool(recovery_attempted),
        }
        if frame_index is not None:
            record["frame_index"] = int(frame_index)
        for key, value in extra.items():
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                if value.dtype == bool:
                    record[key] = value.astype(bool).tolist()
                elif np.issubdtype(value.dtype, np.integer):
                    record[key] = value.astype(int).tolist()
                else:
                    record[key] = value.astype(float).tolist()
            else:
                record[key] = value
        return record

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

    def _compute_visibility(
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
        return visible_ratios

    def _weighted_se3_mean_from_poses(
        self,
        world_poses: list[np.ndarray],
        Ts: list[np.ndarray],
        visible_ratios: np.ndarray,
        select_idx: np.ndarray,
    ) -> np.ndarray:
        selected = np.where(select_idx)[0]
        cam_rotations = np.array([Ts[j][:3, :3] for j in selected])
        W_trans = np.diag([1.0, 1.0, self.depth_direction_trust])
        w_pose = self._visible_ratio_to_precision(visible_ratios[selected])
        return self._se3_split_mean_weighted(
            np.array(world_poses)[select_idx],
            cam_rotations,
            W_trans,
            w_pose,
        )

    def _avg_poses(
        self,
        world_poses: list[np.ndarray],
        masks: list[np.ndarray],
        Ks: list[np.ndarray],
        Ts: list[np.ndarray],
    ) -> np.ndarray:
        visible_ratios = self._compute_visibility(world_poses, masks, Ks, Ts)
        select_idx = visible_ratios > self.visible_ratio_cutoff_low
        if not select_idx.any():
            print(f"Object not visible from any camera, using last pose")
            return self.pose_last, visible_ratios, select_idx
        avg_pose = self._weighted_se3_mean_from_poses(
            world_poses, Ts, visible_ratios, select_idx,
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
