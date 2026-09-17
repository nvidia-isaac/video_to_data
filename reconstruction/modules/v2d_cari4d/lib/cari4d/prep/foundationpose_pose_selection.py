from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag

import cv2
import numpy as np

from lib_mhr.object_symmetry import OBJECT_SYMMETRY_MODE_FINITE, OBJECT_SYMMETRY_MODE_FULL_SO3


@dataclass(frozen=True)
class OracleSelection:
    index: int
    centered_pose: np.ndarray
    output_pose: np.ndarray
    rotation_errors_deg: np.ndarray


@dataclass(frozen=True)
class CandidateFilter:
    iou_mask: np.ndarray
    keep_mask: np.ndarray
    rotation_distances_deg: np.ndarray
    rotation_threshold_deg: float
    translation_distances_m: np.ndarray
    translation_threshold_m: float
    selected_index: int | None


class TrackingState(IntEnum):
    TRACKING = 0
    OCCLUDED = 1
    REACQUIRING = 2


class PoseSource(IntEnum):
    NONE = 0
    TRACKING_CANDIDATE = 1
    REACQUISITION_CANDIDATE = 2
    CARRIED_CONFIRMED = 3
    FIRST_CONFIRMED_BACKFILL = 4
    GT_ROTATION_ORACLE = 5
    INITIAL_REGISTRATION = 6
    FOUNDATIONPOSE_TRACK = 7


class RejectionReason(IntFlag):
    NONE = 0
    OBSERVED_SUPPORT = 1 << 0
    RENDERED_SUPPORT = 1 << 1
    RENDERED_VISIBILITY = 1 << 2
    SILHOUETTE_IOU = 1 << 3
    SILHOUETTE_PRECISION = 1 << 4
    SILHOUETTE_RECALL = 1 << 5
    CENTROID = 1 << 6
    TEMPORAL_ROTATION = 1 << 7
    TEMPORAL_TRANSLATION = 1 << 8
    CONFIRMATION_PENDING = 1 << 9
    EMPTY_OBJECT_MASK = 1 << 10
    INVALID_INITIALIZATION_DEPTH = 1 << 11
    NO_CANDIDATE = 1 << 12


@dataclass(frozen=True)
class ReacquisitionThresholds:
    support_fraction: float = 0.0
    support_floor_pixels: int = 100
    visibility_floor: float = 0.15
    iou_min: float = 0.40
    precision_min: float = 0.70
    recall_min: float = 0.50
    centroid_distance_max: float = 0.10
    confirmation_frames: int = 3
    pending_rotation_deg: float = 45.0
    pending_translation_floor_m: float = 0.15
    pending_translation_cap_m: float = 0.30
    pending_translation_diameter_fraction: float = 0.25
    long_occlusion_frames: int = 30
    insufficient_evidence_frames: int = 2


@dataclass(frozen=True)
class VisibleSilhouetteMetrics:
    observed_visible_pixels: int
    rendered_pixels: np.ndarray
    rendered_visible_pixels: np.ndarray
    rendered_visibility: np.ndarray
    iou: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    centroid_distance: np.ndarray
    tolerance_pixels: int


@dataclass(frozen=True)
class ReacquisitionFilter:
    keep_mask: np.ndarray
    rejection_masks: np.ndarray
    selected_index: int | None


def minimum_visible_support_pixels(image_shape: tuple[int, int], thresholds: ReacquisitionThresholds = ReacquisitionThresholds()) -> int:
    height, width = (int(value) for value in image_shape)
    if height <= 0 or width <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    return max(int(thresholds.support_floor_pixels), int(np.ceil(float(thresholds.support_fraction) * height * width)))


def _binary_mask(value: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2:
        raise ValueError(f"{label} must have shape [H,W], got {value.shape}")
    if value.dtype != np.dtype("bool"):
        if not np.isin(value, (0, 1)).all():
            raise ValueError(f"{label} must be binary")
        value = value.astype(bool)
    return value


def _bbox_diagonal(mask: np.ndarray) -> float:
    vv, uu = np.nonzero(mask)
    if len(uu) == 0:
        return 0.0
    return float(np.hypot(float(uu.max() - uu.min() + 1), float(vv.max() - vv.min() + 1)))


def measure_visible_silhouettes(rendered_masks: np.ndarray, human_mask: np.ndarray, object_mask: np.ndarray) -> VisibleSilhouetteMetrics:
    rendered_masks = np.asarray(rendered_masks)
    if rendered_masks.ndim != 3:
        raise ValueError(f"rendered_masks must have shape [N,H,W], got {rendered_masks.shape}")
    human = _binary_mask(human_mask, "human_mask")
    observed = _binary_mask(object_mask, "object_mask") & ~human
    if rendered_masks.shape[1:] != observed.shape:
        raise ValueError(f"rendered_masks and segmentation masks must share an image grid, got {rendered_masks.shape[1:]} and {observed.shape}")
    if rendered_masks.dtype != np.dtype("bool"):
        if not np.isin(rendered_masks, (0, 1)).all():
            raise ValueError("rendered_masks must be binary")
        rendered_masks = rendered_masks.astype(bool)
    diagonal = _bbox_diagonal(observed)
    tolerance = int(np.clip(round(0.01 * diagonal), 3, 12)) if diagonal > 0.0 else 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    observed_dilated = cv2.dilate(observed.astype(np.uint8), kernel, iterations=1).astype(bool)
    observed_count = int(observed.sum())
    rendered_counts = np.zeros(len(rendered_masks), dtype=np.int32)
    rendered_visible_counts = np.zeros(len(rendered_masks), dtype=np.int32)
    visibility = np.zeros(len(rendered_masks), dtype=np.float32)
    iou = np.zeros(len(rendered_masks), dtype=np.float32)
    precision = np.zeros(len(rendered_masks), dtype=np.float32)
    recall = np.zeros(len(rendered_masks), dtype=np.float32)
    centroid_distance = np.full(len(rendered_masks), np.inf, dtype=np.float32)
    observed_vv, observed_uu = np.nonzero(observed)
    observed_centroid = np.array([observed_uu.mean(), observed_vv.mean()], dtype=np.float64) if observed_count else None
    for index, rendered in enumerate(rendered_masks):
        rendered_visible = rendered & ~human
        rendered_count = int(rendered.sum())
        rendered_visible_count = int(rendered_visible.sum())
        intersection = int((rendered_visible & observed).sum())
        union = int((rendered_visible | observed).sum())
        rendered_counts[index] = rendered_count
        rendered_visible_counts[index] = rendered_visible_count
        visibility[index] = float(rendered_visible_count / rendered_count) if rendered_count else 0.0
        iou[index] = float(intersection / union) if union else 0.0
        precision[index] = float((rendered_visible & observed_dilated).sum() / rendered_visible_count) if rendered_visible_count else 0.0
        rendered_dilated = cv2.dilate(rendered_visible.astype(np.uint8), kernel, iterations=1).astype(bool)
        recall[index] = float((observed & rendered_dilated).sum() / observed_count) if observed_count else 0.0
        if observed_centroid is not None and rendered_visible_count and diagonal > 0.0:
            rendered_vv, rendered_uu = np.nonzero(rendered_visible)
            rendered_centroid = np.array([rendered_uu.mean(), rendered_vv.mean()], dtype=np.float64)
            centroid_distance[index] = float(np.linalg.norm(rendered_centroid - observed_centroid) / diagonal)
    return VisibleSilhouetteMetrics(observed_visible_pixels=observed_count, rendered_pixels=rendered_counts, rendered_visible_pixels=rendered_visible_counts, rendered_visibility=visibility, iou=iou, precision=precision, recall=recall, centroid_distance=centroid_distance, tolerance_pixels=tolerance)


def _rank_metric_indices(indices: np.ndarray, metrics: VisibleSilhouetteMetrics) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return indices
    order = np.lexsort((metrics.centroid_distance[indices], -metrics.recall[indices], -metrics.iou[indices]))
    return indices[order]


def filter_reacquisition_candidates(metrics: VisibleSilhouetteMetrics, image_shape: tuple[int, int], thresholds: ReacquisitionThresholds = ReacquisitionThresholds()) -> ReacquisitionFilter:
    count = len(metrics.iou)
    for label, value in (("rendered_pixels", metrics.rendered_pixels), ("rendered_visible_pixels", metrics.rendered_visible_pixels), ("rendered_visibility", metrics.rendered_visibility), ("precision", metrics.precision), ("recall", metrics.recall), ("centroid_distance", metrics.centroid_distance)):
        if np.asarray(value).shape != (count,):
            raise ValueError(f"{label} must have shape ({count},), got {np.asarray(value).shape}")
    rejection = np.zeros(count, dtype=np.uint32)
    support = minimum_visible_support_pixels(image_shape, thresholds)
    if metrics.observed_visible_pixels < support:
        rejection |= np.uint32(RejectionReason.OBSERVED_SUPPORT)
    rejection[metrics.rendered_visible_pixels < support] |= np.uint32(RejectionReason.RENDERED_SUPPORT)
    rejection[metrics.rendered_visibility < thresholds.visibility_floor] |= np.uint32(RejectionReason.RENDERED_VISIBILITY)
    rejection[metrics.iou < thresholds.iou_min] |= np.uint32(RejectionReason.SILHOUETTE_IOU)
    rejection[metrics.precision < thresholds.precision_min] |= np.uint32(RejectionReason.SILHOUETTE_PRECISION)
    rejection[metrics.recall < thresholds.recall_min] |= np.uint32(RejectionReason.SILHOUETTE_RECALL)
    rejection[metrics.centroid_distance > thresholds.centroid_distance_max] |= np.uint32(RejectionReason.CENTROID)
    keep = rejection == 0
    ranked = _rank_metric_indices(np.flatnonzero(keep), metrics)
    return ReacquisitionFilter(keep_mask=keep, rejection_masks=rejection, selected_index=int(ranked[0]) if len(ranked) else None)


def validate_rigid_transform(pose: np.ndarray, label: str, *, atol: float = 1e-3) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"{label} must have shape (4, 4), got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError(f"{label} contains nonfinite values")
    if not np.allclose(pose[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol, rtol=0.0):
        raise ValueError(f"{label} has invalid homogeneous bottom row {pose[3].tolist()}")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError(f"{label} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"{label} rotation determinant must be +1, got {determinant:.8f}")
    return pose.astype(np.float32)


def validate_rigid_transforms(poses: np.ndarray, label: str) -> np.ndarray:
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError(f"{label} must have shape (N, 4, 4) with N > 0, got {poses.shape}")
    return np.stack([validate_rigid_transform(pose, f"{label}[{index}]") for index, pose in enumerate(poses)]).astype(np.float32)


def rotation_geodesic_deg(poses: np.ndarray, reference_pose: np.ndarray) -> np.ndarray:
    poses = validate_rigid_transforms(poses, "candidate poses")
    reference_pose = validate_rigid_transform(reference_pose, "reference pose")
    relative = np.einsum("ij,njk->nik", reference_pose[:3, :3].T, poses[:, :3, :3])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine)).astype(np.float32)


def symmetry_aware_pose_distances(candidate_pose: np.ndarray, reference_pose: np.ndarray, symmetry_tfs: np.ndarray, symmetry_mode: int = OBJECT_SYMMETRY_MODE_FINITE, symmetry_center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    candidate_pose = validate_rigid_transform(candidate_pose, "candidate pose")
    reference_pose = validate_rigid_transform(reference_pose, "reference pose")
    symmetry_tfs = validate_rigid_transforms(symmetry_tfs, "symmetry transforms")
    symmetry_mode = int(symmetry_mode)
    symmetry_center = np.zeros(3, dtype=np.float32) if symmetry_center is None else np.asarray(symmetry_center, dtype=np.float32)
    if symmetry_center.shape != (3,) or not np.isfinite(symmetry_center).all():
        raise ValueError(f"symmetry_center must be finite with shape [3], got {symmetry_center.shape}")
    if symmetry_mode == OBJECT_SYMMETRY_MODE_FULL_SO3:
        candidate_center = candidate_pose[:3, :3] @ symmetry_center + candidate_pose[:3, 3]
        reference_center = reference_pose[:3, :3] @ symmetry_center + reference_pose[:3, 3]
        return np.zeros(1, dtype=np.float32), np.asarray([np.linalg.norm(candidate_center - reference_center)], dtype=np.float32)
    if symmetry_mode != OBJECT_SYMMETRY_MODE_FINITE:
        raise ValueError(f"Unsupported object symmetry mode {symmetry_mode}")
    equivalents = np.matmul(candidate_pose[None], symmetry_tfs)
    return rotation_geodesic_deg(equivalents, reference_pose), _translation_distances_m(equivalents, reference_pose)


def symmetry_aware_candidate_distances(candidate_poses: np.ndarray, reference_pose: np.ndarray, symmetry_tfs: np.ndarray, *, rotation_threshold_deg: float, translation_threshold_m: float, symmetry_mode: int = OBJECT_SYMMETRY_MODE_FINITE, symmetry_center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = validate_rigid_transforms(candidate_poses, "candidate poses")
    reference_pose = validate_rigid_transform(reference_pose, "reference pose")
    symmetry_tfs = validate_rigid_transforms(symmetry_tfs, "symmetry transforms")
    rotation_threshold_deg = float(rotation_threshold_deg)
    translation_threshold_m = float(translation_threshold_m)
    if rotation_threshold_deg <= 0.0 or translation_threshold_m <= 0.0:
        raise ValueError("pose-distance thresholds must be positive")
    rotation_distances = np.empty(len(candidates), dtype=np.float32)
    translation_distances = np.empty(len(candidates), dtype=np.float32)
    keep = np.zeros(len(candidates), dtype=bool)
    for index, candidate in enumerate(candidates):
        rotations, translations = symmetry_aware_pose_distances(candidate, reference_pose, symmetry_tfs, symmetry_mode=symmetry_mode, symmetry_center=symmetry_center)
        normalized = np.maximum(rotations / rotation_threshold_deg, translations / translation_threshold_m)
        best = int(np.argmin(normalized))
        rotation_distances[index] = rotations[best]
        translation_distances[index] = translations[best]
        keep[index] = bool(np.any((rotations < rotation_threshold_deg) & (translations <= translation_threshold_m)))
    return rotation_distances, translation_distances, keep


def closest_symmetry_equivalent_pose(candidate_pose: np.ndarray, reference_pose: np.ndarray, symmetry_tfs: np.ndarray, *, rotation_scale_deg: float, translation_scale_m: float, symmetry_mode: int = OBJECT_SYMMETRY_MODE_FINITE, symmetry_center: np.ndarray | None = None) -> np.ndarray:
    candidate_pose = validate_rigid_transform(candidate_pose, "candidate pose")
    reference_pose = validate_rigid_transform(reference_pose, "reference pose")
    symmetry_tfs = validate_rigid_transforms(symmetry_tfs, "symmetry transforms")
    symmetry_mode = int(symmetry_mode)
    symmetry_center = np.zeros(3, dtype=np.float32) if symmetry_center is None else np.asarray(symmetry_center, dtype=np.float32)
    if symmetry_center.shape != (3,) or not np.isfinite(symmetry_center).all():
        raise ValueError(f"symmetry_center must be finite with shape [3], got {symmetry_center.shape}")
    if symmetry_mode == OBJECT_SYMMETRY_MODE_FULL_SO3:
        candidate_center = candidate_pose[:3, :3] @ symmetry_center + candidate_pose[:3, 3]
        aligned = candidate_pose.copy()
        aligned[:3, :3] = reference_pose[:3, :3]
        aligned[:3, 3] = candidate_center - aligned[:3, :3] @ symmetry_center
        return validate_rigid_transform(aligned, "closest full-SO(3)-equivalent pose")
    if symmetry_mode != OBJECT_SYMMETRY_MODE_FINITE:
        raise ValueError(f"Unsupported object symmetry mode {symmetry_mode}")
    equivalents = np.matmul(candidate_pose[None], symmetry_tfs)
    rotations = rotation_geodesic_deg(equivalents, reference_pose)
    translations = _translation_distances_m(equivalents, reference_pose)
    normalized = np.maximum(rotations / float(rotation_scale_deg), translations / float(translation_scale_m))
    return validate_rigid_transform(equivalents[int(np.argmin(normalized))], "closest symmetry-equivalent pose")


def _pending_translation_threshold_m(object_diameter_m: float, thresholds: ReacquisitionThresholds) -> float:
    object_diameter_m = float(object_diameter_m)
    if not np.isfinite(object_diameter_m) or object_diameter_m <= 0.0:
        raise ValueError(f"object_diameter_m must be finite and positive, got {object_diameter_m}")
    return max(thresholds.pending_translation_floor_m, min(thresholds.pending_translation_cap_m, thresholds.pending_translation_diameter_fraction * object_diameter_m))


def poses_are_pending_consistent(candidate_pose: np.ndarray, pending_pose: np.ndarray, symmetry_tfs: np.ndarray, object_diameter_m: float, thresholds: ReacquisitionThresholds = ReacquisitionThresholds(), *, symmetry_mode: int = OBJECT_SYMMETRY_MODE_FINITE, symmetry_center: np.ndarray | None = None) -> tuple[bool, float, float]:
    translation_threshold = _pending_translation_threshold_m(object_diameter_m, thresholds)
    rotations, translations = symmetry_aware_pose_distances(candidate_pose, pending_pose, symmetry_tfs, symmetry_mode=symmetry_mode, symmetry_center=symmetry_center)
    valid = (rotations <= thresholds.pending_rotation_deg) & (translations <= translation_threshold)
    normalized = np.maximum(rotations / thresholds.pending_rotation_deg, translations / translation_threshold)
    best = int(np.argmin(normalized))
    return bool(np.any(valid)), float(rotations[best]), float(translations[best])


def advance_pending_pose(candidate_pose: np.ndarray, pending_pose: np.ndarray | None, confirmation_count: int, symmetry_tfs: np.ndarray, object_diameter_m: float, thresholds: ReacquisitionThresholds = ReacquisitionThresholds(), *, symmetry_mode: int = OBJECT_SYMMETRY_MODE_FINITE, symmetry_center: np.ndarray | None = None) -> tuple[np.ndarray, int, bool]:
    candidate_pose = validate_rigid_transform(candidate_pose, "pending candidate pose")
    confirmation_count = int(confirmation_count)
    if pending_pose is None:
        if confirmation_count != 0:
            raise ValueError(f"confirmation_count must be zero without a pending pose, got {confirmation_count}")
        return candidate_pose.copy(), 1, False
    if confirmation_count <= 0:
        raise ValueError(f"confirmation_count must be positive with a pending pose, got {confirmation_count}")
    pending_pose = validate_rigid_transform(pending_pose, "pending reference pose")
    consistent, _, _ = poses_are_pending_consistent(candidate_pose, pending_pose, symmetry_tfs, object_diameter_m, thresholds, symmetry_mode=symmetry_mode, symmetry_center=symmetry_center)
    if not consistent:
        return candidate_pose.copy(), 1, False
    translation_threshold_m = _pending_translation_threshold_m(object_diameter_m, thresholds)
    aligned_candidate = closest_symmetry_equivalent_pose(candidate_pose, pending_pose, symmetry_tfs, rotation_scale_deg=thresholds.pending_rotation_deg, translation_scale_m=translation_threshold_m, symmetry_mode=symmetry_mode, symmetry_center=symmetry_center)
    return aligned_candidate, confirmation_count + 1, True


def adaptive_translation_threshold_m(reliable_translation_steps_m: list[float] | np.ndarray, *, base_threshold_m: float = 0.2, history_size: int = 5, multiplier: float = 2.0) -> float:
    base_threshold_m = float(base_threshold_m)
    history_size = int(history_size)
    multiplier = float(multiplier)
    steps = np.asarray(reliable_translation_steps_m, dtype=np.float64)
    if not np.isfinite(base_threshold_m) or base_threshold_m <= 0.0:
        raise ValueError(f"base_threshold_m must be finite and positive, got {base_threshold_m}")
    if history_size <= 0:
        raise ValueError(f"history_size must be positive, got {history_size}")
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"multiplier must be finite and positive, got {multiplier}")
    if steps.ndim != 1 or not np.isfinite(steps).all() or np.any(steps < 0.0):
        raise ValueError("reliable_translation_steps_m must be a finite non-negative one-dimensional sequence")
    if len(steps) == 0:
        return base_threshold_m
    adaptive_threshold_m = multiplier * float(np.median(steps[-history_size:]))
    return max(base_threshold_m, adaptive_threshold_m)


def _translation_distances_m(candidate_poses_centered: np.ndarray, previous_pose_centered: np.ndarray) -> np.ndarray:
    return np.linalg.norm(candidate_poses_centered[:, :3, 3] - previous_pose_centered[None, :3, 3], axis=1).astype(np.float32)


def select_gt_rotation_oracle(candidate_poses_centered: np.ndarray, transform_to_centered_mesh: np.ndarray, gt_pose_cam: np.ndarray) -> OracleSelection:
    candidates = validate_rigid_transforms(candidate_poses_centered, "FoundationPose candidates")
    transform_to_centered_mesh = validate_rigid_transform(transform_to_centered_mesh, "transform_to_centered_mesh")
    gt_pose_cam = validate_rigid_transform(gt_pose_cam, "ground-truth camera pose")
    output_poses = np.stack([validate_rigid_transform(pose @ transform_to_centered_mesh, f"output candidate[{index}]") for index, pose in enumerate(candidates)])
    rotation_errors_deg = rotation_geodesic_deg(output_poses, gt_pose_cam)
    index = int(np.argmin(rotation_errors_deg))
    return OracleSelection(index=index, centered_pose=candidates[index].copy(), output_pose=output_poses[index].copy(), rotation_errors_deg=rotation_errors_deg)


def filter_ranked_candidates(candidate_poses_centered: np.ndarray, ious: np.ndarray, *, previous_pose_centered: np.ndarray | None, frame_index: int, last_visible_index: int, previous_visibility: float, visibility_threshold: float, translation_threshold_m: float, iou_rank: int = 30) -> CandidateFilter:
    candidates = validate_rigid_transforms(candidate_poses_centered, "clustered FoundationPose candidates")
    ious = np.asarray(ious, dtype=np.float32)
    if ious.shape != (len(candidates),):
        raise ValueError(f"candidate IoUs must have shape ({len(candidates)},), got {ious.shape}")
    if not np.isfinite(ious).all():
        raise ValueError("candidate IoUs contain nonfinite values")
    if int(iou_rank) < 0:
        raise ValueError(f"iou_rank must be non-negative, got {iou_rank}")
    threshold_index = min(int(iou_rank), len(ious) - 1)
    iou_threshold = np.sort(ious)[::-1][threshold_index]
    iou_mask = ious >= iou_threshold
    rotation_distances_deg = np.zeros(len(candidates), dtype=np.float32)
    rotation_threshold_deg = -1.0
    translation_distances_m = np.zeros(len(candidates), dtype=np.float32)
    applied_translation_threshold_m = -1.0
    use_temporal_filter = previous_pose_centered is not None and (float(previous_visibility) > float(visibility_threshold) or int(frame_index) - int(last_visible_index) < 30)
    if use_temporal_filter:
        previous_pose_centered = validate_rigid_transform(previous_pose_centered, "previous centered pose")
        translation_threshold_m = float(translation_threshold_m)
        if not np.isfinite(translation_threshold_m) or translation_threshold_m <= 0.0:
            raise ValueError(f"translation_threshold_m must be finite and positive, got {translation_threshold_m}")
        rotation_threshold_deg = 15.0 + max(0, int(frame_index) - int(last_visible_index)) * 2.5
        applied_translation_threshold_m = translation_threshold_m
        rotation_distances_deg = rotation_geodesic_deg(candidates, previous_pose_centered)
        translation_distances_m = _translation_distances_m(candidates, previous_pose_centered)
        keep_mask = iou_mask & (rotation_distances_deg < rotation_threshold_deg) & (translation_distances_m <= applied_translation_threshold_m)
    else:
        keep_mask = iou_mask.copy()
    kept = np.flatnonzero(keep_mask)
    selected_index = int(kept[0]) if len(kept) else None
    return CandidateFilter(iou_mask=iou_mask, keep_mask=keep_mask, rotation_distances_deg=rotation_distances_deg, rotation_threshold_deg=float(rotation_threshold_deg), translation_distances_m=translation_distances_m, translation_threshold_m=float(applied_translation_threshold_m), selected_index=selected_index)


def temporal_rejection_reasons(candidate_filter: CandidateFilter) -> RejectionReason:
    if candidate_filter.selected_index is not None:
        return RejectionReason.NONE
    rotation_pass = candidate_filter.rotation_distances_deg < candidate_filter.rotation_threshold_deg
    translation_pass = candidate_filter.translation_distances_m <= candidate_filter.translation_threshold_m
    reasons = RejectionReason.NONE
    if not np.any(candidate_filter.iou_mask & rotation_pass):
        reasons |= RejectionReason.TEMPORAL_ROTATION
    if not np.any(candidate_filter.iou_mask & translation_pass):
        reasons |= RejectionReason.TEMPORAL_TRANSLATION
    return reasons or (RejectionReason.TEMPORAL_ROTATION | RejectionReason.TEMPORAL_TRANSLATION)


def filter_retry_candidates(candidate_poses_centered: np.ndarray, ious: np.ndarray, *, previous_pose_centered: np.ndarray, base_rotation_threshold_deg: float, attempt: int, max_attempts: int = 5) -> CandidateFilter:
    candidates = validate_rigid_transforms(candidate_poses_centered, "retry FoundationPose candidates")
    previous_pose_centered = validate_rigid_transform(previous_pose_centered, "previous centered pose")
    ious = np.asarray(ious, dtype=np.float32)
    if ious.shape != (len(candidates),) or not np.isfinite(ious).all():
        raise ValueError(f"retry candidate IoUs must be finite with shape ({len(candidates)},), got {ious.shape}")
    attempt = int(attempt)
    max_attempts = int(max_attempts)
    if attempt <= 0 or max_attempts <= 0 or attempt > max_attempts:
        raise ValueError(f"retry attempt must satisfy 1 <= attempt <= max_attempts, got {attempt}/{max_attempts}")
    rotation_distances_deg = rotation_geodesic_deg(candidates, previous_pose_centered)
    translation_distances_m = _translation_distances_m(candidates, previous_pose_centered)
    if attempt == max_attempts:
        iou_mask = np.ones(len(candidates), dtype=bool)
        keep_mask = iou_mask.copy()
        rotation_threshold_deg = -1.0
        applied_translation_threshold_m = -1.0
    else:
        iou_threshold = 0.08 + (float(ious.mean()) - 0.08) * (1.0 - attempt / max_attempts)
        iou_mask = ious >= iou_threshold
        rotation_threshold_deg = min(float(base_rotation_threshold_deg) + attempt, 20.0)
        applied_translation_threshold_m = -1.0
        keep_mask = iou_mask & (rotation_distances_deg < rotation_threshold_deg)
    kept = np.flatnonzero(keep_mask)
    selected_index = int(kept[0]) if len(kept) else None
    return CandidateFilter(iou_mask=iou_mask, keep_mask=keep_mask, rotation_distances_deg=rotation_distances_deg, rotation_threshold_deg=float(rotation_threshold_deg), translation_distances_m=translation_distances_m, translation_threshold_m=float(applied_translation_threshold_m), selected_index=selected_index)
