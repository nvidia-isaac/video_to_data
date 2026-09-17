from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from lib_mhr.object_symmetry import OBJECT_SYMMETRY_MODE_FINITE, OBJECT_SYMMETRY_MODE_FULL_SO3
from prep.foundationpose_pose_selection import validate_rigid_transform, validate_rigid_transforms


FOUNDATIONPOSE_TRAINING_TIER_REVISION = "foundationpose-training-tiers-v6-exact-object-symmetry-groups"
FOUNDATIONPOSE_TRAINING_TIER_BOUNDARY_REVISION = "foundationpose-training-tier-boundaries-v1-float32-tolerant"
FOUNDATIONPOSE_TRAINING_TIER_ROTATION_TOLERANCE_DEG = 1e-5
FOUNDATIONPOSE_TRAINING_TIER_TRANSLATION_DIAMETER_TOLERANCE = 1e-6
FOUNDATIONPOSE_TRAINING_OBJECT_INITIALIZATION_SCHEMA = "foundationpose-score-ranked-tiers-only-v2"
FOUNDATIONPOSE_TRAINING_TIER_REFERENCE_GT = np.uint8(0)
FOUNDATIONPOSE_TRAINING_TIER_REFERENCE_TOP1 = np.uint8(1)
FOUNDATIONPOSE_TRAINING_MISSING_NONE = np.uint8(0)
FOUNDATIONPOSE_TRAINING_MISSING_EMPTY_MASK = np.uint8(1)
FOUNDATIONPOSE_TRAINING_MISSING_DEPTH_SUPPORT = np.uint8(2)
FOUNDATIONPOSE_TRAINING_MISSING_REASON_NAMES = {
    int(FOUNDATIONPOSE_TRAINING_MISSING_NONE): "none",
    int(FOUNDATIONPOSE_TRAINING_MISSING_EMPTY_MASK): "empty_effective_object_mask",
    int(FOUNDATIONPOSE_TRAINING_MISSING_DEPTH_SUPPORT): "insufficient_valid_object_depth_pixels",
}
FOUNDATIONPOSE_TRAINING_TIER_COUNT = 3
FOUNDATIONPOSE_TRAINING_TIER_FIELDS = (
    "fp_training_tier_poses_world",
    "fp_training_tier_valid",
    "fp_training_tier_rotation_error_deg",
    "fp_training_tier_translation_error_m",
    "fp_training_tier_translation_error_diameter",
    "fp_training_tier_score",
    "fp_training_tier_score_rank",
    "fp_training_tier_source_id",
    "fp_training_tier_symmetry_index",
    "fp_training_top1_tier",
    "fp_training_gt_valid",
    "fp_training_reference_type",
    "fp_training_candidate_count",
    "fp_training_imputed",
    "fp_training_donor_frame_index",
    "fp_training_donor_frame_offset",
    "fp_training_missing_reason",
)
FOUNDATIONPOSE_TRAINING_TIER_H5_PATHS = {
    "fp_training_tier_poses_world": "object/init_tiers/pose",
    "fp_training_tier_valid": "object/init_tiers/valid",
    "fp_training_tier_rotation_error_deg": "object/init_tiers/rotation_error_deg",
    "fp_training_tier_translation_error_m": "object/init_tiers/translation_error_m",
    "fp_training_tier_translation_error_diameter": "object/init_tiers/translation_error_diameter",
    "fp_training_tier_score": "object/init_tiers/score",
    "fp_training_tier_score_rank": "object/init_tiers/score_rank",
    "fp_training_tier_source_id": "object/init_tiers/source_id",
    "fp_training_tier_symmetry_index": "object/init_tiers/symmetry_index",
    "fp_training_top1_tier": "object/init_tiers/top1_tier",
    "fp_training_gt_valid": "object/init_tiers/gt_valid",
    "fp_training_reference_type": "object/init_tiers/reference_type",
    "fp_training_candidate_count": "object/init_tiers/candidate_count",
    "fp_training_imputed": "object/init_tiers/imputed",
    "fp_training_donor_frame_index": "object/init_tiers/donor_frame_index",
    "fp_training_donor_frame_offset": "object/init_tiers/donor_frame_offset",
    "fp_training_missing_reason": "object/init_tiers/missing_reason",
}


@dataclass(frozen=True)
class FoundationPoseTrainingTierThresholds:
    tier1_rotation_deg: float = 10.0
    tier1_translation_diameter: float = 0.50
    tier2_rotation_deg: float = 45.0
    tier2_translation_diameter: float = 1.00

    def translation_limits(self, object_diameter_m: float) -> tuple[float, float]:
        object_diameter_m = float(object_diameter_m)
        if not np.isfinite(object_diameter_m) or object_diameter_m <= 0:
            raise ValueError(f"Object diameter must be finite and positive, got {object_diameter_m}")
        tier1 = float(self.tier1_translation_diameter) * object_diameter_m
        tier2 = float(self.tier2_translation_diameter) * object_diameter_m
        if not 0 < self.tier1_rotation_deg <= self.tier2_rotation_deg or tier1 <= 0 or tier2 <= 0:
            raise ValueError(f"Invalid FoundationPose tier thresholds: {self}, diameter={object_diameter_m}")
        return tier1, tier2


def _rotation_errors_deg(candidate_rotation: np.ndarray, reference_rotations: np.ndarray) -> np.ndarray:
    relative = candidate_rotation[None] @ np.swapaxes(reference_rotations, -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine)).astype(np.float32)


def foundationpose_training_tier_boundary_identity(thresholds: FoundationPoseTrainingTierThresholds = FoundationPoseTrainingTierThresholds()) -> dict[str, Any]:
    return {
        "foundationpose_training_tier_boundary_revision": FOUNDATIONPOSE_TRAINING_TIER_BOUNDARY_REVISION,
        "foundationpose_training_tier_boundary_dtype": "float32",
        "foundationpose_training_tier_boundary_comparison": "error_lte_float32_threshold_plus_tolerance",
        "foundationpose_training_tier_rotation_tolerance_deg": float(FOUNDATIONPOSE_TRAINING_TIER_ROTATION_TOLERANCE_DEG),
        "foundationpose_training_tier_translation_diameter_tolerance": float(FOUNDATIONPOSE_TRAINING_TIER_TRANSLATION_DIAMETER_TOLERANCE),
        "foundationpose_training_tier1_rotation_deg": float(thresholds.tier1_rotation_deg),
        "foundationpose_training_tier1_translation_diameter": float(thresholds.tier1_translation_diameter),
        "foundationpose_training_tier2_rotation_deg": float(thresholds.tier2_rotation_deg),
        "foundationpose_training_tier2_translation_diameter": float(thresholds.tier2_translation_diameter),
    }


def foundationpose_training_tier_masks(rotation_error_deg: np.ndarray, translation_error_diameter: np.ndarray, thresholds: FoundationPoseTrainingTierThresholds = FoundationPoseTrainingTierThresholds()) -> tuple[np.ndarray, np.ndarray]:
    rotation_error_deg = np.asarray(rotation_error_deg, dtype=np.float32)
    translation_error_diameter = np.asarray(translation_error_diameter, dtype=np.float32)
    if rotation_error_deg.shape != translation_error_diameter.shape or not np.isfinite(rotation_error_deg).all() or not np.isfinite(translation_error_diameter).all():
        raise ValueError(f"FoundationPose tier errors must be finite arrays with matching shapes, got {rotation_error_deg.shape} and {translation_error_diameter.shape}")
    limits = foundationpose_training_tier_boundary_identity(thresholds)
    tier1 = (rotation_error_deg <= np.float32(limits["foundationpose_training_tier1_rotation_deg"] + limits["foundationpose_training_tier_rotation_tolerance_deg"])) & (translation_error_diameter <= np.float32(limits["foundationpose_training_tier1_translation_diameter"] + limits["foundationpose_training_tier_translation_diameter_tolerance"]))
    tier2 = (rotation_error_deg <= np.float32(limits["foundationpose_training_tier2_rotation_deg"] + limits["foundationpose_training_tier_rotation_tolerance_deg"])) & (translation_error_diameter <= np.float32(limits["foundationpose_training_tier2_translation_diameter"] + limits["foundationpose_training_tier_translation_diameter_tolerance"]))
    return tier1, tier2


def paired_symmetry_pose_error(candidate_pose: np.ndarray, reference_pose: np.ndarray, symmetry_tfs: np.ndarray, object_diameter_m: float, thresholds: FoundationPoseTrainingTierThresholds = FoundationPoseTrainingTierThresholds(), *, symmetry_mode: int = OBJECT_SYMMETRY_MODE_FINITE, symmetry_center: np.ndarray | None = None) -> tuple[int, float, float, float, int]:
    candidate_pose = validate_rigid_transform(candidate_pose, "FoundationPose training candidate")
    reference_pose = validate_rigid_transform(reference_pose, "FoundationPose training reference")
    symmetry_tfs = validate_rigid_transforms(symmetry_tfs, "FoundationPose training symmetry transforms")
    symmetry_mode = int(symmetry_mode)
    symmetry_center = np.zeros(3, dtype=np.float32) if symmetry_center is None else np.asarray(symmetry_center, dtype=np.float32)
    if symmetry_center.shape != (3,) or not np.isfinite(symmetry_center).all():
        raise ValueError(f"FoundationPose training symmetry center must be finite with shape [3], got {symmetry_center.shape}")
    if symmetry_mode == OBJECT_SYMMETRY_MODE_FULL_SO3:
        candidate_center = candidate_pose[:3, :3] @ symmetry_center + candidate_pose[:3, 3]
        reference_center = reference_pose[:3, :3] @ symmetry_center + reference_pose[:3, 3]
        rotation_errors = np.zeros(1, dtype=np.float32)
        translation_errors = np.asarray([np.linalg.norm(candidate_center - reference_center)], dtype=np.float32)
    elif symmetry_mode == OBJECT_SYMMETRY_MODE_FINITE:
        equivalent_references = validate_rigid_transforms(reference_pose[None] @ symmetry_tfs, "FoundationPose symmetry-equivalent references")
        rotation_errors = _rotation_errors_deg(candidate_pose[:3, :3], equivalent_references[:, :3, :3])
        translation_errors = np.linalg.norm(candidate_pose[None, :3, 3] - equivalent_references[:, :3, 3], axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported object symmetry mode {symmetry_mode}")
    tier1_translation_m, tier2_translation_m = thresholds.translation_limits(object_diameter_m)
    translation_errors_diameter = (translation_errors.astype(np.float64) / float(object_diameter_m)).astype(np.float32)
    tier1, tier2 = foundationpose_training_tier_masks(rotation_errors, translation_errors_diameter, thresholds)
    if np.any(tier1):
        tier = 1
        eligible = np.flatnonzero(tier1)
        difficulty = (rotation_errors[eligible] / thresholds.tier1_rotation_deg) ** 2 + (translation_errors[eligible] / tier1_translation_m) ** 2
    elif np.any(tier2):
        tier = 2
        eligible = np.flatnonzero(tier2)
        difficulty = (rotation_errors[eligible] / thresholds.tier2_rotation_deg) ** 2 + (translation_errors[eligible] / tier2_translation_m) ** 2
    else:
        tier = 3
        eligible = np.arange(len(rotation_errors), dtype=np.int64)
        difficulty = (rotation_errors / 180.0) ** 2 + (translation_errors / max(float(object_diameter_m), tier2_translation_m)) ** 2
    symmetry_index = int(eligible[int(np.argmin(difficulty))])
    rotation_error_deg = float(rotation_errors[symmetry_index])
    translation_error_m = float(translation_errors[symmetry_index])
    return tier, rotation_error_deg, translation_error_m, float(translation_errors_diameter[symmetry_index]), symmetry_index


def select_foundationpose_training_tiers(candidate_poses_world: np.ndarray, candidate_scores: np.ndarray, candidate_source_ids: np.ndarray, gt_pose_world: np.ndarray, gt_valid: bool, symmetry_tfs: np.ndarray, object_diameter_m: float, thresholds: FoundationPoseTrainingTierThresholds = FoundationPoseTrainingTierThresholds(), *, symmetry_mode: int = OBJECT_SYMMETRY_MODE_FINITE, symmetry_center: np.ndarray | None = None) -> dict[str, np.ndarray | np.generic]:
    candidate_poses_world = validate_rigid_transforms(candidate_poses_world, "FoundationPose score-ranked training candidates")
    candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
    candidate_source_ids = np.asarray(candidate_source_ids)
    if candidate_scores.shape != (len(candidate_poses_world),) or not np.isfinite(candidate_scores).all():
        raise ValueError(f"FoundationPose candidate scores must be finite with shape ({len(candidate_poses_world)},), got {candidate_scores.shape}")
    if candidate_source_ids.shape != (len(candidate_poses_world),) or not np.issubdtype(candidate_source_ids.dtype, np.integer):
        raise ValueError(f"FoundationPose candidate source IDs must be integer with shape ({len(candidate_poses_world)},), got {candidate_source_ids.shape} {candidate_source_ids.dtype}")
    if len(candidate_poses_world) == 0:
        raise ValueError("FoundationPose training tier selection requires at least one candidate")
    score_order = np.argsort(-candidate_scores, kind="stable")
    candidates = candidate_poses_world[score_order]
    scores = candidate_scores[score_order]
    source_ids = candidate_source_ids[score_order].astype(np.int32, copy=False)
    reference_pose = validate_rigid_transform(gt_pose_world, "FoundationPose ground-truth reference") if gt_valid else candidates[0]
    reference_type = FOUNDATIONPOSE_TRAINING_TIER_REFERENCE_GT if gt_valid else FOUNDATIONPOSE_TRAINING_TIER_REFERENCE_TOP1

    classifications = [paired_symmetry_pose_error(candidate, reference_pose, symmetry_tfs, object_diameter_m, thresholds, symmetry_mode=symmetry_mode, symmetry_center=symmetry_center) for candidate in candidates]
    candidate_tiers = np.asarray([item[0] for item in classifications], dtype=np.uint8)
    top1_tier = int(candidate_tiers[0])
    tier_cap = top1_tier if gt_valid else FOUNDATIONPOSE_TRAINING_TIER_COUNT
    selected_poses = np.repeat(np.eye(4, dtype=np.float32)[None], FOUNDATIONPOSE_TRAINING_TIER_COUNT, axis=0)
    selected_valid = np.zeros((FOUNDATIONPOSE_TRAINING_TIER_COUNT,), dtype=bool)
    selected_rotation = np.full((FOUNDATIONPOSE_TRAINING_TIER_COUNT,), -1.0, dtype=np.float32)
    selected_translation = np.full((FOUNDATIONPOSE_TRAINING_TIER_COUNT,), -1.0, dtype=np.float32)
    selected_translation_diameter = np.full((FOUNDATIONPOSE_TRAINING_TIER_COUNT,), -1.0, dtype=np.float32)
    selected_scores = np.zeros((FOUNDATIONPOSE_TRAINING_TIER_COUNT,), dtype=np.float32)
    selected_ranks = np.full((FOUNDATIONPOSE_TRAINING_TIER_COUNT,), -1, dtype=np.int32)
    selected_source_ids = np.full((FOUNDATIONPOSE_TRAINING_TIER_COUNT,), -1, dtype=np.int32)
    selected_symmetry_indices = np.full((FOUNDATIONPOSE_TRAINING_TIER_COUNT,), -1, dtype=np.int16)
    for tier in range(1, tier_cap + 1):
        matches = np.flatnonzero(candidate_tiers == tier)
        if len(matches) == 0:
            continue
        rank = int(matches[0])
        classification = classifications[rank]
        slot = tier - 1
        selected_poses[slot] = candidates[rank]
        selected_valid[slot] = True
        selected_rotation[slot] = classification[1]
        selected_translation[slot] = classification[2]
        selected_translation_diameter[slot] = classification[3]
        selected_scores[slot] = scores[rank]
        selected_ranks[slot] = rank
        selected_source_ids[slot] = source_ids[rank]
        selected_symmetry_indices[slot] = classification[4]
    if not selected_valid[top1_tier - 1]:
        raise RuntimeError(f"Top-1 FoundationPose tier {top1_tier} was not retained")
    return {
        "top1_pose_world": candidates[0].astype(np.float32),
        "fp_training_tier_poses_world": selected_poses,
        "fp_training_tier_valid": selected_valid,
        "fp_training_tier_rotation_error_deg": selected_rotation,
        "fp_training_tier_translation_error_m": selected_translation,
        "fp_training_tier_translation_error_diameter": selected_translation_diameter,
        "fp_training_tier_score": selected_scores,
        "fp_training_tier_score_rank": selected_ranks,
        "fp_training_tier_source_id": selected_source_ids,
        "fp_training_tier_symmetry_index": selected_symmetry_indices,
        "fp_training_top1_tier": np.uint8(top1_tier),
        "fp_training_gt_valid": np.bool_(gt_valid),
        "fp_training_reference_type": np.uint8(reference_type),
        "fp_training_candidate_count": np.int32(len(candidates)),
    }


def add_foundationpose_training_tier_provenance(selection: Mapping[str, Any], frame_index: int, donor_frame_index: int, missing_reason: int) -> dict[str, np.ndarray | np.generic]:
    frame_index = int(frame_index)
    donor_frame_index = int(donor_frame_index)
    missing_reason = int(missing_reason)
    if frame_index < 0 or donor_frame_index < 0:
        raise ValueError(f"Frame and donor indices must be nonnegative, got {frame_index}/{donor_frame_index}")
    if missing_reason not in FOUNDATIONPOSE_TRAINING_MISSING_REASON_NAMES:
        raise ValueError(f"Unsupported FoundationPose missing reason {missing_reason}")
    imputed = donor_frame_index != frame_index
    if imputed != (missing_reason != int(FOUNDATIONPOSE_TRAINING_MISSING_NONE)):
        raise ValueError(f"FoundationPose imputation and missing reason disagree: frame={frame_index} donor={donor_frame_index} reason={missing_reason}")
    result = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in selection.items()}
    result["fp_training_imputed"] = np.bool_(imputed)
    result["fp_training_donor_frame_index"] = np.int32(donor_frame_index)
    result["fp_training_donor_frame_offset"] = np.int32(donor_frame_index - frame_index)
    result["fp_training_missing_reason"] = np.uint8(missing_reason)
    return result


def reclassify_foundationpose_training_tier_selection(selection: Mapping[str, Any], gt_pose_world: np.ndarray, gt_valid: bool, symmetry_tfs: np.ndarray, object_diameter_m: float, thresholds: FoundationPoseTrainingTierThresholds = FoundationPoseTrainingTierThresholds(), *, symmetry_mode: int = OBJECT_SYMMETRY_MODE_FINITE, symmetry_center: np.ndarray | None = None, preserve_candidate_count: bool = True) -> dict[str, np.ndarray | np.generic]:
    retained = np.asarray(selection["fp_training_tier_valid"], dtype=bool)
    ranks = np.asarray(selection["fp_training_tier_score_rank"], dtype=np.int32)[retained]
    order = np.argsort(ranks, kind="stable")
    poses = np.asarray(selection["fp_training_tier_poses_world"], dtype=np.float32)[retained][order]
    scores = np.asarray(selection["fp_training_tier_score"], dtype=np.float32)[retained][order]
    source_ids = np.asarray(selection["fp_training_tier_source_id"], dtype=np.int32)[retained][order]
    ranks = ranks[order]
    if len(poses) == 0 or int(ranks[0]) != 0 or len(np.unique(ranks)) != len(ranks):
        raise ValueError("FoundationPose retained tier representatives must contain unique score ranks starting at rank 0")
    if not np.array_equal(np.argsort(-scores, kind="stable"), np.arange(len(scores))):
        raise ValueError("FoundationPose retained tier scores disagree with their score-network ranks")
    selected = select_foundationpose_training_tiers(poses, scores, source_ids, gt_pose_world, bool(gt_valid), symmetry_tfs, object_diameter_m, thresholds, symmetry_mode=symmetry_mode, symmetry_center=symmetry_center)
    selected_valid = np.asarray(selected["fp_training_tier_valid"], dtype=bool)
    local_ranks = np.asarray(selected["fp_training_tier_score_rank"], dtype=np.int32)
    inherited_ranks = local_ranks.copy()
    inherited_ranks[selected_valid] = ranks[local_ranks[selected_valid]]
    selected["fp_training_tier_score_rank"] = inherited_ranks
    if preserve_candidate_count:
        candidate_count = int(selection["fp_training_candidate_count"])
        if candidate_count < int(ranks.max()) + 1:
            raise ValueError(f"FoundationPose candidate count {candidate_count} is inconsistent with retained score rank {int(ranks.max())}")
        selected["fp_training_candidate_count"] = np.int32(candidate_count)
    return selected


def nearest_direct_donor_indices(missing_reason: np.ndarray) -> np.ndarray:
    missing_reason = np.asarray(missing_reason)
    if missing_reason.ndim != 1 or not np.issubdtype(missing_reason.dtype, np.integer) or not np.isin(missing_reason, tuple(FOUNDATIONPOSE_TRAINING_MISSING_REASON_NAMES)).all():
        raise ValueError(f"FoundationPose missing reasons must be a one-dimensional supported integer array, got {missing_reason.shape} {missing_reason.dtype}")
    direct = np.flatnonzero(missing_reason == int(FOUNDATIONPOSE_TRAINING_MISSING_NONE))
    if len(direct) == 0:
        raise ValueError("FoundationPose neighbor imputation requires at least one directly estimated frame")
    donors = np.empty(len(missing_reason), dtype=np.int32)
    for frame_index in range(len(missing_reason)):
        insertion = int(np.searchsorted(direct, frame_index))
        left = int(direct[insertion - 1]) if insertion > 0 else None
        right = int(direct[insertion]) if insertion < len(direct) else None
        if left is None:
            donor = right
        elif right is None:
            donor = left
        else:
            donor = left if frame_index - left <= right - frame_index else right
        donors[frame_index] = int(donor)
    return donors


def impute_foundationpose_training_tiers(direct_selections: list[Mapping[str, Any] | None], missing_reason: np.ndarray, gt_poses_world: np.ndarray, gt_valid: np.ndarray, symmetry_tfs: np.ndarray, object_diameter_m: float, thresholds: FoundationPoseTrainingTierThresholds = FoundationPoseTrainingTierThresholds(), *, symmetry_mode: int = OBJECT_SYMMETRY_MODE_FINITE, symmetry_center: np.ndarray | None = None) -> list[dict[str, np.ndarray | np.generic]]:
    missing_reason = np.asarray(missing_reason, dtype=np.uint8)
    gt_poses_world = validate_rigid_transforms(gt_poses_world, "FoundationPose imputation ground-truth poses")
    gt_valid = np.asarray(gt_valid, dtype=bool)
    if missing_reason.shape != (len(direct_selections),) or gt_poses_world.shape != (len(direct_selections), 4, 4) or gt_valid.shape != (len(direct_selections),):
        raise ValueError(f"FoundationPose imputation arrays differ: selections={len(direct_selections)} missing={missing_reason.shape} gt={gt_poses_world.shape} valid={gt_valid.shape}")
    donors = nearest_direct_donor_indices(missing_reason)
    output = []
    for frame_index, reason in enumerate(missing_reason.tolist()):
        donor_index = int(donors[frame_index])
        donor = direct_selections[donor_index]
        if donor is None:
            raise ValueError(f"FoundationPose direct donor {donor_index} is missing for target frame {frame_index}")
        if reason == int(FOUNDATIONPOSE_TRAINING_MISSING_NONE):
            if donor_index != frame_index:
                raise RuntimeError(f"Direct FoundationPose frame {frame_index} selected a different donor {donor_index}")
            selected = reclassify_foundationpose_training_tier_selection(donor, gt_poses_world[frame_index], bool(gt_valid[frame_index]), symmetry_tfs, object_diameter_m, thresholds, symmetry_mode=symmetry_mode, symmetry_center=symmetry_center)
            output.append(add_foundationpose_training_tier_provenance(selected, frame_index, donor_index, reason))
            continue
        selected = reclassify_foundationpose_training_tier_selection(donor, gt_poses_world[frame_index], bool(gt_valid[frame_index]), symmetry_tfs, object_diameter_m, thresholds, symmetry_mode=symmetry_mode, symmetry_center=symmetry_center, preserve_candidate_count=False)
        output.append(add_foundationpose_training_tier_provenance(selected, frame_index, donor_index, reason))
    return output


def training_tier_metadata(thresholds: FoundationPoseTrainingTierThresholds = FoundationPoseTrainingTierThresholds()) -> dict[str, Any]:
    return {
        "foundationpose_training_object_initialization_schema": FOUNDATIONPOSE_TRAINING_OBJECT_INITIALIZATION_SCHEMA,
        "foundationpose_training_tier_revision": FOUNDATIONPOSE_TRAINING_TIER_REVISION,
        "foundationpose_training_tier_count": FOUNDATIONPOSE_TRAINING_TIER_COUNT,
        "foundationpose_training_tier_score_order": "descending_foundationpose_score_first_candidate_per_tier",
        "foundationpose_training_tier_top1_cap": "gt_valid_frames_keep_only_tiers_not_harder_than_top1",
        "foundationpose_training_tier_invalid_gt_reference": "foundationpose_top1_pseudo_reference_no_top1_cap",
        "foundationpose_training_tier_sampling": "uniform_over_available_tiers_independently_per_training_frame",
        "foundationpose_training_tier_symmetry_semantics": "finite_group_closure_or_exact_full_so3_center_distance",
        "foundationpose_training_object_loss_mask": "object_pose_valid_mask",
        "foundationpose_training_missing_support_policy": "nearest_direct_same_camera_frame_previous_on_tie",
        "foundationpose_training_missing_support_reasons": {str(key): value for key, value in FOUNDATIONPOSE_TRAINING_MISSING_REASON_NAMES.items()},
        "foundationpose_training_imputed_pose_policy": "copy_one_donor_pose_set_then_reclassify_against_target_frame_reference",
        "foundationpose_training_imputed_score_policy": "inherit_donor_scores_score_ranks_and_source_ids",
        "foundationpose_training_imputed_candidate_count": "number_of_retained_donor_tier_representatives_reclassified_at_target",
        "foundationpose_training_tier1_rotation_deg": float(thresholds.tier1_rotation_deg),
        "foundationpose_training_tier1_translation_diameter": float(thresholds.tier1_translation_diameter),
        "foundationpose_training_tier1_translation_policy": "object_diameter_fraction_only",
        "foundationpose_training_tier2_rotation_deg": float(thresholds.tier2_rotation_deg),
        "foundationpose_training_tier2_translation_diameter": float(thresholds.tier2_translation_diameter),
        "foundationpose_training_tier2_translation_policy": "object_diameter_fraction_only",
    }


def validate_foundationpose_training_tier_metadata(metadata: Mapping[str, Any], label: str) -> None:
    expected = training_tier_metadata()
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{label} has stale {key}: expected={value!r}, actual={metadata.get(key)!r}")


def validate_foundationpose_training_tier_arrays(data: Mapping[str, Any], prefix_shape: tuple[int, ...], label: str) -> dict[str, np.ndarray]:
    missing = [key for key in FOUNDATIONPOSE_TRAINING_TIER_FIELDS if key not in data]
    if missing:
        raise KeyError(f"{label} is missing FoundationPose training tier fields: {missing}")
    expected_shapes = {
        "fp_training_tier_poses_world": prefix_shape + (3, 4, 4),
        "fp_training_tier_valid": prefix_shape + (3,),
        "fp_training_tier_rotation_error_deg": prefix_shape + (3,),
        "fp_training_tier_translation_error_m": prefix_shape + (3,),
        "fp_training_tier_translation_error_diameter": prefix_shape + (3,),
        "fp_training_tier_score": prefix_shape + (3,),
        "fp_training_tier_score_rank": prefix_shape + (3,),
        "fp_training_tier_source_id": prefix_shape + (3,),
        "fp_training_tier_symmetry_index": prefix_shape + (3,),
        "fp_training_top1_tier": prefix_shape,
        "fp_training_gt_valid": prefix_shape,
        "fp_training_reference_type": prefix_shape,
        "fp_training_candidate_count": prefix_shape,
        "fp_training_imputed": prefix_shape,
        "fp_training_donor_frame_index": prefix_shape,
        "fp_training_donor_frame_offset": prefix_shape,
        "fp_training_missing_reason": prefix_shape,
    }
    arrays = {key: np.asarray(data[key]) for key in FOUNDATIONPOSE_TRAINING_TIER_FIELDS}
    for key, expected_shape in expected_shapes.items():
        if arrays[key].shape != expected_shape:
            raise ValueError(f"{label}/{key} expected shape {expected_shape}, got {arrays[key].shape}")
    poses = arrays["fp_training_tier_poses_world"].astype(np.float32, copy=False)
    validate_rigid_transforms(poses.reshape(-1, 4, 4), f"{label}/fp_training_tier_poses_world")
    valid = arrays["fp_training_tier_valid"].astype(bool, copy=False)
    if not valid.any(axis=-1).all():
        raise ValueError(f"{label}/fp_training_tier_valid must retain at least one tier per frame/view")
    top1_tier = arrays["fp_training_top1_tier"]
    if not np.issubdtype(top1_tier.dtype, np.integer) or not np.isin(top1_tier, (1, 2, 3)).all():
        raise ValueError(f"{label}/fp_training_top1_tier must contain integer tiers 1, 2, or 3")
    if not np.take_along_axis(valid, (top1_tier.astype(np.int64) - 1)[..., None], axis=-1)[..., 0].all():
        raise ValueError(f"{label}/fp_training_tier_valid must retain each frame/view top-1 tier")
    float_fields = ("fp_training_tier_rotation_error_deg", "fp_training_tier_translation_error_m", "fp_training_tier_translation_error_diameter", "fp_training_tier_score")
    for key in float_fields:
        if not np.issubdtype(arrays[key].dtype, np.floating) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{label}/{key} must contain finite floating-point values")
    for key in ("fp_training_tier_rotation_error_deg", "fp_training_tier_translation_error_m", "fp_training_tier_translation_error_diameter"):
        if np.any(arrays[key][valid] < 0) or np.any(arrays[key][~valid] != -1):
            raise ValueError(f"{label}/{key} must be nonnegative for valid tiers and -1 for unavailable tiers")
    for key in ("fp_training_tier_score_rank", "fp_training_tier_source_id", "fp_training_tier_symmetry_index"):
        if not np.issubdtype(arrays[key].dtype, np.integer) or np.any(arrays[key][valid] < 0) or np.any(arrays[key][~valid] != -1):
            raise ValueError(f"{label}/{key} must be nonnegative for valid tiers and -1 for unavailable tiers")
    selected_top1_rank = np.take_along_axis(arrays["fp_training_tier_score_rank"], (top1_tier.astype(np.int64) - 1)[..., None], axis=-1)[..., 0]
    if np.any(selected_top1_rank != 0):
        raise ValueError(f"{label}/fp_training_tier_score_rank must retain score rank 0 in fp_training_top1_tier")
    gt_valid = arrays["fp_training_gt_valid"].astype(bool, copy=False)
    if np.any(~gt_valid & (top1_tier != 1)):
        raise ValueError(f"{label}/fp_training_top1_tier must be Tier 1 when ground truth is invalid and top-1 is the pseudo-reference")
    tier_numbers = np.arange(1, FOUNDATIONPOSE_TRAINING_TIER_COUNT + 1, dtype=np.uint8)
    if np.any(valid & gt_valid[..., None] & (tier_numbers > top1_tier[..., None])):
        raise ValueError(f"{label}/fp_training_tier_valid violates the ground-truth-valid top-1 tier cap")
    reference_type = arrays["fp_training_reference_type"]
    expected_reference = np.where(gt_valid, FOUNDATIONPOSE_TRAINING_TIER_REFERENCE_GT, FOUNDATIONPOSE_TRAINING_TIER_REFERENCE_TOP1)
    if not np.array_equal(reference_type, expected_reference):
        raise ValueError(f"{label}/fp_training_reference_type disagrees with fp_training_gt_valid")
    candidate_count = arrays["fp_training_candidate_count"]
    if not np.issubdtype(candidate_count.dtype, np.integer) or np.any(candidate_count < 1):
        raise ValueError(f"{label}/fp_training_candidate_count must contain positive integers")
    if np.any(valid.sum(axis=-1) > candidate_count):
        raise ValueError(f"{label}/fp_training_candidate_count is smaller than the number of retained tiers")
    if len(prefix_shape) < 1:
        raise ValueError(f"{label} must have a temporal prefix dimension")
    imputed = arrays["fp_training_imputed"]
    donor_index = arrays["fp_training_donor_frame_index"]
    donor_offset = arrays["fp_training_donor_frame_offset"]
    missing_reason = arrays["fp_training_missing_reason"]
    if imputed.dtype != np.dtype(bool):
        raise ValueError(f"{label}/fp_training_imputed must be Boolean")
    for key, value in (("fp_training_donor_frame_index", donor_index), ("fp_training_donor_frame_offset", donor_offset), ("fp_training_missing_reason", missing_reason)):
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{label}/{key} must be integer")
    if np.any(donor_index < 0) or np.any(donor_index >= prefix_shape[0]):
        raise ValueError(f"{label}/fp_training_donor_frame_index is outside the sequence timeline")
    time_shape = (prefix_shape[0],) + (1,) * (len(prefix_shape) - 1)
    frame_index = np.broadcast_to(np.arange(prefix_shape[0], dtype=np.int64).reshape(time_shape), prefix_shape)
    if not np.array_equal(donor_offset.astype(np.int64), donor_index.astype(np.int64) - frame_index):
        raise ValueError(f"{label}/fp_training_donor_frame_offset does not equal donor minus target frame index")
    if np.any(~imputed & ((donor_index != frame_index) | (donor_offset != 0) | (missing_reason != FOUNDATIONPOSE_TRAINING_MISSING_NONE))):
        raise ValueError(f"{label} direct FoundationPose frames have invalid donor provenance")
    if np.any(imputed & ((donor_index == frame_index) | (donor_offset == 0) | ~np.isin(missing_reason, (FOUNDATIONPOSE_TRAINING_MISSING_EMPTY_MASK, FOUNDATIONPOSE_TRAINING_MISSING_DEPTH_SUPPORT)))):
        raise ValueError(f"{label} imputed FoundationPose frames have invalid donor provenance")
    rotation_error = arrays["fp_training_tier_rotation_error_deg"]
    translation_error_m = arrays["fp_training_tier_translation_error_m"]
    translation_error_diameter = arrays["fp_training_tier_translation_error_diameter"]
    thresholds = FoundationPoseTrainingTierThresholds()
    tier1_match, tier2_match = foundationpose_training_tier_masks(rotation_error, translation_error_diameter, thresholds)
    if np.any(valid[..., 0] & ~tier1_match[..., 0]):
        raise ValueError(f"{label} has a retained Tier 1 pose outside the Tier 1 thresholds")
    if np.any(valid[..., 1] & (tier1_match[..., 1] | ~tier2_match[..., 1])):
        raise ValueError(f"{label} has a retained Tier 2 pose outside the Tier 2-only thresholds")
    if np.any(valid[..., 2] & (tier1_match[..., 2] | tier2_match[..., 2])):
        raise ValueError(f"{label} has a retained Tier 3 pose inside the Tier 1 or Tier 2 thresholds")
    return arrays
