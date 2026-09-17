from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import default_collate
from torchvision.transforms import ToTensor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib_mhr import MHR_REFIT_ADAPTIVE_AUDIT_KEYS, MHR_REFIT_MODES, MHR_REFIT_MODE_FULL, MHR_REFIT_MODE_TRANSLATION_UNIFORM_SCALE, MHRLayer, MHRRefitConfig, assert_mhr_schema, load_mhr_result, refit_mhr_to_vertices, save_mhr_result
from lib_mhr.camera_conventions import MHR_ROOT_JOINT_INDEX, mhr_init_parameter_frame_metadata, mhr_init_root_metadata, mhr_translation_between_frames, sam3d_root_camera_to_world_rot6d, validate_mhr_init_root_metadata, validate_mhr_init_translation_metadata
from lib_mhr.schema import MHR_PARAM_DIMS
from prep.mhr_depth_alignment import sample_mhr_alignment_surface
from prep.mhr_export_utils import (
    camera_calibration,
    frame_names,
    load_edex,
    MHR_CAMERA_NAMES,
    mask_input_identity,
    project_points,
    read_depth_m,
    read_mask,
    read_rgb,
    transform_points,
)
from prep.mhr_depth_filter import filter_depth_like_smpl, filtered_depth_points_like_smpl
from prep.mhr_ffv1_sidecar import RGB as RGB_FFV1_KIND, ffv1_sidecar_kind, validate_ffv1_metadata
from prep.mhr_refit_cache import MHRRefitCache, load_mhr_refit_cache, reconstruct_mhr_refit_target_vertices, save_mhr_refit_cache
from prep.mhr_refit_metadata import build_mhr_refit_metadata
from prep.mhr_sam3d_cache import MHRSAM3DDirectCache, MHR_SAM3D_DIRECT_PARAM_DIMS, MHR_SAM3D_DIRECT_RUNTIME_GEOMETRY_KEYS, decode_mhr_sam3d_direct_geometry, load_mhr_sam3d_direct_cache, save_mhr_sam3d_direct_cache
from lib_mhr.body_pose import compact_model_params_to_cont_body_np
from tools import img_utils
from tools.pipeline_timing import PipelineTimer

SAM3D_ROOT = Path(os.environ.get("SAM3D_BODY_ROOT", ROOT / "sam-3d-body"))
if str(SAM3D_ROOT) not in sys.path:
    sys.path.append(str(SAM3D_ROOT))

from sam_3d_body import load_sam_3d_body
from sam_3d_body.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
from sam_3d_body.data.utils.prepare_batch import NoCollate
from sam_3d_body.utils import recursive_to


SAM3D_ASSETS_ROOT = Path(os.environ.get("MHR_ASSETS_ROOT", SAM3D_ROOT))
DEFAULT_SAM3D_CKPT = SAM3D_ASSETS_ROOT / "checkpoints/sam-3d-body-dinov3/model.ckpt"
DEFAULT_MHR_PATH = SAM3D_ASSETS_ROOT / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
DEFAULT_REFIT_CONFIG = MHRRefitConfig()
MHR_REFIT_CACHE_VERSION = 3
MHR_REFIT_INPUT_PREPROCESS_VERSION = 1
MHR_SAM3D_DIRECT_PREPROCESS_VERSION = 2
SAM3D_HUMAN_PROMPT_BBOX_ONLY = "bbox_only"
SAM3D_HUMAN_PROMPT_SAM2_BBOX_MASK = "sam2_bbox_mask"
SAM3D_HUMAN_PROMPT_MODES = (SAM3D_HUMAN_PROMPT_BBOX_ONLY, SAM3D_HUMAN_PROMPT_SAM2_BBOX_MASK)
DEFAULT_SAM3D_HUMAN_PROMPT_MODE = SAM3D_HUMAN_PROMPT_SAM2_BBOX_MASK


def _sam3d_human_prompt_metadata(human_prompt_mode: str) -> dict[str, Any]:
    if human_prompt_mode not in SAM3D_HUMAN_PROMPT_MODES:
        raise ValueError(f"Unsupported SAM 3D Body human prompt mode: {human_prompt_mode!r}")
    return {"mode": human_prompt_mode, "bbox_source": "effective_human_mask_nonzero_support", "mask_source": "effective_human_mask" if human_prompt_mode == SAM3D_HUMAN_PROMPT_SAM2_BBOX_MASK else "none", "mask_score": 1.0 if human_prompt_mode == SAM3D_HUMAN_PROMPT_SAM2_BBOX_MASK else 0.0}


def _validate_mhr_output(path: str | Path, expected_frames: list[str], expected_camera_id: int, expected_refit_config: MHRRefitConfig, expected_depth_metadata: dict[str, Any] | None = None, expected_human_prompt_mode: str = DEFAULT_SAM3D_HUMAN_PROMPT_MODE) -> dict[str, Any]:
    result = load_mhr_result(path)
    assert_mhr_schema(result)
    validate_mhr_init_root_metadata(result.get("metadata", {}), f"MHR output {path}", required=True)
    validate_mhr_init_translation_metadata(result.get("metadata", {}), f"MHR output {path}", required=True)
    if list(result["frames"]) != list(expected_frames):
        raise ValueError(f"MHR output frames differ for {path}: expected={len(expected_frames)}, actual={len(result['frames'])}")
    if [int(value) for value in result["kids"]] != [int(expected_camera_id)]:
        raise ValueError(f"MHR output camera differs for {path}: expected={[expected_camera_id]}, actual={result['kids']}")
    expected_human_prompt = _sam3d_human_prompt_metadata(expected_human_prompt_mode)
    if result.get("metadata", {}).get("sam3d_human_prompt") != expected_human_prompt:
        raise ValueError(f"MHR output SAM 3D Body human prompt is stale for {path}: expected={expected_human_prompt}, actual={result.get('metadata', {}).get('sam3d_human_prompt')}")
    refit = result.get("metadata", {}).get("mhr_refit", {})
    expected = {"mode": expected_refit_config.mode, "iterations": expected_refit_config.iterations, "sample_count": expected_refit_config.sample_count, "optimization_batch_size": expected_refit_config.optimization_batch_size, "parameter_prior_weight": expected_refit_config.parameter_prior_weight, "max_mean_error_increase_m": expected_refit_config.max_mean_error_increase_m}
    mismatched = {key: (refit.get(key), value) for key, value in expected.items() if refit.get(key) != value}
    if expected_refit_config.adaptive_rescue:
        adaptive = refit.get("adaptive_rescue", {})
        adaptive_expected = {"strong_adam.trigger_mean_error_m": expected_refit_config.strong_refit_mean_threshold_m, "strong_adam.iterations": expected_refit_config.strong_iterations, "strong_adam.learning_rate_multiplier": expected_refit_config.strong_learning_rate_multiplier, "strong_adam.parameter_prior_weight": expected_refit_config.strong_parameter_prior_weight, "strong_adam.cosine_final_learning_rate_ratio": expected_refit_config.strong_cosine_final_learning_rate_ratio, "lbfgs.trigger_mean_error_m": expected_refit_config.lbfgs_mean_threshold_m, "lbfgs.trigger_max_error_m": expected_refit_config.lbfgs_max_threshold_m, "lbfgs.iterations": expected_refit_config.lbfgs_iterations}
        for key, value in adaptive_expected.items():
            group, field = key.split(".", 1)
            if adaptive.get(group, {}).get(field) != value:
                mismatched[key] = (adaptive.get(group, {}).get(field), value)
    if mismatched:
        raise ValueError(f"MHR output refit metadata is stale for {path}: {mismatched}")
    if expected_depth_metadata is not None:
        metadata = result.get("metadata", {})
        for key in ("depth_root", "depth_source"):
            if metadata.get(key) != expected_depth_metadata[key]:
                raise ValueError(f"MHR output {key} is stale for {path}: expected={expected_depth_metadata[key]}, actual={metadata.get(key)}")
        actual_depth_input_identity = metadata.get("depth_input_identity")
        if actual_depth_input_identity is not None and actual_depth_input_identity != expected_depth_metadata["depth_input_identity"]:
            raise ValueError(f"MHR output depth artifact is stale for {path}")
    return result


def _explicit_torch_hub_repo(repo_or_dir: str) -> str:
    if repo_or_dir == "facebookresearch/dinov3":
        return "facebookresearch/dinov3:main"
    return repo_or_dir


def _bbox_from_mask(mask: np.ndarray, fallback_shape: tuple[int, int]) -> np.ndarray:
    if np.sum(mask) < 20:
        h, w = fallback_shape
        return np.array([0, 0, w - 1, h - 1], dtype=np.float32)
    bmin, bmax = img_utils.masks2bbox([mask.astype(np.uint8) * 255])
    return np.array([bmin[0], bmin[1], bmax[0], bmax[1]], dtype=np.float32)


def _prepare_batch(frames: list[np.ndarray], bboxes: list[np.ndarray], transform: Any, K: np.ndarray, *, masks: list[np.ndarray] | None = None, human_prompt_mode: str = DEFAULT_SAM3D_HUMAN_PROMPT_MODE) -> dict[str, Any]:
    prompt_metadata = _sam3d_human_prompt_metadata(human_prompt_mode)
    if len(frames) != len(bboxes):
        raise ValueError(f"SAM 3D Body frame/bbox count differs: frames={len(frames)}, bboxes={len(bboxes)}")
    if human_prompt_mode == SAM3D_HUMAN_PROMPT_SAM2_BBOX_MASK and (masks is None or len(masks) != len(frames)):
        raise ValueError(f"SAM 3D Body {human_prompt_mode} requires one mask per frame: frames={len(frames)}, masks={0 if masks is None else len(masks)}")
    data_list = []
    for index, (image, bbox) in enumerate(zip(frames, bboxes)):
        h, w = image.shape[:2]
        prompt_mask = np.zeros((h, w, 1), dtype=np.uint8)
        if human_prompt_mode == SAM3D_HUMAN_PROMPT_SAM2_BBOX_MASK:
            mask = np.asarray(masks[index])
            if mask.shape != (h, w):
                raise ValueError(f"SAM 3D Body prompt mask shape differs at batch index {index}: expected={(h, w)}, actual={mask.shape}")
            if int(np.count_nonzero(mask)) < 20:
                raise ValueError(f"SAM 3D Body prompt mask has fewer than 20 foreground pixels at batch index {index}")
            prompt_mask = mask.astype(bool).astype(np.uint8)[..., None]
        data_info = {
            "img": image,
            "bbox": bbox,
            "bbox_format": "xyxy",
            "mask": prompt_mask,
            "mask_score": np.array(prompt_metadata["mask_score"], dtype=np.float32),
        }
        data_list.append(transform(data_info))

    batch = default_collate(data_list)
    for key in [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    batch["person_valid"] = torch.ones((1, len(frames)))
    batch["cam_int"] = torch.from_numpy(K).float()[None].to(batch["img"])
    batch["img_ori"] = [NoCollate(frames[0])]
    return batch


def _rot6d_world(global_rot_zyx: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    return sam3d_root_camera_to_world_rot6d(global_rot_zyx, c2w)


def _camera_to_world_geometry(points_cam: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    return transform_points(points_cam.reshape(-1, 3), c2w).reshape(points_cam.shape).astype(np.float32)


@dataclass(frozen=True)
class DepthAlignment:
    matrix_cam: np.ndarray
    scale: float
    translation_cam: np.ndarray
    source_points: int
    target_points: int
    image_correction_cam: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    image_correction_px: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    fallback: str = ""


def _scale_camera_intrinsics_to_depth(K: np.ndarray, mask: np.ndarray, depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K_depth = K.copy()
    if mask.shape != depth_m.shape:
        scale_x = depth_m.shape[1] / mask.shape[1]
        scale_y = depth_m.shape[0] / mask.shape[0]
        K_depth[0] *= scale_x
        K_depth[1] *= scale_y
        mask = cv2.resize(
            mask.astype(np.uint8),
            (depth_m.shape[1], depth_m.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return K_depth, mask.astype(bool)


def _filtered_depth_points(depth_m: np.ndarray, K_depth: np.ndarray, human_mask: np.ndarray) -> np.ndarray:
    return filtered_depth_points_like_smpl(depth_m, K_depth, human_mask, max_points=12000)


def _mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _robust_projected_bbox(points_cam: np.ndarray, K: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray | None:
    uv, valid = project_points(points_cam, K, image_shape)
    uv = uv[valid]
    if len(uv) < 128:
        return None
    x0, x1 = np.quantile(uv[:, 0], [0.01, 0.99])
    y0, y1 = np.quantile(uv[:, 1], [0.01, 0.99])
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _image_plane_center_correction(
    vertices_cam: np.ndarray,
    matrix_cam: np.ndarray,
    K_depth: np.ndarray,
    image_shape: tuple[int, int],
    human_mask: np.ndarray,
    *,
    max_px: float = 80.0,
    max_m: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    del human_mask
    original_bbox = _robust_projected_bbox(vertices_cam, K_depth, image_shape)
    if original_bbox is None:
        return np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32)
    aligned = vertices_cam @ matrix_cam[:3, :3].T + matrix_cam[:3, 3]
    mesh_bbox = _robust_projected_bbox(aligned, K_depth, image_shape)
    if mesh_bbox is None:
        return np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

    # SAM3D is driven by the RGB crop/mask and usually has good 2D placement.
    # Depth ICP should change depth/scale but should not slide the mesh away
    # from that image-plane anchor, which is especially ambiguous in front-view
    # partial occlusions.
    original_center = np.array(
        [(original_bbox[0] + original_bbox[2]) * 0.5, (original_bbox[1] + original_bbox[3]) * 0.5],
        dtype=np.float32,
    )
    mesh_center = np.array(
        [(mesh_bbox[0] + mesh_bbox[2]) * 0.5, (mesh_bbox[1] + mesh_bbox[3]) * 0.5],
        dtype=np.float32,
    )
    delta_px = np.clip(original_center - mesh_center, -max_px, max_px).astype(np.float32)

    z_valid = aligned[:, 2]
    z_valid = z_valid[np.isfinite(z_valid) & (z_valid > 0.2)]
    if len(z_valid) == 0:
        return np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32)
    z_ref = float(np.median(z_valid))
    correction = np.array(
        [
            delta_px[0] * z_ref / K_depth[0, 0],
            delta_px[1] * z_ref / K_depth[1, 1],
            0.0,
        ],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(correction[:2]))
    if norm > max_m:
        correction[:2] *= max_m / norm
    return correction.astype(np.float32), delta_px


def _translation_only_icp_np(
    source: np.ndarray,
    target: np.ndarray,
    *,
    voxel_size: float = 0.01,
    max_iters: tuple[int, int, int] = (25, 10, 5),
    axes: tuple[bool, bool, bool] = (True, True, True),
) -> np.ndarray:
    """Robust translation-only ICP using scipy nearest-neighbor search."""

    source_work = np.asarray(source, dtype=np.float32).copy()
    target = np.asarray(target, dtype=np.float32)
    if len(source_work) == 0 or len(target) == 0:
        return np.zeros(3, dtype=np.float32)

    tree = cKDTree(target)
    total = np.zeros(3, dtype=np.float32)
    axis_mask = np.asarray(axes, dtype=bool)
    for max_iter, radius in zip(max_iters, (voxel_size * 8.0, voxel_size * 4.0, voxel_size)):
        max_dist = radius * 3.0
        for _ in range(max_iter):
            dists, idx = tree.query(source_work, k=1)
            inlier = np.isfinite(dists) & (dists < max_dist)
            if int(inlier.sum()) < 32:
                break
            residual = target[idx[inlier]] - source_work[inlier]
            residual_norm = np.linalg.norm(residual, axis=1)
            trim = np.quantile(residual_norm, 0.8)
            if np.isfinite(trim) and trim > 0:
                residual = residual[residual_norm <= trim]
            if len(residual) == 0:
                break
            delta = np.median(residual, axis=0).astype(np.float32)
            delta[~axis_mask] = 0.0
            if not np.all(np.isfinite(delta)) or float(np.linalg.norm(delta)) < 1e-4:
                break
            source_work += delta
            total += delta
    return total.astype(np.float32)


def _identity_alignment(source_points: int = 0, target_points: int = 0, fallback: str = "") -> DepthAlignment:
    return DepthAlignment(
        matrix_cam=np.eye(4, dtype=np.float32),
        scale=1.0,
        translation_cam=np.zeros(3, dtype=np.float32),
        source_points=int(source_points),
        target_points=int(target_points),
        fallback=fallback,
    )


def _align_frame_to_depth(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    depth_m: np.ndarray,
    human_mask: np.ndarray,
    *,
    sample_seed: int,
    depth_is_filtered: bool = False,
) -> DepthAlignment:
    K_depth, human_mask = _scale_camera_intrinsics_to_depth(K, human_mask, depth_m)
    depth_m = np.asarray(depth_m, dtype=np.float32) if depth_is_filtered else filter_depth_like_smpl(depth_m)
    verts_cam = transform_points(vertices_world, w2c)
    source = sample_mhr_alignment_surface(verts_cam, faces, seed=sample_seed)
    target = _filtered_depth_points(depth_m, K_depth, human_mask)
    if len(source) < 128 or len(target) < 128:
        return _identity_alignment(len(source), len(target), "insufficient_points")

    source_z = float(np.median(source[:, 2]))
    target_z = float(np.median(target[:, 2]))
    if not np.isfinite(source_z) or not np.isfinite(target_z) or source_z <= 0.2:
        return _identity_alignment(len(source), len(target), "invalid_depth_median")

    mat = np.eye(4, dtype=np.float32)
    mat[:3, 3] = np.array([0.0, 0.0, target_z - source_z], dtype=np.float32)
    source_first = source + mat[:3, 3]
    trans1 = _translation_only_icp_np(source_first, target, voxel_size=0.02, max_iters=(25, 10, 5))
    mat[:3, 3] += trans1

    aligned_z = source_z + float(mat[2, 3])
    scale = aligned_z / source_z
    if not np.isfinite(scale):
        scale = 1.0
    scale = float(np.clip(scale, 0.65, 1.5))

    mat2 = np.eye(4, dtype=np.float32)
    mat2[:3, :3] *= scale
    mat2[:3, 3] = np.array([0.0, 0.0, target_z - source_z * scale], dtype=np.float32)
    source_second = source * scale + mat2[:3, 3]
    trans2 = _translation_only_icp_np(source_second, target, voxel_size=0.01, max_iters=(25, 10, 5))
    mat2[:3, 3] += trans2
    image_correction_cam, image_correction_px = _image_plane_center_correction(
        verts_cam,
        mat2,
        K_depth,
        depth_m.shape,
        human_mask,
    )
    mat2[:3, 3] += image_correction_cam

    if (
        not np.all(np.isfinite(mat2))
        or np.linalg.norm(mat2[:3, 3]) > 3.0
        or float(np.mean((source * scale + mat2[:3, 3])[:, 2])) < 0.5
    ):
        return _identity_alignment(len(source), len(target), "rejected_transform")

    return DepthAlignment(
        matrix_cam=mat2.astype(np.float32),
        scale=scale,
        translation_cam=mat2[:3, 3].astype(np.float32),
        source_points=len(source),
        target_points=len(target),
        image_correction_cam=image_correction_cam,
        image_correction_px=image_correction_px,
    )


def _apply_camera_alignment(points_world: np.ndarray, w2c: np.ndarray, c2w: np.ndarray, matrix_cam: np.ndarray) -> np.ndarray:
    points_cam = transform_points(points_world.reshape(-1, 3), w2c)
    points_cam = points_cam @ matrix_cam[:3, :3].T + matrix_cam[:3, 3]
    return transform_points(points_cam, c2w).reshape(points_world.shape).astype(np.float32)


def _refit_aligned_mhr_sequence(mhr_layer: MHRLayer, params_np: dict[str, np.ndarray], target_vertices_world: np.ndarray, config: MHRRefitConfig) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    params = {key: torch.from_numpy(np.asarray(value, dtype=np.float32)).cuda() for key, value in params_np.items()}
    target_vertices = torch.from_numpy(np.asarray(target_vertices_world, dtype=np.float32)).cuda()
    result = refit_mhr_to_vertices(mhr_layer, params, target_vertices, config=config)
    refitted_params = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in result.params.items()}
    decoded_joints = []
    decoded_keypoints = []
    with torch.no_grad():
        for start in range(0, target_vertices.shape[0], config.optimization_batch_size):
            stop = min(start + config.optimization_batch_size, target_vertices.shape[0])
            decoded = mhr_layer.mhr_forward({key: value[start:stop] for key, value in result.params.items()})
            decoded_joints.append(decoded.joints.detach().cpu().numpy().astype(np.float32))
            decoded_keypoints.append(decoded.keypoints.detach().cpu().numpy().astype(np.float32))
    refitted_geometry = {
        "mhr_joints": np.concatenate(decoded_joints, axis=0),
        "mhr_keypoints": np.concatenate(decoded_keypoints, axis=0),
        "initial_mean_error_m": result.initial_mean_error_m.detach().cpu().numpy().astype(np.float32),
        "initial_max_error_m": result.initial_max_error_m.detach().cpu().numpy().astype(np.float32),
        "final_mean_error_m": result.final_mean_error_m.detach().cpu().numpy().astype(np.float32),
        "final_max_error_m": result.final_max_error_m.detach().cpu().numpy().astype(np.float32),
    }
    for key in ("uniform_scale_direction_relative_error", "uniform_scale_final_relative_error"):
        value = getattr(result, key)
        if value is not None:
            refitted_geometry[key] = value.detach().cpu().numpy().astype(np.float32)
    if config.mode == MHR_REFIT_MODE_FULL and config.adaptive_rescue:
        for key in MHR_REFIT_ADAPTIVE_AUDIT_KEYS:
            value = getattr(result, key)
            if value is None:
                raise RuntimeError(f"MHR adaptive refit did not return {key}")
            dtype = np.bool_ if key.endswith("_mask") else np.float32
            refitted_geometry[key] = value.detach().cpu().numpy().astype(dtype)
    return refitted_params, refitted_geometry


def _alignment_arrays(alignments: list[DepthAlignment]) -> dict[str, np.ndarray]:
    return {
        "matrix_cam": np.asarray([item.matrix_cam for item in alignments], dtype=np.float32),
        "scales": np.asarray([item.scale for item in alignments], dtype=np.float32),
        "translations_cam": np.asarray([item.translation_cam for item in alignments], dtype=np.float32),
        "image_corrections_cam": np.asarray([item.image_correction_cam for item in alignments], dtype=np.float32),
        "image_corrections_px": np.asarray([item.image_correction_px for item in alignments], dtype=np.float32),
        "source_points": np.asarray([item.source_points for item in alignments], dtype=np.int32),
        "target_points": np.asarray([item.target_points for item in alignments], dtype=np.int32),
        "fallback": np.asarray([item.fallback for item in alignments], dtype=object),
    }


def _alignments_from_arrays(values: dict[str, np.ndarray]) -> list[DepthAlignment]:
    return [DepthAlignment(matrix_cam=np.asarray(values["matrix_cam"][index], dtype=np.float32), scale=float(values["scales"][index]), translation_cam=np.asarray(values["translations_cam"][index], dtype=np.float32), source_points=int(values["source_points"][index]), target_points=int(values["target_points"][index]), image_correction_cam=np.asarray(values["image_corrections_cam"][index], dtype=np.float32), image_correction_px=np.asarray(values["image_corrections_px"][index], dtype=np.float32), fallback=str(values["fallback"][index])) for index in range(len(values["scales"]))]


def _path_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _depth_output_metadata(export_seq: Path, depth_root: str | Path | None, depth_source: str) -> dict[str, Any]:
    resolved_depth_root = Path(depth_root) if depth_root is not None else export_seq / "depth"
    return {"depth_root": str(depth_root) if depth_root is not None else "export_depth", "depth_source": str(depth_source), "depth_input_identity": _path_identity(resolved_depth_root)}


def default_mhr_sam3d_cache_path(out_file: str | Path) -> Path:
    out_file = Path(out_file)
    return out_file.with_name(f"{out_file.stem}.sam3d-direct.h5")


def _sam3d_input_modality_identity(export_seq: Path, dirname: str, camera_id: int, names: list[str]) -> dict[str, Any]:
    if dirname in {"human_masks", "object_masks"}:
        kind = "human" if dirname == "human_masks" else "object"
        return mask_input_identity(export_seq, kind, camera_id)
    camera_name = MHR_CAMERA_NAMES[camera_id]
    h5_path = export_seq / dirname / f"{camera_name}.h5"
    if h5_path.is_file():
        if dirname == "images" and ffv1_sidecar_kind(h5_path) is not None:
            metadata = validate_ffv1_metadata(h5_path, expected_kind=RGB_FFV1_KIND)
            return {
                "storage": "ffv1-sidecar",
                "metadata": _path_identity(h5_path),
                "sidecar": _path_identity(metadata["sidecar_path"]),
                "sidecar_sha256": metadata["sidecar_sha256"],
                "source_logical_sha256": metadata.get("source_logical_sha256"),
                "frame_count": metadata["frame_count"],
                "height": metadata["height"],
                "width": metadata["width"],
            }
        return {"storage": "h5", "source": _path_identity(h5_path)}
    frame_root = export_seq / dirname / camera_name
    digest = hashlib.sha256()
    for name in names:
        path = frame_root / f"{name}.png"
        stat = path.stat()
        digest.update(f"{name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return {"storage": "png", "root": str(frame_root.resolve()), "frame_count": len(names), "metadata_sha256": digest.hexdigest()}


def _sam3d_direct_cache_metadata(export_seq: Path, camera_id: int, names: list[str], sam3d_ckpt: str | Path, mhr_path: str | Path, start: int, stop: int, human_prompt_mode: str = DEFAULT_SAM3D_HUMAN_PROMPT_MODE) -> dict[str, Any]:
    return {
        "preprocess_version": MHR_SAM3D_DIRECT_PREPROCESS_VERSION,
        "camera_id": int(camera_id),
        "edex": _path_identity(export_seq / "edex"),
        "rgb": _sam3d_input_modality_identity(export_seq, "images", camera_id, names),
        "human_mask": _sam3d_input_modality_identity(export_seq, "human_masks", camera_id, names),
        "sam3d_checkpoint": _path_identity(sam3d_ckpt),
        "mhr_model": _path_identity(mhr_path),
        "start": int(start),
        "stop": int(stop),
        "inference_type": "body",
        "thresh_wrist_angle": 1.4,
        "human_prompt": _sam3d_human_prompt_metadata(human_prompt_mode),
    }


def _canonical_sam3d_direct_predictions(out: dict[str, Any], frame_count: int) -> dict[str, np.ndarray]:
    body_pose = out.get("body_pose", out.get("body_pose_params"))
    hand = out.get("hand", out.get("hand_pose_params"))
    scale = out.get("scale", out.get("scale_params"))
    shape = out.get("shape", out.get("shape_params"))
    face = out.get("expr_params")
    if body_pose is None or hand is None or scale is None or shape is None:
        missing = [name for name, value in (("body_pose", body_pose), ("hand", hand), ("scale", scale), ("shape", shape)) if value is None]
        raise KeyError(f"SAM 3D Body output is missing required predictions: {missing}")
    if face is None:
        face = np.zeros((frame_count, MHR_PARAM_DIMS["mhr_face"]), dtype=np.float32)
    values = {
        "global_rot": out["global_rot"],
        "pred_cam_t": out["pred_cam_t"],
        "body_pose_params": body_pose,
        "hand_pose_params": hand,
        "shape_params": shape,
        "scale_params": scale,
        "expr_params": face,
        "pred_vertices": out["pred_vertices"],
        "pred_joint_coords": out["pred_joint_coords"],
        "pred_keypoints_3d": out["pred_keypoints_3d"],
    }
    return {key: np.asarray(value, dtype=np.float32) for key, value in values.items()}


def _refit_cache_metadata(export_seq: Path, depth_root: str | Path | None, depth_source: str, sam3d_ckpt: str | Path, mhr_path: str | Path, camera_id: int, names: list[str], start: int, stop: int, human_prompt_mode: str = DEFAULT_SAM3D_HUMAN_PROMPT_MODE) -> dict[str, Any]:
    resolved_depth_root = Path(depth_root) if depth_root is not None else export_seq / "depth"
    return {
        "cache_version": MHR_REFIT_CACHE_VERSION,
        "alignment_preprocess_version": MHR_REFIT_INPUT_PREPROCESS_VERSION,
        "sam3d_direct_identity": _sam3d_direct_cache_metadata(export_seq, camera_id, names, sam3d_ckpt, mhr_path, start, stop, human_prompt_mode),
        "depth_root": _path_identity(resolved_depth_root),
        "depth_source": str(depth_source),
        **mhr_init_root_metadata(),
    }


def _save_refitted_sequence(export_seq: Path, out_file: Path, names: list[str], camera_id: int, depth_root: str | Path | None, depth_source: str, refit_config: MHRRefitConfig, alignments: list[DepthAlignment], params_np: dict[str, np.ndarray], refitted: dict[str, np.ndarray], human_prompt_mode: str, refit_cache_file: str | Path | None = None, sam3d_cache_file: str | Path | None = None) -> Path:
    refit_error_keys = ["initial_mean_error_m", "initial_max_error_m", "final_mean_error_m", "final_max_error_m"]
    if refit_config.mode == MHR_REFIT_MODE_FULL and refit_config.adaptive_rescue:
        refit_error_keys.extend(MHR_REFIT_ADAPTIVE_AUDIT_KEYS)
    if refit_config.mode == MHR_REFIT_MODE_TRANSLATION_UNIFORM_SCALE:
        refit_error_keys.extend(("uniform_scale_direction_relative_error", "uniform_scale_final_relative_error"))
    refit_errors = {key: [refitted[key]] for key in refit_error_keys}
    metadata = {
        "source": f"sam3d_body_rgb_mhr_{refit_config.mode}_refit_aligned_to_metric_depth",
        "camera_id": camera_id,
        **_depth_output_metadata(export_seq, depth_root, depth_source),
        "alignment": f"median_z_translation_icp_scale_icp_image_plane_driftlock_then_mhr_{refit_config.mode}_refit",
        "alignment_source_sampling": "8000_deterministic_area_weighted_triangle_surface_points",
        "mean_depth_align_translation_m": float(np.mean([np.linalg.norm(item.translation_cam) for item in alignments])) if alignments else 0.0,
        "mean_depth_align_scale": float(np.mean([item.scale for item in alignments])) if alignments else 1.0,
        "fallback_count": int(sum(bool(item.fallback) for item in alignments)),
        "sam3d_human_prompt": _sam3d_human_prompt_metadata(human_prompt_mode),
        **mhr_init_parameter_frame_metadata(),
    }
    if refit_cache_file is not None:
        metadata["sam3d_refit_cache"] = str(Path(refit_cache_file))
    if sam3d_cache_file is not None:
        metadata["sam3d_direct_cache"] = str(Path(sam3d_cache_file))
    refit_metadata = build_mhr_refit_metadata(refit_config, refit_errors, enabled=True)
    if refit_metadata is None:
        raise RuntimeError("MHR refit metadata was not generated")
    metadata["mhr_refit"] = refit_metadata
    if refit_config.mode == MHR_REFIT_MODE_FULL:
        metadata["mhr_full_refit"] = refit_metadata
    result = {"body_model": "mhr", "frames": names, "kids": [camera_id], "metadata": metadata}
    result.update({key: np.asarray(params_np[key], dtype=np.float32) for key in MHR_PARAM_DIMS})
    result.update({key: np.asarray(refitted[key], dtype=np.float32) for key in ("mhr_joints", "mhr_keypoints")})
    result["metadata"]["mhr_vertex_storage"] = "parameters_only"
    assert_mhr_schema(result)
    save_mhr_result(out_file, result)
    stats_file = out_file.with_suffix(".stats.pkl")
    stats = _alignment_arrays(alignments)
    stats["frames"] = names
    for key, values in refit_errors.items():
        dtype = np.bool_ if key.endswith("_mask") else np.float32
        stats[f"mhr_refit_{key}"] = np.concatenate(values).astype(dtype)
    with stats_file.open("wb") as f:
        pickle.dump(stats, f)
    print(f"saved native MHR init to {out_file}")
    print(f"saved alignment stats to {stats_file}")
    return out_file


def run_cached_sam3d_mhr_refit(export_seq: str | Path, out_file: str | Path, refit_cache_file: str | Path, *, camera_id: int = 0, depth_root: str | Path | None = None, sam3d_ckpt: str | Path = DEFAULT_SAM3D_CKPT, mhr_path: str | Path = DEFAULT_MHR_PATH, sam3d_cache_file: str | Path | None = None, start: int = 0, end: int | None = None, depth_source: str = "metric_depth", refit_config: MHRRefitConfig = DEFAULT_REFIT_CONFIG, human_prompt_mode: str = DEFAULT_SAM3D_HUMAN_PROMPT_MODE, redo: bool = False) -> Path:
    export_seq = Path(export_seq)
    out_file = Path(out_file)
    edex = load_edex(export_seq)
    camera_calibration(edex, camera_id)
    names_all = frame_names(export_seq, camera_id)
    stop = len(names_all) if end is None else min(end, len(names_all))
    names = names_all[start:stop]
    if not names:
        raise ValueError(f"No frames selected from {export_seq} for range [{start}, {stop})")
    expected_depth_metadata = _depth_output_metadata(export_seq, depth_root, depth_source)
    if out_file.is_file() and not redo:
        _validate_mhr_output(out_file, names, camera_id, refit_config, expected_depth_metadata, human_prompt_mode)
        print(f"validated existing MHR output at {out_file}")
        return out_file
    expected_metadata = _refit_cache_metadata(export_seq, depth_root, depth_source, sam3d_ckpt, mhr_path, camera_id, names, start, stop, human_prompt_mode)
    cache = load_mhr_refit_cache(refit_cache_file, expected_frames=names, expected_camera_id=camera_id, expected_metadata=expected_metadata)
    print(f"Loading cached pre-refit MHR inputs from {refit_cache_file}")
    mhr_layer = MHRLayer.from_mhr_assets(mhr_assets_root=SAM3D_ASSETS_ROOT, checkpoint_path=sam3d_ckpt, mhr_model_path=mhr_path, device="cuda")
    target_vertices_world = reconstruct_mhr_refit_target_vertices(mhr_layer, cache, batch_size=refit_config.optimization_batch_size)
    params_np, refitted = _refit_aligned_mhr_sequence(mhr_layer, cache.params, target_vertices_world, refit_config)
    return _save_refitted_sequence(export_seq, out_file, names, camera_id, depth_root, depth_source, refit_config, _alignments_from_arrays(cache.alignment), params_np, refitted, human_prompt_mode, refit_cache_file, sam3d_cache_file)


def run_sam3d_mhr_export(
    export_seq: str | Path,
    out_file: str | Path,
    *,
    camera_id: int = 0,
    depth_root: str | Path | None = None,
    sam3d_ckpt: str | Path = DEFAULT_SAM3D_CKPT,
    mhr_path: str | Path = DEFAULT_MHR_PATH,
    chunk_size: int = 16,
    start: int = 0,
    end: int | None = None,
    align_to_gt_depth: bool = True,
    depth_source: str = "metric_depth",
    refit_config: MHRRefitConfig = DEFAULT_REFIT_CONFIG,
    refit_cache_file: str | Path | None = None,
    sam3d_cache_file: str | Path | None = None,
    redo_refit_cache: bool = False,
    redo_sam3d_cache: bool = False,
    alignment_workers: int = 1,
    prepare_refit_cache_only: bool = False,
    sam3d_direct_only: bool = False,
    refit_preparation_device: str = "cuda",
    require_sam3d_cache: bool = False,
    human_prompt_mode: str = DEFAULT_SAM3D_HUMAN_PROMPT_MODE,
    redo: bool = False,
) -> Path:
    _sam3d_human_prompt_metadata(human_prompt_mode)
    if alignment_workers <= 0:
        raise ValueError(f"alignment_workers must be positive, got {alignment_workers}")
    if refit_preparation_device not in {"cpu", "cuda"}:
        raise ValueError(f"refit_preparation_device must be cpu or cuda, got {refit_preparation_device!r}")
    if prepare_refit_cache_only and refit_cache_file is None:
        raise ValueError("prepare_refit_cache_only requires refit_cache_file")
    if prepare_refit_cache_only and not align_to_gt_depth:
        raise ValueError("prepare_refit_cache_only requires metric-depth alignment")
    if sam3d_direct_only and prepare_refit_cache_only:
        raise ValueError("sam3d_direct_only and prepare_refit_cache_only are mutually exclusive")
    if require_sam3d_cache and redo_sam3d_cache:
        raise ValueError("require_sam3d_cache and redo_sam3d_cache are mutually exclusive")
    export_seq = Path(export_seq)
    out_file = Path(out_file)
    redo_refit_cache = redo_refit_cache or redo_sam3d_cache
    redo = redo or redo_refit_cache or redo_sam3d_cache
    edex = load_edex(export_seq)
    K, w2c = camera_calibration(edex, camera_id)
    c2w = np.linalg.inv(w2c).astype(np.float32)
    names_all = frame_names(export_seq, camera_id)
    stop = len(names_all) if end is None else min(end, len(names_all))
    names = names_all[start:stop]
    if not names:
        raise ValueError(f"No frames selected from {export_seq} for range [{start}, {stop})")
    expected_depth_metadata = _depth_output_metadata(export_seq, depth_root, depth_source) if align_to_gt_depth else None
    if out_file.is_file() and not redo and not prepare_refit_cache_only and not sam3d_direct_only:
        _validate_mhr_output(out_file, names, camera_id, refit_config, expected_depth_metadata, human_prompt_mode)
        print(f"validated existing MHR output at {out_file}")
        return out_file
    sam3d_cache_file = Path(sam3d_cache_file) if sam3d_cache_file is not None else default_mhr_sam3d_cache_path(out_file)
    refit_cache_file = Path(refit_cache_file) if refit_cache_file is not None else None
    if refit_cache_file is not None and refit_cache_file.is_file() and not redo_refit_cache and not sam3d_direct_only:
        if prepare_refit_cache_only:
            expected_metadata = _refit_cache_metadata(export_seq, depth_root, depth_source, sam3d_ckpt, mhr_path, camera_id, names, start, stop, human_prompt_mode)
            load_mhr_refit_cache(refit_cache_file, expected_frames=names, expected_camera_id=camera_id, expected_metadata=expected_metadata)
            print(f"Validated existing SAM3D MHR refit cache at {refit_cache_file}")
            return refit_cache_file
        return run_cached_sam3d_mhr_refit(export_seq, out_file, refit_cache_file, camera_id=camera_id, depth_root=depth_root, sam3d_ckpt=sam3d_ckpt, mhr_path=mhr_path, sam3d_cache_file=sam3d_cache_file, start=start, end=end, depth_source=depth_source, refit_config=refit_config, human_prompt_mode=human_prompt_mode, redo=True)
    sam3d_metadata = _sam3d_direct_cache_metadata(export_seq, camera_id, names, sam3d_ckpt, mhr_path, start, stop, human_prompt_mode)
    direct_predictions: dict[str, np.ndarray]
    if sam3d_cache_file.is_file() and not redo_sam3d_cache:
        direct_cache = load_mhr_sam3d_direct_cache(sam3d_cache_file, expected_frames=names, expected_camera_id=camera_id, expected_metadata=sam3d_metadata)
        direct_predictions = direct_cache.predictions
        print(f"Loading cached direct SAM 3D Body output from {sam3d_cache_file}")
    else:
        if require_sam3d_cache:
            raise FileNotFoundError(f"Required direct SAM 3D Body cache is missing: {sam3d_cache_file}")
        print("Loading SAM3D-body model...")
        device = torch.device("cuda")
        torch_hub_load = torch.hub.load

        def torch_hub_load_with_explicit_ref(repo_or_dir: str, *args: Any, **kwargs: Any) -> Any:
            return torch_hub_load(_explicit_torch_hub_repo(repo_or_dir), *args, **kwargs)

        # Torch Hub resolves unspecified default branches over GitHub before consulting its cache.
        torch.hub.load = torch_hub_load_with_explicit_ref
        try:
            model, model_cfg = load_sam_3d_body(str(sam3d_ckpt), device=device, mhr_path=str(mhr_path))
        finally:
            torch.hub.load = torch_hub_load
        model.eval()
        transform = Compose([GetBBoxCenterScale(), TopdownAffine(input_size=model_cfg.MODEL.IMAGE_SIZE, use_udp=False), VisionTransformWrapper(ToTensor())])
        direct_accum: dict[str, list[np.ndarray]] = {key: [] for key in (*MHR_SAM3D_DIRECT_PARAM_DIMS, *MHR_SAM3D_DIRECT_RUNTIME_GEOMETRY_KEYS)}
        for base in tqdm(range(0, len(names), chunk_size), desc="SAM3D native MHR"):
            chunk_names = names[base : base + chunk_size]
            images, bboxes, masks = [], [], []
            for frame_name in chunk_names:
                image = read_rgb(export_seq, camera_id, frame_name)
                mask = read_mask(export_seq, "human", camera_id, frame_name)
                images.append(image)
                bboxes.append(_bbox_from_mask(mask, image.shape[:2]))
                masks.append(mask)
            batch = recursive_to(_prepare_batch(images, bboxes, transform, K, masks=masks, human_prompt_mode=human_prompt_mode), device)
            with torch.no_grad():
                model._initialize_batch(batch)
                pose_output = model.run_inference(images[0], batch, inference_type="body", transform_hand=None, thresh_wrist_angle=1.4)
            out = recursive_to(recursive_to(pose_output["mhr"], "cpu"), "numpy")
            direct_chunk = _canonical_sam3d_direct_predictions(out, len(chunk_names))
            for key, value in direct_chunk.items():
                direct_accum[key].append(value)
        direct_predictions = {key: np.concatenate(values, axis=0).astype(np.float32) for key, values in direct_accum.items()}
        direct_cache = MHRSAM3DDirectCache(frames=tuple(names), camera_id=camera_id, predictions={key: direct_predictions[key] for key in MHR_SAM3D_DIRECT_PARAM_DIMS}, metadata={**sam3d_metadata, "inference_chunk_size": int(chunk_size), "vertex_storage": "parameters_only"})
        save_mhr_sam3d_direct_cache(sam3d_cache_file, direct_cache, overwrite=redo_sam3d_cache)
        print(f"saved direct SAM 3D Body output cache to {sam3d_cache_file}")
        del model
        torch.cuda.empty_cache()

    if sam3d_direct_only:
        return sam3d_cache_file

    device = torch.device(refit_preparation_device)
    mhr_layer = MHRLayer.from_mhr_assets(mhr_assets_root=SAM3D_ASSETS_ROOT, checkpoint_path=sam3d_ckpt, mhr_model_path=mhr_path, device=refit_preparation_device)
    if not all(key in direct_predictions for key in MHR_SAM3D_DIRECT_RUNTIME_GEOMETRY_KEYS):
        direct_predictions = {**direct_predictions, **decode_mhr_sam3d_direct_geometry(mhr_layer, direct_predictions, batch_size=chunk_size)}
    mhr_faces = None
    if align_to_gt_depth:
        mhr_faces_raw = mhr_layer.mesh_faces(device=device)
        mhr_faces = mhr_faces_raw.detach().cpu().numpy() if torch.is_tensor(mhr_faces_raw) else np.asarray(mhr_faces_raw)
        mhr_faces = np.asarray(mhr_faces, dtype=np.int64)

    accum: dict[str, list[np.ndarray]] = {
        "mhr_global_rot6d": [],
        "mhr_trans": [],
        "mhr_body_pose_cont": [],
        "mhr_hand": [],
        "mhr_shape": [],
        "mhr_scale": [],
        "mhr_face": [],
        "mhr_joints": [],
        "mhr_keypoints": [],
    }
    target_vertex_chunks: list[np.ndarray] = []
    alignments: list[DepthAlignment] = []
    alignment_executor = ThreadPoolExecutor(max_workers=alignment_workers) if alignment_workers > 1 else None
    try:
        for base in tqdm(range(0, len(names), chunk_size), desc="SAM3D MHR postprocess"):
            chunk_names = names[base : base + chunk_size]
            direct_chunk = {key: value[base : base + len(chunk_names)] for key, value in direct_predictions.items()}
            pred_cam_t = direct_chunk["pred_cam_t"]
            vertices_cam = direct_chunk["pred_vertices"] + pred_cam_t[:, None, :]
            joints_cam = direct_chunk["pred_joint_coords"] + pred_cam_t[:, None, :]
            keypoints_cam = direct_chunk["pred_keypoints_3d"] + pred_cam_t[:, None, :]
            if joints_cam.shape[1] <= MHR_ROOT_JOINT_INDEX:
                raise ValueError(f"SAM3D MHR pred_joint_coords needs root joint {MHR_ROOT_JOINT_INDEX}, got {joints_cam.shape}")

            vertices_world = _camera_to_world_geometry(vertices_cam, c2w)
            joints_world = _camera_to_world_geometry(joints_cam, c2w)
            keypoints_world = _camera_to_world_geometry(keypoints_cam, c2w)
            trans_world = mhr_translation_between_frames(pred_cam_t, joints_cam[:, MHR_ROOT_JOINT_INDEX], c2w).astype(np.float32)

            if align_to_gt_depth:
                alignment_inputs = []
                for local_idx, frame_name in enumerate(chunk_names):
                    depth = filter_depth_like_smpl(read_depth_m(export_seq, camera_id, frame_name, depth_root=depth_root), device=refit_preparation_device)
                    mask = read_mask(export_seq, "human", camera_id, frame_name)
                    alignment_inputs.append((vertices_world[local_idx], depth, mask, base + local_idx))

                def align_one(values: tuple[np.ndarray, np.ndarray, np.ndarray, int]) -> DepthAlignment:
                    vertices, depth, mask, sample_seed = values
                    return _align_frame_to_depth(vertices, mhr_faces, w2c, K, depth, mask, sample_seed=sample_seed, depth_is_filtered=True)

                chunk_alignments = list(alignment_executor.map(align_one, alignment_inputs)) if alignment_executor is not None else [align_one(values) for values in alignment_inputs]
                for local_idx, alignment in enumerate(chunk_alignments):
                    vertices_world[local_idx] = _apply_camera_alignment(vertices_world[local_idx], w2c, c2w, alignment.matrix_cam)
                    joints_world[local_idx] = _apply_camera_alignment(joints_world[local_idx], w2c, c2w, alignment.matrix_cam)
                    keypoints_world[local_idx] = _apply_camera_alignment(keypoints_world[local_idx], w2c, c2w, alignment.matrix_cam)
                    trans_world[local_idx] = _apply_camera_alignment(trans_world[local_idx], w2c, c2w, alignment.matrix_cam)
                    alignments.append(alignment)

            params_np = {
                "mhr_global_rot6d": _rot6d_world(direct_chunk["global_rot"], c2w),
                "mhr_trans": trans_world,
                "mhr_body_pose_cont": compact_model_params_to_cont_body_np(direct_chunk["body_pose_params"]),
                "mhr_hand": direct_chunk["hand_pose_params"],
                "mhr_shape": direct_chunk["shape_params"],
                "mhr_scale": direct_chunk["scale_params"],
                "mhr_face": direct_chunk["expr_params"],
            }
            for key in MHR_PARAM_DIMS:
                accum[key].append(params_np[key])
            if align_to_gt_depth:
                target_vertex_chunks.append(vertices_world)
            if not align_to_gt_depth:
                accum["mhr_joints"].append(joints_world)
                accum["mhr_keypoints"].append(keypoints_world)
    finally:
        if alignment_executor is not None:
            alignment_executor.shutdown(wait=True)

    if align_to_gt_depth:
        if mhr_layer is None:
            raise RuntimeError("MHR full-refit layer was not initialized")
        params_np = {key: np.concatenate(accum[key], axis=0).astype(np.float32) for key in MHR_PARAM_DIMS}
        target_vertices_world = np.concatenate(target_vertex_chunks, axis=0).astype(np.float32)
        if refit_cache_file is not None:
            cache = MHRRefitCache(frames=tuple(names), camera_id=camera_id, params=params_np, alignment=_alignment_arrays(alignments), metadata={**_refit_cache_metadata(export_seq, depth_root, depth_source, sam3d_ckpt, mhr_path, camera_id, names, start, stop, human_prompt_mode), "vertex_storage": "parameters_plus_alignment"})
            save_mhr_refit_cache(refit_cache_file, cache, overwrite=redo_refit_cache)
            print(f"saved SAM3D MHR refit cache to {refit_cache_file}")
            if prepare_refit_cache_only:
                return refit_cache_file
        params_np, refitted = _refit_aligned_mhr_sequence(mhr_layer, params_np, target_vertices_world, refit_config)
        return _save_refitted_sequence(export_seq, out_file, names, camera_id, depth_root, depth_source, refit_config, alignments, params_np, refitted, human_prompt_mode, refit_cache_file, sam3d_cache_file)

    metadata = {
        "source": f"sam3d_body_rgb_mhr_{refit_config.mode}_refit_aligned_to_metric_depth" if align_to_gt_depth else "sam3d_body_rgb",
        "camera_id": camera_id,
        "depth_root": str(depth_root) if depth_root is not None else str(export_seq / "depth"),
        "depth_source": str(depth_source),
        "alignment": f"median_z_translation_icp_scale_icp_image_plane_driftlock_then_mhr_{refit_config.mode}_refit" if align_to_gt_depth else "none",
        "alignment_source_sampling": "8000_deterministic_area_weighted_triangle_surface_points" if align_to_gt_depth else "none",
        "mean_depth_align_translation_m": float(np.mean([np.linalg.norm(item.translation_cam) for item in alignments])) if alignments else 0.0,
        "mean_depth_align_scale": float(np.mean([item.scale for item in alignments])) if alignments else 1.0,
        "fallback_count": int(sum(bool(item.fallback) for item in alignments)),
        "sam3d_direct_cache": str(sam3d_cache_file),
        "sam3d_human_prompt": _sam3d_human_prompt_metadata(human_prompt_mode),
        "mhr_vertex_storage": "parameters_only",
        **mhr_init_parameter_frame_metadata(),
    }
    refit_metadata = build_mhr_refit_metadata(refit_config, {}, enabled=align_to_gt_depth)
    if refit_metadata is not None:
        metadata["mhr_refit"] = refit_metadata
        if refit_config.mode == MHR_REFIT_MODE_FULL:
            metadata["mhr_full_refit"] = refit_metadata
    result = {"body_model": "mhr", "frames": names, "kids": [camera_id], "metadata": metadata}
    result.update({key: np.concatenate(value, axis=0).astype(np.float32) for key, value in accum.items()})
    assert_mhr_schema(result)
    save_mhr_result(out_file, result)

    stats_file = out_file.with_suffix(".stats.pkl")
    stats = {"matrix_cam": np.asarray([item.matrix_cam for item in alignments], dtype=np.float32), "scales": np.asarray([item.scale for item in alignments], dtype=np.float32), "translations_cam": np.asarray([item.translation_cam for item in alignments], dtype=np.float32), "image_corrections_cam": np.asarray([item.image_correction_cam for item in alignments], dtype=np.float32), "image_corrections_px": np.asarray([item.image_correction_px for item in alignments], dtype=np.float32), "source_points": np.asarray([item.source_points for item in alignments], dtype=np.int32), "target_points": np.asarray([item.target_points for item in alignments], dtype=np.int32), "fallback": [item.fallback for item in alignments], "frames": names}
    with stats_file.open("wb") as f:
        pickle.dump(stats, f)
    print(f"saved native MHR init to {out_file}")
    print(f"saved alignment stats to {stats_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM3D-body on flat MHR export RGB and save native MHR init.")
    parser.add_argument("export_seq")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--depth-root", default=None, help="Depth root to align to. Defaults to export depth.")
    parser.add_argument("--sam3d-ckpt", default=str(DEFAULT_SAM3D_CKPT))
    parser.add_argument("--mhr-path", default=str(DEFAULT_MHR_PATH))
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--depth-source", default="metric_depth")
    parser.add_argument("--mhr-refit-mode", choices=MHR_REFIT_MODES, default=MHR_REFIT_MODE_FULL)
    parser.add_argument("--refit-cache", default=None, help="H5 cache for original SAM3D MHR parameters and depth-aligned target vertices.")
    parser.add_argument("--refit-only", action="store_true", help="Skip SAM3D inference and refit exclusively from --refit-cache.")
    parser.add_argument("--redo-refit-cache", action="store_true", help="Recompute camera conversion and metric-depth alignment from the direct SAM 3D Body cache, then transactionally replace --refit-cache.")
    parser.add_argument("--sam3d-cache", default=None, help="H5 cache for immutable direct SAM 3D Body network outputs. Defaults beside --out-file.")
    parser.add_argument("--redo-sam3d-cache", action="store_true", help="Rerun SAM 3D Body and transactionally replace its direct-output cache. Ordinary --redo reuses the cache.")
    parser.add_argument("--alignment-workers", type=int, default=1, help="CPU workers for deterministic per-frame depth alignment after GPU depth filtering.")
    parser.add_argument("--refit-batch-size", type=int, default=512)
    parser.add_argument("--prepare-refit-cache-only", action="store_true", help="Run through deterministic depth alignment and publish the refit cache without optimizing MHR parameters.")
    parser.add_argument("--sam3d-direct-only", action="store_true", help="Publish or validate the immutable direct SAM 3D Body cache, then stop before depth alignment and MHR refitting.")
    parser.add_argument("--refit-preparation-device", choices=("cpu", "cuda"), default="cuda", help="Device used for canonical MHR topology and depth filtering while preparing a refit cache.")
    parser.add_argument("--require-sam3d-cache", action="store_true", help="Fail instead of running SAM 3D Body when the direct cache is absent.")
    parser.add_argument("--human-prompt-mode", choices=SAM3D_HUMAN_PROMPT_MODES, default=DEFAULT_SAM3D_HUMAN_PROMPT_MODE, help="Human prompt passed to SAM 3D Body. The production default uses the SAM2-derived bbox and mask.")
    parser.add_argument("--no-align-to-gt-depth", action="store_true")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    if args.refit_only and args.refit_cache is None:
        parser.error("--refit-only requires --refit-cache")
    if args.refit_only and args.no_align_to_gt_depth:
        parser.error("--refit-only requires metric-depth alignment")
    if args.redo_refit_cache and args.refit_cache is None:
        parser.error("--redo-refit-cache requires --refit-cache")
    if args.refit_only and (args.redo_refit_cache or args.redo_sam3d_cache):
        parser.error("--refit-only cannot be combined with cache regeneration")
    if args.refit_only and args.prepare_refit_cache_only:
        parser.error("--refit-only and --prepare-refit-cache-only are mutually exclusive")
    if args.refit_only and args.sam3d_direct_only:
        parser.error("--refit-only and --sam3d-direct-only are mutually exclusive")
    if args.prepare_refit_cache_only and args.sam3d_direct_only:
        parser.error("--prepare-refit-cache-only and --sam3d-direct-only are mutually exclusive")
    if args.prepare_refit_cache_only and args.refit_cache is None:
        parser.error("--prepare-refit-cache-only requires --refit-cache")
    if args.prepare_refit_cache_only and args.no_align_to_gt_depth:
        parser.error("--prepare-refit-cache-only requires metric-depth alignment")
    if args.refit_cache is not None and args.no_align_to_gt_depth:
        parser.error("--refit-cache requires metric-depth alignment")
    if args.require_sam3d_cache and args.redo_sam3d_cache:
        parser.error("--require-sam3d-cache and --redo-sam3d-cache are mutually exclusive")
    common = {"camera_id": args.camera_id, "depth_root": args.depth_root, "sam3d_ckpt": args.sam3d_ckpt, "mhr_path": args.mhr_path, "start": args.start, "end": args.end, "depth_source": args.depth_source, "refit_config": replace(DEFAULT_REFIT_CONFIG, mode=args.mhr_refit_mode, optimization_batch_size=args.refit_batch_size), "human_prompt_mode": args.human_prompt_mode, "redo": args.redo}
    with PipelineTimer("sam3d_mhr_initialization") as profiler:
        profiler.update_metadata({"chunk_size": int(args.chunk_size), "align_to_depth": not bool(args.no_align_to_gt_depth), "refit_only": bool(args.refit_only), "human_prompt_mode": args.human_prompt_mode})
        if args.refit_only:
            run_cached_sam3d_mhr_refit(args.export_seq, args.out_file, args.refit_cache, sam3d_cache_file=args.sam3d_cache, **common)
        else:
            run_sam3d_mhr_export(args.export_seq, args.out_file, chunk_size=args.chunk_size, align_to_gt_depth=not args.no_align_to_gt_depth, refit_cache_file=args.refit_cache, sam3d_cache_file=args.sam3d_cache, redo_refit_cache=args.redo_refit_cache, redo_sam3d_cache=args.redo_sam3d_cache, alignment_workers=args.alignment_workers, prepare_refit_cache_only=args.prepare_refit_cache_only, sam3d_direct_only=args.sam3d_direct_only, refit_preparation_device=args.refit_preparation_device, require_sam3d_cache=args.require_sam3d_cache, **common)


if __name__ == "__main__":
    main()
