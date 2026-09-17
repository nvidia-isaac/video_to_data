from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio
import nvdiffrast.torch as dr
import numpy as np
import torch
import trimesh
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Utils
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, cluster_poses
from lib_mhr import ObjectSymmetry, load_optional_output_object_symmetry, load_optional_output_symmetry_tfs
from lib_mhr.object_texture import concatenate_object_mesh_parts, load_object_mesh_parts
from prep.foundationpose_pose_selection import PoseSource, ReacquisitionThresholds, RejectionReason, TrackingState, adaptive_translation_threshold_m, advance_pending_pose, filter_ranked_candidates, filter_reacquisition_candidates, filter_retry_candidates, measure_visible_silhouettes, minimum_visible_support_pixels, select_gt_rotation_oracle, temporal_rejection_reasons, validate_rigid_transform, validate_rigid_transforms
from prep.mhr_foundationpose_diagnostics import FOUNDATIONPOSE_REACQUISITION_LATENCY_UNKNOWN, FOUNDATIONPOSE_SELECTION_DTYPES, FOUNDATIONPOSE_SELECTION_FIELDS, FOUNDATIONPOSE_SELECTION_REVISION, validate_foundationpose_selection_arrays, validate_foundationpose_selection_metadata
from prep.mhr_geometry_crop import load_object_pose_valid_mask_for_frames
from tools.pipeline_timing import PipelineTimer
from prep.mhr_gt_object_visibility import GTObjectVisibility, GT_OBJECT_VISIBILITY_DEFINITION, GT_OBJECT_VISIBILITY_REVISION, measure_gt_object_visibility, validate_gt_object_visibility_arrays, validate_gt_object_visibility_metadata
from prep.mhr_preprocess_manifest import OBJECT_POSE_VALIDITY_EXPRESSION
from prep.mhr_export_utils import (
    camera_calibration,
    frame_names,
    load_edex,
    mask_input_identity,
    path_identity,
    read_depth_m,
    read_mask,
    read_rgb,
    resolve_object_mesh_path,
)


FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_EVERY_FRAME = "register-every-frame"
FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK = "register-first-then-track"
FOUNDATIONPOSE_SEQUENCE_MODES = frozenset((FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_EVERY_FRAME, FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK))
DEFAULT_FOUNDATIONPOSE_SEQUENCE_MODE = FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_EVERY_FRAME
DEFAULT_WILD_INFERENCE_FOUNDATIONPOSE_SEQUENCE_MODE = FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK


def _open_ffmpeg_writer(path: str | Path):
    os.environ["IMAGEIO_FFMPEG_NO_PREVENT_SIGINT"] = "1"
    return imageio.get_writer(str(path), "ffmpeg", fps=6)


def _atomic_pickle_dump(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_foundationpose_debug_dir(out_file: str | Path, debug_dir: str | Path | None) -> Path:
    return Path(out_file).parent / "debug" if debug_dir is None else Path(debug_dir)


def load_export_object_mesh(export_seq: str | Path) -> trimesh.Trimesh:
    path = resolve_object_mesh_path(export_seq)
    mesh = concatenate_object_mesh_parts(load_object_mesh_parts(path))
    try:
        mesh.visual = mesh.visual.to_color()
    except Exception:
        colors = np.tile(np.array([[180, 180, 180, 255]], dtype=np.uint8), (len(mesh.vertices), 1))
        mesh.visual = trimesh.visual.color.ColorVisuals(vertex_colors=colors)
    return mesh


def _centered_symmetry_tfs(mesh: trimesh.Trimesh, symmetry_tfs: np.ndarray) -> np.ndarray:
    symmetry_tfs = validate_rigid_transforms(symmetry_tfs, "object symmetry transforms")
    center = (np.asarray(mesh.vertices).max(axis=0) + np.asarray(mesh.vertices).min(axis=0)) * 0.5
    to_centered = np.eye(4, dtype=np.float32)
    to_centered[:3, 3] = -center.astype(np.float32)
    centered = to_centered[None] @ symmetry_tfs @ np.linalg.inv(to_centered)[None]
    return validate_rigid_transforms(centered.astype(np.float32), "centered object symmetry transforms")


def _centered_symmetry_center(mesh: trimesh.Trimesh, symmetry_center: np.ndarray) -> np.ndarray:
    mesh_center = (np.asarray(mesh.vertices).max(axis=0) + np.asarray(mesh.vertices).min(axis=0)) * 0.5
    centered = np.asarray(symmetry_center, dtype=np.float32) - mesh_center.astype(np.float32)
    if centered.shape != (3,) or not np.isfinite(centered).all():
        raise ValueError(f"Centered object symmetry center must be finite with shape [3], got {centered.shape}")
    return centered


def _pose_estimator(mesh: trimesh.Trimesh, debug_dir: str | Path) -> FoundationPose:
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    return FoundationPose(
        model_pts=np.asarray(mesh.vertices),
        model_normals=np.asarray(mesh.vertex_normals),
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=0,
        glctx=glctx,
    )


def _has_usable_initialization_depth(depth: np.ndarray, mask: np.ndarray, min_valid_pixels: int = 4) -> bool:
    filtered_depth = Utils.erode_depth(depth, radius=2, device="cuda")
    filtered_depth = Utils.bilateral_filter_depth(filtered_depth, radius=2, device="cuda")
    valid = mask.astype(bool) & (filtered_depth >= 0.001)
    return int(valid.sum()) >= int(min_valid_pixels)


def _prepare_foundationpose_inputs(rgb: np.ndarray, depth: np.ndarray, human_mask: np.ndarray, object_mask: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    human_mask = np.asarray(human_mask)
    object_mask = np.asarray(object_mask)
    K = np.asarray(K)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"FoundationPose RGB must have shape [H, W, 3], got {rgb.shape}")
    if depth.ndim != 2:
        raise ValueError(f"FoundationPose depth must have shape [H, W], got {depth.shape}")
    native_shape = rgb.shape[:2]
    if human_mask.shape != native_shape or object_mask.shape != native_shape:
        raise ValueError(f"FoundationPose masks must match native RGB shape {native_shape}, got human={human_mask.shape}, object={object_mask.shape}")
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"FoundationPose camera intrinsics must be finite [3, 3], got {K.shape}")
    if depth.shape[0] > native_shape[0] or depth.shape[1] > native_shape[1]:
        raise ValueError(f"FoundationPose depth {depth.shape} cannot be upsampled to smaller RGB shape {native_shape}")
    if depth.shape != native_shape:
        depth = cv2.resize(depth, (native_shape[1], native_shape[0]), interpolation=cv2.INTER_NEAREST)
    return rgb, depth, human_mask.astype(bool, copy=False), object_mask.astype(bool, copy=False), K.copy()


def _tensor_to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


FOUNDATIONPOSE_OPTIMIZATIONS = frozenset({"frame-cache", "tensor-residency", "prefetch"})
DEFAULT_FOUNDATIONPOSE_OPTIMIZATIONS = tuple(sorted(FOUNDATIONPOSE_OPTIMIZATIONS))
FOUNDATIONPOSE_OPTIMIZATION_REVISION = "foundationpose-production-opt-v4"


def _estimator_candidates(estimator: FoundationPose, *, cluster_angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if estimator.poses is None:
        raise RuntimeError("FoundationPose registration did not publish candidate poses")
    all_centered = validate_rigid_transforms(_tensor_to_numpy(estimator.poses), "FoundationPose registration candidates")
    symmetry_tfs = validate_rigid_transforms(_tensor_to_numpy(estimator.symmetry_tfs), "FoundationPose centered symmetry transforms")
    clustered_values, clustered_indices = cluster_poses(cluster_angle_deg, 0.1, all_centered, symmetry_tfs, return_indices=True)
    clustered = np.stack(clustered_values).astype(np.float32)
    clustered = validate_rigid_transforms(clustered, "clustered FoundationPose candidates")
    transform_to_centered = validate_rigid_transform(_tensor_to_numpy(estimator.get_tf_to_centered_mesh()), "FoundationPose transform_to_centered_mesh")
    return all_centered, clustered, transform_to_centered, np.asarray(clustered_indices, dtype=np.int64)


def _cluster_candidates(all_centered: np.ndarray, cluster_angle_deg: float, symmetry_tfs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    symmetry_tfs = np.eye(4, dtype=np.float32)[None] if symmetry_tfs is None else validate_rigid_transforms(symmetry_tfs, "cluster symmetry transforms")
    clustered_values, clustered_indices = cluster_poses(cluster_angle_deg, 0.1, all_centered, symmetry_tfs, return_indices=True)
    clustered = validate_rigid_transforms(np.stack(clustered_values).astype(np.float32), "cached clustered FoundationPose candidates")
    return clustered, np.asarray(clustered_indices, dtype=np.int64)


def _estimator_source_ids(estimator: FoundationPose, clustered_indices: np.ndarray) -> np.ndarray:
    if estimator.pose_source_ids is None:
        raise RuntimeError("FoundationPose did not retain candidate source identifiers")
    source_ids = _tensor_to_numpy(estimator.pose_source_ids).astype(np.int64, copy=False)
    clustered_indices = np.asarray(clustered_indices, dtype=np.int64)
    if source_ids.shape != (len(estimator.poses),):
        raise ValueError(f"FoundationPose candidate source identifiers have invalid shape {source_ids.shape}")
    return source_ids[clustered_indices]


def _cluster_candidates_with_source_ids(all_candidates: np.ndarray, source_ids: np.ndarray, symmetry_tfs: np.ndarray, *, cluster_angle_deg: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    all_candidates = validate_rigid_transforms(all_candidates, "raw FoundationPose candidates")
    source_ids = np.asarray(source_ids, dtype=np.int64)
    if source_ids.shape != (len(all_candidates),):
        raise ValueError(f"candidate source identifiers expected {(len(all_candidates),)}, got {source_ids.shape}")
    clustered, clustered_indices = _cluster_candidates(all_candidates, cluster_angle_deg, symmetry_tfs)
    return clustered, source_ids[clustered_indices]


def _scaled_image_grid(K: np.ndarray, image_shape: tuple[int, int], max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image_shape
    scale = min(1.0, float(max_side) / max(height, width))
    scaled_shape = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    scaled_K = np.asarray(K, dtype=np.float32).copy()
    scaled_K[0] *= scaled_shape[1] / width
    scaled_K[1] *= scaled_shape[0] / height
    return scaled_K, scaled_shape


@torch.no_grad()
def _render_candidate_masks(estimator: FoundationPose, candidate_poses_centered: np.ndarray, K: np.ndarray, image_shape: tuple[int, int], *, chunk_size: int = 8) -> np.ndarray:
    candidates = validate_rigid_transforms(candidate_poses_centered, "rendered FoundationPose candidates")
    height, width = (int(value) for value in image_shape)
    if height <= 0 or width <= 0 or int(chunk_size) <= 0:
        raise ValueError(f"invalid candidate render shape/chunk: {image_shape}/{chunk_size}")
    device = estimator.mesh_tensors["pos"].device
    projection = torch.as_tensor(Utils.projection_matrix_from_intrinsics(K, height=height, width=width, znear=0.001, zfar=100.0), dtype=torch.float32, device=device)
    cv_to_gl = torch.as_tensor(Utils.glcam_in_cvcam, dtype=torch.float32, device=device)
    position_homogeneous = Utils.to_homo_torch(estimator.mesh_tensors["pos"])
    masks: list[np.ndarray] = []
    for begin in range(0, len(candidates), int(chunk_size)):
        pose = torch.as_tensor(candidates[begin:begin + int(chunk_size)], dtype=torch.float32, device=device)
        position_clip = ((projection[None] @ cv_to_gl[None] @ pose)[:, None] @ position_homogeneous[None, ..., None])[..., 0]
        raster, _ = dr.rasterize(estimator.glctx, position_clip, estimator.mesh_tensors["faces"], resolution=(height, width))
        masks.append(_tensor_to_numpy(_raster_alpha_to_opencv_mask(raster)).astype(bool))
    return np.concatenate(masks, axis=0)


def _resize_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(np.asarray(mask, dtype=np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def _raster_alpha_to_opencv_mask(raster: torch.Tensor) -> torch.Tensor:
    if raster.ndim != 4 or raster.shape[-1] != 4:
        raise ValueError(f"nvdiffrast raster must have shape [N,H,W,4], got {tuple(raster.shape)}")
    return torch.flip(raster[..., 3] > 0, dims=[1])


def _screen_candidate_indices(estimator: FoundationPose, candidates: np.ndarray, K: np.ndarray, human_mask: np.ndarray, object_mask: np.ndarray, thresholds: ReacquisitionThresholds, *, max_side: int = 384) -> np.ndarray:
    screen_K, screen_shape = _scaled_image_grid(K, human_mask.shape, max_side)
    screen_masks = _render_candidate_masks(estimator, candidates, screen_K, screen_shape)
    screen_metrics = measure_visible_silhouettes(screen_masks, _resize_mask(human_mask, screen_shape), _resize_mask(object_mask, screen_shape))
    relaxed = replace(thresholds, support_floor_pixels=max(16, int(round(thresholds.support_floor_pixels * screen_shape[0] * screen_shape[1] / (human_mask.shape[0] * human_mask.shape[1])))), visibility_floor=0.10, iou_min=0.20, precision_min=0.50, recall_min=0.30, centroid_distance_max=0.25)
    keep = filter_reacquisition_candidates(screen_metrics, screen_shape, relaxed).keep_mask
    return np.flatnonzero(keep).astype(np.int64)


def _full_candidate_data(estimator: FoundationPose, candidates: np.ndarray, source_ids: np.ndarray, candidate_indices: np.ndarray, K: np.ndarray, human_mask: np.ndarray, object_mask: np.ndarray):
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if len(candidate_indices) == 0:
        return np.empty((0, 4, 4), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0, *human_mask.shape), dtype=bool), None
    selected_candidates = validate_rigid_transforms(candidates[candidate_indices], "full-image candidate shortlist")
    masks = _render_candidate_masks(estimator, selected_candidates, K, human_mask.shape)
    metrics = measure_visible_silhouettes(masks, human_mask, object_mask)
    return selected_candidates, np.asarray(source_ids, dtype=np.int64)[candidate_indices], masks, metrics


def _selected_metric_values(metrics, index: int | None) -> dict[str, float | int]:
    if metrics is None or index is None:
        return {"observed_visible_pixels": 0 if metrics is None else int(metrics.observed_visible_pixels), "rendered_visible_pixels": 0, "rendered_visibility": float("nan"), "iou": float("nan"), "precision": float("nan"), "recall": float("nan"), "centroid_distance": float("nan")}
    return {"observed_visible_pixels": int(metrics.observed_visible_pixels), "rendered_visible_pixels": int(metrics.rendered_visible_pixels[index]), "rendered_visibility": float(metrics.rendered_visibility[index]), "iou": float(metrics.iou[index]), "precision": float(metrics.precision[index]), "recall": float(metrics.recall[index]), "centroid_distance": float(metrics.centroid_distance[index])}


def _set_estimator_pose_last(estimator: FoundationPose, pose_centered: np.ndarray) -> None:
    pose_centered = validate_rigid_transform(pose_centered, "selected centered FoundationPose pose")
    if not torch.is_tensor(estimator.pose_last):
        raise TypeError(f"FoundationPose pose_last must be a tensor after registration, got {type(estimator.pose_last).__name__}")
    estimator.pose_last = torch.as_tensor(pose_centered, dtype=estimator.pose_last.dtype, device=estimator.pose_last.device)


@torch.no_grad()
def _render_gt_object_visibility_frame(estimator: FoundationPose, gt_pose_cam: np.ndarray, K: np.ndarray, human_mask: np.ndarray, *, annotation_valid: bool):
    gt_pose_cam = validate_rigid_transform(gt_pose_cam, "ground-truth object camera pose for visibility")
    transform_to_centered = validate_rigid_transform(_tensor_to_numpy(estimator.get_tf_to_centered_mesh()), "FoundationPose transform_to_centered_mesh for visibility")
    gt_pose_centered = validate_rigid_transform(gt_pose_cam @ np.linalg.inv(transform_to_centered), "centered ground-truth object camera pose for visibility")
    height, width = np.asarray(human_mask).shape
    device = estimator.mesh_tensors["pos"].device
    pose = torch.as_tensor(gt_pose_centered[None], dtype=torch.float32, device=device)
    cv_to_gl = torch.as_tensor(Utils.glcam_in_cvcam, dtype=torch.float32, device=device)[None]
    projection = torch.as_tensor(Utils.projection_matrix_from_intrinsics(K, height=height, width=width, znear=0.001, zfar=100.0), dtype=torch.float32, device=device)[None]
    position_homogeneous = Utils.to_homo_torch(estimator.mesh_tensors["pos"])
    position_clip = ((projection @ cv_to_gl @ pose)[:, None] @ position_homogeneous[None, ..., None])[..., 0]
    raster, _ = dr.rasterize(estimator.glctx, position_clip, estimator.mesh_tensors["faces"], resolution=np.asarray([height, width]))
    rendered_object_mask = _tensor_to_numpy(_raster_alpha_to_opencv_mask(raster)[0])
    return measure_gt_object_visibility(rendered_object_mask, human_mask, annotation_valid=annotation_valid)


def _read_foundationpose_frame(export_seq: Path, depth_root: Path, camera_id: int, frame_name: str, K: np.ndarray) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = read_rgb(export_seq, camera_id, frame_name)
    depth = read_depth_m(export_seq, camera_id, frame_name, depth_root=depth_root)
    mask_h = read_mask(export_seq, "human", camera_id, frame_name)
    mask_o = read_mask(export_seq, "object", camera_id, frame_name)
    rgb, depth, mask_h, mask_o, K_frame = _prepare_foundationpose_inputs(rgb, depth, mask_h, mask_o, K)
    depth = depth.astype(np.float32)
    depth[(depth < 0.001) | (depth > 8.0)] = 0.0
    return frame_name, rgb, depth, mask_h, mask_o, K_frame


def _foundationpose_frame_iterator(export_seq: Path, depth_root: Path, camera_id: int, names: list[str], K: np.ndarray, prefetch: bool):
    if not prefetch:
        for frame_name in names:
            yield _read_foundationpose_frame(export_seq, depth_root, camera_id, frame_name, K)
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="foundationpose-input") as executor:
        future = executor.submit(_read_foundationpose_frame, export_seq, depth_root, camera_id, names[0], K)
        for index in range(len(names)):
            current = future.result()
            if index + 1 < len(names):
                future = executor.submit(_read_foundationpose_frame, export_seq, depth_root, camera_id, names[index + 1], K)
            yield current


def _load_gt_camera_poses(export_seq: Path, w2c: np.ndarray, frame_count: int) -> np.ndarray:
    pose_path = export_seq / "poses.npy"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Ground-truth rotation oracle requires {pose_path}")
    poses_world = np.asarray(np.load(pose_path), dtype=np.float32)
    if poses_world.shape != (frame_count, 4, 4):
        raise ValueError(f"{pose_path} must have shape ({frame_count}, 4, 4), got {poses_world.shape}")
    poses_world = validate_rigid_transforms(poses_world, "ground-truth object world poses")
    return validate_rigid_transforms(np.matmul(w2c[None], poses_world), "ground-truth object camera poses")


def _chm_progress_reporter(total: int):
    metric_id = os.environ.get("CHM_PROGRESS_METRIC_ID", "").strip()
    if not metric_id:
        return lambda current, force=False: None
    phase_id = os.environ.get("CHM_PROGRESS_PHASE_ID", "").strip()
    item = os.environ.get("CHM_PROGRESS_ITEM", "").strip()
    attempt_id = os.environ.get("CHM_JOB_ATTEMPT_ID", "").strip()
    interval_seconds = float(os.environ.get("CHM_PROGRESS_INTERVAL_SECONDS", "60"))
    if interval_seconds <= 0:
        raise ValueError(f"CHM_PROGRESS_INTERVAL_SECONDS must be positive, got {interval_seconds}")
    last_emitted = [0.0]

    def emit(current: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and last_emitted[0] and now - last_emitted[0] < interval_seconds:
            return
        payload = {"observedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "current": current, "total": total, "unit": "frames", "metricId": metric_id}
        for key, value in (("phaseId", phase_id), ("item", item), ("attemptId", attempt_id)):
            if value:
                payload[key] = value
        print("CHM_PROGRESS " + json.dumps(payload, separators=(",", ":")), flush=True)
        last_emitted[0] = now

    return emit


def _validate_foundationpose_output(path: str | Path, expected_frames: list[str], expected_camera_id: int, expected_optimizations: frozenset[str], expected_mask_identities: dict[str, object], expected_first_usable_frame_gt_rotation_oracle: bool, expected_depth_root: str | Path | None = None, expected_depth_source: str | None = None, expected_depth_input_identity: dict[str, object] | None = None, expected_sequence_mode: str | None = None) -> dict:
    with Path(path).open("rb") as handle:
        result = pickle.load(handle)
    if list(result.get("frames", [])) != list(expected_frames):
        raise ValueError(f"FoundationPose output frames differ for {path}: expected={len(expected_frames)}, actual={len(result.get('frames', []))}")
    if int(result.get("camera_id", -1)) != int(expected_camera_id):
        raise ValueError(f"FoundationPose output camera differs for {path}: expected={expected_camera_id}, actual={result.get('camera_id')}")
    frame_count = len(expected_frames)
    expected_shapes = {"fp_poses_cam": (frame_count, 1, 4, 4), "obj_pose_world": (frame_count, 4, 4), "obj_rot_init": (frame_count, 3, 3), "obj_t_init": (frame_count, 3)}
    for key, expected_shape in expected_shapes.items():
        value = np.asarray(result.get(key))
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise ValueError(f"FoundationPose output {key} is invalid for {path}: expected={expected_shape}, actual={value.shape}")
    for key in ("fp_translation_threshold_m", "fp_selected_translation_distance_m"):
        value = np.asarray(result.get(key))
        if value.shape != (frame_count,):
            raise ValueError(f"FoundationPose output {key} is invalid for {path}: expected={(frame_count,)}, actual={value.shape}")
    metadata = result.get("metadata", {})
    if metadata.get("foundationpose_optimization_revision") != FOUNDATIONPOSE_OPTIMIZATION_REVISION or set(metadata.get("foundationpose_optimizations", [])) != expected_optimizations:
        raise ValueError(f"FoundationPose output optimization metadata is stale for {path}: revision={metadata.get('foundationpose_optimization_revision')}, optimizations={metadata.get('foundationpose_optimizations')}")
    if metadata.get("mask_identities") != expected_mask_identities:
        raise ValueError(f"FoundationPose output mask identities are stale for {path}")
    if expected_depth_root is not None and metadata.get("depth_root") != str(expected_depth_root):
        raise ValueError(f"FoundationPose output depth root is stale for {path}: expected={expected_depth_root}, actual={metadata.get('depth_root')}")
    if expected_depth_source is not None and metadata.get("depth_source") != str(expected_depth_source):
        raise ValueError(f"FoundationPose output depth source is stale for {path}: expected={expected_depth_source}, actual={metadata.get('depth_source')}")
    actual_depth_input_identity = metadata.get("depth_input_identity")
    if actual_depth_input_identity is not None and expected_depth_input_identity is not None and actual_depth_input_identity != expected_depth_input_identity:
        raise ValueError(f"FoundationPose output depth artifact is stale for {path}")
    actual_oracle_setting = metadata.get("first_usable_frame_gt_rotation_oracle")
    if not isinstance(actual_oracle_setting, bool) or actual_oracle_setting != bool(expected_first_usable_frame_gt_rotation_oracle):
        raise ValueError(f"FoundationPose output GT rotation-oracle setting is stale for {path}: expected={bool(expected_first_usable_frame_gt_rotation_oracle)}, actual={metadata.get('first_usable_frame_gt_rotation_oracle')}")
    if expected_sequence_mode is not None and metadata.get("foundationpose_sequence_mode") != expected_sequence_mode:
        raise ValueError(f"FoundationPose output sequence mode is stale for {path}: expected={expected_sequence_mode}, actual={metadata.get('foundationpose_sequence_mode')}")
    validate_gt_object_visibility_metadata(metadata, f"FoundationPose output {path}")
    validate_gt_object_visibility_arrays(result, (frame_count,), f"FoundationPose output {path}")
    validate_foundationpose_selection_metadata(metadata, f"FoundationPose output {path}")
    validate_foundationpose_selection_arrays(result, (frame_count,), f"FoundationPose output {path}")
    validate_rigid_transforms(np.asarray(result["fp_poses_cam"])[:, 0], f"cached FoundationPose camera poses {path}")
    validate_rigid_transforms(np.asarray(result["obj_pose_world"]), f"cached FoundationPose world poses {path}")
    return result


def run_foundationpose_mhr_export(
    export_seq: str | Path,
    depth_root: str | Path,
    out_file: str | Path,
    *,
    camera_id: int = 0,
    start: int = 0,
    end: int | None = None,
    reinit_every: int | None = None,
    iteration: int = 5,
    viz_file: str | Path | None = None,
    debug_dir: str | Path | None = None,
    both_depth_and_rgb: bool = False,
    rgb_only_score: bool = False,
    viz_every: int = 10,
    depth_source: str = "metric_depth",
    min_object_pixels: int = 50,
    visibility_threshold: float = 0.5,
    iou_rank: int = 30,
    max_attempts: int = 5,
    translation_base_threshold_m: float = 0.2,
    translation_history_size: int = 5,
    translation_history_multiplier: float = 2.0,
    random_seed: int = 0,
    reacquisition_thresholds: ReacquisitionThresholds = ReacquisitionThresholds(),
    candidate_screen_max_side: int = 384,
    foundationpose_optimizations: tuple[str, ...] = DEFAULT_FOUNDATIONPOSE_OPTIMIZATIONS,
    first_usable_frame_gt_rotation_oracle: bool = True,
    sequence_mode: str = DEFAULT_FOUNDATIONPOSE_SEQUENCE_MODE,
    redo: bool = False,
    profiler: PipelineTimer | None = None,
) -> Path:
    setup_started = profiler.start() if profiler is not None else 0.0
    export_seq = Path(export_seq)
    depth_root = Path(depth_root)
    out_file = Path(out_file)
    debug_dir = _resolve_foundationpose_debug_dir(out_file, debug_dir)
    sequence_mode = str(sequence_mode)
    if sequence_mode not in FOUNDATIONPOSE_SEQUENCE_MODES:
        raise ValueError(f"Unknown FoundationPose sequence mode: {sequence_mode}")
    if sequence_mode == FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK and first_usable_frame_gt_rotation_oracle:
        raise ValueError("register-first-then-track is a ground-truth-free inference mode; disable the first-usable-frame GT rotation oracle")
    foundationpose_optimizations = frozenset(foundationpose_optimizations)
    unknown_optimizations = foundationpose_optimizations - FOUNDATIONPOSE_OPTIMIZATIONS
    if unknown_optimizations:
        raise ValueError(f"Unknown FoundationPose optimizations: {sorted(unknown_optimizations)}")
    if both_depth_and_rgb:
        raise ValueError("The score-ordered retry exporter alternates one depth- or RGB-refined candidate pool per registration; --both-depth-and-rgb is incompatible with this selection revision")
    max_attempts = int(max_attempts)
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")
    use_frame_cache = "frame-cache" in foundationpose_optimizations
    use_tensor_residency = "tensor-residency" in foundationpose_optimizations
    use_prefetch = "prefetch" in foundationpose_optimizations

    edex = load_edex(export_seq)
    K, w2c = camera_calibration(edex, camera_id)
    c2w = np.linalg.inv(w2c).astype(np.float32)
    names_all = frame_names(export_seq, camera_id)
    stop = len(names_all) if end is None else min(end, len(names_all))
    names = names_all[start:stop]
    mask_identities = {kind: mask_input_identity(export_seq, kind, camera_id) for kind in ("human", "object")}
    depth_input_identity = path_identity(depth_root)
    if out_file.is_file() and not redo:
        _validate_foundationpose_output(out_file, names, camera_id, foundationpose_optimizations, mask_identities, first_usable_frame_gt_rotation_oracle, depth_root, depth_source, depth_input_identity, sequence_mode)
        print(f"validated existing FoundationPose output at {out_file}")
        return out_file
    object_mesh_path = resolve_object_mesh_path(export_seq)
    mesh = load_export_object_mesh(export_seq)
    object_symmetry_path = export_seq / "object_mesh" / "output_symmetry.json"
    object_symmetry = load_optional_output_object_symmetry(object_symmetry_path, object_mesh_path=object_mesh_path)
    object_symmetry = ObjectSymmetry(np.eye(4, dtype=np.float32)[None]) if object_symmetry is None else object_symmetry
    object_symmetry_tfs = validate_rigid_transforms(object_symmetry.transforms, "object symmetry transforms")
    debug_dir.mkdir(parents=True, exist_ok=True)
    if viz_file:
        Path(viz_file).parent.mkdir(parents=True, exist_ok=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_ffmpeg_writer(viz_file) if viz_file else None
    estimator = _pose_estimator(mesh, debug_dir)
    centered_symmetry_tfs = _centered_symmetry_tfs(mesh, object_symmetry_tfs)
    centered_symmetry_center = _centered_symmetry_center(mesh, object_symmetry.center)
    identity_cluster_symmetry_tfs = np.eye(4, dtype=np.float32)[None]
    object_diameter_m = float(estimator.diameter)
    if sequence_mode == FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK and reinit_every is not None:
        raise ValueError("register-first-then-track does not support periodic re-registration")
    if sequence_mode == FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_EVERY_FRAME and reinit_every not in (None, 1):
        raise ValueError(f"fp_hy3d-compatible MHR export registers every usable frame; reinit_every must be 1 or unset, got {reinit_every}")
    candidate_screen_max_side = int(candidate_screen_max_side)
    if candidate_screen_max_side <= 0:
        raise ValueError(f"candidate_screen_max_side must be positive, got {candidate_screen_max_side}")
    translation_base_threshold_m = float(translation_base_threshold_m)
    translation_history_size = int(translation_history_size)
    translation_history_multiplier = float(translation_history_multiplier)
    adaptive_translation_threshold_m([], base_threshold_m=translation_base_threshold_m, history_size=translation_history_size, multiplier=translation_history_multiplier)
    gt_object_pose_path = export_seq / "poses.npy"
    gt_object_pose_available = gt_object_pose_path.is_file()
    if first_usable_frame_gt_rotation_oracle and not gt_object_pose_available:
        raise FileNotFoundError(f"Ground-truth first-frame rotation oracle requires {gt_object_pose_path}; use --no-first-usable-frame-gt-rotation-oracle for inference-only data")
    gt_poses_cam_all = _load_gt_camera_poses(export_seq, w2c, len(names_all)) if gt_object_pose_available else None
    gt_visibility_object_pose_valid_all = load_object_pose_valid_mask_for_frames(export_seq, names_all) if gt_object_pose_available else np.zeros(len(names_all), dtype=bool)
    rng = np.random.default_rng(random_seed)
    report_progress = _chm_progress_reporter(len(names))

    viz_every = max(1, int(viz_every))
    bbox_to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    poses_cam: list[np.ndarray | None] = []
    poses_world: list[np.ndarray | None] = []
    fallback_reasons: list[str] = []
    candidate_counts: list[int] = []
    selected_candidate_indices: list[int] = []
    selected_ious: list[float] = []
    selected_visibilities: list[float] = []
    retry_counts: list[int] = []
    reliable: list[bool] = []
    oracle_rotation_errors_deg: list[float] = []
    translation_thresholds_m: list[float] = []
    selected_translation_distances_m: list[float] = []
    gt_object_visibility_ratios: list[float] = []
    gt_object_visibility_valid: list[bool] = []
    gt_object_rendered_pixels: list[int] = []
    gt_object_visible_pixels: list[int] = []
    selection_diagnostics: dict[str, list] = {key: [] for key in FOUNDATIONPOSE_SELECTION_FIELDS}
    registration_call_count = 0
    tracking_call_count = 0
    initial_registration_frame_index = -1
    selection_state = TrackingState.REACQUIRING
    last_confirmed_pose_centered: np.ndarray | None = None
    last_confirmed_pose_cam: np.ndarray | None = None
    last_confirmed_pose_world: np.ndarray | None = None
    last_confirmed_index = -1
    confirmed_pose_source = PoseSource.NONE
    pending_pose_centered: np.ndarray | None = None
    pending_confirmation_count = 0
    reacquisition_visible_count = 0
    insufficient_evidence_count = 0
    reliable_translation_steps_m: list[float] = []
    visibility_last = 0.0
    last_visible_index = -1
    oracle_applied = False
    oracle_frame_index = -1
    mesh_transform_to_centered = validate_rigid_transform(_tensor_to_numpy(estimator.get_tf_to_centered_mesh()), "FoundationPose transform_to_centered_mesh")

    def append_frame_record(*, pose_cam, pose_world, fallback_reason: str, candidate_count: int, selected_candidate_index: int, selected_iou: float, selected_visibility: float, retry_count: int, reliable_frame: bool, translation_threshold_m: float, translation_distance_m: float, metric_values: dict[str, float | int], state: TrackingState, pending_source: PoseSource, emitted_source: PoseSource, gap_length: int, confirmation_count: int, rejection_mask: int, reacquisition_latency: int, oracle_rotation_error_deg: float = float("nan")) -> None:
        poses_cam.append(None if pose_cam is None else np.asarray(pose_cam, dtype=np.float32).copy())
        poses_world.append(None if pose_world is None else np.asarray(pose_world, dtype=np.float32).copy())
        fallback_reasons.append(str(fallback_reason))
        candidate_counts.append(int(candidate_count))
        selected_candidate_indices.append(int(selected_candidate_index))
        selected_ious.append(float(selected_iou))
        selected_visibilities.append(float(selected_visibility))
        retry_counts.append(int(retry_count))
        reliable.append(bool(reliable_frame))
        oracle_rotation_errors_deg.append(float(oracle_rotation_error_deg))
        translation_thresholds_m.append(float(translation_threshold_m))
        selected_translation_distances_m.append(float(translation_distance_m))
        selection_diagnostics["fp_tracking_state"].append(int(state))
        selection_diagnostics["fp_confirmed_pose_source"].append(int(confirmed_pose_source))
        selection_diagnostics["fp_pending_pose_source"].append(int(pending_source))
        selection_diagnostics["fp_emitted_pose_source"].append(int(emitted_source))
        selection_diagnostics["fp_gap_length"].append(min(int(gap_length), np.iinfo(np.uint16).max))
        selection_diagnostics["fp_observed_visible_pixels"].append(int(metric_values["observed_visible_pixels"]))
        selection_diagnostics["fp_rendered_visible_pixels"].append(int(metric_values["rendered_visible_pixels"]))
        selection_diagnostics["fp_rendered_visibility"].append(float(metric_values["rendered_visibility"]))
        selection_diagnostics["fp_full_image_iou"].append(float(metric_values["iou"]))
        selection_diagnostics["fp_tolerant_precision"].append(float(metric_values["precision"]))
        selection_diagnostics["fp_tolerant_recall"].append(float(metric_values["recall"]))
        selection_diagnostics["fp_centroid_distance"].append(float(metric_values["centroid_distance"]))
        selection_diagnostics["fp_confirmation_count"].append(int(confirmation_count))
        selection_diagnostics["fp_rejection_mask"].append(int(rejection_mask))
        selection_diagnostics["fp_reliable"].append(bool(reliable_frame))
        selection_diagnostics["fp_reacquisition_latency"].append(int(reacquisition_latency))

    def append_visualization(frame_index: int, rgb: np.ndarray, object_mask: np.ndarray, K_frame: np.ndarray, pose_cam: np.ndarray | None) -> None:
        if writer is None or frame_index % viz_every != 0 or pose_cam is None:
            return
        center_pose = pose_cam @ np.linalg.inv(bbox_to_origin)
        vis = Utils.draw_posed_3d_box(K_frame, img=rgb.copy(), ob_in_cam=center_pose, bbox=bbox)
        vis = Utils.draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K_frame, thickness=3, transparency=0)
        mask_vis = rgb.copy()
        mask_vis[object_mask] = (0.5 * mask_vis[object_mask] + np.array([255, 80, 20]) * 0.5).astype(np.uint8)
        frame = np.concatenate([rgb, vis, mask_vis], axis=1)
        writer.append_data(cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3)))

    frame_iterator = _foundationpose_frame_iterator(export_seq, depth_root, camera_id, names, K, use_prefetch)
    if profiler is not None:
        profiler.record("setup", setup_started)
        profiler.update_metadata({"frames": len(names), "iteration": int(iteration), "max_attempts": int(max_attempts), "prefetch": bool(use_prefetch), "frame_cache": bool(use_frame_cache), "tensor_residency": bool(use_tensor_residency)})
    frame_loop_started = profiler.start() if profiler is not None else 0.0
    for idx, frame_values in enumerate(tqdm(frame_iterator, total=len(names), desc="FoundationPose MHR export")):
        report_progress(idx)
        frame_name, rgb, depth, mask_h, mask_o, K_frame = frame_values
        observed_visible_pixels = int((mask_o & ~mask_h).sum())
        support_pixels = minimum_visible_support_pixels(mask_o.shape, reacquisition_thresholds)
        gt_visibility = _render_gt_object_visibility_frame(estimator, gt_poses_cam_all[start + idx], K_frame, mask_h, annotation_valid=bool(gt_visibility_object_pose_valid_all[start + idx])) if gt_poses_cam_all is not None else GTObjectVisibility(float("nan"), False, 0, 0)
        gt_object_visibility_ratios.append(gt_visibility.ratio)
        gt_object_visibility_valid.append(gt_visibility.valid)
        gt_object_rendered_pixels.append(gt_visibility.rendered_pixels)
        gt_object_visible_pixels.append(gt_visibility.visible_pixels)

        if observed_visible_pixels < support_pixels and (sequence_mode == FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_EVERY_FRAME or last_confirmed_pose_centered is None):
            insufficient_evidence_count += 1
            pending_pose_centered = None
            pending_confirmation_count = 0
            if insufficient_evidence_count >= reacquisition_thresholds.insufficient_evidence_frames:
                selection_state = TrackingState.OCCLUDED
            gap_length = idx - last_confirmed_index if last_confirmed_index >= 0 else idx + 1
            metric_values = _selected_metric_values(None, None)
            metric_values["observed_visible_pixels"] = observed_visible_pixels
            fallback_reason = "carry_confirmed_insufficient_object_support" if last_confirmed_pose_cam is not None else "pending_insufficient_object_support"
            low_support_rejection = RejectionReason.OBSERVED_SUPPORT
            if not np.any(mask_o):
                low_support_rejection |= RejectionReason.EMPTY_OBJECT_MASK
            append_frame_record(pose_cam=last_confirmed_pose_cam, pose_world=last_confirmed_pose_world, fallback_reason=fallback_reason, candidate_count=0, selected_candidate_index=-1, selected_iou=float("nan"), selected_visibility=float("nan"), retry_count=0, reliable_frame=False, translation_threshold_m=float("nan"), translation_distance_m=float("nan"), metric_values=metric_values, state=selection_state, pending_source=PoseSource.NONE, emitted_source=PoseSource.CARRIED_CONFIRMED if last_confirmed_pose_cam is not None else PoseSource.NONE, gap_length=gap_length, confirmation_count=0, rejection_mask=int(low_support_rejection), reacquisition_latency=FOUNDATIONPOSE_REACQUISITION_LATENCY_UNKNOWN)
            continue
        insufficient_evidence_count = 0

        if sequence_mode == FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK and last_confirmed_pose_centered is not None:
            previous_pose_centered = last_confirmed_pose_centered.copy()
            tracked_pose_cam_returned = validate_rigid_transform(np.asarray(estimator.track_one(rgb=rgb, depth=depth, K=K_frame, iteration=iteration), dtype=np.float32).reshape(4, 4), f"tracked camera pose frame {frame_name}")
            tracking_call_count += 1
            tracked_pose_centered = validate_rigid_transform(_tensor_to_numpy(estimator.pose_last).reshape(4, 4), f"tracked centered pose frame {frame_name}")
            tracked_pose_cam = validate_rigid_transform(tracked_pose_centered @ mesh_transform_to_centered, f"tracked camera pose from pose_last frame {frame_name}")
            if not np.allclose(tracked_pose_cam_returned, tracked_pose_cam, atol=1e-5, rtol=1e-5):
                raise ValueError(f"FoundationPose track_one return and pose_last disagree at frame {frame_name}")
            tracked_pose_world = validate_rigid_transform(c2w @ tracked_pose_cam, f"tracked world pose frame {frame_name}")
            _tracked_candidates, _tracked_source_ids, _tracked_masks, tracked_metrics = _full_candidate_data(estimator, tracked_pose_centered[None], np.array([-1], dtype=np.int64), np.array([0], dtype=np.int64), K_frame, mask_h, mask_o)
            metric_values = _selected_metric_values(tracked_metrics, 0)
            selected_visibility = float(metric_values["rendered_visibility"])
            last_confirmed_pose_centered = tracked_pose_centered
            last_confirmed_pose_cam = tracked_pose_cam
            last_confirmed_pose_world = tracked_pose_world
            last_confirmed_index = idx
            confirmed_pose_source = PoseSource.FOUNDATIONPOSE_TRACK
            selection_state = TrackingState.TRACKING
            if np.isfinite(selected_visibility):
                visibility_last = selected_visibility
                if selected_visibility > visibility_threshold:
                    last_visible_index = idx
            append_frame_record(pose_cam=tracked_pose_cam, pose_world=tracked_pose_world, fallback_reason="", candidate_count=1, selected_candidate_index=-1, selected_iou=float(metric_values["iou"]), selected_visibility=selected_visibility, retry_count=0, reliable_frame=True, translation_threshold_m=float("nan"), translation_distance_m=float(np.linalg.norm(tracked_pose_centered[:3, 3] - previous_pose_centered[:3, 3])), metric_values=metric_values, state=selection_state, pending_source=PoseSource.NONE, emitted_source=PoseSource.FOUNDATIONPOSE_TRACK, gap_length=1, confirmation_count=1, rejection_mask=int(RejectionReason.NONE), reacquisition_latency=1)
            append_visualization(idx, rgb, mask_o, K_frame, tracked_pose_cam)
            continue

        prepared_frame = estimator.prepare_registration_frame(K_frame, rgb, depth, mask_o.astype(bool), tensor_residency=use_tensor_residency) if use_frame_cache else None
        usable_initialization_depth = int((mask_o.astype(bool) & (prepared_frame["depth"] >= 0.001)).sum()) >= 4 if prepared_frame is not None else _has_usable_initialization_depth(depth, mask_o)
        if last_confirmed_pose_centered is None and not usable_initialization_depth:
            metric_values = _selected_metric_values(None, None)
            metric_values["observed_visible_pixels"] = observed_visible_pixels
            append_frame_record(pose_cam=None, pose_world=None, fallback_reason="pending_invalid_depth", candidate_count=0, selected_candidate_index=-1, selected_iou=float("nan"), selected_visibility=float("nan"), retry_count=0, reliable_frame=False, translation_threshold_m=float("nan"), translation_distance_m=float("nan"), metric_values=metric_values, state=TrackingState.REACQUIRING, pending_source=PoseSource.NONE, emitted_source=PoseSource.NONE, gap_length=idx + 1, confirmation_count=0, rejection_mask=int(RejectionReason.INVALID_INITIALIZATION_DEPTH), reacquisition_latency=FOUNDATIONPOSE_REACQUISITION_LATENCY_UNKNOWN)
            continue

        register_kwargs = {}
        if use_frame_cache:
            register_kwargs["prepared_frame"] = prepared_frame
        if use_tensor_residency:
            register_kwargs["tensor_residency"] = True

        def registration_attempt(rgb_only_current: bool, seed: int, cluster_angle_deg: float, cluster_symmetry_tfs: np.ndarray, *, screen_all: bool = False):
            nonlocal registration_call_count
            estimator.register(K=K_frame, rgb=rgb, depth=depth, ob_mask=mask_o.astype(bool), iteration=iteration, rgb_only=rgb_only_current, seed=seed, both_depth_and_rgb=False, **register_kwargs)
            registration_call_count += 1
            all_candidates = validate_rigid_transforms(_tensor_to_numpy(estimator.poses), "FoundationPose registration candidates")
            all_candidate_source_ids = _estimator_source_ids(estimator, np.arange(len(all_candidates), dtype=np.int64))
            clustered_candidates, clustered_source_ids = _cluster_candidates_with_source_ids(all_candidates, all_candidate_source_ids, cluster_symmetry_tfs, cluster_angle_deg=cluster_angle_deg)
            candidate_transform_to_centered = validate_rigid_transform(_tensor_to_numpy(estimator.get_tf_to_centered_mesh()), "FoundationPose transform_to_centered_mesh")
            candidate_indices = np.arange(len(clustered_candidates), dtype=np.int64) if screen_all else _screen_candidate_indices(estimator, clustered_candidates, K_frame, mask_h, mask_o, reacquisition_thresholds, max_side=candidate_screen_max_side)
            native_candidates, native_source_ids, native_masks, native_metrics = _full_candidate_data(estimator, clustered_candidates, clustered_source_ids, candidate_indices, K_frame, mask_h, mask_o)
            return all_candidates, all_candidate_source_ids, clustered_candidates, candidate_transform_to_centered, native_candidates, native_source_ids, native_masks, native_metrics

        rgb_only_current = bool(rgb_only_score)
        all_candidates_centered, all_source_ids, candidates_centered, transform_to_centered, full_candidates, full_source_ids, full_masks, full_metrics = registration_attempt(rgb_only_current, 0, 10.0, identity_cluster_symmetry_tfs)
        candidate_count = len(candidates_centered)
        gap_length = idx - last_confirmed_index if last_confirmed_index >= 0 else idx + 1
        frame_translation_threshold_m = adaptive_translation_threshold_m(reliable_translation_steps_m, base_threshold_m=translation_base_threshold_m, history_size=translation_history_size, multiplier=translation_history_multiplier)
        selected_local_index: int | None = None
        selected_pose_centered: np.ndarray | None = None
        selected_source = PoseSource.NONE
        selected_metrics = full_metrics
        selected_source_ids = full_source_ids
        oracle_rotation_error_deg = float("nan")
        rejection_mask = int(RejectionReason.NONE)
        reliable_frame = False
        retry_count = 0
        reacquisition_latency = FOUNDATIONPOSE_REACQUISITION_LATENCY_UNKNOWN
        tracking_allowed = last_confirmed_pose_centered is not None and (selection_state == TrackingState.TRACKING or gap_length < reacquisition_thresholds.long_occlusion_frames)
        if first_usable_frame_gt_rotation_oracle and not oracle_applied and bool(gt_visibility_object_pose_valid_all[start + idx]):
            oracle = select_gt_rotation_oracle(all_candidates_centered, transform_to_centered, gt_poses_cam_all[start + idx])
            _oracle_candidates, oracle_source_ids, _oracle_masks, oracle_metrics = _full_candidate_data(estimator, all_candidates_centered, all_source_ids, np.array([oracle.index], dtype=np.int64), K_frame, mask_h, mask_o)
            selected_local_index = 0
            selected_pose_centered = oracle.centered_pose
            selected_source = PoseSource.GT_ROTATION_ORACLE
            selected_metrics = oracle_metrics
            selected_source_ids = oracle_source_ids
            oracle_rotation_error_deg = float(oracle.rotation_errors_deg[oracle.index])
            reliable_frame = True
            reacquisition_latency = 1
            oracle_applied = True
            oracle_frame_index = idx
        if selected_pose_centered is None and sequence_mode == FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK:
            if not len(all_candidates_centered):
                raise RuntimeError(f"FoundationPose registration produced no candidates at frame {frame_name}")
            initial_candidates, initial_source_ids, _initial_masks, initial_metrics = _full_candidate_data(estimator, all_candidates_centered, all_source_ids, np.array([0], dtype=np.int64), K_frame, mask_h, mask_o)
            selected_local_index = 0
            selected_pose_centered = initial_candidates[0].copy()
            selected_source = PoseSource.INITIAL_REGISTRATION
            selected_metrics = initial_metrics
            selected_source_ids = initial_source_ids
            candidate_count = len(all_candidates_centered)
            reliable_frame = True
            reacquisition_latency = 1
            initial_registration_frame_index = idx
        if selected_pose_centered is None and tracking_allowed and full_metrics is not None:
            support_keep = (full_metrics.rendered_visible_pixels >= support_pixels) & (full_metrics.rendered_visibility >= reacquisition_thresholds.visibility_floor)
            tracking_indices = np.flatnonzero(support_keep)
            if len(tracking_indices):
                tracking_filter = filter_ranked_candidates(full_candidates[tracking_indices], full_metrics.iou[tracking_indices], previous_pose_centered=last_confirmed_pose_centered, frame_index=idx, last_visible_index=last_visible_index, previous_visibility=visibility_last, visibility_threshold=visibility_threshold, translation_threshold_m=frame_translation_threshold_m, iou_rank=min(iou_rank, len(tracking_indices) - 1))
                if tracking_filter.selected_index is not None:
                    selected_local_index = int(tracking_indices[tracking_filter.selected_index])
                    selected_pose_centered = full_candidates[selected_local_index].copy()
                    selected_source = PoseSource.TRACKING_CANDIDATE
                    reliable_frame = tracking_filter.rotation_threshold_deg >= 0.0
                else:
                    rejection_mask |= int(temporal_rejection_reasons(tracking_filter))
        if selected_pose_centered is None and tracking_allowed:
            base_retry_rotation_threshold_deg = 15.0 + max(0, idx - last_confirmed_index) * 2.5
            while selected_pose_centered is None and retry_count < max_attempts:
                retry_count += 1
                rgb_only_current = not rgb_only_current
                screen_all = retry_count == max_attempts
                _retry_all_candidates, _retry_all_source_ids, retry_candidates_centered, transform_to_centered, retry_full_candidates, retry_full_source_ids, _retry_full_masks, retry_full_metrics = registration_attempt(rgb_only_current, int(rng.integers(60000)), 5.0, identity_cluster_symmetry_tfs, screen_all=screen_all)
                candidate_count = len(retry_candidates_centered)
                selected_metrics = retry_full_metrics
                selected_source_ids = retry_full_source_ids
                selected_local_index = None
                if retry_full_metrics is None or not len(retry_full_candidates):
                    continue
                retry_indices = np.arange(len(retry_full_candidates), dtype=np.int64)
                retry_filter = filter_retry_candidates(retry_full_candidates, retry_full_metrics.iou, previous_pose_centered=last_confirmed_pose_centered, base_rotation_threshold_deg=base_retry_rotation_threshold_deg, attempt=retry_count, max_attempts=max_attempts)
                if retry_filter.selected_index is not None:
                    selected_local_index = int(retry_indices[retry_filter.selected_index])
                    selected_pose_centered = retry_full_candidates[selected_local_index].copy()
                    selected_source = PoseSource.TRACKING_CANDIDATE
                    reliable_frame = False
        pending_tracking_pose_centered = None
        if pending_pose_centered is not None and selected_pose_centered is not None and selected_source == PoseSource.TRACKING_CANDIDATE:
            pending_tracking_pose_centered = selected_pose_centered.copy()
            selected_pose_centered = None
            selected_source = PoseSource.NONE
            reliable_frame = False
        if selected_pose_centered is None:
            if selection_state != TrackingState.REACQUIRING:
                reacquisition_visible_count = 0
            selection_state = TrackingState.REACQUIRING
            reacquisition_visible_count += 1
            individual = None
            current_pending = pending_tracking_pose_centered
            if current_pending is None:
                individual = filter_reacquisition_candidates(full_metrics, mask_h.shape, reacquisition_thresholds) if full_metrics is not None and len(full_candidates) else None
            if current_pending is None and individual is not None and individual.selected_index is not None:
                selected_local_index = int(individual.selected_index)
                current_pending = full_candidates[selected_local_index].copy()
            if current_pending is not None:
                pending_pose_centered, pending_confirmation_count, _ = advance_pending_pose(current_pending, pending_pose_centered, pending_confirmation_count, centered_symmetry_tfs, object_diameter_m, reacquisition_thresholds, symmetry_mode=object_symmetry.mode, symmetry_center=centered_symmetry_center)
                if pending_confirmation_count >= reacquisition_thresholds.confirmation_frames:
                    selected_pose_centered = pending_pose_centered.copy()
                    selected_source = PoseSource.REACQUISITION_CANDIDATE
                    reliable_frame = True
                    reacquisition_latency = reacquisition_visible_count
                else:
                    rejection_mask |= int(RejectionReason.CONFIRMATION_PENDING)
            else:
                pending_pose_centered = None
                pending_confirmation_count = 0
                if individual is None:
                    rejection_mask |= int(RejectionReason.NO_CANDIDATE)
                else:
                    rejection_counts = np.asarray([bin(int(value)).count("1") for value in individual.rejection_masks], dtype=np.int32)
                    best_rejected = int(np.argmin(rejection_counts))
                    selected_local_index = best_rejected
                    rejection_mask |= int(individual.rejection_masks[best_rejected])
        selected_translation_distance_m = float(np.linalg.norm(selected_pose_centered[:3, 3] - last_confirmed_pose_centered[:3, 3])) if selected_pose_centered is not None and selected_source == PoseSource.TRACKING_CANDIDATE and last_confirmed_pose_centered is not None else float("nan")
        if selected_pose_centered is not None:
            previous_confirmed_pose = last_confirmed_pose_centered
            previous_confirmed_index = last_confirmed_index
            last_confirmed_pose_centered = validate_rigid_transform(selected_pose_centered, f"confirmed centered pose frame {frame_name}")
            last_confirmed_pose_cam = validate_rigid_transform(last_confirmed_pose_centered @ transform_to_centered, f"confirmed camera pose frame {frame_name}")
            last_confirmed_pose_world = validate_rigid_transform(c2w @ last_confirmed_pose_cam, f"confirmed world pose frame {frame_name}")
            last_confirmed_index = idx
            confirmed_pose_source = selected_source
            if selected_source == PoseSource.TRACKING_CANDIDATE and reliable_frame and previous_confirmed_pose is not None and previous_confirmed_index == idx - 1:
                reliable_translation_steps_m.append(float(np.linalg.norm(last_confirmed_pose_centered[:3, 3] - previous_confirmed_pose[:3, 3])))
            selection_state = TrackingState.TRACKING
            pending_pose_centered = None
            pending_confirmation_count = 0
            emitted_pose_cam = last_confirmed_pose_cam
            emitted_pose_world = last_confirmed_pose_world
            emitted_source = selected_source
            fallback_reason = ""
        else:
            emitted_pose_cam = last_confirmed_pose_cam
            emitted_pose_world = last_confirmed_pose_world
            emitted_source = PoseSource.CARRIED_CONFIRMED if emitted_pose_cam is not None else PoseSource.NONE
            fallback_reason = "carry_confirmed_reacquisition_unconfirmed" if emitted_pose_cam is not None else "pending_reacquisition_unconfirmed"
        if last_confirmed_pose_centered is not None:
            _set_estimator_pose_last(estimator, last_confirmed_pose_centered)
        metric_values = _selected_metric_values(selected_metrics, selected_local_index)
        selected_iou = float(metric_values["iou"])
        selected_visibility = float(metric_values["rendered_visibility"])
        if selected_pose_centered is not None and np.isfinite(selected_visibility):
            visibility_last = selected_visibility
            if selected_visibility > visibility_threshold:
                last_visible_index = idx
        selected_candidate_index = int(selected_source_ids[selected_local_index]) if selected_local_index is not None else -1
        confirmation_count = reacquisition_thresholds.confirmation_frames if selected_source == PoseSource.REACQUISITION_CANDIDATE else 1 if selected_source in (PoseSource.GT_ROTATION_ORACLE, PoseSource.INITIAL_REGISTRATION) else pending_confirmation_count
        append_frame_record(pose_cam=emitted_pose_cam, pose_world=emitted_pose_world, fallback_reason=fallback_reason, candidate_count=candidate_count, selected_candidate_index=selected_candidate_index, selected_iou=selected_iou, selected_visibility=selected_visibility, retry_count=retry_count, reliable_frame=reliable_frame, translation_threshold_m=frame_translation_threshold_m if tracking_allowed else float("nan"), translation_distance_m=selected_translation_distance_m, metric_values=metric_values, state=selection_state, pending_source=PoseSource.REACQUISITION_CANDIDATE if pending_pose_centered is not None else PoseSource.NONE, emitted_source=emitted_source, gap_length=gap_length, confirmation_count=confirmation_count, rejection_mask=rejection_mask, reacquisition_latency=reacquisition_latency, oracle_rotation_error_deg=oracle_rotation_error_deg)
        append_visualization(idx, rgb, mask_o, K_frame, emitted_pose_cam)

    if profiler is not None:
        profiler.record("frame_registration_and_selection", frame_loop_started)
    finalize_started = profiler.start() if profiler is not None else 0.0

    if writer is not None:
        writer.close()
        print(f"saved visualization to {viz_file}")

    failure_diagnostics_file = out_file.with_suffix(out_file.suffix + ".failure.pkl")
    first_valid = next((pose for pose in poses_cam if pose is not None), None)
    first_valid_world = next((pose for pose in poses_world if pose is not None), None)
    if first_valid is None or first_valid_world is None:
        failure_diagnostics = {
            "frames": names,
            "camera_id": int(camera_id),
            "candidate_count": np.asarray(candidate_counts, dtype=np.int32),
            "fallback_reasons": fallback_reasons,
            "selection": {key: np.asarray(selection_diagnostics[key], dtype=FOUNDATIONPOSE_SELECTION_DTYPES[key]) for key in FOUNDATIONPOSE_SELECTION_FIELDS},
            "metadata": {
                "reason": "no_confirmed_pose",
                "minimum_visible_support_pixels": int(minimum_visible_support_pixels((mask_h.shape[0], mask_h.shape[1]), reacquisition_thresholds)),
                "reacquisition_confirmation_frames": int(reacquisition_thresholds.confirmation_frames),
                "reacquisition_iou_min": float(reacquisition_thresholds.iou_min),
                "reacquisition_precision_min": float(reacquisition_thresholds.precision_min),
                "reacquisition_recall_min": float(reacquisition_thresholds.recall_min),
                "reacquisition_centroid_distance_max": float(reacquisition_thresholds.centroid_distance_max),
            },
        }
        _atomic_pickle_dump(failure_diagnostics_file, failure_diagnostics)
        raise RuntimeError(
            f"No FoundationPose candidate was confirmed for camera {camera_id}; diagnostics saved to {failure_diagnostics_file}"
        )
    for i, pose in enumerate(poses_cam):
        if pose is None:
            poses_cam[i] = np.asarray(first_valid, dtype=np.float32).copy()
            poses_world[i] = np.asarray(first_valid_world, dtype=np.float32).copy()
            selection_diagnostics["fp_emitted_pose_source"][i] = int(PoseSource.FIRST_CONFIRMED_BACKFILL)
            if fallback_reasons[i] == "pending_insufficient_object_support":
                fallback_reasons[i] = "carry_first_valid_insufficient_object_support"
            elif fallback_reasons[i] == "pending_invalid_depth":
                fallback_reasons[i] = "carry_first_valid_invalid_depth"
            elif fallback_reasons[i] == "pending_reacquisition_unconfirmed":
                fallback_reasons[i] = "carry_first_valid_reacquisition_unconfirmed"
            else:
                raise RuntimeError(f"Unknown pending FoundationPose fallback reason: {fallback_reasons[i]}")

    poses_cam_arr = np.stack([np.asarray(pose, dtype=np.float32) for pose in poses_cam]).astype(np.float32)
    poses_world_arr = np.stack([np.asarray(pose, dtype=np.float32) for pose in poses_world]).astype(np.float32)
    diagnostics = (candidate_counts, selected_candidate_indices, selected_ious, selected_visibilities, retry_counts, reliable, oracle_rotation_errors_deg, translation_thresholds_m, selected_translation_distances_m, fallback_reasons, gt_object_visibility_ratios, gt_object_visibility_valid, gt_object_rendered_pixels, gt_object_visible_pixels, *selection_diagnostics.values())
    if any(len(values) != len(names) for values in diagnostics):
        raise RuntimeError(f"FoundationPose diagnostics do not cover all frames: expected {len(names)}, got {[len(values) for values in diagnostics]}")
    selection_arrays = {key: np.asarray(selection_diagnostics[key], dtype=FOUNDATIONPOSE_SELECTION_DTYPES[key]) for key in FOUNDATIONPOSE_SELECTION_FIELDS}
    result = {
        "frames": names,
        "camera_id": camera_id,
        "fp_poses_cam": poses_cam_arr[:, None],
        "fp_candidate_count": np.asarray(candidate_counts, dtype=np.int32),
        "fp_selected_candidate_index": np.asarray(selected_candidate_indices, dtype=np.int32),
        "fp_selected_iou": np.asarray(selected_ious, dtype=np.float32),
        "fp_selected_visibility": np.asarray(selected_visibilities, dtype=np.float32),
        "fp_retry_count": np.asarray(retry_counts, dtype=np.int32),
        "fp_reliable": np.asarray(reliable, dtype=bool),
        "fp_oracle_rotation_error_deg": np.asarray(oracle_rotation_errors_deg, dtype=np.float32),
        "fp_translation_threshold_m": np.asarray(translation_thresholds_m, dtype=np.float32),
        "fp_selected_translation_distance_m": np.asarray(selected_translation_distances_m, dtype=np.float32),
        "gt_object_visibility_ratio": np.asarray(gt_object_visibility_ratios, dtype=np.float32),
        "gt_object_visibility_valid": np.asarray(gt_object_visibility_valid, dtype=bool),
        "gt_object_rendered_pixels": np.asarray(gt_object_rendered_pixels, dtype=np.int32),
        "gt_object_visible_pixels": np.asarray(gt_object_visible_pixels, dtype=np.int32),
        "obj_pose_world": poses_world_arr,
        "obj_rot_init": poses_world_arr[:, :3, :3],
        "obj_t_init": poses_world_arr[:, :3, 3],
        "metadata": {
            "source": f"foundationpose_register_first_then_track_{depth_source}" if sequence_mode == FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK else f"foundationpose_gt_first_rotation_oracle_score_retry_{depth_source}" if first_usable_frame_gt_rotation_oracle else f"foundationpose_monocular_score_retry_{depth_source}",
            "depth_root": str(depth_root),
            "depth_source": str(depth_source),
            "depth_input_identity": depth_input_identity,
            "mesh": str(resolve_object_mesh_path(export_seq)),
            "mask_identities": mask_identities,
            "gt_object_pose_source": str(gt_object_pose_path) if gt_object_pose_available else None,
            "gt_object_pose_available": bool(gt_object_pose_available),
            "gt_object_visibility_revision": GT_OBJECT_VISIBILITY_REVISION,
            "gt_object_visibility_definition": GT_OBJECT_VISIBILITY_DEFINITION,
            "gt_object_visibility_annotation_validity": OBJECT_POSE_VALIDITY_EXPRESSION if gt_object_pose_available else "unavailable_without_ground_truth_object_pose",
            "gt_object_visibility_uses_object_segmentation": False,
            "gt_object_visibility_uses_foundationpose_prediction": False,
            "selection_pipeline": "score_ranked_first_usable_frame_registration_then_foundationpose_tracking" if sequence_mode == FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK else "gt_first_rotation_oracle_then_score_ordered_full_image_silhouette_temporal_retry" if first_usable_frame_gt_rotation_oracle else "monocular_score_ordered_full_image_silhouette_temporal_retry",
            "foundationpose_selection_revision": FOUNDATIONPOSE_SELECTION_REVISION,
            "foundationpose_selection_uses_depth": True,
            "foundationpose_selection_uses_synchronized_cameras": False,
            "foundationpose_selection_uses_gt_pose": bool(first_usable_frame_gt_rotation_oracle),
            "foundationpose_selection_uses_foundationpose_score": True,
            "foundationpose_candidate_generation_uses_aligned_depth": True,
            "foundationpose_candidate_generation_uses_rgb_refinement": True,
            "foundationpose_candidate_generation_uses_depth_refinement": True,
            "foundationpose_candidate_generation_uses_object_symmetry": False,
            "foundationpose_temporal_symmetry_mode": object_symmetry.mode_name,
            "foundationpose_temporal_symmetry_center_centered_mesh": centered_symmetry_center.tolist(),
            "foundationpose_initial_refinement_mode": "rgb" if rgb_only_score else "depth",
            "foundationpose_retry_refinement_modes": ["rgb", "depth"],
            "first_usable_frame_gt_rotation_oracle": bool(first_usable_frame_gt_rotation_oracle),
            "gt_rotation_oracle_frame_index": int(oracle_frame_index),
            "gt_rotation_oracle_frame": names[oracle_frame_index] if oracle_frame_index >= 0 else None,
            "gt_translation_oracle": False,
            "foundationpose_sequence_mode": sequence_mode,
            "register_every_usable_frame": sequence_mode == FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_EVERY_FRAME,
            "track_after_first_registration": sequence_mode == FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK,
            "initial_registration_frame_index": int(initial_registration_frame_index),
            "initial_registration_frame": names[initial_registration_frame_index] if initial_registration_frame_index >= 0 else None,
            "foundationpose_optimizations": sorted(foundationpose_optimizations),
            "foundationpose_optimization_revision": FOUNDATIONPOSE_OPTIMIZATION_REVISION,
            "registration_call_count": int(registration_call_count),
            "tracking_call_count": int(tracking_call_count),
            "cluster_angle_deg": 10.0,
            "retry_cluster_angle_deg": 5.0,
            "cluster_translation_threshold_m": 0.1,
            "normal_cluster_symmetry_mode": "identity_only",
            "retry_cluster_symmetry_mode": "identity_only",
            "iou_rank": int(iou_rank),
            "max_attempts": int(max_attempts),
            "visibility_threshold": float(visibility_threshold),
            "translation_base_threshold_m": translation_base_threshold_m,
            "translation_history_size": translation_history_size,
            "translation_history_multiplier": translation_history_multiplier,
            "retry_exhaustion_policy": "accept_first_score_ordered_candidate_on_final_attempt",
            "random_seed": int(random_seed),
            "initial_registration_seed": 0,
            "both_depth_and_rgb": False,
            "rgb_only_score": bool(rgb_only_score),
            "foundationpose_score_used_for_selection": True,
            "min_object_pixels": int(min_object_pixels),
            "candidate_screen_max_side": candidate_screen_max_side,
            "reacquisition_support_fraction": reacquisition_thresholds.support_fraction,
            "reacquisition_support_floor_pixels": reacquisition_thresholds.support_floor_pixels,
            "reacquisition_visibility_floor": reacquisition_thresholds.visibility_floor,
            "reacquisition_iou_min": reacquisition_thresholds.iou_min,
            "reacquisition_precision_min": reacquisition_thresholds.precision_min,
            "reacquisition_recall_min": reacquisition_thresholds.recall_min,
            "reacquisition_centroid_distance_max": reacquisition_thresholds.centroid_distance_max,
            "reacquisition_confirmation_frames": reacquisition_thresholds.confirmation_frames,
            "reacquisition_long_occlusion_frames": reacquisition_thresholds.long_occlusion_frames,
            "reacquisition_insufficient_evidence_frames": reacquisition_thresholds.insufficient_evidence_frames,
            "pending_consistency_rotation_deg": reacquisition_thresholds.pending_rotation_deg,
            "pending_consistency_translation_floor_m": reacquisition_thresholds.pending_translation_floor_m,
            "pending_consistency_translation_cap_m": reacquisition_thresholds.pending_translation_cap_m,
            "pending_consistency_translation_diameter_fraction": reacquisition_thresholds.pending_translation_diameter_fraction,
            "threshold_relaxation": True,
            "unrestricted_candidate_acceptance": True,
            "retry_frame_count": int(np.count_nonzero(np.asarray(retry_counts) > 0)),
            "retry_attempt_count": int(np.asarray(retry_counts).sum()),
            "retry_exhausted_count": int(np.count_nonzero(np.asarray(retry_counts) == max_attempts)),
            "reliable_frame_count": int(np.asarray(reliable).sum()),
            "object_pose_fallback_count": int(sum(bool(item) for item in fallback_reasons)),
            "object_pose_fallback_reasons": fallback_reasons,
            "invalid_initialization_depth_fallback_count": int(sum("invalid_depth" in item for item in fallback_reasons)),
            "insufficient_object_support_fallback_count": int(sum("insufficient_object_support" in item for item in fallback_reasons)),
        },
    }
    result.update(selection_arrays)
    validate_foundationpose_selection_metadata(result["metadata"], "generated FoundationPose output")
    validate_foundationpose_selection_arrays(result, (len(names),), "generated FoundationPose output")
    report_progress(len(names), force=True)
    _atomic_pickle_dump(out_file, result)
    failure_diagnostics_file.unlink(missing_ok=True)
    print(f"saved FoundationPose init to {out_file}")
    if profiler is not None:
        profiler.record("output_finalize", finalize_started)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FoundationPose on a flat MHR export using GT mesh and aligned estimated depth.")
    parser.add_argument("export_seq")
    parser.add_argument("--depth-root", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--reinit-every", type=int, default=None)
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--viz-file", default=None)
    parser.add_argument("--debug-dir", default=None)
    refinement_group = parser.add_mutually_exclusive_group()
    refinement_group.add_argument("--both-depth-and-rgb", dest="both_depth_and_rgb", action="store_true", help="Refine both depth and RGB hypotheses instead of fp_hy3d.py's single-mode refinement.")
    refinement_group.add_argument("--depth-only", dest="both_depth_and_rgb", action="store_false", help="Explicitly select fp_hy3d.py's single-mode refinement behavior.")
    parser.set_defaults(both_depth_and_rgb=False)
    score_group = parser.add_mutually_exclusive_group()
    score_group.add_argument("--rgb-score", action="store_true", help="Use RGB-mode refinement and scoring for the initial registration instead of the depth-mode default.")
    score_group.add_argument("--depth-score", action="store_true", help="Explicitly select depth-mode refinement and scoring for the initial registration.")
    parser.add_argument("--viz-every", type=int, default=10, help="Write one visualization frame every N processed frames.")
    parser.add_argument("--depth-source", default="metric_depth")
    parser.add_argument("--min-object-pixels", type=int, default=50)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--iou-rank", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--translation-base-threshold-m", type=float, default=0.2)
    parser.add_argument("--translation-history-size", type=int, default=5)
    parser.add_argument("--translation-history-multiplier", type=float, default=2.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--candidate-screen-max-side", type=int, default=384)
    parser.add_argument("--foundationpose-optimizations", nargs="*", choices=sorted(FOUNDATIONPOSE_OPTIMIZATIONS), default=DEFAULT_FOUNDATIONPOSE_OPTIMIZATIONS)
    sequence_group = parser.add_mutually_exclusive_group()
    sequence_group.add_argument("--register-first-then-track", dest="sequence_mode", action="store_const", const=FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK, help="Register on the first usable frame, then run FoundationPose tracking on every later frame (default for inference).")
    sequence_group.add_argument("--register-every-frame", dest="sequence_mode", action="store_const", const=FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_EVERY_FRAME, help="Run independent FoundationPose registration and temporal candidate selection on every usable frame.")
    parser.set_defaults(sequence_mode=DEFAULT_WILD_INFERENCE_FOUNDATIONPOSE_SEQUENCE_MODE)
    oracle_group = parser.add_mutually_exclusive_group()
    oracle_group.add_argument("--first-usable-frame-gt-rotation-oracle", dest="first_usable_frame_gt_rotation_oracle", action="store_true")
    oracle_group.add_argument("--no-first-usable-frame-gt-rotation-oracle", dest="first_usable_frame_gt_rotation_oracle", action="store_false")
    parser.set_defaults(first_usable_frame_gt_rotation_oracle=None)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    first_usable_frame_gt_rotation_oracle = args.first_usable_frame_gt_rotation_oracle if args.first_usable_frame_gt_rotation_oracle is not None else args.sequence_mode == FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_EVERY_FRAME

    with PipelineTimer("foundationpose_initialization") as profiler:
        run_foundationpose_mhr_export(
            args.export_seq,
            args.depth_root,
            args.out_file,
            camera_id=args.camera_id,
            start=args.start,
            end=args.end,
            reinit_every=args.reinit_every,
            iteration=args.iteration,
            viz_file=args.viz_file,
            debug_dir=args.debug_dir,
            both_depth_and_rgb=args.both_depth_and_rgb,
            rgb_only_score=args.rgb_score,
            viz_every=args.viz_every,
            depth_source=args.depth_source,
            min_object_pixels=args.min_object_pixels,
            visibility_threshold=args.visibility_threshold,
            iou_rank=args.iou_rank,
            max_attempts=args.max_attempts,
            translation_base_threshold_m=args.translation_base_threshold_m,
            translation_history_size=args.translation_history_size,
            translation_history_multiplier=args.translation_history_multiplier,
            random_seed=args.random_seed,
            candidate_screen_max_side=args.candidate_screen_max_side,
            foundationpose_optimizations=tuple(args.foundationpose_optimizations),
            first_usable_frame_gt_rotation_oracle=first_usable_frame_gt_rotation_oracle,
            sequence_mode=args.sequence_mode,
            redo=args.redo,
            profiler=profiler,
        )


if __name__ == "__main__":
    main()
