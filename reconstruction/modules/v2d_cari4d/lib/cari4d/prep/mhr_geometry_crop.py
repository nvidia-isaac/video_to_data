from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from prep.mhr_preprocess_manifest import OBJECT_POSE_VALIDITY_EXPRESSION, build_validity_manifest


MHR_GEOMETRY_CROP_REVISION = "canonical-object-mask-depth-tested-fallback-v2"


@dataclass(frozen=True)
class GeometryGuidedCrop:
    top_left: np.ndarray
    bottom_right: np.ndarray
    crop_size: float
    human_crop_mask: np.ndarray
    object_crop_mask: np.ndarray
    diagnostics: dict[str, object]


def _validate_mask(mask: np.ndarray, name: str) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"{name} expected a two-dimensional mask, got {mask.shape}")
    if not np.isfinite(mask).all():
        raise ValueError(f"{name} contains nonfinite values")
    return mask


def _mask_bbox(mask: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    vs, us = np.where(mask)
    if len(us) == 0:
        raise ValueError(f"{name} is empty")
    return np.array([us.min(), vs.min()], dtype=np.float32), np.array([us.max(), vs.max()], dtype=np.float32)


def _dilation_radius(silhouette: np.ndarray, min_dilation_px: int, relative_dilation: float, name: str) -> int:
    bmin, bmax = _mask_bbox(silhouette, f"{name} geometry silhouette")
    diagonal = float(np.linalg.norm(bmax - bmin))
    return max(int(min_dilation_px), int(np.ceil(diagonal * relative_dilation)))


def _dilated_geometry_gate(silhouette: np.ndarray, min_dilation_px: int, relative_dilation: float, name: str) -> tuple[np.ndarray, int]:
    radius = _dilation_radius(silhouette, min_dilation_px, relative_dilation, name)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(np.asarray(silhouette, dtype=np.uint8), kernel).astype(bool), radius


def load_pose_valid_masks_for_frames(export_seq: str | Path, frames: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load human and object validity aligned to an arbitrary source-frame sequence."""
    export_seq = Path(export_seq)
    manifest = build_validity_manifest(export_seq)
    manifest_indices = {str(frame): index for index, frame in enumerate(manifest["frames"])}
    frame_indices = []
    for frame in frames:
        stem = Path(str(frame)).stem
        if stem not in manifest_indices:
            raise ValueError(f"Cannot map frame {frame!r} to the validity timeline for {export_seq}")
        frame_indices.append(manifest_indices[stem])
    indices = np.asarray(frame_indices, dtype=np.int64)
    human = np.asarray(manifest["human_pose_valid_mask"], dtype=bool)[indices].astype(bool, copy=False)
    object_mask = np.asarray(manifest["object_pose_valid_mask"], dtype=bool)[indices].astype(bool, copy=False)
    return human, object_mask


def load_object_pose_valid_mask_for_frames(export_seq: str | Path, frames: Sequence[str]) -> np.ndarray:
    """Load object validity from the optional pose mask and conservative failure ranges."""
    return load_pose_valid_masks_for_frames(export_seq, frames)[1]


def rasterize_mesh_silhouette(vertices_camera: np.ndarray, faces: np.ndarray, K: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    vertices_camera = np.asarray(vertices_camera, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    K = np.asarray(K, dtype=np.float64)
    if vertices_camera.ndim != 2 or vertices_camera.shape[1] != 3 or len(vertices_camera) == 0:
        raise ValueError(f"vertices_camera expected shape [V,3], got {vertices_camera.shape}")
    if not np.isfinite(vertices_camera).all():
        raise ValueError("vertices_camera contains nonfinite values")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"faces expected shape [F,3], got {faces.shape}")
    if faces.min() < 0 or faces.max() >= len(vertices_camera):
        raise ValueError(f"faces contain indices outside [0,{len(vertices_camera)})")
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"K expected a finite 3x3 matrix, got {K.shape}")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")

    positive = vertices_camera[:, 2] > 1e-6
    visible_faces = faces[np.all(positive[faces], axis=1)]
    if len(visible_faces) == 0:
        raise ValueError("mesh has no triangles fully in front of the camera")
    projected_h = vertices_camera @ K.T
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    triangles = projected[visible_faces]
    inside_or_crossing = ~((triangles[:, :, 0] < 0).all(axis=1) | (triangles[:, :, 0] >= width).all(axis=1) | (triangles[:, :, 1] < 0).all(axis=1) | (triangles[:, :, 1] >= height).all(axis=1))
    triangles = triangles[inside_or_crossing]
    if len(triangles) == 0:
        raise ValueError("mesh geometry projects entirely outside the image")
    triangles[:, :, 0] = np.clip(triangles[:, :, 0], -2 * width, 3 * width)
    triangles[:, :, 1] = np.clip(triangles[:, :, 1], -2 * height, 3 * height)
    silhouette = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(silhouette, np.rint(triangles).astype(np.int32), 1)
    if not np.any(silhouette):
        raise ValueError("mesh geometry produced an empty image silhouette")
    return silhouette.astype(bool)


def filter_crop_masks_by_geometry(mask_h: np.ndarray, mask_o: np.ndarray, human_silhouette: np.ndarray, object_silhouette: np.ndarray, *, object_pose_valid: bool, object_visible_silhouette: np.ndarray | None = None, min_dilation_px: int = 16, relative_dilation: float = 0.02) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    mask_h = _validate_mask(mask_h, "mask_h")
    mask_o = _validate_mask(mask_o, "mask_o")
    human_silhouette = _validate_mask(human_silhouette, "human_silhouette").astype(bool)
    object_silhouette = _validate_mask(object_silhouette, "object_silhouette").astype(bool)
    if mask_h.shape != mask_o.shape or mask_h.shape != human_silhouette.shape or mask_h.shape != object_silhouette.shape:
        raise ValueError(f"crop masks and geometry silhouettes must have the same shape, got {mask_h.shape}, {mask_o.shape}, {human_silhouette.shape}, {object_silhouette.shape}")
    if min_dilation_px < 0 or relative_dilation < 0:
        raise ValueError(f"dilation settings must be nonnegative, got min={min_dilation_px}, relative={relative_dilation}")
    if not isinstance(object_pose_valid, (bool, np.bool_)):
        raise TypeError(f"object_pose_valid must be Boolean, got {type(object_pose_valid).__name__}")
    if object_visible_silhouette is not None:
        object_visible_silhouette = _validate_mask(object_visible_silhouette, "object_visible_silhouette").astype(bool)
        if object_visible_silhouette.shape != mask_h.shape:
            raise ValueError(f"visible object silhouette and crop masks must have the same shape, got {object_visible_silhouette.shape} and {mask_h.shape}")

    human_gate, human_radius = _dilated_geometry_gate(human_silhouette, min_dilation_px, relative_dilation, "human")
    object_gate, object_radius = _dilated_geometry_gate(object_silhouette, min_dilation_px, relative_dilation, "object") if bool(object_pose_valid) else (np.zeros(mask_h.shape, dtype=bool), 0)
    raw_h = mask_h > 127
    raw_o = mask_o > 127
    crop_h = raw_h & human_gate
    crop_o = raw_o & object_gate if bool(object_pose_valid) else raw_o.copy()
    object_crop_support_source = "raw_object_mask_geometry_intersection" if bool(object_pose_valid) else "raw_object_mask_fallback"
    if np.any(raw_h) and not np.any(crop_h):
        raise ValueError("geometry-guided human crop support is empty")
    if np.any(raw_o) and not np.any(crop_o):
        if bool(object_pose_valid):
            if object_visible_silhouette is None:
                raise ValueError("valid object pose with empty object-mask geometry intersection requires a depth-tested visible object silhouette")
            crop_o = object_visible_silhouette.copy()
            object_crop_support_source = "gt_depth_tested_object_silhouette_fallback"
        else:
            crop_o = raw_o.copy()
            object_crop_support_source = "raw_object_mask_fallback"
    elif not np.any(raw_o):
        object_crop_support_source = "empty_effective_object_mask"
    if not np.any(crop_h | crop_o):
        raise ValueError("geometry-guided combined crop support is empty because both effective entity masks are empty")
    diagnostics = {
        "human_dilation_radius_px": human_radius,
        "object_dilation_radius_px": object_radius,
        "human_raw_pixels": int(raw_h.sum()),
        "object_raw_pixels": int(raw_o.sum()),
        "human_crop_pixels": int(crop_h.sum()),
        "object_crop_pixels": int(crop_o.sum()),
        "human_removed_pixels": int((raw_h & ~crop_h).sum()),
        "object_removed_pixels": int((raw_o & ~crop_o).sum()),
        "human_effective_mask_empty": not bool(np.any(raw_h)),
        "object_effective_mask_empty": not bool(np.any(raw_o)),
        "object_pose_valid": bool(object_pose_valid),
        "object_pose_validity_expression": OBJECT_POSE_VALIDITY_EXPRESSION,
        "object_crop_support_source": object_crop_support_source,
        "object_raw_human_overlap_pixels": int((raw_o & raw_h).sum()),
        "object_crop_revision": MHR_GEOMETRY_CROP_REVISION,
        "object_human_overlap_rule": "preserve_canonical_object_mask_and_depth_test_only_gt_fallback",
    }
    return crop_h, crop_o, diagnostics


def build_geometry_guided_crop(mask_h: np.ndarray, mask_o: np.ndarray, human_vertices_camera: np.ndarray, human_faces: np.ndarray, object_vertices_camera: np.ndarray | None, object_faces: np.ndarray | None, K: np.ndarray, *, human_pose_valid: bool, object_pose_valid: bool, pad: float, min_crop_size: float = 8.0, min_dilation_px: int = 16, relative_dilation: float = 0.02, raster_context=None) -> GeometryGuidedCrop:
    if not np.isfinite(pad) or pad <= 0 or not np.isfinite(min_crop_size) or min_crop_size <= 0:
        raise ValueError(f"pad and min_crop_size must be finite and positive, got pad={pad}, min_crop_size={min_crop_size}")
    image_shape = _validate_mask(mask_h, "mask_h").shape
    _validate_mask(mask_o, "mask_o")
    if not isinstance(human_pose_valid, (bool, np.bool_)):
        raise TypeError(f"human_pose_valid must be Boolean, got {type(human_pose_valid).__name__}")
    human_silhouette = rasterize_mesh_silhouette(human_vertices_camera, human_faces, K, image_shape)
    object_visible_silhouette = None
    if bool(object_pose_valid):
        if object_vertices_camera is None or object_faces is None:
            raise ValueError("valid object pose requires object geometry for crop support")
        object_silhouette = rasterize_mesh_silhouette(object_vertices_camera, object_faces, K, image_shape)
        object_gate, _object_radius = _dilated_geometry_gate(object_silhouette, min_dilation_px, relative_dilation, "object")
        needs_fallback = bool(np.any(np.asarray(mask_o) > 127) and not np.any((np.asarray(mask_o) > 127) & object_gate))
        if needs_fallback:
            if bool(human_pose_valid):
                from prep.mhr_mesh_visibility import render_joint_visible_masks

                _visible_human, visible_object = render_joint_visible_masks(np.asarray(human_vertices_camera)[None], human_faces, np.asarray(object_vertices_camera)[None], object_faces, K, image_shape, raster_context=raster_context)
                object_visible_silhouette = visible_object[0]
            else:
                object_visible_silhouette = object_silhouette
    else:
        object_silhouette = np.zeros(image_shape, dtype=bool)
    crop_h, crop_o, diagnostics = filter_crop_masks_by_geometry(mask_h, mask_o, human_silhouette, object_silhouette, object_pose_valid=object_pose_valid, object_visible_silhouette=object_visible_silhouette, min_dilation_px=min_dilation_px, relative_dilation=relative_dilation)
    diagnostics["human_pose_valid"] = bool(human_pose_valid)
    bmin, bmax = _mask_bbox(crop_h | crop_o, "geometry-guided crop support")
    center = (bmin + bmax) / 2.0
    crop_size = max(float(np.max(bmax - bmin) * pad), float(min_crop_size))
    top_left = (center - crop_size / 2.0).astype(np.float32)
    bottom_right = (center + crop_size / 2.0).astype(np.float32)
    diagnostics.update({"support_bbox_uvuv": np.concatenate([bmin, bmax]).astype(float).tolist(), "crop_bbox_uvuv": np.concatenate([top_left, bottom_right]).astype(float).tolist(), "crop_size_px": crop_size})
    return GeometryGuidedCrop(top_left=top_left, bottom_right=bottom_right, crop_size=crop_size, human_crop_mask=crop_h, object_crop_mask=crop_o, diagnostics=diagnostics)
