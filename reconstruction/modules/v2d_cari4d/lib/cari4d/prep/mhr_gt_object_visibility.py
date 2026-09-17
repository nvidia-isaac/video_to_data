from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import numpy as np


GT_OBJECT_VISIBILITY_REVISION = "gt-object-full-image-effective-human-occlusion-v4"
GT_OBJECT_VISIBILITY_DEFINITION = "full_native_image_rendered_gt_object_pixels_not_occluded_by_effective_human_segmentation_over_all_rendered_gt_object_pixels"
GT_OBJECT_VISIBILITY_FIELDS = (
    "gt_object_visibility_ratio",
    "gt_object_visibility_valid",
    "gt_object_rendered_pixels",
    "gt_object_visible_pixels",
)
GT_OBJECT_VISIBILITY_H5_PATHS = {
    "gt_object_visibility_ratio": "object/gt_visibility_ratio",
    "gt_object_visibility_valid": "object/gt_visibility_valid",
    "gt_object_rendered_pixels": "object/gt_rendered_pixels",
    "gt_object_visible_pixels": "object/gt_visible_pixels",
}


class GTObjectVisibility(NamedTuple):
    ratio: float
    valid: bool
    rendered_pixels: int
    visible_pixels: int


def validate_gt_object_visibility_metadata(metadata: Mapping[str, Any], label: str, camera_order: list[int] | None = None) -> None:
    if metadata.get("gt_object_visibility_revision") != GT_OBJECT_VISIBILITY_REVISION:
        raise ValueError(f"{label} has unsupported ground-truth object visibility revision: {metadata.get('gt_object_visibility_revision')}")
    if metadata.get("gt_object_visibility_definition") != GT_OBJECT_VISIBILITY_DEFINITION:
        raise ValueError(f"{label} has unsupported ground-truth object visibility definition: {metadata.get('gt_object_visibility_definition')}")
    annotation_validity = metadata.get("gt_object_visibility_annotation_validity")
    if not isinstance(annotation_validity, str) or not annotation_validity.strip():
        raise ValueError(f"{label} is missing the ground-truth object visibility annotation-validity expression")
    for key in ("gt_object_visibility_uses_object_segmentation", "gt_object_visibility_uses_foundationpose_prediction"):
        if metadata.get(key) is not False:
            raise ValueError(f"{label}/{key} must be false")
    if camera_order is not None and [int(kid) for kid in metadata.get("gt_object_visibility_camera_order", [])] != [int(kid) for kid in camera_order]:
        raise ValueError(f"{label} camera order does not match kids: {metadata.get('gt_object_visibility_camera_order')} != {camera_order}")


def measure_gt_object_visibility(rendered_object_mask: np.ndarray, human_mask: np.ndarray, *, annotation_valid: bool) -> GTObjectVisibility:
    rendered = np.asarray(rendered_object_mask)
    human = np.asarray(human_mask)
    if rendered.ndim != 2 or human.shape != rendered.shape:
        raise ValueError(f"Ground-truth object and human masks must be matching [H,W] arrays, got {rendered.shape} and {human.shape}")
    if rendered.dtype != np.dtype("bool"):
        if not np.isin(rendered, (0, 1)).all():
            raise ValueError("Ground-truth rendered-object mask must be binary")
        rendered = rendered.astype(bool)
    if human.dtype != np.dtype("bool"):
        if not np.isin(human, (0, 1)).all():
            raise ValueError("Human segmentation mask must be binary")
        human = human.astype(bool)
    if not isinstance(annotation_valid, (bool, np.bool_)):
        raise TypeError(f"annotation_valid must be Boolean, got {type(annotation_valid).__name__}")
    rendered_pixels = int(rendered.sum())
    visible_pixels = int((rendered & ~human).sum())
    valid = bool(annotation_valid) and rendered_pixels > 0
    ratio = float(visible_pixels / rendered_pixels) if valid else float("nan")
    return GTObjectVisibility(ratio=ratio, valid=valid, rendered_pixels=rendered_pixels, visible_pixels=visible_pixels)


def validate_gt_object_visibility_arrays(data: Mapping[str, Any], expected_shape: tuple[int, ...], label: str) -> dict[str, np.ndarray]:
    missing = [key for key in GT_OBJECT_VISIBILITY_FIELDS if key not in data]
    if missing:
        raise ValueError(f"{label} is missing ground-truth object visibility fields: {missing}")
    ratio = np.asarray(data["gt_object_visibility_ratio"])
    valid = np.asarray(data["gt_object_visibility_valid"])
    rendered = np.asarray(data["gt_object_rendered_pixels"])
    visible = np.asarray(data["gt_object_visible_pixels"])
    for key, value in (("gt_object_visibility_ratio", ratio), ("gt_object_visibility_valid", valid), ("gt_object_rendered_pixels", rendered), ("gt_object_visible_pixels", visible)):
        if value.shape != expected_shape:
            raise ValueError(f"{label}/{key} must have shape {expected_shape}, got {value.shape}")
    if ratio.dtype != np.dtype("float32"):
        raise TypeError(f"{label}/gt_object_visibility_ratio must be float32, got {ratio.dtype}")
    if valid.dtype != np.dtype("bool"):
        raise TypeError(f"{label}/gt_object_visibility_valid must be bool, got {valid.dtype}")
    for key, value in (("gt_object_rendered_pixels", rendered), ("gt_object_visible_pixels", visible)):
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"{label}/{key} must be integer, got {value.dtype}")
        if value.size and (int(value.min()) < 0 or int(value.max()) > np.iinfo(np.int32).max):
            raise ValueError(f"{label}/{key} contains a count outside int32 range")
    rendered = rendered.astype(np.int32, copy=False)
    visible = visible.astype(np.int32, copy=False)
    if np.any(visible > rendered):
        raise ValueError(f"{label}/gt_object_visible_pixels exceeds rendered pixels")
    if np.any(valid & (rendered <= 0)):
        raise ValueError(f"{label} marks empty ground-truth object renders as valid")
    if np.any(~np.isfinite(ratio[valid])) or np.any((ratio[valid] < 0.0) | (ratio[valid] > 1.0)):
        raise ValueError(f"{label}/gt_object_visibility_ratio contains an invalid value on a valid frame")
    if np.any(~np.isnan(ratio[~valid])):
        raise ValueError(f"{label}/gt_object_visibility_ratio must be NaN when visibility is invalid")
    if np.any(valid):
        expected = visible[valid].astype(np.float64) / rendered[valid].astype(np.float64)
        if not np.allclose(ratio[valid], expected, rtol=0.0, atol=1e-7):
            raise ValueError(f"{label}/gt_object_visibility_ratio does not match visible/rendered pixel counts")
    return {
        "gt_object_visibility_ratio": ratio,
        "gt_object_visibility_valid": valid,
        "gt_object_rendered_pixels": rendered,
        "gt_object_visible_pixels": visible,
    }
