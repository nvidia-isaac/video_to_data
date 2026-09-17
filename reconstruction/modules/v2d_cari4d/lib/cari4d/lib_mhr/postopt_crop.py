from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


POSTOPT_RENDER_SIZE = 256
OBJECT_ONLY_MIN_PIXELS = 200
MASK_THRESHOLD = 127
POSTOPT_PADDING = 1.1
POSTOPT_CROP_CONTRACT = "cari4d.smplh_postopt_full_resolution_crop.v1"


@dataclass(frozen=True)
class PostoptCrop:
    human_mask: np.ndarray
    object_mask: np.ndarray
    keep_mask: np.ndarray
    image_ref: np.ndarray
    K_roi: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    crop_center: np.ndarray
    radius: float
    top_left: np.ndarray
    bottom_right: np.ndarray
    support_source: Literal["object", "human_object"]


def _validate_inputs(human_mask: np.ndarray, object_mask: np.ndarray, K_full: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    human = np.asarray(human_mask)
    obj = np.asarray(object_mask)
    intrinsics = np.asarray(K_full)
    if human.ndim != 2 or obj.ndim != 2 or human.shape != obj.shape:
        raise ValueError(f"human_mask and object_mask must be same-shape 2D arrays, got {human.shape} and {obj.shape}")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"K_full must be a finite 3x3 matrix, got {intrinsics.shape}")
    if human.dtype == np.bool_ or (human.size and np.nanmax(human) <= 1):
        human = human.astype(np.uint8) * 255
    if obj.dtype == np.bool_ or (obj.size and np.nanmax(obj) <= 1):
        obj = obj.astype(np.uint8) * 255
    return human, obj, intrinsics


def _crop_like_public_cari4d(image: np.ndarray, center: np.ndarray, crop_size: float) -> np.ndarray:
    height, width = image.shape[:2]
    top_left = np.round(center - crop_size / 2).astype(int)
    bottom_right = np.round(center + crop_size / 2).astype(int)
    u1, v1 = max(0, top_left[0]), max(0, top_left[1])
    u2, v2 = min(width - 1, bottom_right[0]), min(height - 1, bottom_right[1])
    cropped = image[v1:v2, u1:u2]
    pad_left, pad_top = max(0, -top_left[0]), max(0, -top_left[1])
    pad_right, pad_bottom = max(0, bottom_right[0] - width + 1), max(0, bottom_right[1] - height + 1)
    return np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)))


def build_postopt_crop(human_mask: np.ndarray, object_mask: np.ndarray, K_full: np.ndarray) -> PostoptCrop:
    """Build the public CARI4D SMPL-H post-optimization silhouette crop in OpenCV (u, v) coordinates."""
    human, obj, intrinsics = _validate_inputs(human_mask, object_mask, K_full)
    object_foreground = obj > MASK_THRESHOLD
    if int(object_foreground.sum()) >= OBJECT_ONLY_MIN_PIXELS:
        support = object_foreground
        support_source = "object"
    else:
        support = (human > MASK_THRESHOLD) | object_foreground
        support_source = "human_object"
    v_ids, u_ids = np.where(support)
    if len(u_ids) == 0:
        raise ValueError("public CARI4D post-optimization crop support is empty")
    bbox_min = np.array([u_ids.min(), v_ids.min()])
    bbox_max = np.array([u_ids.max(), v_ids.max()])
    crop_center = (bbox_max + bbox_min) / 2
    radius = float(np.max(bbox_max - bbox_min) * POSTOPT_PADDING / 2)
    if radius <= 0:
        raise ValueError("public CARI4D post-optimization crop support has zero spatial extent")
    top_left = crop_center - radius
    bottom_right = crop_center + radius
    crop_size = float(np.mean(bottom_right - top_left))
    scale = POSTOPT_RENDER_SIZE / crop_size
    focal = np.array([intrinsics[0, 0], intrinsics[1, 1]])
    principal_point = np.array([intrinsics[0, 2], intrinsics[1, 2]])
    focal_roi = focal * scale
    principal_roi = (principal_point - top_left) * scale
    K_roi = np.array([[focal_roi[0], 0, principal_roi[0]], [0, focal_roi[1], principal_roi[1]], [0, 0, 1.0]])
    human_crop = cv2.resize(_crop_like_public_cari4d(human, crop_center, radius * 2), (POSTOPT_RENDER_SIZE, POSTOPT_RENDER_SIZE))
    object_crop = cv2.resize(_crop_like_public_cari4d(obj, crop_center, radius * 2), (POSTOPT_RENDER_SIZE, POSTOPT_RENDER_SIZE))
    person_mask = (human_crop > MASK_THRESHOLD).astype(np.float32)
    obj_mask = (object_crop > MASK_THRESHOLD).astype(np.float32)
    mask_inverse = -(person_mask > 0.5).astype(np.float32)
    mask_inverse[obj_mask > 0.5] = 1.0
    return PostoptCrop(human_mask=person_mask, object_mask=obj_mask, keep_mask=mask_inverse >= 0, image_ref=obj_mask > 0, K_roi=K_roi, bbox_min=bbox_min, bbox_max=bbox_max, crop_center=crop_center, radius=radius, top_left=top_left, bottom_right=bottom_right, support_source=support_source)


def stack_postopt_crops(crops: list[PostoptCrop], K_full: np.ndarray) -> dict[str, np.ndarray | str]:
    if not crops:
        raise ValueError("post-optimization crop sequence is empty")
    intrinsics = np.asarray(K_full, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        intrinsics = np.repeat(intrinsics[None], len(crops), axis=0)
    if intrinsics.shape != (len(crops), 3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"K_full must have shape [3,3] or [{len(crops)},3,3], got {intrinsics.shape}")
    return {
        "postopt_crop_contract": POSTOPT_CROP_CONTRACT,
        "K_full": intrinsics,
        "postopt_K_rois": np.stack([crop.K_roi for crop in crops]).astype(np.float32),
        "postopt_human_mask": np.stack([crop.human_mask for crop in crops]).astype(np.float32),
        "postopt_object_mask": np.stack([crop.object_mask for crop in crops]).astype(np.float32),
        "postopt_keep_mask": np.stack([crop.keep_mask for crop in crops]),
        "postopt_image_ref": np.stack([crop.image_ref for crop in crops]),
        "postopt_crop_bbox_min": np.stack([crop.bbox_min for crop in crops]).astype(np.int32),
        "postopt_crop_bbox_max": np.stack([crop.bbox_max for crop in crops]).astype(np.int32),
        "postopt_crop_center": np.stack([crop.crop_center for crop in crops]).astype(np.float32),
        "postopt_crop_radius": np.asarray([crop.radius for crop in crops], dtype=np.float32),
        "postopt_crop_support_source": np.asarray([crop.support_source for crop in crops], dtype="U12"),
    }
