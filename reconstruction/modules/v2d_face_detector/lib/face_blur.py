"""Face-local Gaussian blurring with landmark-oriented feathered masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from .face_detection import FaceDetection
except ImportError:  # Support direct imports in lightweight tests.
    from face_detection import FaceDetection


_FILTER_SUPPORT_SIGMAS = 3.0


@dataclass(frozen=True)
class _FaceBlurRegion:
    x0: int
    y0: int
    x1: int
    y1: int
    mask: np.ndarray
    center_global: tuple[float, float]
    center_local: tuple[int, int]
    axes: tuple[int, int]
    angle: float
    feather_sigma: float
    blur_sigma: float


def _build_face_blur_region(
    frame_shape: tuple[int, ...],
    detection: FaceDetection,
    *,
    ellipse_scale: float,
    feather_fraction: float,
    blur_sigma_fraction: float,
) -> _FaceBlurRegion | None:
    """Build an unclipped ellipse on a filter-padded, frame-clipped canvas."""
    frame_height, frame_width = frame_shape[:2]
    x, y, box_width, box_height = detection.bbox_xywh.astype(float)
    center_global = (x + box_width / 2.0, y + box_height / 2.0)
    axes = (
        max(1, int(round(box_width * 0.5 * ellipse_scale))),
        max(1, int(round(box_height * 0.5 * ellipse_scale))),
    )

    right_eye, left_eye = detection.landmarks[:2]
    eye_delta = left_eye - right_eye
    angle = float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0])))
    angle_radians = np.deg2rad(angle)
    cos_angle = float(np.cos(angle_radians))
    sin_angle = float(np.sin(angle_radians))
    extent_x = float(
        np.hypot(axes[0] * cos_angle, axes[1] * sin_angle)
    )
    extent_y = float(
        np.hypot(axes[0] * sin_angle, axes[1] * cos_angle)
    )

    feather_sigma = max(0.5, min(box_width, box_height) * feather_fraction)
    blur_sigma = max(1.0, max(box_width, box_height) * blur_sigma_fraction)
    padding = int(
        np.ceil(_FILTER_SUPPORT_SIGMAS * max(feather_sigma, blur_sigma))
    )
    x0 = max(0, int(np.floor(center_global[0] - extent_x)) - padding)
    x1 = min(
        frame_width,
        int(np.ceil(center_global[0] + extent_x)) + padding + 1,
    )
    y0 = max(0, int(np.floor(center_global[1] - extent_y)) - padding)
    y1 = min(
        frame_height,
        int(np.ceil(center_global[1] + extent_y)) + padding + 1,
    )
    if x1 <= x0 or y1 <= y0:
        return None

    center_local = (
        int(round(center_global[0] - x0)),
        int(round(center_global[1] - y0)),
    )
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    cv2.ellipse(
        mask,
        center_local,
        axes,
        angle,
        0,
        360,
        1.0,
        thickness=-1,
    )
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_sigma)
    mask = np.clip(mask, 0.0, 1.0)
    return _FaceBlurRegion(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        mask=mask,
        center_global=center_global,
        center_local=center_local,
        axes=axes,
        angle=angle,
        feather_sigma=feather_sigma,
        blur_sigma=blur_sigma,
    )


def _blur_one_face(
    frame: np.ndarray,
    detection: FaceDetection,
    *,
    ellipse_scale: float,
    feather_fraction: float,
    blur_sigma_fraction: float,
) -> np.ndarray:
    region = _build_face_blur_region(
        frame.shape,
        detection,
        ellipse_scale=ellipse_scale,
        feather_fraction=feather_fraction,
        blur_sigma_fraction=blur_sigma_fraction,
    )
    if region is None:
        return frame

    roi = frame[region.y0:region.y1, region.x0:region.x1]
    mask = region.mask[..., None]
    blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=region.blur_sigma)
    blended = np.rint(
        roi.astype(np.float32) * (1.0 - mask)
        + blurred.astype(np.float32) * mask
    ).astype(frame.dtype)
    output = frame.copy()
    output[region.y0:region.y1, region.x0:region.x1] = blended
    return output


def blur_faces(
    frame_rgb: np.ndarray,
    detections: list[FaceDetection],
    *,
    ellipse_scale: float = 1.0,
    feather_fraction: float = 0.12,
    blur_sigma_fraction: float = 0.18,
) -> np.ndarray:
    """Blur every detection while preserving pixels outside padded mask support."""
    if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise TypeError(
            "blur_faces expects an HxWx3 uint8 RGB frame; "
            f"got shape={frame_rgb.shape}, dtype={frame_rgb.dtype}"
        )
    if not 0.0 < ellipse_scale <= 1.0:
        raise ValueError(
            f"ellipse_scale must be in the interval (0, 1], got {ellipse_scale}"
        )
    output = frame_rgb.copy()
    for detection in detections:
        output = _blur_one_face(
            output,
            detection,
            ellipse_scale=ellipse_scale,
            feather_fraction=feather_fraction,
            blur_sigma_fraction=blur_sigma_fraction,
        )
    return output
