"""Generic face detections and an OpenCV YuNet implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """One face observation in source-image pixel coordinates."""

    bbox_xywh: np.ndarray
    landmarks: np.ndarray
    score: float
    track_id: int | None = None
    provenance: str = "detected"

    def __post_init__(self) -> None:
        bbox = np.asarray(self.bbox_xywh, dtype=np.float32)
        landmarks = np.asarray(self.landmarks, dtype=np.float32)
        if bbox.shape != (4,):
            raise ValueError(f"bbox_xywh must have shape (4,), got {bbox.shape}")
        if landmarks.shape != (5, 2):
            raise ValueError(
                f"landmarks must have shape (5, 2), got {landmarks.shape}"
            )
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"face width and height must be positive, got {bbox}")
        object.__setattr__(self, "bbox_xywh", bbox)
        object.__setattr__(self, "landmarks", landmarks)
        object.__setattr__(self, "score", float(self.score))

    def with_track(self, track_id: int) -> "FaceDetection":
        return replace(self, track_id=int(track_id))

    def to_json(self) -> dict:
        return {
            "bbox_xywh": [float(value) for value in self.bbox_xywh],
            "landmarks": self.landmarks.astype(float).tolist(),
            "score": self.score,
            "track_id": self.track_id,
            "provenance": self.provenance,
        }


class FaceDetector(Protocol):
    """Detector interface used by multiview orchestration and tests."""

    def detect(self, frame_rgb: np.ndarray) -> list[FaceDetection]:
        """Return all faces in an RGB uint8 frame."""


def bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    lx, ly, lw, lh = np.asarray(left, dtype=np.float32)
    rx, ry, rw, rh = np.asarray(right, dtype=np.float32)
    ix0, iy0 = max(lx, rx), max(ly, ry)
    ix1, iy1 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, float(ix1 - ix0)) * max(0.0, float(iy1 - iy0))
    union = float(lw * lh + rw * rh - intersection)
    return intersection / union if union > 0 else 0.0


def non_max_suppression(
    detections: list[FaceDetection], iou_threshold: float
) -> list[FaceDetection]:
    """Deterministic score-ordered NMS across multiscale detections."""
    kept: list[FaceDetection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if all(
            bbox_iou(detection.bbox_xywh, previous.bbox_xywh) < iou_threshold
            for previous in kept
        ):
            kept.append(detection)
    return kept


class YuNetFaceDetector:
    """OpenCV FaceDetectorYN wrapper with multiscale inference."""

    def __init__(
        self,
        model_path: str,
        *,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        scales: tuple[float, ...] = (1.0, 0.5),
    ) -> None:
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError(f"scales must contain positive values, got {scales}")
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.scales = tuple(float(scale) for scale in scales)
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            int(top_k),
        )

    def detect(self, frame_rgb: np.ndarray) -> list[FaceDetection]:
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise TypeError(
                "YuNet expects an HxWx3 uint8 RGB frame; "
                f"got shape={frame_rgb.shape}, dtype={frame_rgb.dtype}"
            )
        height, width = frame_rgb.shape[:2]
        candidates: list[FaceDetection] = []
        for scale in self.scales:
            scaled_width = max(1, int(round(width * scale)))
            scaled_height = max(1, int(round(height * scale)))
            if (scaled_width, scaled_height) == (width, height):
                scaled_rgb = frame_rgb
            else:
                scaled_rgb = cv2.resize(
                    frame_rgb,
                    (scaled_width, scaled_height),
                    interpolation=cv2.INTER_AREA,
                )
            scaled_bgr = cv2.cvtColor(scaled_rgb, cv2.COLOR_RGB2BGR)
            self._detector.setInputSize((scaled_width, scaled_height))
            _retval, faces = self._detector.detect(scaled_bgr)
            if faces is None:
                continue
            inverse_scale = 1.0 / scale
            for row in np.asarray(faces):
                if float(row[14]) < self.score_threshold:
                    continue
                candidates.append(
                    FaceDetection(
                        bbox_xywh=np.asarray(row[:4]) * inverse_scale,
                        landmarks=np.asarray(row[4:14]).reshape(5, 2)
                        * inverse_scale,
                        score=float(row[14]),
                    )
                )
        return non_max_suppression(candidates, self.nms_threshold)
