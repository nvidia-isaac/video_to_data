"""Offline temporal association and robust smoothing for face detections."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

try:
    from .face_detection import FaceDetection, bbox_iou
except ImportError:  # Support direct imports in lightweight tests.
    from face_detection import FaceDetection, bbox_iou


def _normalized_center_distance(left: FaceDetection, right: FaceDetection) -> float:
    left_center = left.bbox_xywh[:2] + left.bbox_xywh[2:] / 2
    right_center = right.bbox_xywh[:2] + right.bbox_xywh[2:] / 2
    scale = max(float(np.linalg.norm(left.bbox_xywh[2:])), 1.0)
    return float(np.linalg.norm(left_center - right_center) / scale)


def associate_tracks(
    frames: list[list[FaceDetection]],
    *,
    max_gap_frames: int = 5,
    min_iou: float = 0.2,
    max_center_distance: float = 0.75,
) -> tuple[list[list[FaceDetection]], dict[int, list[FaceDetection | None]]]:
    """Greedily associate observations while retaining every detection."""
    n_frames = len(frames)
    tracks: dict[int, list[FaceDetection | None]] = {}
    last_seen: dict[int, tuple[int, FaceDetection]] = {}
    assigned_frames: list[list[FaceDetection]] = [[] for _ in frames]
    next_track_id = 0

    for frame_index, detections in enumerate(frames):
        candidates: list[tuple[float, int, int]] = []
        for track_id, (previous_index, previous) in last_seen.items():
            if frame_index - previous_index > max_gap_frames + 1:
                continue
            for detection_index, detection in enumerate(detections):
                iou = bbox_iou(previous.bbox_xywh, detection.bbox_xywh)
                distance = _normalized_center_distance(previous, detection)
                if iou >= min_iou or distance <= max_center_distance:
                    quality = max(iou, 1.0 - distance)
                    candidates.append((quality, track_id, detection_index))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        assignments: dict[int, int] = {}
        for _quality, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id in used_tracks or detection_index in used_detections:
                continue
            assignments[detection_index] = track_id
            used_tracks.add(track_id)
            used_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = [None] * n_frames
            tracked = detection.with_track(track_id)
            tracks[track_id][frame_index] = tracked
            last_seen[track_id] = (frame_index, tracked)
            assigned_frames[frame_index].append(tracked)

    return assigned_frames, tracks


def _to_vector(detection: FaceDetection) -> np.ndarray:
    x, y, width, height = detection.bbox_xywh
    center = np.array([x + width / 2, y + height / 2], dtype=np.float32)
    return np.concatenate([center, [width, height], detection.landmarks.reshape(-1)])


def _from_vector(
    vector: np.ndarray,
    *,
    score: float,
    track_id: int,
    provenance: str,
) -> FaceDetection:
    center_x, center_y, width, height = vector[:4]
    bbox = np.array(
        [center_x - width / 2, center_y - height / 2, width, height],
        dtype=np.float32,
    )
    return FaceDetection(
        bbox_xywh=bbox,
        landmarks=vector[4:].reshape(5, 2),
        score=score,
        track_id=track_id,
        provenance=provenance,
    )


def _fill_track(
    values: list[FaceDetection | None],
    *,
    max_gap_frames: int,
    edge_fill_frames: int,
) -> list[FaceDetection | None]:
    filled = list(values)
    observed = [index for index, value in enumerate(values) if value is not None]
    if not observed:
        return filled

    first, last = observed[0], observed[-1]
    first_value = values[first]
    last_value = values[last]
    for index in range(max(0, first - edge_fill_frames), first):
        filled[index] = replace(first_value, provenance="interpolated")
    for index in range(last + 1, min(len(values), last + edge_fill_frames + 1)):
        filled[index] = replace(last_value, provenance="interpolated")

    for left_index, right_index in zip(observed, observed[1:]):
        gap = right_index - left_index - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        left, right = values[left_index], values[right_index]
        left_vector, right_vector = _to_vector(left), _to_vector(right)
        for offset in range(1, gap + 1):
            ratio = offset / (gap + 1)
            vector = left_vector * (1.0 - ratio) + right_vector * ratio
            filled[left_index + offset] = _from_vector(
                vector,
                score=min(left.score, right.score),
                track_id=int(left.track_id),
                provenance="interpolated",
            )
    return filled


def _smooth_track(
    values: list[FaceDetection | None], *, median_window: int, ema_alpha: float
) -> list[FaceDetection | None]:
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError("median_window must be a positive odd integer")
    radius = median_window // 2
    median_vectors: list[np.ndarray | None] = [None] * len(values)
    for index, value in enumerate(values):
        if value is None:
            continue
        neighbors = [
            _to_vector(values[neighbor])
            for neighbor in range(max(0, index - radius), min(len(values), index + radius + 1))
            if values[neighbor] is not None
        ]
        median_vector = np.median(np.stack(neighbors), axis=0)
        current = _to_vector(value)
        # Smoothing must never reduce face coverage from the current observation.
        median_vector[2:4] = np.maximum(median_vector[2:4], current[2:4])
        median_vectors[index] = median_vector

    smoothed: list[FaceDetection | None] = [None] * len(values)
    previous: np.ndarray | None = None
    for index, (value, median_vector) in enumerate(zip(values, median_vectors)):
        if value is None:
            previous = None
            continue
        vector = median_vector if previous is None else (
            ema_alpha * median_vector + (1.0 - ema_alpha) * previous
        )
        current = _to_vector(value)
        vector[2:4] = np.maximum(vector[2:4], current[2:4])
        previous = vector.copy()
        smoothed[index] = _from_vector(
            vector,
            score=value.score,
            track_id=int(value.track_id),
            provenance=value.provenance,
        )
    return smoothed


def filter_detections(
    frames: list[list[FaceDetection]],
    *,
    max_gap_frames: int = 5,
    edge_fill_frames: int = 2,
    min_iou: float = 0.2,
    max_center_distance: float = 0.75,
    median_window: int = 3,
    ema_alpha: float = 0.6,
) -> tuple[list[list[FaceDetection]], list[list[FaceDetection]]]:
    """Return track-labelled raw observations and filtered observations."""
    raw_frames, tracks = associate_tracks(
        frames,
        max_gap_frames=max_gap_frames,
        min_iou=min_iou,
        max_center_distance=max_center_distance,
    )
    filtered_frames: list[list[FaceDetection]] = [[] for _ in frames]
    for track_id, values in tracks.items():
        filled = _fill_track(
            values,
            max_gap_frames=max_gap_frames,
            edge_fill_frames=edge_fill_frames,
        )
        smoothed = _smooth_track(
            filled,
            median_window=median_window,
            ema_alpha=ema_alpha,
        )
        for frame_index, detection in enumerate(smoothed):
            if detection is not None:
                filtered_frames[frame_index].append(detection)
    for detections in filtered_frames:
        detections.sort(key=lambda detection: int(detection.track_id))
    return raw_frames, filtered_frames
