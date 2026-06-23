# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ByteTrack multi-object tracker.

Implements the two-stage association strategy from:
    Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.

Key idea: low-confidence detections (from partial occlusion) are used in a second
association pass to maintain tracks that would otherwise be lost.

Uses a Kalman filter with constant-velocity model for bbox state prediction.
"""

from __future__ import annotations

import numpy as np

from .tracker import Track, TrackState, bbox_iou_matrix, hungarian_assign


def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to Kalman measurement [cx, cy, a, h]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2
    cy = bbox[1] + h / 2
    a = w / (h + 1e-6)
    return np.array([cx, cy, a, h])


def _z_to_bbox(z: np.ndarray) -> np.ndarray:
    """Convert Kalman measurement [cx, cy, a, h] to [x1, y1, x2, y2]."""
    w = z[2] * z[3]
    h = z[3]
    return np.array([
        z[0] - w / 2,
        z[1] - h / 2,
        z[0] + w / 2,
        z[1] + h / 2,
    ])


class KalmanBoxTracker:
    """Per-track Kalman filter over [cx, cy, aspect_ratio, h, vcx, vcy, va, vh]."""

    _DIM_Z = 4
    _DIM_X = 8

    def __init__(self, bbox: np.ndarray):
        self.x = np.zeros(self._DIM_X)
        self.x[:self._DIM_Z] = _bbox_to_z(bbox)

        # State transition (constant velocity)
        self.F = np.eye(self._DIM_X)
        for i in range(self._DIM_Z):
            self.F[i, self._DIM_Z + i] = 1.0

        # Measurement matrix
        self.H = np.zeros((self._DIM_Z, self._DIM_X))
        self.H[:self._DIM_Z, :self._DIM_Z] = np.eye(self._DIM_Z)

        # Covariance
        self.P = np.eye(self._DIM_X) * 10.0
        self.P[self._DIM_Z:, self._DIM_Z:] *= 1000.0

        # Process noise
        self.Q = np.eye(self._DIM_X)
        self.Q[self._DIM_Z:, self._DIM_Z:] *= 0.01

        # Measurement noise
        self.R = np.eye(self._DIM_Z)
        self.R[2, 2] *= 10.0  # aspect ratio is less certain

    def predict(self) -> np.ndarray:
        """Advance state and return predicted bbox [x1, y1, x2, y2]."""
        # Prevent negative height
        if self.x[3] + self.x[7] <= 0:
            self.x[7] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_bbox()

    def update(self, bbox: np.ndarray) -> None:
        """Correct state with observed bbox."""
        z = _bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self._DIM_X) - K @ self.H
        self.P = I_KH @ self.P

    def get_bbox(self) -> np.ndarray:
        """Current state as [x1, y1, x2, y2]."""
        return _z_to_bbox(self.x[:self._DIM_Z])


class _STrack:
    """Internal tracked object pairing a Track with its Kalman state."""

    def __init__(self, track: Track, kalman: KalmanBoxTracker):
        self.track = track
        self.kalman = kalman


class ByteTracker:
    """ByteTrack multi-object tracker with two-stage association.

    Args:
        track_thresh: Confidence split between high and low detections.
        match_thresh: IoU threshold for first (high-confidence) association.
        second_match_thresh: IoU threshold for second (low-confidence) association.
        max_lost: Frames a track survives without a match before removal.
        min_hits: Minimum detections before a tentative track is confirmed.
    """

    def __init__(
        self,
        track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        second_match_thresh: float = 0.5,
        max_lost: int = 30,
        min_hits: int = 3,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.max_lost = max_lost
        self.min_hits = min_hits

        self._next_id = 0
        self._active: list[_STrack] = []
        self._lost: list[_STrack] = []
        self._tentative: list[_STrack] = []
        self._finished: list[_STrack] = []

    def _new_strack(self, frame_idx: int, bbox: np.ndarray, score: float) -> _STrack:
        track = Track(track_id=self._next_id, state=TrackState.TENTATIVE)
        track.append(frame_idx, bbox, score)
        kalman = KalmanBoxTracker(bbox)
        self._next_id += 1
        return _STrack(track, kalman)

    def _predicted_bboxes(self, stracks: list[_STrack]) -> np.ndarray:
        if not stracks:
            return np.empty((0, 4))
        return np.array([st.kalman.predict() for st in stracks])

    def update(self, frame_idx: int, bboxes: np.ndarray, scores: np.ndarray):
        """Process one frame of detections (all above det_thresh).

        The caller should pass ALL detections above det_thresh; this method
        splits them into high/low confidence internally.
        """
        # Split detections by confidence
        high_mask = scores >= self.track_thresh
        low_mask = ~high_mask

        high_bboxes = bboxes[high_mask] if high_mask.any() else np.empty((0, 4))
        high_scores = scores[high_mask] if high_mask.any() else np.empty(0)
        low_bboxes = bboxes[low_mask] if low_mask.any() else np.empty((0, 4))
        low_scores = scores[low_mask] if low_mask.any() else np.empty(0)

        # Predict positions for all tracked objects
        established = self._active + self._lost
        pred_bboxes = self._predicted_bboxes(established)

        # --- First association: high-confidence detections vs all tracks ---
        if len(established) > 0 and len(high_bboxes) > 0:
            iou = bbox_iou_matrix(pred_bboxes, high_bboxes)
            matches_1, unmatched_t_1, unmatched_d_1 = hungarian_assign(
                iou, self.match_thresh
            )
        elif len(established) > 0:
            matches_1 = []
            unmatched_t_1 = list(range(len(established)))
            unmatched_d_1 = []
        else:
            matches_1 = []
            unmatched_t_1 = []
            unmatched_d_1 = list(range(len(high_bboxes)))

        new_active: list[_STrack] = []

        for t_idx, d_idx in matches_1:
            st = established[t_idx]
            st.kalman.update(high_bboxes[d_idx])
            st.track.append(frame_idx, high_bboxes[d_idx], float(high_scores[d_idx]))
            new_active.append(st)

        # --- Second association: low-confidence detections vs unmatched tracks ---
        remaining_stracks = [established[i] for i in unmatched_t_1]

        if len(remaining_stracks) > 0 and len(low_bboxes) > 0:
            remaining_pred = np.array([st.kalman.get_bbox() for st in remaining_stracks])
            iou_2 = bbox_iou_matrix(remaining_pred, low_bboxes)
            matches_2, unmatched_t_2, _ = hungarian_assign(
                iou_2, self.second_match_thresh
            )
        elif len(remaining_stracks) > 0:
            matches_2 = []
            unmatched_t_2 = list(range(len(remaining_stracks)))
        else:
            matches_2 = []
            unmatched_t_2 = []

        for t_idx, d_idx in matches_2:
            st = remaining_stracks[t_idx]
            st.kalman.update(low_bboxes[d_idx])
            st.track.append(frame_idx, low_bboxes[d_idx], float(low_scores[d_idx]))
            new_active.append(st)

        # Handle tracks that matched neither stage
        new_lost: list[_STrack] = []
        for t_idx in unmatched_t_2:
            st = remaining_stracks[t_idx]
            st.track.frames_since_match += 1
            if st.track.frames_since_match > self.max_lost:
                st.track.state = TrackState.FINISHED
                self._finished.append(st)
            else:
                st.track.state = TrackState.LOST
                new_lost.append(st)

        # --- Tentative track association with unmatched high-conf detections ---
        unmatched_high_bboxes = high_bboxes[unmatched_d_1] if unmatched_d_1 else np.empty((0, 4))
        unmatched_high_scores = high_scores[unmatched_d_1] if unmatched_d_1 else np.empty(0)

        new_tentative: list[_STrack] = []

        if len(self._tentative) > 0 and len(unmatched_high_bboxes) > 0:
            tent_pred = self._predicted_bboxes(self._tentative)
            iou_3 = bbox_iou_matrix(tent_pred, unmatched_high_bboxes)
            matches_3, unmatched_tent, unmatched_d_3 = hungarian_assign(
                iou_3, self.match_thresh
            )
        elif len(self._tentative) > 0:
            matches_3 = []
            unmatched_tent = list(range(len(self._tentative)))
            unmatched_d_3 = []
        else:
            matches_3 = []
            unmatched_tent = []
            unmatched_d_3 = list(range(len(unmatched_high_bboxes)))

        for t_idx, d_idx in matches_3:
            st = self._tentative[t_idx]
            st.kalman.update(unmatched_high_bboxes[d_idx])
            st.track.append(
                frame_idx, unmatched_high_bboxes[d_idx],
                float(unmatched_high_scores[d_idx]),
            )
            if st.track.length >= self.min_hits:
                st.track.state = TrackState.ACTIVE
                new_active.append(st)
            else:
                new_tentative.append(st)

        for t_idx in unmatched_tent:
            st = self._tentative[t_idx]
            st.track.state = TrackState.FINISHED
            self._finished.append(st)

        # Only high-confidence unmatched detections start new tracks
        for d_idx in unmatched_d_3:
            new_tentative.append(
                self._new_strack(
                    frame_idx,
                    unmatched_high_bboxes[d_idx],
                    float(unmatched_high_scores[d_idx]),
                )
            )

        self._active = new_active
        self._lost = new_lost
        self._tentative = new_tentative

    def finalize(self) -> list[Track]:
        """Return all tracks sorted by track_id."""
        all_stracks = (
            self._finished + self._active + self._lost + self._tentative
        )
        tracks = [st.track for st in all_stracks]
        tracks.sort(key=lambda t: t.track_id)
        return tracks
