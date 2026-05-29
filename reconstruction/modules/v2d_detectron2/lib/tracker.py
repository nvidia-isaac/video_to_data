# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from scipy.optimize import linear_sum_assignment

from v2d.mv.math.numpy_fn import linear_one_euro_filter


class TrackState(Enum):
    TENTATIVE = auto()
    ACTIVE = auto()
    LOST = auto()
    FINISHED = auto()


@dataclass
class Track:
    track_id: int
    state: TrackState = TrackState.ACTIVE
    frames_since_match: int = 0
    history: list[tuple[int, np.ndarray, float]] = field(default_factory=list)

    @property
    def first_bbox(self) -> np.ndarray:
        return self.history[0][1]

    @property
    def last_bbox(self) -> np.ndarray:
        return self.history[-1][1]

    @property
    def last_score(self) -> float:
        return self.history[-1][2]

    @property
    def start_frame(self) -> int:
        return self.history[0][0]

    @property
    def end_frame(self) -> int:
        return self.history[-1][0]

    @property
    def length(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:
        motion = self.total_motion()
        return (
            f"Track(id={self.track_id}, frames={self.length}, "
            f"span={self.start_frame}-{self.end_frame}, "
            f"motion={motion:.0f}px, state={self.state.name})"
        )

    def append(self, frame_idx: int, bbox: np.ndarray, score: float):
        self.history.append((frame_idx, bbox, score))
        self.state = TrackState.ACTIVE
        self.frames_since_match = 0

    def get_bboxes(self) -> np.ndarray:
        """Return (N, 4) array of all bboxes in this track."""
        return np.array([h[1] for h in self.history])

    def get_scores(self) -> np.ndarray:
        return np.array([h[2] for h in self.history])

    def get_frame_indices(self) -> np.ndarray:
        return np.array([h[0] for h in self.history])

    def interpolate(self, total_frames: int):
        """Fill gaps in-place with linearly interpolated bboxes (score=-1).

        After calling, self.history has exactly total_frames entries
        covering frames 0..total_frames-1. Interpolated entries use score=-1
        to distinguish them from real detections. Frames before the first or
        after the last observation are held (nearest bbox, score=-1).
        """
        frame_indices = self.get_frame_indices()
        bboxes = self.get_bboxes()
        scores = self.get_scores()

        observed = {int(f): (b, s) for f, b, s in zip(frame_indices, bboxes, scores)}
        first, last = int(frame_indices[0]), int(frame_indices[-1])

        new_history: list[tuple[int, np.ndarray, float]] = []

        for f in range(total_frames):
            if f in observed:
                new_history.append((f, observed[f][0], float(observed[f][1])))
            elif f < first:
                new_history.append((f, bboxes[0].copy(), -1.0))
            elif f > last:
                new_history.append((f, bboxes[-1].copy(), -1.0))
            else:
                # Find surrounding observations
                lo = max(fi for fi in frame_indices if fi < f)
                hi = min(fi for fi in frame_indices if fi > f)
                t = (f - lo) / (hi - lo)
                bbox = observed[lo][0] * (1 - t) + observed[hi][0] * t
                new_history.append((f, bbox, -1.0))

        self.history = new_history

    def _smoothed_centers(self, f_cutoff: float = 5.0, f_sample: float = 30.0) -> np.ndarray:
        """Bbox centers low-passed by a One Euro filter with beta=0 (plain low-pass).

        f_cutoff defaults high (5 Hz at 30 fps) so only very-high-frequency noise —
        e.g. bbox flicker from intermittent occlusion — is removed; real person
        motion (sub-Hz) passes through.
        """
        bboxes = self.get_bboxes()
        centers = np.column_stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2,
            (bboxes[:, 1] + bboxes[:, 3]) / 2,
        ])
        if centers.shape[0] < 2:
            return centers
        # linear_one_euro_filter expects time on the last axis -> (2, T)
        smoothed = linear_one_euro_filter(
            centers.T, min_f_cutoff=f_cutoff, beta=0.0, f_sample=f_sample,
        ).T
        return smoothed

    def total_motion(self) -> float:
        """Sum of frame-to-frame bbox center displacements (in pixels), after smoothing.

        Centers are low-passed first so occlusion-induced bbox flicker doesn't inflate
        the score. A truly stationary (but jittery) track yields ~0.
        """
        if self.length < 2:
            return 0.0
        centers = self._smoothed_centers()
        deltas = np.diff(centers, axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    def trajectory_span(self) -> float:
        """Diagonal of the axis-aligned bbox enclosing (smoothed) bbox centers.

        Unlike total_motion, this is immune to back-and-forth flicker: a track that
        jitters in one spot has near-zero span regardless of cumulative motion.
        """
        if self.length < 1:
            return 0.0
        centers = self._smoothed_centers()
        return float(np.hypot(np.ptp(centers[:, 0]), np.ptp(centers[:, 1])))



def bbox_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of [x1, y1, x2, y2] boxes.

    Returns shape (len(boxes_a), len(boxes_b)).
    """
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / (union + 1e-6)


def greedy_assign(
    iou_matrix: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy matching by descending IoU.

    Returns (matches, unmatched_track_indices, unmatched_det_indices).
    """
    if iou_matrix.size == 0:
        return (
            [],
            list(range(iou_matrix.shape[0])),
            list(range(iou_matrix.shape[1])),
        )

    matches = []
    matched_tracks = set()
    matched_dets = set()

    flat_order = np.argsort(iou_matrix.ravel())[::-1]
    n_cols = iou_matrix.shape[1]

    for idx in flat_order:
        t = int(idx // n_cols)
        d = int(idx % n_cols)
        if iou_matrix[t, d] < threshold:
            break
        if t in matched_tracks or d in matched_dets:
            continue
        matches.append((t, d))
        matched_tracks.add(t)
        matched_dets.add(d)

    unmatched_tracks = [i for i in range(iou_matrix.shape[0]) if i not in matched_tracks]
    unmatched_dets = [i for i in range(iou_matrix.shape[1]) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def hungarian_assign(
    iou_matrix: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Optimal matching via the Hungarian algorithm on an IoU cost matrix.

    Returns (matches, unmatched_track_indices, unmatched_det_indices).
    """
    if iou_matrix.size == 0:
        return (
            [],
            list(range(iou_matrix.shape[0])),
            list(range(iou_matrix.shape[1])),
        )

    cost = 1.0 - iou_matrix
    row_indices, col_indices = linear_sum_assignment(cost)

    matches = []
    matched_tracks = set()
    matched_dets = set()
    for r, c in zip(row_indices, col_indices):
        if iou_matrix[r, c] >= threshold:
            matches.append((int(r), int(c)))
            matched_tracks.add(int(r))
            matched_dets.add(int(c))

    unmatched_tracks = [i for i in range(iou_matrix.shape[0]) if i not in matched_tracks]
    unmatched_dets = [i for i in range(iou_matrix.shape[1]) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


class IoUTracker:
    """Simple IoU-based multi-object tracker with lost-track buffer.

    Args:
        iou_threshold: Minimum IoU to match a detection to a track.
        max_lost: Frames a track survives without a match before removal.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost: int = 30,
        min_hits: int = 3,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits
        self._next_id = 0
        self.active_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.tentative_tracks: list[Track] = []
        self.finished_tracks: list[Track] = []

    def _new_track(self, frame_idx: int, bbox: np.ndarray, score: float) -> Track:
        track = Track(track_id=self._next_id, state=TrackState.TENTATIVE)
        track.append(frame_idx, bbox, score)
        self._next_id += 1
        return track

    def update(self, frame_idx: int, bboxes: np.ndarray, scores: np.ndarray):
        """Process one frame of detections.

        Two-stage matching:
        1. Match detections against active + lost tracks (established tracks
           get priority).
        2. Match remaining detections against tentative tracks.
        Unmatched detections from stage 2 create new tentative tracks.
        Tentative tracks that reach *min_hits* consecutive matches are promoted
        to active.

        Args:
            frame_idx: Current frame number.
            bboxes: (M, 4) array of [x1, y1, x2, y2] detections.
            scores: (M,) confidence scores.
        """
        # Stage 1: match against established (active + lost) tracks
        established = self.active_tracks + self.lost_tracks

        if len(established) > 0 and len(bboxes) > 0:
            track_boxes = np.array([t.last_bbox for t in established])
            iou = bbox_iou_matrix(track_boxes, bboxes)
            matches_1, unmatched_t_1, unmatched_d_1 = greedy_assign(iou, self.iou_threshold)
        elif len(established) > 0:
            matches_1, unmatched_t_1, unmatched_d_1 = [], list(range(len(established))), []
        else:
            matches_1, unmatched_t_1, unmatched_d_1 = [], [], list(range(len(bboxes)))

        new_active = []
        new_lost = []

        for t_idx, d_idx in matches_1:
            track = established[t_idx]
            track.append(frame_idx, bboxes[d_idx], scores[d_idx])
            new_active.append(track)

        for t_idx in unmatched_t_1:
            track = established[t_idx]
            track.frames_since_match += 1
            if track.frames_since_match > self.max_lost:
                track.state = TrackState.FINISHED
                self.finished_tracks.append(track)
            else:
                track.state = TrackState.LOST
                new_lost.append(track)

        # Stage 2: match remaining detections against tentative tracks
        remaining_bboxes = bboxes[unmatched_d_1] if len(unmatched_d_1) > 0 else np.empty((0, 4))
        remaining_scores = scores[unmatched_d_1] if len(unmatched_d_1) > 0 else np.empty(0)

        new_tentative = []

        if len(self.tentative_tracks) > 0 and len(remaining_bboxes) > 0:
            tent_boxes = np.array([t.last_bbox for t in self.tentative_tracks])
            iou_2 = bbox_iou_matrix(tent_boxes, remaining_bboxes)
            matches_2, unmatched_t_2, unmatched_d_2 = greedy_assign(iou_2, self.iou_threshold)
        elif len(self.tentative_tracks) > 0:
            matches_2, unmatched_t_2, unmatched_d_2 = [], list(range(len(self.tentative_tracks))), []
        else:
            matches_2, unmatched_t_2, unmatched_d_2 = [], [], list(range(len(remaining_bboxes)))

        for t_idx, d_idx in matches_2:
            track = self.tentative_tracks[t_idx]
            track.append(frame_idx, remaining_bboxes[d_idx], float(remaining_scores[d_idx]))
            if track.length >= self.min_hits:
                track.state = TrackState.ACTIVE
                new_active.append(track)
            else:
                new_tentative.append(track)

        # Unmatched tentative tracks are dropped immediately
        for t_idx in unmatched_t_2:
            track = self.tentative_tracks[t_idx]
            track.state = TrackState.FINISHED
            self.finished_tracks.append(track)

        # Unmatched detections start new tentative tracks
        for d_idx in unmatched_d_2:
            new_tentative.append(
                self._new_track(frame_idx, remaining_bboxes[d_idx], float(remaining_scores[d_idx]))
            )

        self.active_tracks = new_active
        self.lost_tracks = new_lost
        self.tentative_tracks = new_tentative

    def finalize(self) -> list[Track]:
        """Return all tracks (active + lost + tentative + finished), sorted by track_id."""
        all_tracks = (
            self.finished_tracks + self.active_tracks
            + self.lost_tracks + self.tentative_tracks
        )
        all_tracks.sort(key=lambda t: t.track_id)
        return all_tracks

    @staticmethod
    def _track_merge_score(a: Track, b: Track) -> float:
        """Compute merge affinity between two tracks.

        For overlapping tracks: mean IoU across shared frames.
        For non-overlapping tracks: IoU between A's last bbox and B's first bbox.
        Returns -1 if tracks have no overlapping frames and no observations to compare.
        """
        a_by_frame = {int(h[0]): h[1] for h in a.history}
        b_by_frame = {int(h[0]): h[1] for h in b.history}
        shared_frames = sorted(set(a_by_frame) & set(b_by_frame))

        if shared_frames:
            a_boxes = np.array([a_by_frame[f] for f in shared_frames])
            b_boxes = np.array([b_by_frame[f] for f in shared_frames])
            ious = bbox_iou_matrix(a_boxes, b_boxes)
            return float(np.mean(np.diag(ious)))

        return float(bbox_iou_matrix(
            a.last_bbox.reshape(1, 4),
            b.first_bbox.reshape(1, 4),
        )[0, 0])

    @staticmethod
    def merge_fragmented_tracks(
        tracks: list[Track],
        max_gap: int = 10,
        iou_threshold: float = 0.5,
    ) -> list[Track]:
        """Merge tracks that likely represent the same object across ID switches.

        Two tracks are merged when:
        - Track A ends within *max_gap* frames of track B starting (or they overlap)
        - Their merge score (mean IoU over shared frames, or boundary IoU
          if no overlap) >= *iou_threshold*

        Merging is greedy: the best pair (highest score) is merged first, then
        the search repeats until no more pairs qualify.

        Returns a new list of tracks (merged tracks get the earlier track's ID).
        """
        tracks = list(tracks)
        changed = True
        while changed:
            changed = False
            tracks.sort(key=lambda t: t.start_frame)
            best_score = -1.0
            best_pair = None
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    a, b = tracks[i], tracks[j]
                    gap = b.start_frame - a.end_frame
                    if gap > max_gap:
                        continue
                    score = IoUTracker._track_merge_score(a, b)
                    if score >= iou_threshold and score > best_score:
                        best_score = score
                        best_pair = (i, j)

            if best_pair is not None:
                i, j = best_pair
                a, b = tracks[i], tracks[j]
                # For overlapping frames, keep A's observation (earlier track)
                b_frames = {int(h[0]) for h in b.history}
                a_frames = {int(h[0]) for h in a.history}
                a.history.extend(h for h in b.history if int(h[0]) not in a_frames)
                a.history.sort(key=lambda h: h[0])
                tracks.pop(j)
                changed = True

        return tracks

    @staticmethod
    def select_primary_track(
        tracks: list[Track],
        image_size: tuple[int, int] | None = None,
        method: str = "combined",
        weights: dict[str, float] | None = None,
    ) -> Track:
        """Select the primary subject from a set of tracks.

        Args:
            tracks: List of completed tracks.
            image_size: (height, width) of the frames, needed for center-based methods.
            method:
                "longest"  — track with the most frames.
                "center"   — track whose avg bbox center is closest to image center.
                "motion"   — track with the most total bbox-center displacement.
                "combined" — weighted sum of normalized length, center proximity,
                             motion, and trajectory span. Tune via *weights* dict.
            weights: Weights for "combined" mode.
                Keys: "length", "center", "motion", "span".
                Defaults to {"length": 0.2, "center": 0.2, "motion": 0.3, "span": 0.3}.
        """
        if not tracks:
            raise ValueError("No tracks to select from")

        if method == "longest":
            return max(tracks, key=lambda t: t.length)

        if method == "motion":
            return max(tracks, key=lambda t: t.total_motion())

        if method in ("center", "combined"):
            if image_size is None:
                raise ValueError("image_size required for center-based selection")

        h, w = image_size
        img_cx, img_cy = w / 2, h / 2

        def center_dist(t: Track) -> float:
            bboxes = t.get_bboxes()
            cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
            cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
            return float(np.mean(np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)))

        if method == "center":
            return min(tracks, key=center_dist)

        if method == "combined":
            w_cfg = weights or {"length": 0.2, "center": 0.2, "motion": 0.3, "span": 0.3}

            lengths = np.array([t.length for t in tracks], dtype=float)
            dists = np.array([center_dist(t) for t in tracks], dtype=float)
            motions = np.array([t.total_motion() for t in tracks], dtype=float)
            spans = np.array([t.trajectory_span() for t in tracks], dtype=float)

            def normalize(a: np.ndarray) -> np.ndarray:
                r = a.max() - a.min()
                return (a - a.min()) / r if r > 0 else np.zeros_like(a)

            scores = (
                w_cfg.get("length", 0) * normalize(lengths)
                + w_cfg.get("center", 0) * (1.0 - normalize(dists))
                + w_cfg.get("motion", 0) * normalize(motions)
                + w_cfg.get("span", 0) * normalize(spans)
            )
            return tracks[int(np.argmax(scores))]

        raise ValueError(f"Unknown selection method: {method}")
