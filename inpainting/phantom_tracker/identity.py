"""Explicit bimanual identity assignment for Phantom's DINO detections.

Phantom's non-EPIC path selects one top-scoring ``a hand`` box and assigns it
to the configured target side.  TACO is bimanual, so silently running that
path twice would often feed the same box to both hands.  This module extends
the policy deterministically: the calibrated TACO view initializes image-left
as anatomical left, temporal proximity preserves the labels, and geometrically
ambiguous frames are rejected instead of being guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np


SIDES = ("left", "right")


def box_centers(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return (boxes[:, :2] + boxes[:, 2:]) * 0.5


def filter_box_area(
    boxes: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
    *,
    minimum_fraction: float,
    maximum_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject boxes whose area is implausible for a hand in the full frame.

    Grounding DINO sometimes labels the union of both arms and the torso as
    ``a hand``. Phantom's original single-hand top-score rule accepts that box;
    the explicit area gate prevents it from silently becoming one side of a
    bimanual track.
    """

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not 0.0 <= minimum_fraction < maximum_fraction <= 1.0:
        raise ValueError("box area fractions must satisfy 0 <= min < max <= 1")
    if not len(boxes):
        return boxes, scores
    area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    fraction = area / float(width * height)
    keep = (fraction >= minimum_fraction) & (fraction <= maximum_fraction)
    return boxes[keep], scores[keep]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Return deterministic score-descending NMS indices."""

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if not len(boxes):
        return np.empty(0, dtype=np.int64)
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("boxes and scores must be finite")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("boxes must have positive width and height")

    # Stable tie break: lower original index wins equal scores.
    order = np.lexsort((np.arange(len(scores)), -scores))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        rest = order[1:]
        if not rest.size:
            break
        x0 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y0 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y1 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[iou <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


@dataclass(frozen=True)
class Assignment:
    """One frame's anatomical side assignment."""

    indices: dict[str, int]
    ambiguous: bool
    reason: str


class BimanualIdentityTracker:
    """Track anatomical identities without consulting ground truth.

    The initial TACO horizontal convention is fixed and explicit: image-left
    is anatomical left. Once both identities have been observed,
    minimum-displacement association is used.
    A small best-vs-second-best gap is treated as ambiguous.
    """

    def __init__(
        self,
        *,
        initialization_separation_ratio: float = 0.08,
        ambiguity_margin_ratio: float = 0.04,
        max_jump_ratio: float = 0.45,
    ) -> None:
        if initialization_separation_ratio <= 0:
            raise ValueError("initialization_separation_ratio must be positive")
        if ambiguity_margin_ratio <= 0:
            raise ValueError("ambiguity_margin_ratio must be positive")
        if max_jump_ratio <= 0:
            raise ValueError("max_jump_ratio must be positive")
        self.initialization_separation_ratio = initialization_separation_ratio
        self.ambiguity_margin_ratio = ambiguity_margin_ratio
        self.max_jump_ratio = max_jump_ratio
        self._last: dict[str, np.ndarray] = {}

    @property
    def last_centers(self) -> dict[str, np.ndarray]:
        return {side: center.copy() for side, center in self._last.items()}

    def assign(self, boxes: np.ndarray, width: int, height: int) -> Assignment:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if len(boxes) == 0:
            return Assignment({}, False, "no_detection")
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("boxes must be finite xyxy boxes with positive area")

        centers = box_centers(boxes)
        diagonal = float(np.hypot(width, height))
        ambiguity_margin = self.ambiguity_margin_ratio * diagonal
        max_jump = self.max_jump_ratio * diagonal

        if len(self._last) < 2:
            result = self._initialize(centers, width, ambiguity_margin)
        else:
            candidates: list[tuple[float, int, int]] = []
            for left_idx, right_idx in permutations(range(len(centers)), 2):
                cost = float(
                    np.linalg.norm(centers[left_idx] - self._last["left"])
                    + np.linalg.norm(centers[right_idx] - self._last["right"])
                )
                candidates.append((cost, left_idx, right_idx))
            if not candidates:  # one detection
                result = self._assign_single(centers[0], max_jump, ambiguity_margin)
            else:
                candidates.sort(key=lambda item: (item[0], item[1], item[2]))
                best = candidates[0]
                second = candidates[1] if len(candidates) > 1 else None
                individual_jumps = (
                    np.linalg.norm(centers[best[1]] - self._last["left"]),
                    np.linalg.norm(centers[best[2]] - self._last["right"]),
                )
                if max(individual_jumps) > max_jump:
                    result = Assignment({}, True, "temporal_jump_exceeds_limit")
                elif second is not None and second[0] - best[0] <= ambiguity_margin:
                    result = Assignment({}, True, "temporal_assignment_ambiguous")
                else:
                    result = Assignment(
                        {"left": int(best[1]), "right": int(best[2])},
                        False,
                        "temporal_assignment",
                    )

        if not result.ambiguous:
            for side, index in result.indices.items():
                self._last[side] = centers[index].copy()
        return result

    def _initialize(
        self, centers: np.ndarray, width: int, ambiguity_margin: float
    ) -> Assignment:
        if len(centers) == 1:
            center_x = float(centers[0, 0])
            center_margin = self.initialization_separation_ratio * width
            if center_x < width * 0.5 - center_margin:
                return Assignment({"left": 0}, False, "taco_horizontal_initialization")
            if center_x > width * 0.5 + center_margin:
                return Assignment({"right": 0}, False, "taco_horizontal_initialization")
            return Assignment({}, True, "single_detection_near_center")

        # The caller supplies confidence-descending, NMS-filtered boxes. Use
        # exactly its top two for initialization so a low-score distant false
        # positive cannot win merely by being far from the real hands.
        first, second = 0, 1
        if centers[first, 0] <= centers[second, 0]:
            image_left, image_right = first, second
        else:
            image_left, image_right = second, first
        separation = float(centers[image_right, 0] - centers[image_left, 0])
        minimum = max(self.initialization_separation_ratio * width, ambiguity_margin)
        if separation <= minimum:
            return Assignment({}, True, "initial_hands_not_horizontally_separated")
        return Assignment(
            {"left": int(image_left), "right": int(image_right)},
            False,
            "taco_horizontal_initialization",
        )

    def _assign_single(
        self, center: np.ndarray, max_jump: float, ambiguity_margin: float
    ) -> Assignment:
        distances = {
            side: float(np.linalg.norm(center - self._last[side])) for side in SIDES
        }
        ordered = sorted(distances.items(), key=lambda item: (item[1], item[0]))
        if ordered[0][1] > max_jump:
            return Assignment({}, True, "single_detection_jump_exceeds_limit")
        if ordered[1][1] - ordered[0][1] <= ambiguity_margin:
            return Assignment({}, True, "single_detection_identity_ambiguous")
        return Assignment({ordered[0][0]: 0}, False, "single_temporal_assignment")
