from __future__ import annotations

import unittest

import numpy as np

from inpainting.phantom_tracker.identity import BimanualIdentityTracker, filter_box_area, nms


class NmsTests(unittest.TestCase):
    def test_suppresses_duplicate_and_keeps_stable_score_order(self) -> None:
        boxes = np.array(
            [[0, 0, 20, 20], [1, 1, 21, 21], [80, 0, 100, 20]], dtype=float
        )
        scores = np.array([0.9, 0.8, 0.9])
        np.testing.assert_array_equal(nms(boxes, scores, 0.5), [0, 2])

    def test_rejects_invalid_box(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive width"):
            nms(np.array([[1, 1, 1, 2]]), np.array([0.5]), 0.5)

    def test_area_filter_rejects_union_of_both_arms(self) -> None:
        boxes, scores = filter_box_area(
            np.array([[0, 0, 900, 900], [100, 100, 300, 300]], dtype=float),
            np.array([0.9, 0.5]),
            1000,
            1000,
            minimum_fraction=0.001,
            maximum_fraction=0.12,
        )
        np.testing.assert_array_equal(boxes, [[100, 100, 300, 300]])
        np.testing.assert_array_equal(scores, [0.5])


class IdentityTests(unittest.TestCase):
    def test_initial_taco_mapping_is_explicit(self) -> None:
        tracker = BimanualIdentityTracker()
        assignment = tracker.assign(
            np.array([[50, 50, 150, 150], [850, 50, 950, 150]]), 1000, 500
        )
        self.assertFalse(assignment.ambiguous)
        self.assertEqual(assignment.indices, {"left": 0, "right": 1})
        self.assertEqual(assignment.reason, "taco_horizontal_initialization")

    def test_temporal_assignment_survives_detector_order_change(self) -> None:
        tracker = BimanualIdentityTracker()
        tracker.assign(
            np.array([[50, 50, 150, 150], [850, 50, 950, 150]]), 1000, 500
        )
        assignment = tracker.assign(
            np.array([[840, 55, 940, 155], [60, 55, 160, 155]]), 1000, 500
        )
        self.assertEqual(assignment.indices, {"right": 0, "left": 1})

    def test_crossing_is_rejected_as_ambiguous(self) -> None:
        tracker = BimanualIdentityTracker(ambiguity_margin_ratio=0.1)
        tracker.assign(
            np.array([[50, 50, 150, 150], [850, 50, 950, 150]]), 1000, 500
        )
        assignment = tracker.assign(
            np.array([[450, 50, 550, 150], [451, 50, 551, 150]]), 1000, 500
        )
        self.assertTrue(assignment.ambiguous)
        self.assertEqual(assignment.indices, {})

    def test_single_center_detection_is_not_guessed(self) -> None:
        tracker = BimanualIdentityTracker()
        assignment = tracker.assign(np.array([[450, 20, 550, 120]]), 1000, 500)
        self.assertTrue(assignment.ambiguous)
        self.assertEqual(assignment.reason, "single_detection_near_center")


if __name__ == "__main__":
    unittest.main()
