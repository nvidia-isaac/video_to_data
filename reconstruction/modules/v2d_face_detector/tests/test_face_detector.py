from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
COMMON_DIR = Path(__file__).resolve().parents[2] / "v2d_common"
sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(COMMON_DIR))

from face_blur import _build_face_blur_region, blur_faces
from face_detection import FaceDetection, YuNetFaceDetector, non_max_suppression
from temporal_filter import associate_tracks, filter_detections


def detection(x: float, *, score: float = 0.9) -> FaceDetection:
    return FaceDetection(
        bbox_xywh=np.array([x, 10, 20, 24], dtype=np.float32),
        landmarks=np.array(
            [[x + 6, 17], [x + 14, 17], [x + 10, 22], [x + 7, 28], [x + 13, 28]],
            dtype=np.float32,
        ),
        score=score,
    )


def test_nms_keeps_high_score_and_distinct_faces():
    kept = non_max_suppression(
        [detection(10, score=0.7), detection(11, score=0.9), detection(70)],
        0.3,
    )
    assert len(kept) == 2
    assert kept[0].score == 0.9
    assert kept[1].bbox_xywh[0] == 70


def test_yunet_multiscale_mapping_color_conversion_and_confidence_filtering():
    class FakeYuNet:
        def __init__(self):
            self.input_sizes = []
            self.first_pixels = []

        def setInputSize(self, size):
            self.input_sizes.append(size)

        def detect(self, bgr):
            self.first_pixels.append(bgr[0, 0].tolist())
            if bgr.shape[:2] == (80, 100):
                rows = [
                    [5, 5, 10, 12, *([1, 2] * 5), 0.55],
                    [5, 5, 10, 12, *([1, 2] * 5), 0.90],
                ]
            else:
                rows = [[20, 10, 10, 12, *([21, 12] * 5), 0.80]]
            return 1, np.asarray(rows, dtype=np.float32)

    backend = YuNetFaceDetector.__new__(YuNetFaceDetector)
    backend.score_threshold = 0.6
    backend.nms_threshold = 0.3
    backend.scales = (1.0, 0.5)
    backend._detector = FakeYuNet()
    frame = np.full((80, 100, 3), [1, 2, 3], dtype=np.uint8)

    result = backend.detect(frame)

    assert backend._detector.input_sizes == [(100, 80), (50, 40)]
    assert backend._detector.first_pixels == [[3, 2, 1], [3, 2, 1]]
    assert len(result) == 2
    np.testing.assert_allclose(result[1].bbox_xywh, [40, 20, 20, 24])
    np.testing.assert_allclose(result[1].landmarks[0], [42, 24])


def test_temporal_association_uses_center_distance_when_iou_is_zero():
    _, tracks = associate_tracks(
        [[detection(0)], [detection(21)]],
        min_iou=0.2,
        max_center_distance=1.0,
    )
    assert len(tracks) == 1


def test_temporal_filter_fills_gap_smooths_jitter_and_keeps_isolated():
    frames = [
        [detection(10)],
        [detection(13)],
        [],
        [detection(11)],
        [],
        [],
        [detection(90)],
    ]
    raw, filtered = filter_detections(
        frames, max_gap_frames=2, edge_fill_frames=0, ema_alpha=0.6
    )
    assert raw[0][0].track_id == raw[1][0].track_id
    assert filtered[2][0].provenance == "interpolated"
    assert filtered[6], "isolated recall-oriented detection must be retained"
    raw_jitter = np.std([10.0, 13.0, 11.0])
    filtered_jitter = np.std(
        [filtered[index][0].bbox_xywh[0] for index in (0, 1, 3)]
    )
    assert filtered_jitter < raw_jitter
    assert all(item.bbox_xywh[2] >= 20 for frame in filtered for item in frame)


def test_temporal_filter_fills_at_most_two_edge_frames_and_five_internal_frames():
    frames = [[], [], [detection(10)], *([[]] * 5), [detection(12)], [], [], []]
    _, filtered = filter_detections(frames)
    assert all(filtered[index] for index in range(11))
    assert filtered[11] == []

    long_gap = [[detection(10)], *([[]] * 6), [detection(12)]]
    _, filtered_long_gap = filter_detections(long_gap)
    assert filtered_long_gap[3] == []


def test_blur_uses_padded_support_and_preserves_pixels_beyond_it():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(80, 100, 3), dtype=np.uint8)
    face = detection(30)
    region = _build_face_blur_region(
        frame.shape,
        face,
        ellipse_scale=1.0,
        feather_fraction=0.12,
        blur_sigma_fraction=0.18,
    )
    assert region is not None

    result = blur_faces(frame, [face])

    assert not np.array_equal(result[10:35, 30:50], frame[10:35, 30:50])
    outside = np.ones(frame.shape[:2], dtype=bool)
    outside[region.y0:region.y1, region.x0:region.x1] = False
    np.testing.assert_array_equal(result[outside], frame[outside])


def test_padded_mask_fades_before_non_frame_boundaries_without_resizing_ellipse():
    tilted_face = FaceDetection(
        bbox_xywh=np.array([80, 70, 40, 60], dtype=np.float32),
        landmarks=np.array(
            [[90, 85], [110, 105], [100, 102], [94, 116], [106, 116]],
            dtype=np.float32,
        ),
        score=0.9,
    )
    region = _build_face_blur_region(
        (220, 220, 3),
        tilted_face,
        ellipse_scale=1.0,
        feather_fraction=0.12,
        blur_sigma_fraction=0.18,
    )
    assert region is not None

    assert region.axes == (20, 30)
    assert region.angle == pytest.approx(45.0)
    assert region.center_global == pytest.approx((100.0, 100.0))
    assert region.center_local[0] + region.x0 == pytest.approx(100.0, abs=0.5)
    assert region.center_local[1] + region.y0 == pytest.approx(100.0, abs=0.5)
    boundary = np.concatenate(
        [
            region.mask[0],
            region.mask[-1],
            region.mask[:, 0],
            region.mask[:, -1],
        ]
    )
    assert float(boundary.max()) < 1e-3


def test_blur_clips_at_frame_border_without_recentering_ellipse():
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 256, size=(40, 50, 3), dtype=np.uint8)
    edge_face = detection(-8)
    region = _build_face_blur_region(
        frame.shape,
        edge_face,
        ellipse_scale=1.0,
        feather_fraction=0.12,
        blur_sigma_fraction=0.18,
    )
    assert region is not None

    result = blur_faces(frame, [edge_face])

    assert region.x0 == 0
    assert region.center_global[0] == pytest.approx(2.0)
    assert region.center_local[0] == 2
    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
    assert not np.array_equal(result[:, :14], frame[:, :14])
    if region.x1 < frame.shape[1]:
        np.testing.assert_array_equal(result[:, region.x1:], frame[:, region.x1:])


def test_face_detection_validation():
    with pytest.raises(ValueError, match="shape"):
        FaceDetection(np.ones(3), np.ones((5, 2)), 0.9)
    with pytest.raises(ValueError, match="positive"):
        FaceDetection(np.array([0, 0, 0, 1]), np.ones((5, 2)), 0.9)
    with pytest.raises(ValueError, match="ellipse_scale"):
        blur_faces(np.zeros((20, 20, 3), np.uint8), [], ellipse_scale=1.1)
