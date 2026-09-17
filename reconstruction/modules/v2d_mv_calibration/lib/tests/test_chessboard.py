import cv2
import numpy as np
import pytest

from v2d.mv.calibration.lib import chessboard


def test_legacy_detector_preserves_corner_subpix_refinement(monkeypatch):
    board_size = (3, 4)
    gray = np.zeros((32, 32), dtype=np.uint8)
    corners = np.arange(24, dtype=np.float32).reshape(12, 1, 2)
    refined = corners + 0.25
    calls = {}

    def fake_find(image, size, flags):
        calls["find"] = (image, size, flags)
        return True, corners

    def fake_refine(image, initial, window, zero_zone, criteria):
        calls["refine"] = (image, initial, window, zero_zone, criteria)
        return refined

    monkeypatch.setattr(chessboard.cv2, "findChessboardCorners", fake_find)
    monkeypatch.setattr(chessboard.cv2, "cornerSubPix", fake_refine)
    monkeypatch.setattr(
        chessboard.cv2,
        "findChessboardCornersSBWithMeta",
        lambda *args: pytest.fail("marker detector must not run in legacy mode"),
    )

    result = chessboard._detect_chessboard(gray, board_size)

    expected_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS
    )
    assert calls["find"][1:] == (board_size, expected_flags)
    assert calls["refine"][2:] == (
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
    )
    np.testing.assert_array_equal(result, refined.squeeze(1))


def test_marker_detector_uses_sb_native_coordinates(monkeypatch):
    board_size = (3, 4)
    gray = np.zeros((32, 32), dtype=np.uint8)
    corners = np.arange(24, dtype=np.float32).reshape(12, 1, 2)
    meta = np.ones((4, 3), dtype=np.uint8)
    meta[1, 1] = 4
    calls = {}

    def fake_find(image, size, flags):
        calls["find"] = (image, size, flags)
        return True, corners, meta

    monkeypatch.setattr(chessboard.cv2, "findChessboardCornersSBWithMeta", fake_find)
    monkeypatch.setattr(
        chessboard.cv2,
        "findChessboardCorners",
        lambda *args: pytest.fail("legacy detector must not run in marker mode"),
    )
    monkeypatch.setattr(
        chessboard.cv2,
        "cornerSubPix",
        lambda *args: pytest.fail("marker mode must use SB native coordinates"),
    )

    result = chessboard._detect_chessboard(
        gray,
        board_size,
        use_marker_chessboard=True,
    )

    expected_flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
        | cv2.CALIB_CB_MARKER
    )
    assert calls["find"][1:] == (board_size, expected_flags)
    np.testing.assert_array_equal(result, corners.reshape(-1, 2))


def test_marker_detector_transposes_metadata_aligned_corner_grid(monkeypatch):
    board_size = (3, 4)
    input_grid = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    meta = np.ones((3, 4), dtype=np.uint8)
    meta[1, 2] = 4
    monkeypatch.setattr(
        chessboard.cv2,
        "findChessboardCornersSBWithMeta",
        lambda *args: (True, input_grid.reshape(-1, 1, 2), meta),
    )

    result = chessboard._detect_chessboard(
        np.zeros((32, 32), dtype=np.uint8),
        board_size,
        use_marker_chessboard=True,
    )

    expected = input_grid.transpose(1, 0, 2).reshape(-1, 2)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("ret", "corners", "meta"),
    [
        (False, None, None),
        (
            True,
            np.zeros((12, 1, 2), dtype=np.float32),
            np.ones((4, 3), dtype=np.uint8),
        ),
        (
            True,
            np.zeros((12, 1, 2), dtype=np.float32),
            np.array([[4, 4, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8),
        ),
        (
            True,
            np.zeros((11, 1, 2), dtype=np.float32),
            np.array([[4, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8),
        ),
        (
            True,
            np.zeros((12, 1, 2), dtype=np.float32),
            np.array([[4] + [1] * 11], dtype=np.uint8),
        ),
    ],
)
def test_marker_detector_rejects_invalid_detection(monkeypatch, ret, corners, meta):
    monkeypatch.setattr(
        chessboard.cv2,
        "findChessboardCornersSBWithMeta",
        lambda *args: (ret, corners, meta),
    )

    result = chessboard._detect_chessboard(
        np.zeros((32, 32), dtype=np.uint8),
        (3, 4),
        use_marker_chessboard=True,
    )

    assert result is None


def test_marker_detector_reports_incompatible_opencv(monkeypatch):
    monkeypatch.setattr(chessboard.cv2, "findChessboardCornersSBWithMeta", None)

    with pytest.raises(RuntimeError, match="findChessboardCornersSBWithMeta"):
        chessboard._detect_chessboard(
            np.zeros((32, 32), dtype=np.uint8),
            (3, 4),
            use_marker_chessboard=True,
        )


def _make_marker_chessboard() -> np.ndarray:
    columns, rows = 7, 11
    square_size = 60
    border = 120
    image = np.full(
        (rows * square_size + 2 * border, columns * square_size + 2 * border),
        255,
        dtype=np.uint8,
    )
    for y in range(rows):
        for x in range(columns):
            if x % 2 == y % 2:
                image[
                    border + y * square_size : border + (y + 1) * square_size,
                    border + x * square_size : border + (x + 1) * square_size,
                ] = 0

    # Center, one cell left, and one cell diagonally above-left. Each dot
    # contrasts with its containing checkerboard cell.
    for x, y in ((3, 5), (2, 5), (2, 4)):
        cell_is_black = x % 2 == y % 2
        cv2.circle(
            image,
            (
                border + int((x + 0.5) * square_size),
                border + int((y + 0.5) * square_size),
            ),
            int(0.17 * square_size),
            255 if cell_is_black else 0,
            -1,
        )
    return image


def test_marker_detector_has_consistent_order_across_right_angle_rotations():
    image = _make_marker_chessboard()
    height, width = image.shape
    rotations = [
        (image, lambda points: points),
        (
            cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
            lambda points: np.stack(
                [points[:, 1], height - 1 - points[:, 0]], axis=-1
            ),
        ),
        (
            cv2.rotate(image, cv2.ROTATE_180),
            lambda points: np.stack(
                [width - 1 - points[:, 0], height - 1 - points[:, 1]], axis=-1
            ),
        ),
        (
            cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
            lambda points: np.stack(
                [width - 1 - points[:, 1], points[:, 0]], axis=-1
            ),
        ),
    ]

    canonical = None
    for rotated, to_original_coordinates in rotations:
        corners = chessboard._detect_chessboard(
            rotated,
            (6, 10),
            use_marker_chessboard=True,
        )
        assert corners is not None
        corners = to_original_coordinates(corners)
        if canonical is None:
            canonical = corners
        else:
            np.testing.assert_allclose(corners, canonical, atol=0.15)


def test_worker_forwards_marker_mode(monkeypatch):
    class FakeFrameSource:
        n_frames = 1

        def __getitem__(self, index):
            assert index == 0
            return np.zeros((8, 8), dtype=np.uint8)

        def close(self):
            pass

    seen = []
    monkeypatch.setattr(
        chessboard.FrameSource,
        "from_path",
        lambda *args, **kwargs: FakeFrameSource(),
    )

    def fake_detect(gray, board_size, use_marker_chessboard=False):
        seen.append(use_marker_chessboard)
        return np.zeros((board_size[0] * board_size[1], 2), dtype=np.float32)

    monkeypatch.setattr(chessboard, "_detect_chessboard", fake_detect)

    correspondences, frame_indices = chessboard._chessboard_detect_worker(
        ["cam0", "cam1"],
        (3, 4),
        0,
        1,
        use_marker_chessboard=True,
    )

    assert seen == [True, True]
    assert frame_indices == [0]
    assert len(correspondences) == 1
    assert len(correspondences[0]) == 2
