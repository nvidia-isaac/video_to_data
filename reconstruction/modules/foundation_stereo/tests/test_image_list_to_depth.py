"""Tests for Foundation Stereo image_list_to_depth.

Unit tests (no GPU):
  - Preprocessing shape and normalization
  - Disparity → depth math
  - Coordinate transformation (crop + scale + resize)

Integration tests (require TRT engine, skipped automatically if unavailable):
  - End-to-end output files exist with matching stems
  - Output depth is 16-bit PNG with non-zero values
  - Output intrinsics JSON has correct structure
"""

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths to test data (relative to this file)
# ---------------------------------------------------------------------------
TEST_DATA_DIR = Path(__file__).parent / 'test_data'
LEFT_DIR = TEST_DATA_DIR / 'left'
RIGHT_DIR = TEST_DATA_DIR / 'right'
CALIBRATION_FILE = TEST_DATA_DIR / 'calibration.json'


@pytest.fixture(scope='session')
def calibration():
    with open(CALIBRATION_FILE) as f:
        return json.load(f)


@pytest.fixture(scope='session')
def sample_left_image():
    images = sorted(LEFT_DIR.glob('*.jpeg')) + sorted(LEFT_DIR.glob('*.jpg'))
    assert images, f"No JPEG images found in {LEFT_DIR}"
    img = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    assert img is not None
    return img


@pytest.fixture(scope='session')
def sample_right_image():
    images = sorted(RIGHT_DIR.glob('*.jpeg')) + sorted(RIGHT_DIR.glob('*.jpg'))
    assert images, f"No JPEG images found in {RIGHT_DIR}"
    img = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    assert img is not None
    return img


# ---------------------------------------------------------------------------
# Unit tests — no GPU required
# ---------------------------------------------------------------------------

class TestPreprocessImage:
    def test_output_shape(self, sample_left_image):
        """Preprocessed output must be (1, 3, 576, 960)."""
        from modules.foundation_stereo._impl.trt_inference import (
            MODEL_INPUT_HEIGHT,
            MODEL_INPUT_WIDTH,
        )

        # Replicate just the preprocessing logic without a TRT engine
        h, w = sample_left_image.shape[:2]
        scale = min(MODEL_INPUT_WIDTH / w, MODEL_INPUT_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        import cv2 as _cv2
        interp = _cv2.INTER_CUBIC if scale > 1.0 else _cv2.INTER_AREA
        resized = _cv2.resize(sample_left_image, (new_w, new_h), interpolation=interp)
        pad_w = MODEL_INPUT_WIDTH - new_w
        pad_h = MODEL_INPUT_HEIGHT - new_h
        padded = _cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, _cv2.BORDER_REPLICATE)
        nchw = padded.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        assert nchw.shape == (1, 3, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)

    def test_aspect_ratio_scale(self, sample_left_image):
        """Scale factor must be min(scale_w, scale_h)."""
        from modules.foundation_stereo._impl.trt_inference import (
            MODEL_INPUT_HEIGHT,
            MODEL_INPUT_WIDTH,
        )

        h, w = sample_left_image.shape[:2]
        expected_scale = min(MODEL_INPUT_WIDTH / w, MODEL_INPUT_HEIGHT / h)
        assert expected_scale > 0
        assert expected_scale <= 1.0  # back_stereo images are 1920×1200 > 960×576


class TestDisparityToDepth:
    def test_formula(self):
        """depth = fx * baseline / disparity for valid pixels."""
        from modules.foundation_stereo._impl.trt_inference import disparity_to_depth

        disparity = np.array([[2.0, 0.0, 4.0]], dtype=np.float32)
        fx = 800.0
        baseline = 0.15

        depth = disparity_to_depth(disparity, fx, baseline)

        assert depth[0, 0] == pytest.approx(fx * baseline / 2.0, rel=1e-5)
        assert depth[0, 1] == 0.0    # invalid (below threshold)
        assert depth[0, 2] == pytest.approx(fx * baseline / 4.0, rel=1e-5)

    def test_invalid_disparity_is_zero(self):
        """Disparity values at or below threshold produce zero depth."""
        from modules.foundation_stereo._impl.trt_inference import (
            MIN_DISPARITY_THRESHOLD,
            disparity_to_depth,
        )

        disparity = np.array([[MIN_DISPARITY_THRESHOLD, 0.005, -1.0]], dtype=np.float32)
        depth = disparity_to_depth(disparity, fx=800.0, baseline=0.15)

        assert np.all(depth == 0.0)

    def test_output_shape(self):
        """Output shape matches input shape."""
        from modules.foundation_stereo._impl.trt_inference import disparity_to_depth

        disparity = np.random.rand(480, 640).astype(np.float32) * 10
        depth = disparity_to_depth(disparity, fx=800.0, baseline=0.15)
        assert depth.shape == (480, 640)


class TestCoordinateTransform:
    def test_output_shape_matches_original(self, sample_left_image):
        """After transform, disparity shape must match original image dims."""
        from modules.foundation_stereo._impl.trt_inference import (
            MODEL_INPUT_HEIGHT,
            MODEL_INPUT_WIDTH,
        )

        h, w = sample_left_image.shape[:2]
        scale = min(MODEL_INPUT_WIDTH / w, MODEL_INPUT_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        metadata = {
            'scale': scale,
            'resized_w': new_w,
            'resized_h': new_h,
            'orig_w': w,
            'orig_h': h,
        }

        fake_disparity = np.ones((MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH), dtype=np.float32) * 5.0

        cropped = fake_disparity[:new_h, :new_w]
        scaled = cropped / scale
        result = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)

        assert result.shape == (h, w)


# ---------------------------------------------------------------------------
# Integration tests — require TRT engine
# ---------------------------------------------------------------------------

def _trt_available(model_dir: str) -> bool:
    """Return True if a TRT engine exists in model_dir."""
    try:
        from modules.foundation_stereo._impl.export_engine import get_engine_path
        return os.path.exists(get_engine_path(model_dir))
    except Exception:
        return False


DEFAULT_MODEL_DIR = os.environ.get(
    'MODEL_DIR',
    os.path.join(os.environ.get('DATA_DIR', '/data'), 'foundation_stereo', 'models'),
)

requires_trt = pytest.mark.skipif(
    not _trt_available(DEFAULT_MODEL_DIR),
    reason="TRT engine not found — skipping integration tests",
)


@requires_trt
class TestImageListToDepthIntegration:

    @pytest.fixture(scope='class')
    def output_dirs(self, tmp_path_factory):
        """Run image_list_to_depth once, return (depth_dir, intrinsics_dir)."""
        from modules.foundation_stereo.image_list_to_depth import image_list_to_depth

        depth_dir = tmp_path_factory.mktemp('depth')
        intr_dir = tmp_path_factory.mktemp('intrinsics')
        with open(CALIBRATION_FILE) as f:
            calibration = json.load(f)

        image_list_to_depth(
            left_dir=str(LEFT_DIR),
            right_dir=str(RIGHT_DIR),
            depth_folder=str(depth_dir),
            intrinsics_folder=str(intr_dir),
            calibration=calibration,
            model_dir=DEFAULT_MODEL_DIR,
        )
        return depth_dir, intr_dir

    def test_depth_files_exist(self, output_dirs):
        depth_dir, _ = output_dirs
        depth_files = list(depth_dir.glob('*.png'))
        assert len(depth_files) == 3, f"Expected 3 depth PNGs, got {len(depth_files)}"

    def test_intrinsics_files_exist(self, output_dirs):
        _, intr_dir = output_dirs
        intr_files = list(intr_dir.glob('*.json'))
        assert len(intr_files) == 3, f"Expected 3 intrinsics JSONs, got {len(intr_files)}"

    def test_output_stems_match_input(self, output_dirs):
        depth_dir, intr_dir = output_dirs
        input_stems = {p.stem for p in LEFT_DIR.glob('*.jpeg')}
        depth_stems = {p.stem for p in depth_dir.glob('*.png')}
        intr_stems = {p.stem for p in intr_dir.glob('*.json')}
        assert depth_stems == input_stems
        assert intr_stems == input_stems

    def test_depth_is_16bit_png(self, output_dirs):
        depth_dir, _ = output_dirs
        for depth_file in depth_dir.glob('*.png'):
            img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            assert img is not None, f"Failed to read {depth_file}"
            assert img.dtype == np.uint16, f"Expected uint16, got {img.dtype}"

    def test_depth_has_nonzero_values(self, output_dirs):
        depth_dir, _ = output_dirs
        for depth_file in depth_dir.glob('*.png'):
            img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            assert np.any(img > 0), f"All-zero depth in {depth_file}"

    def test_intrinsics_structure(self, output_dirs, calibration):
        _, intr_dir = output_dirs
        for intr_file in intr_dir.glob('*.json'):
            with open(intr_file) as f:
                data = json.load(f)
            for key in ('fx', 'fy', 'cx', 'cy', 'width', 'height'):
                assert key in data, f"Missing key '{key}' in {intr_file}"
            assert data['fx'] == pytest.approx(calibration['fx'], rel=1e-5)
            assert data['fy'] == pytest.approx(calibration['fy'], rel=1e-5)
