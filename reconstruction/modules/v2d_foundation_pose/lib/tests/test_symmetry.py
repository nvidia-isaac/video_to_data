import json
import sys
from pathlib import Path

import numpy as np
import pytest


LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

from symmetry import (  # noqa: E402
    MAX_SYMMETRY_GROUP_SIZE,
    canonicalize_pose,
    load_symmetry_spec,
)


def _write(tmp_path, continuous, discrete=None):
    path = tmp_path / "symmetry.json"
    path.write_text(json.dumps({
        "symmetries_discrete": discrete or [],
        "symmetries_continuous": continuous,
    }))
    return path


def _pose(yaw_deg=0.0, translation=(0.0, 0.0, 0.0)):
    angle = np.deg2rad(yaw_deg)
    pose = np.eye(4)
    pose[:3, :3] = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    pose[:3, 3] = translation
    return pose


def test_three_centered_axes_are_full_rotation_without_enumeration(tmp_path):
    spec = load_symmetry_spec(_write(tmp_path, [
        {"axis": [0, 0, 1]},
        {"axis": [1, 0, 0]},
        {"axis": [0, 1, 0]},
    ]))

    assert spec.full_rotational is True
    assert spec.mode == "full_rotational"
    assert spec.candidate_count == 1
    assert spec.transforms.shape == (1, 4, 4)
    assert spec.provenance()["candidate_guard_limit"] == MAX_SYMMETRY_GROUP_SIZE


def test_default_axial_sampling_has_36_candidates(tmp_path):
    spec = load_symmetry_spec(_write(tmp_path, [{"axis": [0, 0, 2]}]))

    assert spec.full_rotational is False
    assert spec.mode == "axial"
    assert spec.continuous_step_deg == 10.0
    assert spec.candidate_count == 36


def test_parallel_and_antiparallel_centered_axes_are_deduplicated(tmp_path):
    spec = load_symmetry_spec(_write(tmp_path, [
        {"axis": [0, 0, 1]},
        {"axis": [0, 0, 2]},
        {"axis": [0, 0, -1]},
    ]))

    assert spec.full_rotational is False
    assert spec.candidate_count == 36


def test_nonparallel_centered_axes_trigger_full_rotation(tmp_path):
    spec = load_symmetry_spec(_write(tmp_path, [
        {"axis": [0, 0, 1], "offset": [0, 0, 0]},
        {"axis": [1, 1, 0], "offset": [0, 0, 0]},
    ]))

    assert spec.full_rotational is True
    assert spec.candidate_count == 1


def test_candidate_guard_fails_before_cartesian_allocation(tmp_path):
    path = _write(tmp_path, [
        {"axis": [0, 0, 1], "offset": [1, 0, 0]},
        {"axis": [1, 0, 0], "offset": [0, 1, 0]},
        {"axis": [0, 1, 0], "offset": [0, 0, 1]},
    ])

    with pytest.raises(ValueError, match="46656 candidates.*4096"):
        load_symmetry_spec(path)


@pytest.mark.parametrize("step", [0, -1, 361, float("nan")])
def test_invalid_sampling_interval_is_rejected(tmp_path, step):
    path = _write(tmp_path, [{"axis": [0, 0, 1]}])
    with pytest.raises(ValueError, match="symmetry step"):
        load_symmetry_spec(path, continuous_step_deg=step)


def test_vectorized_canonicalization_matches_exhaustive_choice(tmp_path):
    spec = load_symmetry_spec(
        _write(tmp_path, [{"axis": [0, 0, 1]}]), continuous_step_deg=30.0,
    )
    pose = _pose(77.0, (1.0, 2.0, 3.0))
    reference = _pose(-14.0)

    actual = canonicalize_pose(pose, spec.transforms, reference)
    exhaustive = []
    for transform in spec.transforms:
        candidate = pose @ transform
        relative = reference[:3, :3].T @ candidate[:3, :3]
        angle = np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        exhaustive.append((angle, candidate))
    expected = min(exhaustive, key=lambda item: item[0])[1]

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(actual[:3, 3], pose[:3, 3])
