import json
import sys
from pathlib import Path

import numpy as np


LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

from compare_object_poses import (
    _symmetry_spec,
    compare_pose_arrays,
    compare_pose_files,
)


def _poses(count=8):
    return np.repeat(np.eye(4)[None], count, axis=0)


def test_identical_pose_arrays_pass():
    poses = _poses()
    result = compare_pose_arrays(
        poses, poses.copy(),
        mesh_vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        mesh_diameter=np.sqrt(2),
    )
    assert result["status"] == "PASS"
    assert result["translation_m"]["max"] == 0
    assert result["rotation_deg"]["max"] == 0
    assert result["normalized_adds"]["max"] == 0


def test_discrete_symmetry_minimizes_rotation():
    legacy = _poses()
    commercial = _poses()
    half_turn = np.eye(4)
    half_turn[:3, :3] = np.diag([-1, -1, 1])
    commercial[:] = half_turn
    result = compare_pose_arrays(
        legacy, commercial,
        mesh_vertices=np.array([[1, 0, 0], [-1, 0, 0]], dtype=float),
        mesh_diameter=2.0,
        symmetries=[np.eye(4), half_turn],
    )
    assert result["status"] == "PASS"
    assert result["rotation_deg"]["max"] == 0


def test_independent_continuous_axes_use_full_rotation_fast_path(tmp_path):
    symmetry_path = tmp_path / "symmetry.json"
    symmetry_path.write_text(json.dumps({
        "symmetries_discrete": [],
        "symmetries_continuous": [
            {"axis": [0.0, 0.0, 1.0]},
            {"axis": [1.0, 0.0, 0.0]},
            {"axis": [0.0, 1.0, 0.0]},
        ],
    }))

    symmetries, full_rotational = _symmetry_spec(symmetry_path)

    assert full_rotational is True
    assert len(symmetries) == 1

    legacy = _poses(3)
    commercial = _poses(3)
    commercial[:, :3, :3] = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    commercial[:, 0, 3] = 0.01
    result = compare_pose_arrays(
        legacy,
        commercial,
        mesh_vertices=np.array([[0.0, 0.0, 0.0]]),
        mesh_diameter=1.0,
        symmetries=symmetries,
        full_rotational_symmetry=full_rotational,
    )

    assert result["status"] == "PASS"
    assert result["full_rotational_symmetry"] is True
    assert result["symmetry_candidate_count"] == 1
    assert result["rotation_deg"]["max"] == 0.0
    assert result["translation_m"]["max"] == 0.01


def test_parallel_continuous_axes_keep_sampled_axial_symmetry(tmp_path):
    symmetry_path = tmp_path / "symmetry.json"
    symmetry_path.write_text(json.dumps({
        "symmetries_discrete": [],
        "symmetries_continuous": [
            {"axis": [0.0, 0.0, 1.0]},
            {"axis": [0.0, 0.0, -1.0]},
        ],
    }))

    symmetries, full_rotational = _symmetry_spec(symmetry_path, step_deg=90.0)

    assert full_rotational is False
    assert len(symmetries) == 4


def test_default_axial_symmetry_sampling_is_ten_degrees(tmp_path):
    symmetry_path = tmp_path / "symmetry.json"
    symmetry_path.write_text(json.dumps({
        "symmetries_continuous": [{"axis": [0.0, 0.0, 1.0]}],
    }))

    symmetries, full_rotational = _symmetry_spec(symmetry_path)

    assert full_rotational is False
    assert len(symmetries) == 36


def _global_fraction_tolerances():
    return {
        "translation_m": {"median_max": 1.0, "p95_max": 1.0},
        "rotation_deg": {"median_max": 180.0, "p95_max": 180.0},
        "normalized_adds": {"median_max": 1.0, "p95_max": 1.0},
        "valid_coverage_min": 0.0,
        "newly_invalid_fraction_max": 1.0,
        "divergent_frame_fraction": {
            "max": 0.05,
            "translation_m": 0.08,
            "rotation_deg": 45.0,
            "normalized_adds": 0.25,
        },
    }


def test_more_than_five_percent_divergent_frames_and_new_invalid_frames_fail():
    legacy = _poses(100)
    commercial = _poses(100)
    commercial[:6, 0, 3] = 0.2
    commercial_valid = np.ones(100, dtype=bool)
    commercial_valid[-1] = False
    result = compare_pose_arrays(
        legacy, commercial,
        commercial_valid=commercial_valid,
        mesh_vertices=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        mesh_diameter=1.0,
        tolerances={
            **_global_fraction_tolerances(),
            "newly_invalid_fraction_max": 0.0,
        },
    )
    assert result["status"] == "FAIL"
    assert "divergent_frame_fraction" in result["failures"]
    assert "newly_invalid_frames" in result["failures"]
    assert result["divergent_frames"] == 6
    assert result["divergent_frame_fraction"] == 6 / 99
    assert result["divergence_segments"] == [
        {"start_frame": 0, "end_frame": 5, "frame_count": 6}
    ]


def test_exactly_five_percent_divergent_frames_pass_regardless_of_adjacency():
    legacy = _poses(100)
    mesh_vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    for indices in (range(5), (0, 20, 40, 60, 80)):
        commercial = _poses(100)
        commercial[list(indices), 0, 3] = 0.2
        result = compare_pose_arrays(
            legacy, commercial,
            mesh_vertices=mesh_vertices,
            mesh_diameter=1.0,
            tolerances=_global_fraction_tolerances(),
        )
        assert result["status"] == "PASS"
        assert result["divergent_frames"] == 5
        assert result["divergent_frame_fraction"] == 0.05


def test_frozen_sustained_divergence_config_uses_global_fraction():
    legacy = _poses(100)
    commercial = _poses(100)
    commercial[:5, 0, 3] = 0.2
    tolerances = _global_fraction_tolerances()
    divergence = tolerances.pop("divergent_frame_fraction")
    tolerances["sustained_divergence"] = {
        "length": 5,
        **{key: value for key, value in divergence.items() if key != "max"},
    }
    result = compare_pose_arrays(
        legacy, commercial,
        mesh_vertices=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        mesh_diameter=1.0,
        tolerances=tolerances,
    )
    assert result["status"] == "PASS"
    assert "sustained_divergence" not in result["tolerances"]
    assert result["tolerances"]["divergent_frame_fraction"]["max"] == 0.05
    assert result["divergent_frame_fraction"] == 0.05


def test_default_median_translation_threshold_is_three_point_five_cm():
    legacy = _poses(9)
    commercial = _poses(9)
    commercial[:, 0, 3] = 0.034
    mesh_vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)

    passing = compare_pose_arrays(
        legacy, commercial,
        mesh_vertices=mesh_vertices,
        mesh_diameter=1.0,
    )
    assert passing["status"] == "PASS"
    assert passing["tolerances"]["translation_m"]["median_max"] == 0.035

    commercial[:, 0, 3] = 0.036
    failing = compare_pose_arrays(
        legacy, commercial,
        mesh_vertices=mesh_vertices,
        mesh_diameter=1.0,
    )
    assert failing["status"] == "FAIL"
    assert "translation_m_median" in failing["failures"]


def test_original_mesh_pose_is_converted_to_aligned_frame(tmp_path):
    angle = np.deg2rad(90.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    centroid = np.array([0.4, -0.2, 0.7])
    alignment = np.eye(4)
    alignment[:3, :3] = rotation
    aligned_to_original = np.eye(4)
    aligned_to_original[:3, :3] = rotation.T
    aligned_to_original[:3, 3] = centroid

    legacy = _poses(6)
    legacy[:, :3, 3] = np.array([1.0, 2.0, 3.0])
    commercial = legacy @ aligned_to_original
    legacy_path = tmp_path / "legacy.npy"
    commercial_path = tmp_path / "commercial.npy"
    np.save(legacy_path, legacy)
    np.save(commercial_path, commercial)
    mesh_path = tmp_path / "mesh.glb"
    import trimesh
    trimesh.creation.box(extents=[1.0, 2.0, 3.0]).export(mesh_path)
    symmetry_path = tmp_path / "symmetry.json"
    symmetry_path.write_text(json.dumps({
        "alignment": {
            "centroid": centroid.tolist(),
            "rotation": alignment.reshape(-1).tolist(),
        },
        "symmetries_discrete": [],
        "symmetries_continuous": [],
    }))

    result = compare_pose_files(
        legacy_pose_path=legacy_path,
        commercial_pose_path=commercial_path,
        mesh_path=mesh_path,
        symmetry_path=symmetry_path,
        output_path=tmp_path / "comparison.json",
        legacy_pose_frame="original",
    )

    assert result["status"] == "PASS"
    assert result["legacy_alignment_applied"] is True
    assert result["comparison_pose_frame"] == "aligned"
    assert result["rotation_deg"]["max"] == 0
    assert result["translation_m"]["max"] == 0
    assert result["symmetry_sha256"] is not None
