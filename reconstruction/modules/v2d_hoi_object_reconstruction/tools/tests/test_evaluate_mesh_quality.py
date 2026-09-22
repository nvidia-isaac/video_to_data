# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import trimesh

from reconstruction.modules.v2d_hoi_object_reconstruction.tools.evaluate_mesh_quality import (
    EvaluationConfig,
    aligned_bounding_box_metrics,
    align_rigid_multistart,
    area_weighted_surface_moments,
    apply_rigid_transform,
    compute_surface_metrics,
    evaluate_mesh_pair,
    evaluate_paths,
    load_mesh,
    sample_surface,
    summarize_results,
)


def test_surface_sampling_is_deterministic() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2)

    first = sample_surface(mesh, 128, seed=17)
    second = sample_surface(mesh, 128, seed=17)

    np.testing.assert_array_equal(first, second)


def test_area_weighted_surface_moments_are_exact_under_similarity() -> None:
    reference = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
    candidate = reference.copy()
    candidate.apply_scale(1.5)
    candidate.apply_translation([4.0, -2.0, 0.75])

    reference_center, reference_radius = area_weighted_surface_moments(reference)
    candidate_center, candidate_radius = area_weighted_surface_moments(candidate)

    np.testing.assert_allclose(
        candidate_center,
        reference_center * 1.5 + [4.0, -2.0, 0.75],
        atol=1e-12,
    )
    assert candidate_radius / reference_radius == pytest.approx(1.5, abs=1e-12)


def test_area_weighted_surface_moments_ignore_triangle_density() -> None:
    reference = trimesh.creation.icosphere(subdivisions=1, radius=0.7)
    subdivided = reference.subdivide()

    reference_center, reference_radius = area_weighted_surface_moments(reference)
    subdivided_center, subdivided_radius = area_weighted_surface_moments(subdivided)

    np.testing.assert_allclose(subdivided_center, reference_center, atol=1e-12)
    assert subdivided_radius == pytest.approx(reference_radius, abs=1e-12)


def test_aligned_bounds_report_axis_and_diagonal_errors() -> None:
    reference = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
    candidate = trimesh.creation.box(extents=[1.1, 2.4, 2.7])

    metrics = aligned_bounding_box_metrics(reference, candidate, np.eye(3))

    np.testing.assert_allclose(
        metrics["candidate_to_reference_extent_ratios"], [1.1, 1.2, 0.9]
    )
    np.testing.assert_allclose(
        metrics["symmetric_extent_errors_pct"],
        [10.0, 20.0, 100.0 / 9.0],
    )
    assert metrics["maximum_symmetric_extent_error_pct"] == pytest.approx(20.0)
    expected_diagonal_ratio = np.linalg.norm([1.1, 2.4, 2.7]) / np.linalg.norm(
        [1.0, 2.0, 3.0]
    )
    assert metrics["candidate_to_reference_diagonal_ratio"] == pytest.approx(
        expected_diagonal_ratio
    )


def test_identical_points_have_zero_chamfer_and_full_coverage() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    metrics = compute_surface_metrics(
        points,
        points.copy(),
        reference_diagonal=np.sqrt(3.0),
        threshold_percents=(2.0,),
    )

    assert metrics["chamfer_mean_pct_diag"] == pytest.approx(0.0)
    assert metrics["precision_2pct"] == pytest.approx(100.0)
    assert metrics["recall_2pct"] == pytest.approx(100.0)
    assert metrics["fscore_2pct"] == pytest.approx(100.0)


def test_precision_and_recall_keep_their_directional_meaning() -> None:
    reference = np.array(
        [
            [0.00, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.50, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [1.00, 0.0, 0.0],
        ]
    )
    candidate = np.array(
        [
            [0.00, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.04, 0.0, 0.0],
        ]
    )

    metrics = compute_surface_metrics(
        reference,
        candidate,
        reference_diagonal=1.0,
        threshold_percents=(5.0,),
    )

    assert metrics["precision_5pct"] == pytest.approx(100.0)
    assert metrics["recall_5pct"] == pytest.approx(20.0)


def test_rigid_alignment_recovers_rotation_and_translation() -> None:
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(200, 3)) * np.array([1.0, 2.0, 4.0])
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    candidate = apply_rigid_transform(
        reference,
        rotation,
        np.array([3.0, -1.5, 0.75]),
    )

    recovered_rotation, recovered_translation, score = align_rigid_multistart(
        reference,
        candidate,
        iterations=50,
        starts=25,
        trim_fraction=1.0,
    )
    aligned = apply_rigid_transform(
        candidate, recovered_rotation, recovered_translation
    )

    assert score < 1e-10
    np.testing.assert_allclose(aligned, reference, atol=1e-9)


def test_scale_is_preserved_for_delivered_metric_and_removed_for_shape_metric() -> None:
    reference = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    candidate = reference.copy()
    candidate.apply_scale(1.5)
    config = EvaluationConfig(
        metric_samples=2_000,
        alignment_samples=500,
        icp_iterations=15,
        alignment_starts=1,
        trim_fraction=1.0,
        threshold_percents=(2.0,),
        seed=11,
    )

    result = evaluate_mesh_pair(
        reference,
        candidate,
        object_id="scaled_sphere",
        method="synthetic",
        config=config,
    )

    assert result["scale_ratio_to_reference"] == pytest.approx(1.5, abs=1e-12)
    assert result["symmetric_scale_error_pct"] == pytest.approx(50.0, abs=1e-10)
    assert result["as_delivered"]["chamfer_mean_pct_diag"] > 10.0
    assert result["shape_scale_normalized"]["chamfer_mean_pct_diag"] < 2.0
    assert result["shape_scale_normalized"]["chamfer_mean_pct_diag"] < (
        result["as_delivered"]["chamfer_mean_pct_diag"] * 0.2
    )


def test_load_mesh_applies_scene_node_transforms(tmp_path) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(extents=[1.0, 1.0, 1.0]),
        node_name="right",
        transform=trimesh.transformations.translation_matrix([2.0, 0.0, 0.0]),
    )
    scene.add_geometry(
        trimesh.creation.box(extents=[0.5, 0.5, 0.5]),
        node_name="left",
        transform=trimesh.transformations.translation_matrix([-1.0, 0.0, 0.0]),
    )
    path = tmp_path / "transformed_scene.glb"
    path.write_bytes(scene.export(file_type="glb"))

    loaded = load_mesh(path)

    np.testing.assert_allclose(loaded.bounds[0], [-1.25, -0.5, -0.5])
    np.testing.assert_allclose(loaded.bounds[1], [2.5, 0.5, 0.5])


def test_path_evaluation_records_exact_input_hashes(tmp_path) -> None:
    reference = tmp_path / "reference.glb"
    candidate = tmp_path / "candidate.glb"
    mesh_bytes = trimesh.creation.icosphere(subdivisions=1).export(file_type="glb")
    reference.write_bytes(mesh_bytes)
    candidate.write_bytes(mesh_bytes)
    config = EvaluationConfig(
        metric_samples=100,
        alignment_samples=50,
        icp_iterations=5,
        alignment_starts=1,
        trim_fraction=1.0,
        threshold_percents=(2.0, 5.0),
        seed=3,
    )

    result = evaluate_paths(
        reference,
        candidate,
        object_id="hashed_mesh",
        method="synthetic",
        config=config,
    )

    expected = hashlib.sha256(mesh_bytes).hexdigest()
    assert result["reference"]["sha256"] == expected
    assert result["candidate"]["sha256"] == expected


def test_summary_discovers_custom_threshold_metrics() -> None:
    result = {
        "method": "synthetic",
        "status": "evaluated",
        "scale_ratio_to_reference": 1.1,
        "symmetric_scale_error_pct": 10.0,
        "candidate": {"watertight": True},
        "as_delivered": {
            "chamfer_mean_pct_diag": 2.0,
            "precision_3p5pct": 70.0,
            "recall_3p5pct": 80.0,
            "fscore_3p5pct": 74.6666666667,
        },
        "shape_scale_normalized": {"chamfer_mean_pct_diag": 1.0},
    }

    summary = summarize_results([result])["methods"]["synthetic"]

    assert summary["as_delivered_precision_3p5pct"]["mean"] == 70.0
    assert summary["as_delivered_recall_3p5pct"]["mean"] == 80.0
    assert summary["as_delivered_fscore_3p5pct"]["mean"] == pytest.approx(
        74.6666666667
    )
