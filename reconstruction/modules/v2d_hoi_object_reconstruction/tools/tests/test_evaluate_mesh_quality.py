# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from reconstruction.modules.v2d_hoi_object_reconstruction.tools.evaluate_mesh_quality import (
    EvaluationConfig,
    align_rigid_multistart,
    apply_rigid_transform,
    compute_surface_metrics,
    evaluate_mesh_pair,
    load_mesh,
    sample_surface,
    summarize_results,
)


def test_surface_sampling_is_deterministic() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2)

    first = sample_surface(mesh, 128, seed=17)
    second = sample_surface(mesh, 128, seed=17)

    np.testing.assert_array_equal(first, second)


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

    assert result["scale_ratio_to_reference"] == pytest.approx(1.5, abs=0.01)
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
