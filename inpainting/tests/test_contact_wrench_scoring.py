from __future__ import annotations

import hashlib

import numpy as np
import pytest

from inpainting import contact_wrench_scoring as wrench


def test_shared_basis_is_fixed_unit_length_and_hash_identified() -> None:
    first = wrench.shared_wrench_basis()
    second = wrench.shared_wrench_basis()

    assert first is second
    assert first.directions.shape == (512, 6)
    assert first.directions.dtype == np.float64
    assert not first.directions.flags.writeable
    np.testing.assert_allclose(
        np.linalg.norm(first.directions, axis=-1),
        1.0,
        atol=2e-15,
        rtol=0.0,
    )
    assert first.sha256 == wrench.DEFAULT_WRENCH_BASIS_SHA256
    assert first.sha256 == hashlib.sha256(
        np.asarray(first.directions, dtype="<f8", order="C").tobytes()
    ).hexdigest()
    assert first.sha256 == (
        "0f2dd35bc3c467700aefeef3d6a217b9ba371563bcbec5b55b314cf5b8ae43ee"
    )
    provenance = first.provenance()
    assert provenance["seed"] == 0
    assert provenance["shape"] == [512, 6]
    assert provenance["sha256"] == first.sha256
    assert provenance["torque_basis_radius"] == 1.0

    custom_first = wrench.shared_wrench_basis(num_samples=16, seed=7)
    custom_second = wrench.shared_wrench_basis(num_samples=16, seed=7)
    assert custom_first is custom_second
    assert custom_first.directions.shape == (16, 6)
    assert custom_first.sha256 != first.sha256


def test_frisvad_tangents_are_orthonormal_for_both_poles_and_generic_normal() -> None:
    normals = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    tangent_1, tangent_2 = wrench.compute_tangent_basis(normals)

    np.testing.assert_allclose(
        np.linalg.norm(tangent_1, axis=-1), 1.0, atol=1e-14
    )
    np.testing.assert_allclose(
        np.linalg.norm(tangent_2, axis=-1), 1.0, atol=1e-14
    )
    np.testing.assert_allclose(
        np.sum(normals * tangent_1, axis=-1), 0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        np.sum(normals * tangent_2, axis=-1), 0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        np.sum(tangent_1 * tangent_2, axis=-1), 0.0, atol=1e-14
    )


def test_friction_cone_edges_match_v2d_phase_and_append_normal_math() -> None:
    normals = np.asarray([[[0.0, 0.0, 1.0]]])
    cos_t, sin_t = wrench.friction_cone_phases(4)

    edges = wrench.compute_friction_cone_edges(
        normals,
        cos_t,
        sin_t,
        friction_coefficients=0.5,
    )

    expected = np.asarray(
        [
            [0.5, 0.0, 1.0],
            [0.0, 0.5, 1.0],
            [-0.5, 0.0, 1.0],
            [0.0, -0.5, 1.0],
        ]
    )
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    expected = np.concatenate((expected, [[0.0, 0.0, 1.0]]), axis=0)
    np.testing.assert_allclose(edges[0, 0], expected, atol=1e-15)


def test_high_level_scorer_subtracts_com_and_negates_outward_mesh_normal() -> None:
    points_object = np.asarray([[2.0, 1.0, 0.0]])
    outward_normals = np.asarray([[1.0, 0.0, 0.0]])
    basis = wrench.generate_deterministic_wrench_basis(num_samples=1, seed=4)
    # Replace the random direction with an exact probe while retaining an
    # identified WrenchBasis contract for this focused geometry test.
    probe = np.asarray([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.5]])
    probe /= np.linalg.norm(probe, axis=-1, keepdims=True)
    probe.setflags(write=False)
    probe_basis = wrench.WrenchBasis(
        directions=probe,
        sha256="test-probe",
        seed=basis.seed,
        generator="test",
    )

    scores = wrench.score_contact_wrench_candidates(
        points_object,
        outward_normals,
        object_com_object=[1.0, 0.0, 0.0],
        object_radius_m=2.0,
        basis=probe_basis,
        friction_coefficient=0.0,
        num_friction_cone_edges=1,
        low_quantile=0.0,
    )

    # COM-relative p=[1,1,0], inward f=[-1,0,0],
    # p x f / radius = [0,0,0.5].
    primitive = np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    expected_support = max(float(probe[0] @ primitive), 0.0)
    np.testing.assert_allclose(scores.supports, [expected_support], atol=1e-15)
    assert scores.low_quantile_support == pytest.approx(expected_support)
    assert scores.mean_support == pytest.approx(expected_support)
    assert scores.support_coverage == pytest.approx(1.0)


def test_candidate_metrics_are_batched_and_share_one_basis() -> None:
    points = np.asarray(
        [
            [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]],
            [[-0.005, 0.0, 0.0], [0.005, 0.0, 0.0]],
        ]
    )
    outward = np.asarray(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )

    scores = wrench.score_contact_wrench_candidates(
        points,
        outward,
        object_com_object=np.zeros(3),
        object_radius_m=0.1,
        low_quantile=0.1,
    )

    assert scores.supports.shape == (2, 512)
    assert np.asarray(scores.low_quantile_support).shape == (2,)
    assert np.asarray(scores.mean_support).shape == (2,)
    assert np.asarray(scores.support_coverage).shape == (2,)
    np.testing.assert_allclose(
        scores.mean_support,
        scores.supports.mean(axis=-1),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        scores.low_quantile_support,
        np.quantile(scores.supports, 0.1, axis=-1),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        scores.support_coverage,
        (scores.supports > 1.0e-3).mean(axis=-1),
        atol=0.0,
        rtol=0.0,
    )
    assert scores.basis_sha256 == wrench.DEFAULT_WRENCH_BASIS_SHA256
    assert scores.chord_reference_match is None


def test_chord_reference_match_exactly_reproduces_one_body_reward_formula() -> None:
    reference = np.asarray([0.0, 1.0, 2.0, 0.0005])
    candidate = np.asarray([9.0, 0.8, 2.5, 0.0005])

    score = wrench.compute_chord_reference_match(
        candidate,
        reference,
        tolerance=0.1,
        variance=0.1,
        support_threshold=1.0e-3,
    )

    expected = (np.exp(-0.01 / 0.1) + np.exp(-0.09 / 0.1)) / 2.0
    assert score == pytest.approx(expected)


def test_reference_match_broadcasts_one_reference_over_candidate_batch() -> None:
    points = np.asarray(
        [
            [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]],
            [[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]],
        ]
    )
    outward = np.asarray(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    without_reference = wrench.score_contact_wrench_candidates(
        points,
        outward,
        object_com_object=np.zeros(3),
        object_radius_m=0.1,
    )

    with_reference = wrench.score_contact_wrench_candidates(
        points,
        outward,
        object_com_object=np.zeros(3),
        object_radius_m=0.1,
        reference_supports=without_reference.supports[0],
    )

    assert np.asarray(with_reference.chord_reference_match).shape == (2,)
    assert with_reference.chord_reference_match[0] == pytest.approx(1.0)
    assert 0.0 <= with_reference.chord_reference_match[1] <= 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"object_radius_m": 0.0}, "object_radius_m"),
        ({"object_com_object": [0.0, 0.0]}, "object_com_object"),
        ({"low_quantile": 1.1}, "low_quantile"),
        ({"friction_coefficient": -0.1}, "friction_coefficient"),
    ],
)
def test_high_level_scorer_rejects_ambiguous_geometry(
    kwargs: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "object_com_object": np.zeros(3),
        "object_radius_m": 0.1,
    }
    arguments.update(kwargs)
    with pytest.raises(wrench.ContactWrenchScoringError, match=message):
        wrench.score_contact_wrench_candidates(
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            **arguments,
        )
