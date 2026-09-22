# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests pinning the object-tracking metric definitions."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass

import numpy as np
import pytest

from flash_chord.evaluation.metrics import (
    ADD_THRESHOLDS_M,
    MANIPTRANS_ORIENTATION_THRESHOLD_DEG,
    MANIPTRANS_POSITION_THRESHOLD_M,
    SPIDER_ORIENTATION_THRESHOLD_RAD,
    SPIDER_POSITION_THRESHOLD_M,
    TrackingThresholds,
    add_auc,
    chord_success,
    compute_object_tracking_metrics,
    maniptrans_success,
    object_add,
    object_mppe_cm,
    quaternion_geodesic_angle,
    spider_centered_position_error,
    spider_success,
    tracking_cause_masks,
)

STEPS = 12
WORLDS = 5
BODIES = 2
VERTICES = 40

TRAINED_THRESHOLDS = TrackingThresholds(object_position_m=0.10, object_orientation_rad=0.50)
WRIST_GATED_THRESHOLDS = TrackingThresholds(
    wrist_position_m=0.15,
    object_position_m=0.10,
    object_orientation_rad=0.50,
)


@dataclass
class Completion:
    completion_step: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    reference_progress: np.ndarray


def identity_reference(steps: int = STEPS, bodies: int = BODIES) -> np.ndarray:
    """Reference poses whose positions drift and whose orientations stay at identity."""
    reference = np.zeros((steps, bodies, 7))
    reference[..., 3] = 0.0
    reference[..., 6] = 1.0
    for body in range(bodies):
        reference[:, body, 0] = np.linspace(0.0, 0.3, steps) + body
        reference[:, body, 1] = np.linspace(0.0, -0.2, steps)
    return reference


def broadcast_cohort(reference: np.ndarray, worlds: int = WORLDS) -> np.ndarray:
    return np.repeat(reference[:, None], worlds, axis=1)


def unit_vertices(bodies: int = BODIES, count: int = VERTICES) -> np.ndarray:
    generator = np.random.default_rng(0)
    return generator.normal(size=(bodies, count, 3)) * 0.05


def completed(worlds: int = WORLDS) -> Completion:
    return Completion(
        completion_step=np.full(worlds, STEPS - 1, dtype=np.int32),
        terminated=np.zeros(worlds, dtype=np.bool_),
        truncated=np.ones(worlds, dtype=np.bool_),
        reference_progress=np.ones(worlds, dtype=np.float32),
    )


def clean_tracking_error(steps: int = STEPS, worlds: int = WORLDS) -> np.ndarray:
    """Per-step wrist/object errors comfortably inside every trained limit."""
    return np.zeros((steps, worlds, 4))


def quaternion_about_z(angle: float) -> np.ndarray:
    """xyzw quaternion for a rotation of ``angle`` radians about +Z."""
    return np.array([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)])


def spider_reference_angle(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> float:
    """Oracle geodesic angle following the originating wxyz axis-angle formulation."""

    def to_wxyz(value: np.ndarray) -> np.ndarray:
        return np.array([value[3], value[0], value[1], value[2]])

    a, b = to_wxyz(a_xyzw), to_wxyz(b_xyzw)
    conjugate = np.array([b[0], -b[1], -b[2], -b[3]])
    relative = np.array(
        [
            conjugate[0] * a[0] - conjugate[1] * a[1] - conjugate[2] * a[2] - conjugate[3] * a[3],
            conjugate[0] * a[1] + conjugate[1] * a[0] + conjugate[2] * a[3] - conjugate[3] * a[2],
            conjugate[0] * a[2] - conjugate[1] * a[3] + conjugate[2] * a[0] + conjugate[3] * a[1],
            conjugate[0] * a[3] + conjugate[1] * a[2] - conjugate[2] * a[1] + conjugate[3] * a[0],
        ]
    )
    axis_norm = np.linalg.norm(relative[1:])
    if axis_norm == 0.0:
        return 0.0
    speed = 2.0 * np.arctan2(axis_norm, relative[0])
    if speed > np.pi:
        speed -= 2.0 * np.pi
    return float(abs(speed))


def test_perfect_tracking_scores_every_metric_at_one():
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    metrics = compute_object_tracking_metrics(
        achieved,
        reference,
        unit_vertices(),
        np.arange(BODIES),
        clean_tracking_error(),
        TRAINED_THRESHOLDS,
        completed(),
        ("top", "bottom"),
    )
    assert metrics.chord_sr == 1.0
    assert metrics.add_auc == pytest.approx(1.0)
    assert metrics.spider_sr_uncentered == 1.0
    assert metrics.maniptrans_sr == 1.0
    assert metrics.mean_add_m == pytest.approx(0.0, abs=1e-12)
    assert metrics.mppe_cm == pytest.approx(0.0, abs=1e-12)
    assert metrics.position_mean_centered is False


def test_mppe_preserves_translation_and_reports_centimetres():
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    achieved[..., :3] += [0.03, 0.04, 0.0]
    assert object_mppe_cm(achieved, reference) == pytest.approx(5.0)


@pytest.mark.parametrize("angle", [np.pi / 2.0, np.pi])
def test_mppe_rotation_uses_five_centimetre_axis_keypoints(angle):
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    achieved[..., 3:] = quaternion_about_z(angle)
    # Four points move by the chord length; the two points on the Z axis stay fixed.
    expected_cm = (4.0 / 6.0) * 2.0 * 5.0 * np.sin(angle / 2.0)
    assert object_mppe_cm(achieved, reference) == pytest.approx(expected_cm)
    achieved[..., 3:] *= -1.0
    assert object_mppe_cm(achieved, reference) == pytest.approx(expected_cm)


@pytest.mark.parametrize("world_chunk", [1, 32])
def test_mppe_takes_worst_body_before_averaging_frames_and_worlds(world_chunk):
    reference = identity_reference(steps=2, bodies=2)
    achieved = broadcast_cohort(reference, worlds=2)
    achieved[0, 0, 0, 0] += 0.04
    achieved[1, 0, 1, 0] += 0.08
    achieved[0, 1, 1, 0] += 0.02
    achieved[1, 1, 0, 0] += 0.06
    assert object_mppe_cm(achieved, reference, world_chunk=world_chunk) == pytest.approx(5.0)


def test_mppe_rejects_empty_cohort():
    reference = identity_reference()
    with pytest.raises(ValueError, match="at least one frame, finite world, and object body"):
        object_mppe_cm(broadcast_cohort(reference, worlds=0), reference)


def test_constant_translational_bias_is_scored_not_removed():
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    achieved[..., 0] += 0.05

    _, position_error, _ = spider_success(achieved, reference)
    assert position_error == pytest.approx(np.full(WORLDS, 0.05))

    centered = spider_centered_position_error(achieved, reference)
    assert centered == pytest.approx(np.zeros(WORLDS), abs=1e-12)


def test_five_centimetre_bias_separates_the_two_success_rates():
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    achieved[..., 0] += 0.05

    spider_mask, _, _ = spider_success(achieved, reference)
    maniptrans_mask, position_error, _ = maniptrans_success(achieved, reference, np.arange(BODIES))

    assert 0.05 <= SPIDER_POSITION_THRESHOLD_M
    assert 0.05 > MANIPTRANS_POSITION_THRESHOLD_M
    assert spider_mask.all()
    assert not maniptrans_mask.any()
    assert position_error == pytest.approx(np.full(WORLDS, 0.05))


def test_pure_rotation_offset_produces_non_zero_add():
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    achieved[..., 3:] = quaternion_about_z(np.radians(20.0))

    add = object_add(achieved, reference, unit_vertices())
    assert np.all(add > 0.0)

    _, position_error, orientation_error = spider_success(achieved, reference)
    assert position_error == pytest.approx(np.zeros(WORLDS), abs=1e-12)
    assert orientation_error == pytest.approx(np.full(WORLDS, np.radians(20.0)))


def test_add_auc_matches_a_hand_computed_step_curve():
    add = np.full((STEPS, WORLDS, 1), 0.045)
    auc, per_body = add_auc(add)
    accuracy = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert ADD_THRESHOLDS_M.size == accuracy.size
    expected = float(np.trapezoid(accuracy, x=np.linspace(0.0, 1.0, accuracy.size)))
    assert expected == pytest.approx(0.5625)
    assert auc == pytest.approx(0.5625)
    assert per_body == pytest.approx((0.5625,))


def test_geodesic_angle_matches_the_reference_axis_angle_oracle():
    generator = np.random.default_rng(7)
    samples = generator.normal(size=(64, 2, 4))
    samples /= np.linalg.norm(samples, axis=-1, keepdims=True)
    computed = quaternion_geodesic_angle(samples[:, 0], samples[:, 1])
    expected = np.array([spider_reference_angle(a, b) for a, b in samples])
    assert computed == pytest.approx(expected, abs=1e-9)


def test_geodesic_angle_is_sign_invariant_and_bounded():
    generator = np.random.default_rng(11)
    samples = generator.normal(size=(32, 4))
    samples /= np.linalg.norm(samples, axis=-1, keepdims=True)
    identity = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (32, 1))
    angle = quaternion_geodesic_angle(samples, identity)
    assert np.all(angle >= 0.0) and np.all(angle <= np.pi + 1e-12)
    assert quaternion_geodesic_angle(-samples, identity) == pytest.approx(angle)


def test_maniptrans_requires_every_object_to_pass():
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    achieved[:, :, 1, 0] += 0.05

    success, _, _ = maniptrans_success(achieved, reference, np.array([0, 1]))
    assert not success.any()


def test_maniptrans_averages_parts_within_one_object():
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    achieved[:, :, 1, 0] += 0.05

    success, _, _ = maniptrans_success(achieved, reference, np.array([0, 0]))
    assert success.all()
    assert 0.025 < MANIPTRANS_POSITION_THRESHOLD_M


def test_orientation_threshold_applies_to_the_frame_average():
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    achieved[..., 3:] = quaternion_about_z(0.0)
    achieved[0, :, :, 3:] = quaternion_about_z(1.2)

    _, _, orientation_error = spider_success(achieved, reference)
    assert 1.2 > SPIDER_ORIENTATION_THRESHOLD_RAD
    assert orientation_error == pytest.approx(np.full(WORLDS, 1.2 / STEPS))
    mask, _, _ = spider_success(achieved, reference)
    assert mask.all()

    _, _, degrees = maniptrans_success(achieved, reference, np.arange(BODIES))
    assert degrees == pytest.approx(np.full(WORLDS, np.degrees(1.2) / STEPS))
    assert np.degrees(1.2) / STEPS < MANIPTRANS_ORIENTATION_THRESHOLD_DEG


@pytest.mark.parametrize("world_chunk", [1, 2, WORLDS, WORLDS * 3])
def test_add_is_invariant_to_world_chunking(world_chunk):
    generator = np.random.default_rng(3)
    reference = identity_reference()
    achieved = broadcast_cohort(reference).copy()
    achieved[..., :3] += generator.normal(size=achieved[..., :3].shape) * 0.01
    perturbation = generator.normal(size=(STEPS, WORLDS, BODIES, 4))
    achieved[..., 3:] = perturbation / np.linalg.norm(perturbation, axis=-1, keepdims=True)

    vertices = unit_vertices()
    baseline = object_add(achieved, reference, vertices, world_chunk=WORLDS)
    assert object_add(achieved, reference, vertices, world_chunk=world_chunk) == pytest.approx(baseline)


def test_chord_success_requires_reaching_the_end_without_crossing_a_limit():
    completion = completed()
    completion.truncated[1] = False
    completion.reference_progress[2] = 0.5
    tracking_error = clean_tracking_error()
    tracking_error[4, 0, 2] = 0.11
    tracking_error[7, 3, 3] = 0.51

    success, causes = chord_success(tracking_error, TRAINED_THRESHOLDS, completion)
    assert success.tolist() == [False, False, False, False, True]
    assert causes[:, 1].tolist() == [True, False, False, True, False]
    assert not causes[:, 0].any()


def test_object_tracking_errors_are_reported_alongside_the_success_rates():
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    achieved[:, :, 0, 0] += 0.04
    achieved[:, :, 1, 3:] = quaternion_about_z(np.radians(10.0))

    metrics = compute_object_tracking_metrics(
        achieved,
        reference,
        unit_vertices(),
        np.arange(BODIES),
        clean_tracking_error(),
        TRAINED_THRESHOLDS,
        completed(),
        ("top", "bottom"),
    )
    assert metrics.object_position_error_per_body_m == pytest.approx((0.04, 0.0), abs=1e-12)
    assert metrics.object_orientation_error_per_body_rad == pytest.approx((0.0, np.radians(10.0)))
    assert metrics.object_position_error_m == pytest.approx(0.02)
    assert metrics.object_orientation_error_rad == pytest.approx(np.radians(5.0))
    assert metrics.object_orientation_error_deg == pytest.approx(5.0)
    # Every world is identical here, so the across-world spread must vanish.
    assert metrics.object_position_error_std_m == pytest.approx(0.0, abs=1e-12)
    assert metrics.object_orientation_error_std_deg == pytest.approx(0.0, abs=1e-9)


def test_a_disabled_limit_never_contributes_a_crossing():
    tracking_error = clean_tracking_error()
    tracking_error[:, :, 1] = 3.0

    causes = tracking_cause_masks(tracking_error, TRAINED_THRESHOLDS)
    assert not causes.any()
    assert TRAINED_THRESHOLDS.wrist_orientation_rad is None


def test_chord_criteria_ignore_wrist_tracking():
    tracking_error = clean_tracking_error()
    tracking_error[6, 2, 0] = 10.0
    tracking_error[6, 2, 1] = 3.0

    success, causes = chord_success(tracking_error, TRAINED_THRESHOLDS, completed())
    assert success.all()
    assert not causes.any()
    assert TRAINED_THRESHOLDS.gated_causes() == ("object",)


def test_a_single_frame_crossing_fails_the_whole_world():
    tracking_error = clean_tracking_error()
    tracking_error[6, 2, 2] = 0.11

    success, causes = chord_success(tracking_error, TRAINED_THRESHOLDS, completed())
    assert success.tolist() == [True, True, False, True, True]
    assert causes[2].tolist() == [False, True]


def test_wrist_limits_still_gate_when_supplied():
    tracking_error = clean_tracking_error()
    tracking_error[6, 2, 0] = 0.16

    success, causes = chord_success(tracking_error, WRIST_GATED_THRESHOLDS, completed())
    assert success.tolist() == [True, True, False, True, True]
    assert causes[2].tolist() == [True, False]
    assert WRIST_GATED_THRESHOLDS.gated_causes() == ("wrist", "object")


def test_a_self_terminated_rollout_is_rejected():
    completion = completed()
    completion.terminated[3] = True
    with pytest.raises(ValueError, match="must disable the tracking-failure termination"):
        chord_success(clean_tracking_error(), TRAINED_THRESHOLDS, completion)


def test_termination_cause_fraction_is_reported_per_cause():
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    tracking_error = clean_tracking_error()
    tracking_error[3, 0, 2] = 0.2

    metrics = compute_object_tracking_metrics(
        achieved,
        reference,
        unit_vertices(),
        np.arange(BODIES),
        tracking_error,
        TRAINED_THRESHOLDS,
        completed(),
        ("top", "bottom"),
    )
    assert metrics.chord_sr == pytest.approx((WORLDS - 1) / WORLDS)
    assert metrics.gated_causes == ("object",)
    assert metrics.termination_cause_fraction == {"object": 1.0 / WORLDS}


def test_mismatched_reference_shape_is_rejected():
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    with pytest.raises(ValueError, match="reference_pose_w must have shape"):
        spider_success(achieved, reference[:-1])


def test_vertex_array_must_cover_every_body():
    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    with pytest.raises(ValueError, match="object_vertices_o must have shape"):
        object_add(achieved, reference, unit_vertices(bodies=BODIES - 1))


def test_evaluation_report_is_world_readable_and_scalars_are_not_duplicated_by_reset_mode(tmp_path):
    """The OSMO upload worker runs as another user, so the report must not stay at mkstemp's 0600."""
    from flash_chord.evaluation.harness import (
        EvaluationOutcome,
        ReferenceTrackingMetrics,
        evaluation_scalars,
        write_evaluation_report,
    )

    reference = identity_reference()
    achieved = broadcast_cohort(reference)
    achieved[..., 0] += 0.03
    metrics = compute_object_tracking_metrics(
        achieved,
        reference,
        unit_vertices(),
        np.arange(BODIES),
        clean_tracking_error(),
        TRAINED_THRESHOLDS,
        completed(),
        ("top", "bottom"),
    )
    outcome = EvaluationOutcome(
        metrics=metrics,
        reference_tracking=ReferenceTrackingMetrics(
            success_rate=1.0,
            reference_end_fraction=1.0,
            failure_cause_fraction={"object": 0.0},
        ),
        checkpoint="ckpt",
        checkpoint_sha256="0" * 64,
        environment_steps=1,
        parquet="p",
        world_count=WORLDS,
        start_frame=0,
        step_count=STEPS,
        reset_mode="sampled_settled",
        settling_steps=50,
        motion_start_frame=0,
        motion_end_frame=-1,
        sim_control_fps=20.0,
        sim_physics_fps=200.0,
        object_position_threshold_m=0.1,
        object_orientation_threshold_rad=0.5,
        non_finite_worlds=(),
    )
    written = write_evaluation_report(tmp_path / "evaluation.json", outcome)
    assert stat.S_IMODE(written.stat().st_mode) == 0o644
    assert json.loads(written.read_text())["metrics"]["chord_sr"] == 1.0
    assert json.loads(written.read_text())["metrics"]["mppe_cm"] == pytest.approx(3.0)
    scalars = evaluation_scalars(outcome)
    assert "evaluation/reference_trajectory_success" not in scalars
    assert "evaluation/reference_end_fraction" not in scalars
    assert not any(name.startswith("evaluation/reference_failure_fraction/") for name in scalars)
    assert scalars["evaluation/chord_sr"] == 1.0
    assert scalars["evaluation/mppe_cm"] == pytest.approx(3.0)
    assert not any(name.startswith("evaluation_sampled_settled/") for name in scalars)


def test_standard_deviations_track_spread_across_worlds_and_bodies():
    """SPIDER reports error mean +/- std across trajectories; dexmachina reports ADD std pooled."""
    reference = identity_reference()
    achieved = broadcast_cohort(reference).copy()
    offsets = np.array([0.00, 0.01, 0.02, 0.03, 0.04])
    achieved[:, :, :, 0] += offsets[None, :, None]

    metrics = compute_object_tracking_metrics(
        achieved,
        reference,
        unit_vertices(),
        np.arange(BODIES),
        clean_tracking_error(),
        TRAINED_THRESHOLDS,
        completed(),
        ("top", "bottom"),
    )
    assert metrics.object_position_error_m == pytest.approx(offsets.mean())
    assert metrics.object_position_error_std_m == pytest.approx(offsets.std())
    assert metrics.add_std_m > 0.0
    assert len(metrics.add_std_per_body_m) == BODIES
    assert metrics.mean_add_m == pytest.approx(offsets.mean(), abs=1e-9)
