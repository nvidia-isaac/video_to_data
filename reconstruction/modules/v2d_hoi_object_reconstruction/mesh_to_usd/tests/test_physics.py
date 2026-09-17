import math

import pytest

from physics import (
    PoseSettleTracker,
    SettleTracker,
    aggregate_pose_results,
    classify_drop_test_outcome,
    classify_standing,
    ground_metrics,
    is_within_ground_tolerance,
    physics_contact_config,
    tilt_from_vertical_degrees,
)


def test_pose_settle_tracker_accepts_small_stationary_contact_jitter():
    tracker = PoseSettleTracker(
        required_frames=3,
        position_tolerance=0.01,
        angular_tolerance_degrees=2.0,
    )

    assert not tracker.observe(
        contact_active=True,
        position=(0.0, 0.0, 0.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    assert not tracker.observe(
        contact_active=True,
        position=(0.002, 0.0, 0.0),
        orientation_wxyz=(0.999961923, 0.008726535, 0.0, 0.0),
    )
    assert tracker.observe(
        contact_active=True,
        position=(-0.002, 0.0, 0.0),
        orientation_wxyz=(0.999961923, -0.008726535, 0.0, 0.0),
    )
    assert tracker.position_span == pytest.approx(0.004)
    assert tracker.angular_span_degrees == pytest.approx(2.0, abs=1e-5)


def test_pose_settle_tracker_rejects_opposite_extrema_in_window():
    tracker = PoseSettleTracker(
        required_frames=3,
        position_tolerance=0.01,
        angular_tolerance_degrees=2.0,
    )

    for position in (0.0, 0.009, -0.009):
        tracker.observe(
            contact_active=True,
            position=(position, 0.0, 0.0),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )

    assert not tracker.settled
    assert tracker.position_span == pytest.approx(0.018)


def test_pose_settle_tracker_rejects_planar_translation_and_yaw():
    tracker = PoseSettleTracker(
        required_frames=3,
        position_tolerance=0.01,
        angular_tolerance_degrees=2.0,
    )

    yaw_quaternions = (
        (1.0, 0.0, 0.0, 0.0),
        (0.996194698, 0.0, 0.0, 0.087155743),
        (0.984807753, 0.0, 0.0, 0.173648178),
    )
    for position, orientation in zip((0.0, 0.02, 0.04), yaw_quaternions):
        tracker.observe(
            contact_active=True,
            position=(position, 0.0, 0.0),
            orientation_wxyz=orientation,
        )

    assert not tracker.settled
    assert tracker.position_span == pytest.approx(0.04)
    assert tracker.angular_span_degrees == pytest.approx(20.0, abs=1e-5)


def test_pose_settle_tracker_rejects_vertical_motion():
    tracker = PoseSettleTracker(
        required_frames=3,
        position_tolerance=0.01,
        angular_tolerance_degrees=2.0,
    )

    for height in (0.0, 0.006, 0.012):
        tracker.observe(
            contact_active=True,
            position=(0.0, 0.0, height),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )

    assert not tracker.settled
    assert tracker.position_span == pytest.approx(0.012)


def test_pose_settle_tracker_rejects_excessive_support_wobble():
    tracker = PoseSettleTracker(
        required_frames=3,
        position_tolerance=0.02,
        angular_tolerance_degrees=10.0,
    )

    for orientation in (
        (1.0, 0.0, 0.0, 0.0),
        (0.996194698, 0.087155743, 0.0, 0.0),
        (0.994521895, 0.104528463, 0.0, 0.0),
    ):
        tracker.observe(
            contact_active=True,
            position=(0.0, 0.0, 0.0),
            orientation_wxyz=orientation,
        )

    assert not tracker.settled
    assert tracker.angular_span_degrees == pytest.approx(12.0, abs=1e-5)


def test_settle_tracker_revalidates_after_post_settle_motion():
    tracker = SettleTracker(
        required_frames=3,
        linear_speed_limit=0.01,
        angular_speed_limit=0.05,
    )

    for _ in range(3):
        tracker.observe(
            contact_active=True,
            linear_speed=0.0,
            angular_speed=0.0,
        )
    assert tracker.settled

    tracker.observe(
        contact_active=True,
        linear_speed=0.02,
        angular_speed=0.0,
    )
    assert not tracker.settled
    assert tracker.consecutive_frames == 0

    for _ in range(2):
        tracker.observe(
            contact_active=True,
            linear_speed=0.0,
            angular_speed=0.0,
        )
    assert not tracker.settled

    tracker.observe(
        contact_active=True,
        linear_speed=0.0,
        angular_speed=0.0,
    )
    assert tracker.settled


def test_physics_contact_config_is_scale_aware_and_bounded():
    tennis_ball = physics_contact_config(0.066)
    assert tennis_ball["rest_offset_m"] == 0.0
    assert tennis_ball["contact_offset_m"] == pytest.approx(0.00132)
    assert tennis_ball["solver_position_iterations"] == 32
    assert tennis_ball["solver_velocity_iterations"] == 4
    assert tennis_ball["ccd_enabled"] is True

    assert physics_contact_config(0.001)["contact_offset_m"] == 0.001
    assert physics_contact_config(10.0)["contact_offset_m"] == 0.005


def test_ground_metrics_distinguish_gap_from_penetration():
    floating = ground_metrics((-0.1, -0.1, 0.002), (0.1, 0.1, 0.102), 0.0)
    assert floating["clearance"] == pytest.approx(0.002)
    assert floating["gap"] == pytest.approx(0.002)
    assert floating["penetration"] == 0.0
    assert floating["gap_ratio"] == pytest.approx(0.01)

    penetrating = ground_metrics((-0.1, -0.1, -0.003), (0.1, 0.1, 0.097), 0.0)
    assert penetrating["clearance"] == pytest.approx(-0.003)
    assert penetrating["gap"] == 0.0
    assert penetrating["penetration"] == pytest.approx(0.003)
    assert penetrating["penetration_ratio"] == pytest.approx(0.015)


def test_ground_tolerance_accepts_small_gap_or_penetration():
    assert is_within_ground_tolerance(0.01, 0.02)
    assert is_within_ground_tolerance(-0.01, 0.02)
    assert not is_within_ground_tolerance(0.021, 0.02)


def test_identity_is_upright():
    assert tilt_from_vertical_degrees((1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0)


def test_quarter_turn_is_fallen():
    quaternion = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    result = classify_standing(
        quaternion_wxyz=quaternion,
        settled=True,
        contact_seen=True,
        touching_ground=True,
        max_tilt_degrees=15.0,
    )
    assert result.tilt_degrees == pytest.approx(90.0)
    assert not result.standing


def test_drop_outcome_keeps_rigid_and_standing_results_separate():
    outcome = classify_drop_test_outcome(
        contact_seen=True,
        settled=True,
        timed_out=False,
        standing=False,
        standing_required=True,
    )
    assert outcome.rigid_body_test_passed
    assert not outcome.standing_requirement_passed
    assert not outcome.passed
    assert outcome.failure_reasons == ("optional_standing_requirement_not_met",)


def test_pose_aggregation_passes_when_any_candidate_passes():
    result = aggregate_pose_results(
        [
            {
                "pose_id": "principal_axis_0_positive",
                "rigid_body_test_passed": True,
                "standing": False,
                "tilt_degrees": 90.0,
            },
            {
                "pose_id": "principal_axis_0_negative",
                "rigid_body_test_passed": True,
                "standing": True,
                "tilt_degrees": 2.0,
            },
        ],
        standing_required=True,
    )

    assert result["passed"]
    assert result["rigid_body_pass_count"] == 2
    assert result["standing_pass_count"] == 1
    assert result["representative_pose_id"] == "principal_axis_0_negative"


def test_pose_aggregation_reports_missing_standing_candidate():
    result = aggregate_pose_results(
        [
            {
                "pose_id": "principal_axis_0_positive",
                "rigid_body_test_passed": True,
                "standing": False,
                "tilt_degrees": 90.0,
            }
        ],
        standing_required=True,
    )

    assert not result["passed"]
    assert result["rigid_body_test_passed"]
    assert result["failure_reasons"] == [
        "no_pose_candidate_met_standing_requirement"
    ]


def test_pose_aggregation_requires_one_candidate_to_pass_all_checks():
    result = aggregate_pose_results(
        [
            {
                "pose_id": "rigid_only",
                "rigid_body_test_passed": True,
                "standing": False,
                "tilt_degrees": 90.0,
            },
            {
                "pose_id": "standing_only",
                "rigid_body_test_passed": False,
                "standing": True,
                "tilt_degrees": 0.0,
            },
        ],
        standing_required=True,
    )

    assert not result["passed"]
    assert result["rigid_body_test_passed"]
    assert result["standing"]
    assert not result["standing_requirement_passed"]
    assert result["failure_reasons"] == [
        "no_pose_candidate_met_all_required_checks"
    ]
