from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from inpainting import grasp_refinement as refinement


def _target(frame_count: int = 12) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(
            "v2d.inpainting.parallel-jaw-target/v1"
        ),
        "tracker": np.asarray("v2d"),
        "coordinate_frame": np.asarray("world"),
        "frame_indices": np.arange(frame_count, dtype=np.int32),
    }
    for side_index, side in enumerate(("left", "right")):
        arrays[f"{side}_valid"] = np.ones(frame_count, dtype=np.bool_)
        positions = np.zeros((frame_count, 3), dtype=np.float32)
        positions[:, 0] = np.arange(frame_count, dtype=np.float32) * 0.01
        positions[:, 1] = float(side_index)
        quaternions = np.zeros((frame_count, 4), dtype=np.float32)
        quaternions[:, 0] = 1.0
        arrays[f"{side}_position"] = positions
        arrays[f"{side}_wxyz"] = quaternions
        arrays[f"{side}_aperture_m"] = np.full(
            frame_count,
            0.08,
            dtype=np.float32,
        )
    return arrays


def _candidate(
    index: int,
    contacts: np.ndarray,
    *,
    pose: np.ndarray | None = None,
    aperture_m: float | None = None,
    antipodal_score: float = 1.0,
) -> refinement.CandidateContacts:
    pose = np.eye(4) if pose is None else pose
    if aperture_m is None:
        aperture_m = float(np.linalg.norm(contacts[1] - contacts[0]))
    return refinement.CandidateContacts(
        candidate_index=index,
        T_object_gripper=pose,
        valid=True,
        contact_points_object=np.asarray(contacts, dtype=np.float64),
        contact_normals_object=np.array(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        aperture_m=aperture_m,
        antipodal_score=antipodal_score,
        ray_yz_gripper=np.zeros(2),
    )


def test_strict_target_load_rejects_extra_keys(tmp_path: Path) -> None:
    target = _target(3)
    path = tmp_path / "parallel_jaw_trajectory.npz"
    np.savez_compressed(path, **target)
    loaded = refinement.load_parallel_jaw_target(path, expected_frames=3)
    assert set(loaded) == set(target)

    target["implicit_mount_offset"] = np.eye(4)
    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(invalid, **target)
    with pytest.raises(
        refinement.GraspRefinementError,
        match="unexpected=.*implicit_mount_offset",
    ):
        refinement.load_parallel_jaw_target(invalid)


def test_wxyz_matrix_round_trip_and_batch() -> None:
    xyzw = Rotation.from_euler(
        "xyz",
        [[20, -15, 40], [0, 0, 180]],
        degrees=True,
    ).as_quat()
    wxyz = np.column_stack((xyzw[:, 3], xyzw[:, :3]))
    matrices = refinement.wxyz_to_matrix(wxyz)
    recovered = refinement.matrix_to_wxyz(matrices)

    np.testing.assert_allclose(
        refinement.wxyz_to_matrix(recovered),
        matrices,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        refinement.wxyz_to_matrix([1.0, 0.0, 0.0, 0.0]),
        np.eye(3),
    )


def test_foundationpose_batch_forms_world_from_camera_inverse(tmp_path: Path) -> None:
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    values = [
        {
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "translation": [1.0, 2.0, 4.0],
            "scale": [1.0, 1.0, 1.0],
        },
        {
            "rotation": [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            "translation": [2.0, 2.0, 4.0],
            "scale": [1.0, 1.0, 1.0],
        },
    ]
    for frame, value in enumerate(values):
        (pose_dir / f"{frame:06d}.json").write_text(json.dumps(value))
    camera_from_world = np.repeat(np.eye(4)[None], 2, axis=0)
    camera_from_world[:, :3, 3] = [1.0, 2.0, 3.0]

    world_from_object = refinement.load_foundationpose_wxyz_batch(
        pose_dir,
        camera_from_world,
        expected_frames=2,
    )

    np.testing.assert_allclose(world_from_object[0, :3, 3], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(world_from_object[1, :3, 3], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        world_from_object[1, :3, :3],
        Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        atol=1e-12,
    )


def test_events_are_hysteretic_multiple_and_mark_initial_contact() -> None:
    aperture = np.array(
        [0.02, 0.025, 0.05, 0.045, 0.02, 0.025, 0.06, 0.08, 0.02, 0.02, 0.08]
    )
    events = refinement.segment_grasp_events(
        aperture,
        close_threshold_m=0.03,
        open_threshold_m=0.07,
        min_duration_frames=2,
    )

    assert events == (
        refinement.GraspEvent(0, 6, 0, True),
        refinement.GraspEvent(8, 9, 8, False),
    )


def test_event_gap_merge_happens_before_minimum_duration_filter() -> None:
    events = refinement.segment_grasp_events(
        [0.08, 0.02, 0.08, 0.02, 0.08],
        close_threshold_m=0.03,
        open_threshold_m=0.07,
        min_duration_frames=3,
        max_gap_frames=1,
    )
    assert events == (refinement.GraspEvent(1, 3, 1, False),)


def test_contact_derivation_uses_finite_pad_sweep_and_opposing_normals() -> None:
    mesh = trimesh.creation.box(extents=[0.04, 0.02, 0.04])
    sweep = refinement.ParallelJawSweepVolume(
        x_bounds_m=(-0.05, 0.05),
        pad_y_bounds_m=(-0.008, 0.008),
        pad_z_bounds_m=(-0.015, 0.015),
        samples_y=5,
        samples_z=5,
    )

    contact = refinement.derive_parallel_jaw_candidate_contacts(
        mesh,
        np.eye(4),
        sweep_volume=sweep,
    )[0]

    assert contact.valid
    assert contact.aperture_m == pytest.approx(0.04)
    assert contact.antipodal_score == pytest.approx(1.0)
    assert contact.contact_points_object is not None
    np.testing.assert_allclose(
        contact.contact_points_object[:, 0],
        [-0.02, 0.02],
        atol=1e-8,
    )

    shifted = np.eye(4)
    shifted[1, 3] = 0.1
    missed = refinement.derive_parallel_jaw_candidate_contacts(
        mesh,
        shifted,
        sweep_volume=sweep,
    )[0]
    assert not missed.valid
    assert "no opposing" in str(missed.reason)


def test_scoring_uses_unordered_contacts_and_hard_filters() -> None:
    human = np.array([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]])
    exact_swapped = _candidate(0, human[::-1])
    wrong = _candidate(
        1,
        np.array([[-0.02, 0.03, 0.0], [0.02, 0.03, 0.0]]),
    )
    too_wide = _candidate(2, human, aperture_m=0.2)

    scores = refinement.score_grasp_candidates(
        [exact_swapped, wrong, too_wide],
        human,
        confidences=[0.2, 1.0, 1.0],
        aperture_limits_m=(0.005, 0.08),
        registration_weight=1.0,
    )

    assert refinement.select_best_candidate(scores).candidate_index == 0
    assert scores[0].contact_distance_m == pytest.approx(0.0)
    assert not scores[2].feasible
    assert np.isinf(scores[2].total_cost)


def test_translation_registered_residual_and_fallback_weight() -> None:
    candidate_pair = np.array([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]])
    human = candidate_pair + [0.1, -0.02, 0.03]
    residual, translation, swapped = (
        refinement.translation_registered_contact_residual(candidate_pair, human)
    )
    assert residual == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(translation, [0.1, -0.02, 0.03])
    assert not swapped

    scores = refinement.score_grasp_candidates(
        [_candidate(0, candidate_pair)],
        human,
        aperture_limits_m=(0.005, 0.08),
        weights=refinement.GraspScoreWeights(
            confidence=0.0,
            pose_translation=0.0,
            pose_rotation=0.0,
            approach=0.0,
            registration_weight_fallback=0.0,
        ),
    )
    assert scores[0].registration_weight == 0.0
    assert scores[0].total_cost == pytest.approx(0.0, abs=1e-12)


def test_parallel_jaw_pose_prior_treats_pi_roll_as_equivalent() -> None:
    identity = np.eye(3)
    flipped = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    distance, symmetry_flipped = refinement.parallel_jaw_rotation_distance(
        flipped,
        identity,
    )
    assert distance == pytest.approx(0.0, abs=1e-12)
    assert symmetry_flipped

    human_pose = np.eye(4)
    flipped_pose = np.eye(4)
    flipped_pose[:3, :3] = flipped
    contacts = np.array([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]])
    scores = refinement.score_grasp_candidates(
        [
            _candidate(0, contacts, pose=np.eye(4)),
            _candidate(1, contacts[::-1], pose=flipped_pose),
        ],
        contacts,
        human_pose_object=human_pose,
        aperture_limits_m=(0.005, 0.08),
        registration_weight=1.0,
    )
    assert scores[0].total_cost == pytest.approx(scores[1].total_cost, abs=1e-12)


def test_wrench_low_tail_reward_is_primary_and_reference_is_optional() -> None:
    contacts = np.array([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]])
    candidates = [_candidate(0, contacts), _candidate(1, contacts)]
    metrics = {
        0: refinement.CandidateWrenchMetrics(
            candidate_index=0,
            low_quantile_support=0.01,
            mean_support=0.3,
            support_coverage=0.9,
        ),
        1: refinement.CandidateWrenchMetrics(
            candidate_index=1,
            low_quantile_support=0.12,
            mean_support=0.2,
            support_coverage=0.8,
        ),
    }
    scores = refinement.score_grasp_candidates(
        candidates,
        contacts,
        confidences=[1.0, 1.0],
        aperture_limits_m=(0.005, 0.08),
        registration_weight=1.0,
        wrench_metrics=metrics,
        weights=refinement.GraspScoreWeights(
            contact=0.0,
            confidence=0.0,
            pose_translation=0.0,
            pose_rotation=0.0,
            approach=0.0,
            wrench_low_tail_support=1.0,
        ),
    )

    assert refinement.select_best_candidate(scores).candidate_index == 1
    assert scores[1].wrench_low_quantile_support == pytest.approx(0.12)
    assert scores[1].wrench_reference_match is None
    assert scores[1].wrench_weighted_reward == pytest.approx(0.12)
    assert scores[1].total_cost == pytest.approx(-0.12)


def test_wrench_reference_weight_requires_exact_reference_metrics() -> None:
    contacts = np.array([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]])
    with pytest.raises(
        refinement.GraspRefinementError,
        match="exact reference metrics",
    ):
        refinement.score_grasp_candidates(
            [_candidate(0, contacts)],
            contacts,
            aperture_limits_m=(0.005, 0.08),
            wrench_metrics={
                0: refinement.CandidateWrenchMetrics(
                    candidate_index=0,
                    low_quantile_support=0.1,
                    mean_support=0.2,
                    support_coverage=1.0,
                )
            },
            weights=refinement.GraspScoreWeights(
                wrench_reference_match=0.5,
            ),
        )


def test_phase_correction_locks_hold_slerps_and_preserves_untouched_frames() -> None:
    target = _target(12)
    original = {key: value.copy() for key, value in target.items()}
    world_from_object = np.repeat(np.eye(4)[None], 12, axis=0)
    world_from_object[:, 1, 3] = np.linspace(0.2, 0.31, 12)
    object_from_gripper = np.eye(4)
    object_from_gripper[:3, :3] = Rotation.from_euler(
        "z", 90, degrees=True
    ).as_matrix()
    object_from_gripper[:3, 3] = [0.0, 0.04, 0.0]
    event = refinement.GraspEvent(4, 6, 5, False)

    output = refinement.apply_phase_aware_corrections(
        target,
        side="left",
        T_world_object=world_from_object,
        corrections=[
            refinement.GraspCorrection(event, object_from_gripper, 0.04)
        ],
        approach_blend_frames=2,
        release_blend_frames=2,
    )

    assert set(output) == set(original)
    for key in output:
        assert output[key].dtype == original[key].dtype
    for frame in (0, 1, 9, 10, 11):
        np.testing.assert_array_equal(
            output["left_position"][frame],
            original["left_position"][frame],
        )
        np.testing.assert_array_equal(
            output["left_wxyz"][frame],
            original["left_wxyz"][frame],
        )
        assert output["left_aperture_m"][frame] == original["left_aperture_m"][frame]
    for key in (
        "right_position",
        "right_wxyz",
        "right_aperture_m",
        "tracker",
        "frame_indices",
    ):
        np.testing.assert_array_equal(output[key], original[key])

    hold_poses = refinement.pose_matrix(
        output["left_position"][4:7],
        output["left_wxyz"][4:7],
    )
    np.testing.assert_allclose(
        hold_poses,
        world_from_object[4:7] @ object_from_gripper,
        atol=2e-6,
    )
    np.testing.assert_allclose(output["left_aperture_m"][4:7], 0.04)

    approach_rotation = refinement.wxyz_to_matrix(output["left_wxyz"][3])
    approach_angle = Rotation.from_matrix(approach_rotation).magnitude()
    assert 0.0 < approach_angle < np.pi / 2.0
    assert 0.04 < output["left_aperture_m"][3] < 0.08
    refinement.validate_parallel_jaw_target(output, expected_frames=12)


def test_starts_in_contact_is_object_locked_from_frame_zero() -> None:
    target = _target(4)
    object_poses = np.repeat(np.eye(4)[None], 4, axis=0)
    object_poses[:, 2, 3] = [0.1, 0.2, 0.3, 0.4]
    grasp = np.eye(4)
    grasp[0, 3] = 0.02

    output = refinement.apply_phase_aware_corrections(
        target,
        side="right",
        T_world_object=object_poses,
        corrections=[
            refinement.GraspCorrection(
                refinement.GraspEvent(0, 2, 0, True),
                grasp,
                0.03,
            )
        ],
        approach_blend_frames=3,
        release_blend_frames=0,
    )
    poses = refinement.pose_matrix(
        output["right_position"][:3],
        output["right_wxyz"][:3],
    )
    np.testing.assert_allclose(poses, object_poses[:3] @ grasp, atol=2e-7)


def test_base_local_offset_is_exact_and_ignores_non_anchor_object_jitter() -> None:
    target = _target(8)
    base_positions = np.column_stack(
        (
            np.linspace(0.0, 0.14, 8),
            np.linspace(-0.03, 0.04, 8),
            np.linspace(0.2, 0.27, 8),
        )
    )
    base_rotations = Rotation.from_euler(
        "z",
        np.linspace(-20.0, 50.0, 8)[:, None],
        degrees=True,
    ).as_matrix()
    target["left_position"] = base_positions.astype(np.float32)
    target["left_wxyz"] = refinement.matrix_to_wxyz(base_rotations).astype(
        np.float32
    )
    base_poses = refinement.pose_matrix(
        target["left_position"],
        target["left_wxyz"],
    )

    object_poses = np.repeat(np.eye(4)[None], 8, axis=0)
    object_poses[:, :3, :3] = Rotation.from_euler(
        "xyz",
        np.column_stack(
            (
                np.linspace(0.0, 140.0, 8),
                np.linspace(0.0, -70.0, 8),
                np.linspace(0.0, 100.0, 8),
            )
        ),
        degrees=True,
    ).as_matrix()
    object_poses[:, :3, 3] = np.column_stack(
        (
            np.linspace(-0.5, 1.5, 8),
            np.linspace(0.8, -0.6, 8),
            np.linspace(0.1, 1.2, 8),
        )
    )
    anchor = 4
    object_gripper = (
        np.linalg.inv(object_poses[anchor])
        @ base_poses[anchor]
    )
    object_gripper[:3, 3] += [0.025, -0.012, 0.031]
    # This is the gripper's parallel-jaw-equivalent pi roll.  The local-offset
    # mode must nevertheless use the selected input transform literally.
    object_gripper[:3, :3] = (
        object_gripper[:3, :3]
        @ Rotation.from_euler("z", 180.0, degrees=True).as_matrix()
    )
    event = refinement.GraspEvent(2, 5, anchor, False)

    output = refinement.apply_phase_aware_corrections(
        target,
        side="left",
        T_world_object=object_poses,
        corrections=[
            refinement.GraspCorrection(
                event,
                object_gripper,
                0.035,
                refinement.GraspPropagationMode.BASE_LOCAL_OFFSET,
            )
        ],
        approach_blend_frames=1,
        release_blend_frames=1,
    )

    anchor_delta = (
        np.linalg.inv(base_poses[anchor])
        @ (object_poses[anchor] @ object_gripper)
    )
    expected_hold = base_poses[event.start_frame : event.end_frame + 1] @ anchor_delta
    actual_hold = refinement.pose_matrix(
        output["left_position"][event.start_frame : event.end_frame + 1],
        output["left_wxyz"][event.start_frame : event.end_frame + 1],
    )
    np.testing.assert_allclose(actual_hold, expected_hold, atol=3e-6)
    np.testing.assert_allclose(
        output["left_aperture_m"][event.start_frame : event.end_frame + 1],
        0.035,
        atol=1e-7,
    )

    # The deliberately erratic non-anchor object poses are not propagated.
    object_locked = (
        object_poses[event.start_frame : event.end_frame + 1]
        @ object_gripper
    )
    assert np.max(
        np.linalg.norm(
            actual_hold[:, :3, 3] - object_locked[:, :3, 3],
            axis=1,
        )
    ) > 0.5
