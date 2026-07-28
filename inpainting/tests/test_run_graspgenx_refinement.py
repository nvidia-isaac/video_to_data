from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from inpainting.contracts import TRACKING_SCHEMA
from inpainting import grasp_refinement
from inpainting import run_graspgenx_refinement as runner


def _target(frame_count: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {
        "schema_version": np.asarray(
            "v2d.inpainting.parallel-jaw-target/v1"
        ),
        "tracker": np.asarray("v2d"),
        "coordinate_frame": np.asarray("world"),
        "frame_indices": np.arange(frame_count, dtype=np.int32),
    }
    for side in ("left", "right"):
        result[f"{side}_valid"] = np.ones(frame_count, dtype=np.bool_)
        result[f"{side}_position"] = np.zeros(
            (frame_count, 3),
            dtype=np.float32,
        )
        wxyz = np.zeros((frame_count, 4), dtype=np.float32)
        wxyz[:, 0] = 1.0
        result[f"{side}_wxyz"] = wxyz
        result[f"{side}_aperture_m"] = np.full(
            frame_count,
            0.08,
            dtype=np.float32,
        )
    return result


def _tracking(frame_count: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRACKING_SCHEMA),
        "tracker": np.asarray("v2d"),
        "coordinate_frame": np.asarray("world"),
        "frame_indices": np.arange(frame_count, dtype=np.int32),
    }
    for side in ("left", "right"):
        joints = np.zeros((frame_count, 21, 3), dtype=np.float32)
        joints[:, 4] = [-0.02, 0.015, 0.0]
        joints[:, 8] = [0.02, 0.015, 0.0]
        result[f"{side}_valid"] = np.ones(frame_count, dtype=np.bool_)
        result[f"{side}_wrist_position"] = np.zeros(
            (frame_count, 3),
            dtype=np.float32,
        )
        wrist_wxyz = np.zeros((frame_count, 4), dtype=np.float32)
        wrist_wxyz[:, 0] = 1.0
        result[f"{side}_wrist_wxyz"] = wrist_wxyz
        result[f"{side}_joints_3d"] = joints
    return result


def _write_fixture(
    directory: Path,
    *,
    robot_profile: str = "galbot_one_golf",
    frame_count: int = 7,
) -> dict[str, Path | int | float]:
    directory.mkdir(parents=True, exist_ok=True)
    profile = runner.resolve_robot_profile(robot_profile)
    target_path = directory / "base.npz"
    np.savez_compressed(target_path, **_target(frame_count))
    tracking_path = directory / "tracking.npz"
    np.savez_compressed(tracking_path, **_tracking(frame_count))
    calibration_path = directory / "T_camera_world.npy"
    np.save(
        calibration_path,
        np.repeat(np.eye(4)[None], frame_count, axis=0),
    )
    poses = directory / "poses"
    poses.mkdir()
    for frame in range(frame_count):
        (poses / f"{frame:06d}.json").write_text(
            json.dumps(
                {
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                }
            )
        )
    mesh_path = directory / "object.ply"
    trimesh.creation.box(extents=[0.04, 0.02, 0.04]).export(mesh_path)
    candidate = np.eye(4, dtype=np.float32)
    candidate[2, 3] = -profile.gripper_base_to_contact_z_m
    candidates_path = directory / "candidates.npz"
    np.savez_compressed(
        candidates_path,
        object_to_gripper_base=candidate[None],
        confidence=np.asarray([0.9], dtype=np.float32),
    )
    (directory / "candidates.json").write_text(
        json.dumps(
            {
                "schema_version": "v2d.inpainting.graspgenx-candidates/v1",
                "gripper": profile.profile_facts,
            }
        )
    )
    return {
        "target": target_path,
        "tracking": tracking_path,
        "calibration": calibration_path,
        "poses": poses,
        "mesh": mesh_path,
        "candidates": candidates_path,
        "frame_count": frame_count,
        "depth": profile.gripper_base_to_contact_z_m,
    }


def _execute(
    fixture: dict[str, Path | int | float],
    *,
    side: str,
    output: Path,
    base_target: Path | None = None,
    robot_profile: str = "galbot_one_golf",
    propagation_mode: str = "object_lock",
    overwrite: bool = False,
    **execute_kwargs: object,
) -> dict:
    depth = float(fixture["depth"])
    return runner.execute(
        base_target=base_target or Path(fixture["target"]),
        tracking=Path(fixture["tracking"]),
        T_camera_world=Path(fixture["calibration"]),
        foundationpose_poses=Path(fixture["poses"]),
        mesh=Path(fixture["mesh"]),
        candidates=Path(fixture["candidates"]),
        robot_profile=robot_profile,
        side=side,
        object_name="test_object",
        event_start=2,
        event_end=4,
        event_anchor=3,
        starts_in_contact=False,
        output_target=output,
        overwrite=overwrite,
        approach_blend_frames=0,
        release_blend_frames=0,
        propagation_mode=propagation_mode,
        pad_y_bounds_m=(-0.005, 0.005),
        pad_z_bounds_m=(depth, depth + 0.001),
        sweep_samples_y=5,
        sweep_samples_z=3,
        **execute_kwargs,
    )


def test_profile_transforms_and_candidate_matched_sweep_bounds() -> None:
    galbot = runner.resolve_robot_profile("galbot_one_golf")
    np.testing.assert_allclose(
        galbot.sweep_volume.x_bounds_m,
        [-0.06245438313670121, 0.06245438313670121],
    )
    np.testing.assert_allclose(
        galbot.sweep_volume.pad_y_bounds_m,
        [-0.01015, 0.01015],
    )
    np.testing.assert_allclose(
        galbot.sweep_volume.pad_z_bounds_m,
        [0.099338875, 0.158861285],
    )
    np.testing.assert_allclose(
        galbot.T_gripper_base_tcp("left")[:3, :3],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    )
    assert galbot.T_gripper_base_tcp("left")[2, 3] == pytest.approx(0.13996)

    yam = runner.resolve_robot_profile("yam_bimanual")
    np.testing.assert_allclose(
        yam.sweep_volume.x_bounds_m,
        [-0.04745052945624144, 0.04745052945624144],
    )
    np.testing.assert_allclose(yam.sweep_volume.pad_y_bounds_m, [-0.034, 0.034])
    np.testing.assert_allclose(
        yam.sweep_volume.pad_z_bounds_m,
        [0.07456, 0.14256],
    )
    np.testing.assert_allclose(
        yam.T_gripper_base_semantic_contact()[:3, :3],
        np.eye(3),
    )
    np.testing.assert_allclose(
        yam.T_gripper_base_tcp("left")[:3, :3],
        np.diag([-1.0, -1.0, 1.0]),
    )
    np.testing.assert_allclose(
        yam.T_gripper_base_tcp("right")[:3, :3],
        np.eye(3),
    )


def test_candidate_index_allowlist_is_recorded_and_validated(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path / "fixture")
    output = tmp_path / "selected.npz"

    metadata = _execute(
        fixture,
        side="left",
        output=output,
        candidate_index_allowlist=[0],
    )

    assert metadata["selection"]["selected_candidate_index"] == 0
    assert metadata["selection"]["hard_filters"][
        "candidate_index_allowlist"
    ] == [0]

    with pytest.raises(
        runner.RefinementRunError,
        match=r"outside \[0, 1\)",
    ):
        _execute(
            fixture,
            side="left",
            output=tmp_path / "invalid.npz",
            candidate_index_allowlist=[1],
        )


def test_midpoint_surface_registration_preserves_pair_when_independent_collapses() -> None:
    mesh = trimesh.creation.box(extents=[0.04, 0.02, 0.04])
    raw_pair = np.array(
        [[0.10, 0.10, 0.10], [0.12, 0.12, 0.12]],
        dtype=np.float64,
    )

    independent, _, _ = runner._project_to_mesh(mesh, raw_pair)
    registered = runner._surface_register_contact_pair(mesh, raw_pair)

    assert np.linalg.norm(independent[1] - independent[0]) == pytest.approx(0.0)
    np.testing.assert_allclose(
        registered.scoring_pair_object[1]
        - registered.scoring_pair_object[0],
        raw_pair[1] - raw_pair[0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.mean(registered.scoring_pair_object, axis=0),
        registered.projected_midpoint_object,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        registered.scoring_pair_object - raw_pair,
        np.repeat(registered.common_translation_object[None], 2, axis=0),
        atol=1e-12,
    )
    assert registered.midpoint_projection_distance_m == pytest.approx(
        np.linalg.norm(registered.common_translation_object)
    )


def test_execute_projects_contacts_selects_and_preserves_exact_schema(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path / "inputs")
    output = tmp_path / "refined.npz"

    metadata = _execute(fixture, side="left", output=output)

    assert output.is_file()
    metadata_path = output.with_suffix(".json")
    assert metadata_path.is_file()
    assert json.loads(metadata_path.read_text()) == metadata
    assert metadata["state"] == "complete"
    assert metadata["selection"]["selected_candidate_index"] == 0
    np.testing.assert_allclose(
        metadata["human_contacts"]["raw_hand_to_mesh_distance_m"],
        [0.005, 0.005],
        atol=1e-7,
    )
    scoring = metadata["human_contacts"]["scoring_contact_estimation"]
    assert scoring["strategy"] == runner.CONTACT_PAIR_REGISTRATION_STRATEGY
    assert scoring["raw_aperture_m"] == pytest.approx(0.04)
    assert scoring["scoring_aperture_m"] == pytest.approx(0.04)
    assert scoring["preserves_raw_pair_vector"] is True
    assert (
        metadata["human_contacts"]["independent_nearest_point_diagnostics"][
            "used_for_scoring"
        ]
        is False
    )
    assert metadata["contact_registration"]["translation_magnitude_m"] > 0.0
    assert metadata["profile"]["overrides"]["pad_y_bounds_m"] == [-0.005, 0.005]
    assert "semantic_target_to_tcp_rotation" in metadata["profile"]

    base = grasp_refinement.load_parallel_jaw_target(Path(fixture["target"]))
    refined = grasp_refinement.load_parallel_jaw_target(output)
    assert len(refined) == 12
    assert refined["tracker"].item() == "v2d"
    for key in refined:
        assert refined[key].dtype == base[key].dtype
    for key in ("right_position", "right_wxyz", "right_aperture_m"):
        np.testing.assert_array_equal(refined[key], base[key])
    for frame in (0, 1, 5, 6):
        np.testing.assert_array_equal(
            refined["left_position"][frame],
            base["left_position"][frame],
        )
        np.testing.assert_array_equal(
            refined["left_wxyz"][frame],
            base["left_wxyz"][frame],
        )
    expected_position = np.asarray(
        metadata["frame_conversion"]["T_object_semantic"]
    )[:3, 3]
    np.testing.assert_allclose(
        refined["left_position"][2:5],
        np.repeat(expected_position[None], 3, axis=0),
        atol=2e-7,
    )
    np.testing.assert_allclose(refined["left_aperture_m"][2:5], 0.04, atol=1e-7)


def test_contact_wrench_mode_preserves_mesh_contacts_via_object_pose_translation(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path / "inputs")
    quarter_turn = float(np.sqrt(0.5))
    for pose_path in Path(fixture["poses"]).glob("*.json"):
        pose_payload = json.loads(pose_path.read_text())
        pose_payload["rotation"] = [
            quarter_turn,
            0.0,
            0.0,
            quarter_turn,
        ]
        pose_path.write_text(json.dumps(pose_payload))
    output = tmp_path / "wrench_refined.npz"

    metadata = _execute(
        fixture,
        side="left",
        output=output,
        contact_wrench_mode="low_tail",
        post_selection_registration_mode="object_pose_translation",
        post_selection_registration_cap_m=0.002,
        wrench_direction_count=64,
        wrench_direction_seed=7,
        wrench_friction_coefficient=0.2,
        wrench_low_quantile=0.2,
    )

    wrench = metadata["contact_wrench_scoring"]
    assert wrench["enabled"] is True
    assert wrench["mode"] == "low_tail"
    assert wrench["requires_simulator"] is False
    assert wrench["basis"]["shape"] == [64, 6]
    assert wrench["basis"]["seed"] == 7
    assert wrench["configuration"]["friction_coefficient"] == pytest.approx(0.2)
    assert wrench["selected_before_registration"][
        "low_quantile_support"
    ] == pytest.approx(
        wrench["selected_after_mesh_revalidation"]["low_quantile_support"]
    )

    registration = metadata["contact_registration"]
    assert registration["mode"] == "object_pose_translation"
    assert registration["was_capped_or_disabled"] is True
    assert registration["translation_magnitude_m"] == pytest.approx(0.002)
    assert registration["mesh_consistency"][
        "contacts_rederived_after_symmetry"
    ] is True
    np.testing.assert_allclose(
        registration["T_object_gripper_base_registered"],
        metadata["selection"]["T_object_gripper_base_symmetry_aligned"],
        atol=1e-12,
    )
    anchor_before = np.asarray(
        registration["mesh_consistency"]["T_world_object_anchor_before"]
    )
    anchor_corrected = np.asarray(
        registration["mesh_consistency"][
            "T_world_object_anchor_for_propagation"
        ]
    )
    np.testing.assert_allclose(
        registration["translation_world_m"],
        anchor_before[:3, :3]
        @ np.asarray(registration["translation_object_m"]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        anchor_corrected[:3, 3] - anchor_before[:3, 3],
        registration["translation_world_m"],
        atol=1e-12,
    )


def test_contact_wrench_mode_rejects_legacy_gripper_translation(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path / "inputs")
    with pytest.raises(
        runner.RefinementRunError,
        match="mesh-consistent post-selection registration",
    ):
        _execute(
            fixture,
            side="left",
            output=tmp_path / "invalid.npz",
            contact_wrench_mode="low_tail",
        )


def test_wrench_reference_requires_matching_full_wrench_contract(
    tmp_path: Path,
) -> None:
    basis = runner.contact_wrench_scoring.shared_wrench_basis(32, 11)
    object_com = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    radius = 0.25
    reference_path = tmp_path / "reference.npz"
    payload = {
        "schema_version": np.asarray(
            runner.contact_wrench_scoring.REFERENCE_SCHEMA_VERSION
        ),
        "scorer_version": np.asarray(
            runner.contact_wrench_scoring.SCORER_VERSION
        ),
        "supports": np.linspace(0.0, 1.0, 32, dtype=np.float64),
        "basis_sha256": np.asarray(basis.sha256),
        "object_com_object_m": object_com,
        "object_radius_m": np.asarray(radius),
        "friction_coefficient": np.asarray(0.1),
        "friction_cone_edges": np.asarray(8),
        "normal_conversion": np.asarray(
            runner.contact_wrench_scoring.NORMAL_CONVERSION_CONVENTION
        ),
        "torque_normalization": np.asarray(
            runner.contact_wrench_scoring.TORQUE_NORMALIZATION_CONVENTION
        ),
        "friction_cone_phase_convention": np.asarray(
            runner.contact_wrench_scoring.FRICTION_CONE_PHASE_CONVENTION
        ),
    }
    np.savez_compressed(reference_path, **payload)

    supports = runner._load_wrench_reference_supports(
        reference_path,
        basis_sha256=basis.sha256,
        direction_count=32,
        object_com_object=object_com,
        object_radius_m=radius,
        friction_coefficient=0.1,
        friction_cone_edges=8,
    )
    np.testing.assert_array_equal(supports, payload["supports"])

    payload["object_radius_m"] = np.asarray(radius + 0.01)
    np.savez_compressed(reference_path, **payload)
    with pytest.raises(
        runner.RefinementRunError,
        match="object_radius_m does not match",
    ):
        runner._load_wrench_reference_supports(
            reference_path,
            basis_sha256=basis.sha256,
            direction_count=32,
            object_com_object=object_com,
            object_radius_m=radius,
            friction_coefficient=0.1,
            friction_cone_edges=8,
        )


def test_object_pose_translation_supports_anchor_only_base_local_propagation(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path / "inputs")
    metadata = _execute(
        fixture,
        side="left",
        output=tmp_path / "base_local_wrench.npz",
        propagation_mode="base_local_offset",
        contact_wrench_mode="low_tail",
        post_selection_registration_mode="object_pose_translation",
        wrench_direction_count=64,
    )

    assert metadata["trajectory_correction"]["foundationpose_dependency"] == (
        "anchor_frame_only"
    )
    assert metadata["trajectory_correction"]["mesh_contact_scope"] == (
        "anchor_frame_only"
    )
    assert "guaranteed at the anchor" in (
        metadata["contact_registration"]["mesh_consistency"]["policy"]
    )


def test_execute_scores_midpoint_registered_pair_not_independent_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path / "inputs")
    tracking_path = Path(fixture["tracking"])
    with np.load(tracking_path, allow_pickle=False) as archive:
        tracking = {key: np.asarray(archive[key]) for key in archive.files}
    tracking["left_joints_3d"][:, 4] = [0.10, 0.10, 0.10]
    tracking["left_joints_3d"][:, 8] = [0.12, 0.12, 0.12]
    np.savez_compressed(tracking_path, **tracking)

    captured: dict[str, np.ndarray] = {}
    original_score = runner.grasp_refinement.score_grasp_candidates

    def capture_scoring_pair(*args: object, **kwargs: object) -> object:
        captured["pair"] = np.asarray(args[1], dtype=np.float64).copy()
        return original_score(*args, **kwargs)

    monkeypatch.setattr(
        runner.grasp_refinement,
        "score_grasp_candidates",
        capture_scoring_pair,
    )
    metadata = _execute(
        fixture,
        side="left",
        output=tmp_path / "refined.npz",
    )

    contact_metadata = metadata["human_contacts"]
    scoring = contact_metadata["scoring_contact_estimation"]
    diagnostics = contact_metadata["independent_nearest_point_diagnostics"]
    np.testing.assert_allclose(captured["pair"], scoring["scoring_pair_object_m"])
    assert diagnostics["projected_aperture_m"] == pytest.approx(0.0)
    assert scoring["raw_aperture_m"] == pytest.approx(
        np.sqrt(3.0) * 0.02
    )
    assert scoring["scoring_aperture_m"] == pytest.approx(
        scoring["raw_aperture_m"]
    )
    assert not np.allclose(
        captured["pair"],
        diagnostics["projected_object_m"],
    )


def test_second_side_invocation_preserves_first_side_refinement(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path / "inputs")
    left_output = tmp_path / "left.npz"
    left_metadata = _execute(
        fixture,
        side="left",
        output=left_output,
        propagation_mode="base_local_offset",
    )
    left_arrays = grasp_refinement.load_parallel_jaw_target(left_output)

    both_output = tmp_path / "both.npz"
    metadata = _execute(
        fixture,
        side="right",
        output=both_output,
        base_target=left_output,
        propagation_mode="base_local_offset",
    )
    both_arrays = grasp_refinement.load_parallel_jaw_target(both_output)

    for key in ("left_position", "left_wxyz", "left_aperture_m"):
        np.testing.assert_array_equal(both_arrays[key], left_arrays[key])
    assert not np.array_equal(
        both_arrays["right_position"],
        left_arrays["right_position"],
    )
    assert (
        left_metadata["trajectory_correction"]["propagation_mode"]
        == "base_local_offset"
    )
    assert (
        metadata["trajectory_correction"]["propagation_mode"]
        == "base_local_offset"
    )
    assert (
        metadata["trajectory_correction"]["foundationpose_dependency"]
        == "anchor_frame_only"
    )
    assert metadata["trajectory_correction"]["input_can_be_prior_refinement_output"]


def test_existing_outputs_require_overwrite_and_pair_is_replaced(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path / "inputs")
    output = tmp_path / "refined.npz"
    metadata_path = output.with_suffix(".json")
    output.write_bytes(b"old npz")
    metadata_path.write_text("old json")

    with pytest.raises(runner.ArtifactExistsError, match="refusing to overwrite"):
        _execute(fixture, side="left", output=output)
    assert output.read_bytes() == b"old npz"
    assert metadata_path.read_text() == "old json"

    _execute(fixture, side="left", output=output, overwrite=True)
    grasp_refinement.load_parallel_jaw_target(output)
    assert json.loads(metadata_path.read_text())["state"] == "complete"


def test_starts_in_contact_flag_must_match_frame_zero(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "inputs")
    with pytest.raises(runner.RefinementRunError, match="exactly when"):
        runner.execute(
            base_target=Path(fixture["target"]),
            tracking=Path(fixture["tracking"]),
            T_camera_world=Path(fixture["calibration"]),
            foundationpose_poses=Path(fixture["poses"]),
            mesh=Path(fixture["mesh"]),
            candidates=Path(fixture["candidates"]),
            robot_profile="galbot_one_golf",
            side="left",
            object_name="test",
            event_start=0,
            event_end=2,
            event_anchor=1,
            starts_in_contact=False,
            output_target=tmp_path / "invalid.npz",
        )
