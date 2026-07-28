"""CPU tests for MECKA contracts and retarget geometry."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from inpainting import run_mecka_panda_pipeline as pipeline
from inpainting.adapters import mecka, mecka_lerobot, mecka_parallel_jaw
from inpainting.adapters.mecka import build_tracking_arrays
from inpainting.adapters.mecka_parallel_jaw import (
    MAX_ROTATION_STEP_DEG,
    PALM_RATIO_MAX,
    ROTATION_ALPHA,
    TIP_RATIO_MIN,
    closest_parallel_jaw_equivalent,
    palm_landmark_frame,
    retarget_hand_sequence,
    retarget_tracking_arrays,
    smooth_rotation,
    thumb_index_pose,
)
from inpainting.mecka_panda.composite import (
    COMPOSITE_SCHEMA,
    depth_visible_robot_mask,
)
from inpainting.mecka_panda.contracts import (
    ROBOT_RENDER_SCHEMA,
    ContractError,
    validate_parallel_jaw_arrays,
    validate_tracking_arrays,
)
from inpainting.panda_renderer.kinematics import (
    IKCandidateError,
    PandaIK,
    SSIKUnavailable,
    build_panda_model,
    validate_arm_candidate,
)
from inpainting.panda_renderer.render import DEFAULT_PANDA_DIR


def _hand(offset: float = 0.0) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[:, 0] = np.linspace(-0.04, 0.04, 21) + offset
    points[:, 1] = np.linspace(0.02, 0.12, 21)
    points[:, 2] = 0.55
    points[4] = [0.04 + offset, 0.08, 0.55]
    points[[8, 12, 16, 20]] = [
        [-0.02 + offset, 0.09, 0.55],
        [-0.01 + offset, 0.10, 0.55],
        [0.00 + offset, 0.10, 0.55],
        [0.01 + offset, 0.09, 0.55],
    ]
    points[[5, 9, 13, 17]] = [
        [-0.02 + offset, 0.04, 0.55],
        [-0.01 + offset, 0.05, 0.55],
        [0.00 + offset, 0.05, 0.55],
        [0.01 + offset, 0.04, 0.55],
    ]
    points[1] = [0.03 + offset, 0.03, 0.55]
    return points


def _table(frame_count: int = 7) -> pd.DataFrame:
    identity_xyzw = np.tile([0.0, 0.0, 0.0, 1.0], (21, 1)).reshape(-1)
    rows = []
    for frame in range(frame_count):
        rows.append(
            {
                "frame_index": frame,
                "observation.state.hand_left_cam": _hand(-0.1).reshape(-1),
                "observation.state.hand_right_cam": _hand(0.1).reshape(-1),
                "observation.state.hand_left_cam_rotation": identity_xyzw,
                "observation.state.hand_right_cam_rotation": identity_xyzw,
            }
        )
    return pd.DataFrame(rows)


def _pipeline_args(output: Path) -> Namespace:
    background = output / "background.mp4"
    background.parent.mkdir(parents=True, exist_ok=True)
    background.write_bytes(b"test")
    return Namespace(
        dataset="s3://dataset-a",
        episode=1,
        credentials=None,
        output_dir=output,
        background=background,
        background_start_frame=0,
        mask_preview=None,
        mask_start_frame=0,
        object_mask=None,
        object_depth=None,
        start_frame=0,
        max_frames=10,
        stage=None,
        ik="dls",
        emit_depth=False,
        rig_config=output / "rig.json",
        panda_dir=DEFAULT_PANDA_DIR,
        orientation_weight=0.5,
        max_joint_step_rad=0.3,
        jump_k=6.0,
        max_gap=15,
        smooth_window=11,
        smooth_poly=2,
        palm_ratio_max=PALM_RATIO_MAX,
        tip_ratio_min=TIP_RATIO_MIN,
        rotation_alpha=ROTATION_ALPHA,
        max_rotation_step_deg=MAX_ROTATION_STEP_DEG,
        depth_guard_m=0.003,
        overwrite=False,
        execute=False,
    )


def _source_info(output: Path) -> pipeline.SourceInfo:
    return pipeline.SourceInfo(
        kind="lerobot",
        episode_index=1,
        task_id="task_1",
        frame_count=10,
        width=64,
        height=48,
        fps=30.0,
        source_video=output / "tracking" / mecka_lerobot.VIDEO_FILENAME,
        source_parquet="s3://dataset-a rows [10, 20)",
    )


def _write_pipeline_cache(output: Path, args: Namespace) -> None:
    paths = pipeline._layout(output)
    metadata = {
        "tracking": (
            paths["tracking"] / mecka.METADATA_FILENAME,
            {
                "schema_version": mecka_lerobot.RUN_SCHEMA,
                "state": "complete",
                "episode_index": 1,
                "task_id": "task_1",
                "frame_window": {"start": 0, "count": 10},
                "source": {"dataset_uri": "s3://dataset-a"},
            },
        ),
        "retarget": (
            paths["retarget"] / mecka_parallel_jaw.METADATA_FILENAME,
            {
                "schema_version": mecka_parallel_jaw.RUN_SCHEMA,
                "state": "complete",
                "algorithm": {
                    "version": "thumb-index-palm-default/v1",
                    "conditioning": {
                        "jump_k": args.jump_k,
                        "max_gap": args.max_gap,
                        "smooth_window": args.smooth_window,
                        "smooth_poly": args.smooth_poly,
                    },
                    "orientation_stability": {
                        "palm_ratio_max": args.palm_ratio_max,
                        "tip_ratio_min": args.tip_ratio_min,
                        "rotation_alpha": args.rotation_alpha,
                        "max_rotation_step_deg": args.max_rotation_step_deg,
                    },
                },
            },
        ),
        "render": (
            paths["render"] / "render_metadata.json",
            {
                "schema_version": ROBOT_RENDER_SCHEMA,
                "state": "complete",
                "depth_emitted": args.emit_depth,
                "ik": {
                    "backend": args.ik,
                    "orientation_weight": args.orientation_weight,
                    "max_joint_step_limit_rad": args.max_joint_step_rad,
                },
            },
        ),
        "composite": (
            paths["composite"] / "final_overlay.json",
            {
                "schema_version": COMPOSITE_SCHEMA,
                "state": "complete",
                "depth_guard_m": args.depth_guard_m,
                "base_start_frame": args.background_start_frame,
            },
        ),
        "review": (
            paths["review_metadata"],
            {
                "schema_version": pipeline.REVIEW_SCHEMA,
                "state": "complete",
                "geometry": {"frame_count": 10},
            },
        ),
    }
    for path, payload in metadata.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_pipeline_cache_binds_source_and_invalidates_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    args = _pipeline_args(output)
    source = _source_info(output)
    monkeypatch.setattr(pipeline, "_resolve_source", lambda *_: source)
    _write_pipeline_cache(output, args)

    plan = pipeline.build_plan(args)
    assert {stage["state"] for stage in plan["stages"]} == {"skipped_complete"}
    assert plan["blockers"] == []

    args.rotation_alpha = 0.3
    changed = pipeline.build_plan(args)
    states = {stage["name"]: stage["state"] for stage in changed["stages"]}
    assert states["tracking"] == "skipped_complete"
    assert all(
        states[stage] == "pending"
        for stage in ("retarget", "render", "composite", "review")
    )
    assert any("--overwrite" in blocker for blocker in changed["blockers"])

    args.rotation_alpha = ROTATION_ALPHA
    args.dataset = "s3://dataset-b"
    changed_source = pipeline.build_plan(args)
    assert all(stage["state"] == "pending" for stage in changed_source["stages"])


def test_pipeline_blocks_selected_stage_with_stale_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    args = _pipeline_args(output)
    source = _source_info(output)
    monkeypatch.setattr(pipeline, "_resolve_source", lambda *_: source)
    _write_pipeline_cache(output, args)
    args.dataset = "s3://dataset-b"
    args.stage = ["render"]
    plan = pipeline.build_plan(args)
    assert plan["stages"][0]["state"] == "pending"
    assert any("stale dependency 'tracking'" in value for value in plan["blockers"])


def test_mecka_tracking_to_parallel_jaw_contract() -> None:
    tracking = build_tracking_arrays(_table())
    assert validate_tracking_arrays(tracking) == 7
    target, diagnostics = retarget_tracking_arrays(tracking, smooth_window=5)
    assert validate_parallel_jaw_arrays(target) == 7
    assert target["left_valid"].all()
    assert target["right_valid"].all()
    assert diagnostics["left"]["jumps_removed"] == 0
    assert np.isfinite(target["left_position"]).all()
    assert np.allclose(np.linalg.norm(target["right_wxyz"], axis=1), 1.0)
    expected_position = 0.5 * (_hand(-0.1)[4] + _hand(-0.1)[8])
    expected_aperture = np.linalg.norm(_hand(-0.1)[4] - _hand(-0.1)[8])
    assert np.allclose(target["left_position"], expected_position)
    assert np.allclose(target["left_aperture_m"], expected_aperture)
    rotations = Rotation.from_quat(target["left_wxyz"], scalar_first=True).as_matrix()
    assert np.allclose(
        np.einsum("nij,njk->nik", rotations.transpose(0, 2, 1), rotations),
        np.eye(3),
        atol=1e-6,
    )
    assert np.allclose(np.linalg.det(rotations), 1.0)


def test_closed_pinch_uses_stable_palm_default() -> None:
    hands = np.stack([_hand() for _ in range(4)])
    center = np.array([0.0, 0.085, 0.55])
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    for frame, direction in enumerate(directions):
        hands[frame, 4] = center + 0.0005 * direction
        hands[frame, 8] = center - 0.0005 * direction
    hands[0, 4] = center
    hands[0, 8] = center
    _, quaternions, _, diagnostics = retarget_hand_sequence(
        hands,
        np.ones(4, dtype=np.bool_),
        is_right=True,
    )
    rotations = Rotation.from_quat(quaternions, scalar_first=True).as_matrix()
    expected = palm_landmark_frame(hands[0], is_right=True)
    assert np.allclose(rotations, expected, atol=1e-7)
    assert diagnostics["mode_counts"] == {"palm": 4, "blend": 0, "pinch": 0}
    assert diagnostics["fallbacks"]["coincident_thumb_index"] == 1
    palm_outward = np.mean(hands[0, [5, 9, 13, 17]], axis=0) - hands[0, 0]
    assert np.dot(rotations[0, :, 0], palm_outward) > 0.0


@pytest.mark.parametrize("is_right", [False, True])
def test_open_pinch_frame_respects_handedness(is_right: bool) -> None:
    hand = _hand()
    if not is_right:
        hand[:, 0] *= -1.0
    position, raw_rotation, aperture, ratio = thumb_index_pose(
        hand,
        is_right=is_right,
    )
    expected_jaw = (1.0 if is_right else -1.0) * (hand[4] - hand[8])
    expected_jaw /= np.linalg.norm(expected_jaw)
    assert np.dot(raw_rotation[:, 1], expected_jaw) > 1.0 - 1e-7
    assert np.dot(raw_rotation[:, 0], position - hand[0]) > 0.0
    assert aperture == pytest.approx(np.linalg.norm(hand[4] - hand[8]))
    assert ratio >= TIP_RATIO_MIN

    _, quaternions, _, diagnostics = retarget_hand_sequence(
        hand[None, :, :],
        np.array([True]),
        is_right=is_right,
    )
    actual = Rotation.from_quat(quaternions[0], scalar_first=True).as_matrix()
    expected = closest_parallel_jaw_equivalent(
        raw_rotation,
        palm_landmark_frame(hand, is_right=is_right),
    )
    assert np.allclose(actual, expected)
    assert diagnostics["mode_counts"]["pinch"] == 1


def test_ratio_boundaries_select_palm_and_pinch_frames() -> None:
    for ratio, expected_mode in (
        (PALM_RATIO_MAX, "palm"),
        (TIP_RATIO_MIN, "pinch"),
    ):
        hand = _hand()
        hand[5] = [-0.02, 0.04, 0.55]
        hand[17] = [0.02, 0.04, 0.55]
        center = np.array([0.0, 0.085, 0.55])
        aperture = ratio * np.linalg.norm(hand[5] - hand[17])
        hand[4] = center + np.array([aperture / 2.0, 0.0, 0.0])
        hand[8] = center - np.array([aperture / 2.0, 0.0, 0.0])
        _, quaternions, _, diagnostics = retarget_hand_sequence(
            hand[None, :, :],
            np.array([True]),
            is_right=True,
        )
        assert diagnostics["mode_counts"][expected_mode] == 1
        actual = Rotation.from_quat(quaternions[0], scalar_first=True).as_matrix()
        palm = palm_landmark_frame(hand, is_right=True)
        if expected_mode == "palm":
            expected = palm
        else:
            _, raw, _, _ = thumb_index_pose(hand, is_right=True)
            expected = closest_parallel_jaw_equivalent(raw, palm)
        assert np.allclose(actual, expected, atol=1e-7)


def test_parallel_jaw_symmetry_and_low_pass_are_explicit() -> None:
    target = Rotation.from_euler("xyz", [20.0, -15.0, 35.0], degrees=True).as_matrix()
    equivalent = target @ np.diag([1.0, -1.0, -1.0])
    assert np.allclose(
        closest_parallel_jaw_equivalent(target, equivalent),
        equivalent,
    )
    smoothed = smooth_rotation(
        np.eye(3),
        Rotation.from_euler("z", 10.0, degrees=True).as_matrix(),
    )
    step_deg = np.degrees(Rotation.from_matrix(smoothed).magnitude())
    assert step_deg == pytest.approx(ROTATION_ALPHA * 10.0)


def test_target_rotation_limit_survives_invalid_gap() -> None:
    hands = np.stack([_hand(), _hand(), _hand()])
    wrist = hands[2, 0].copy()
    turn = Rotation.from_euler("z", 90.0, degrees=True)
    hands[2] = turn.apply(hands[2] - wrist) + wrist
    valid = np.array([True, False, True])
    _, quaternions, _, diagnostics = retarget_hand_sequence(
        hands,
        valid,
        is_right=True,
    )
    assert np.isnan(quaternions[1]).all()
    rotations = Rotation.from_quat(quaternions[valid], scalar_first=True).as_matrix()
    step_deg = np.degrees(
        Rotation.from_matrix(rotations[0].T @ rotations[1]).magnitude()
    )
    assert 5.9 <= step_deg <= MAX_ROTATION_STEP_DEG + 1e-7
    assert np.dot(quaternions[0], quaternions[2]) >= 0.0
    assert diagnostics["max_rotation_step_deg"] <= MAX_ROTATION_STEP_DEG + 1e-7


def test_retarget_rejects_invalid_stability_thresholds() -> None:
    with pytest.raises(ValueError, match="ratio thresholds"):
        retarget_hand_sequence(
            _hand()[None, :, :],
            np.array([True]),
            is_right=True,
            palm_ratio_max=0.8,
            tip_ratio_min=0.7,
        )


def test_tracking_rejects_noncontiguous_frames() -> None:
    tracking = build_tracking_arrays(_table())
    tracking["frame_indices"][-1] = 99
    with pytest.raises(ContractError, match="contiguous"):
        validate_tracking_arrays(tracking)


def test_missing_hand_remains_invalid_nan() -> None:
    table = _table()
    table.at[3, "observation.state.hand_left_cam"] = np.zeros(63)
    tracking = build_tracking_arrays(table)
    assert not tracking["left_valid"][3]
    assert np.isnan(tracking["left_joints_3d"][3]).all()
    target, _ = retarget_tracking_arrays(tracking)
    assert not target["left_valid"][3]
    assert np.isnan(target["left_position"][3]).all()


def test_depth_visibility_guard() -> None:
    robot_mask = np.array([[True, True, True]])
    robot_depth = np.array([[0.8, 1.0, 1.2]], dtype=np.float32)
    object_mask = np.array([[True, True, False]])
    object_depth = np.array([[0.9, 0.9, np.inf]], dtype=np.float32)
    visible = depth_visible_robot_mask(
        robot_mask, robot_depth, object_mask, object_depth, depth_guard_m=0.003
    )
    assert visible.tolist() == [[True, False, True]]


def test_ik_candidate_gate_rejects_joint_step() -> None:
    ranges = np.tile([-1.0, 1.0], (7, 1))
    previous = np.zeros(7)
    assert validate_arm_candidate(
        np.full(7, 0.05),
        previous_q=previous,
        joint_ranges=ranges,
        max_joint_step_rad=0.05,
    ) == pytest.approx(0.05)
    with pytest.raises(IKCandidateError, match="exceeds") as error:
        validate_arm_candidate(
            np.full(7, 0.051),
            previous_q=previous,
            joint_ranges=ranges,
            max_joint_step_rad=0.05,
        )
    assert error.value.reason == "continuity"


def _panda_case() -> tuple[PandaIK, np.ndarray, np.ndarray]:
    if not (DEFAULT_PANDA_DIR / "panda.xml").is_file():
        pytest.skip("local MuJoCo Menagerie Panda assets are unavailable")
    model = build_panda_model(DEFAULT_PANDA_DIR, 60.0, 64, 64)
    solver = PandaIK(model, np.array([0.0, 0.4, 0.4]), np.array([0.0, 0.0, 0.6]))
    target = solver.fingertip_center().copy()
    hand_rotation = solver.data.body(solver.hand_id).xmat.reshape(3, 3).copy()
    semantic_rotation = hand_rotation @ solver.semantic_to_hand
    return solver, target, semantic_rotation


def test_panda_dls_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    solver, target, semantic_rotation = _panda_case()
    residual = solver.solve_dls(
        target,
        semantic_rotation,
        0.05,
        previous_q=None,
        elbow_outward=np.array([1.0, 0.0, 0.0]),
        iterations=4,
    )
    assert residual < 1e-6

    previous = solver.data.qpos[solver.arm_qadr].copy()
    rejected_ssik = previous + 0.1
    seed_before = solver._ssik_seed.copy()
    monkeypatch.setattr(
        solver,
        "_ssik_candidate",
        lambda *args, **kwargs: rejected_ssik.copy(),
    )
    result = solver.solve_target(
        target,
        semantic_rotation,
        0.05,
        previous_q=previous,
        elbow_outward=np.array([1.0, 0.0, 0.0]),
        backend="hybrid",
        max_joint_step_rad=0.05,
        iterations=4,
    )
    current = solver.data.qpos[solver.arm_qadr].copy()
    assert result.backend == "dls"
    assert result.ssik_status == "rejected_continuity"
    assert np.max(np.abs(current - previous)) <= 0.05 + 1e-8
    assert np.array_equal(solver._ssik_seed, seed_before)


def test_panda_hybrid_accepts_bounded_ssik(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver, target, semantic_rotation = _panda_case()
    previous = solver.data.qpos[solver.arm_qadr].copy()
    candidate = previous + 0.01
    monkeypatch.setattr(
        solver,
        "_ssik_candidate",
        lambda *args, **kwargs: candidate.copy(),
    )
    monkeypatch.setattr(
        solver,
        "solve_dls",
        lambda *args, **kwargs: pytest.fail("bounded SSIK must not call DLS"),
    )
    result = solver.solve_target(
        target,
        semantic_rotation,
        0.05,
        previous_q=previous,
        elbow_outward=np.array([1.0, 0.0, 0.0]),
        backend="hybrid",
        max_joint_step_rad=0.05,
    )
    assert result.backend == "ssik"
    assert result.ssik_status == "accepted"
    assert result.joint_step_rad == pytest.approx(0.01)
    assert np.allclose(solver.data.qpos[solver.arm_qadr], candidate)
    assert np.allclose(solver._ssik_seed, candidate)


def test_panda_hybrid_reports_ssik_unavailable_and_uses_dls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver, target, semantic_rotation = _panda_case()
    previous = solver.data.qpos[solver.arm_qadr].copy()

    def unavailable(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise SSIKUnavailable("SSIK is not installed")

    monkeypatch.setattr(solver, "_ssik_candidate", unavailable)
    result = solver.solve_target(
        target,
        semantic_rotation,
        0.05,
        previous_q=previous,
        elbow_outward=np.array([1.0, 0.0, 0.0]),
        backend="hybrid",
        max_joint_step_rad=0.05,
        iterations=4,
    )
    assert result.backend == "dls"
    assert result.ssik_status == "unavailable"


def test_panda_hybrid_degrades_to_dls_on_any_ssik_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver, target, semantic_rotation = _panda_case()
    previous = solver.data.qpos[solver.arm_qadr].copy()

    def broken(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        # Not a RuntimeError: an optional backend must not be able to fail the
        # frame no matter which exception type it raises.
        raise AssertionError("ssik internals blew up")

    monkeypatch.setattr(solver, "_ssik_candidate", broken)
    result = solver.solve_target(
        target,
        semantic_rotation,
        0.05,
        previous_q=previous,
        elbow_outward=np.array([1.0, 0.0, 0.0]),
        backend="hybrid",
        max_joint_step_rad=0.05,
        iterations=4,
    )
    assert result.backend == "dls"
    assert result.ssik_status == "error"


def test_compatibility_ssik_accepts_first_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without previous_q there is no transition, so nothing may be rejected."""
    solver, target, semantic_rotation = _panda_case()
    # Far enough from the current pose that gating it against the home pose
    # would reject it, which is what the first frame of a real episode does.
    candidate = solver.data.qpos[solver.arm_qadr].copy() + 0.5
    monkeypatch.setattr(
        solver,
        "_ssik_candidate",
        lambda *args, **kwargs: candidate.copy(),
    )
    residual = solver.solve_ssik(
        target,
        semantic_rotation,
        0.05,
        previous_q=None,
        max_joint_step_rad=0.05,
    )
    assert residual is not None
    assert np.allclose(solver.data.qpos[solver.arm_qadr], candidate)

    # A real transition of the same size must still be rejected.
    previous = solver.data.qpos[solver.arm_qadr].copy()
    assert (
        solver.solve_ssik(
            target,
            semantic_rotation,
            0.05,
            previous_q=previous - 0.5,
            max_joint_step_rad=0.05,
        )
        is None
    )


def test_panda_solver_rolls_back_dls_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver, target, semantic_rotation = _panda_case()
    arm_before = solver.data.qpos[solver.arm_qadr].copy()
    fingers_before = solver.data.qpos[solver.finger_qadr].copy()
    seed_before = solver._ssik_seed.copy()

    def fail_after_mutation(*args: object, **kwargs: object) -> float:
        del args, kwargs
        solver.data.qpos[solver.arm_qadr] = arm_before + 0.2
        solver.data.qpos[solver.finger_qadr] = 0.04
        solver._ssik_seed = seed_before + 0.3
        raise RuntimeError("injected DLS failure")

    monkeypatch.setattr(solver, "solve_dls", fail_after_mutation)
    with pytest.raises(RuntimeError, match="injected"):
        solver.solve_target(
            target,
            semantic_rotation,
            0.05,
            previous_q=arm_before,
            elbow_outward=np.array([1.0, 0.0, 0.0]),
            backend="dls",
            max_joint_step_rad=0.05,
        )
    assert np.array_equal(solver.data.qpos[solver.arm_qadr], arm_before)
    assert np.array_equal(solver.data.qpos[solver.finger_qadr], fingers_before)
    assert np.array_equal(solver._ssik_seed, seed_before)
