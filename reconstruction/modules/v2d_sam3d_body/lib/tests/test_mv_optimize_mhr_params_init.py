import math

import pytest

torch = pytest.importorskip("torch")
p3d_transforms = pytest.importorskip("pytorch3d.transforms")
mv_opt = pytest.importorskip("v2d.sam3d_body.lib.mv_optimize_mhr_params")

euler_angles_to_matrix = p3d_transforms.euler_angles_to_matrix
matrix_to_rotation_6d = p3d_transforms.matrix_to_rotation_6d
matrix_to_euler_angles = p3d_transforms.matrix_to_euler_angles


def _assert_valid_rotation(rotmat: torch.Tensor):
    eye = torch.eye(3, dtype=rotmat.dtype, device=rotmat.device)
    eye = eye.expand(rotmat.shape[:-2] + (3, 3))
    assert torch.isfinite(rotmat).all()
    assert torch.allclose(rotmat @ rotmat.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(
        torch.linalg.det(rotmat),
        torch.ones(rotmat.shape[:-2], dtype=rotmat.dtype, device=rotmat.device),
        atol=1e-5,
    )


def test_so3_mean_handles_opposing_lie_down_rotations():
    eulers = torch.tensor(
        [
            [[math.pi / 2, 0.0, 0.0]],
            [[-math.pi / 2, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    rotmats = euler_angles_to_matrix(eulers, "XYZ")

    old_avg_6d = matrix_to_rotation_6d(rotmats).mean(dim=0)
    assert old_avg_6d[..., 3:].norm(dim=-1).max() < 1e-8

    avg_euler = mv_opt.average_euler_angles(eulers, "XYZ")
    avg_rotmat = euler_angles_to_matrix(avg_euler, "XYZ")
    _assert_valid_rotation(avg_rotmat)


def _rotation_angle(rot_a: torch.Tensor, rot_b: torch.Tensor) -> torch.Tensor:
    rel = rot_a @ rot_b.transpose(-1, -2)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    return torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))


def _mhr_inputs_with_root_angle(angle: float) -> dict:
    return {
        "global_trans": torch.zeros(1, 3, dtype=torch.float64),
        "global_rot": torch.tensor([[angle, 0.0, 0.0]], dtype=torch.float64),
        "body_pose_params": torch.zeros(1, 133, dtype=torch.float64),
        "hand_pose_params": torch.zeros(1, 108, dtype=torch.float64),
        "scale_params": torch.zeros(1, 28, dtype=torch.float64),
        "shape_params": torch.zeros(1, 45, dtype=torch.float64),
    }


def test_average_mhr_inputs_preserves_dominant_root_orientation_cluster():
    inputs = [
        _mhr_inputs_with_root_angle(math.pi / 2),
        _mhr_inputs_with_root_angle(math.pi / 2 + 0.05),
        _mhr_inputs_with_root_angle(math.pi / 2 - 0.05),
        _mhr_inputs_with_root_angle(-math.pi / 2),
    ]

    result = mv_opt.average_mhr_inputs(inputs)
    avg_rot = euler_angles_to_matrix(result["global_rot"], "ZYX")
    target_rot = euler_angles_to_matrix(
        torch.tensor([[math.pi / 2, 0.0, 0.0]], dtype=torch.float64),
        "ZYX",
    )

    assert _rotation_angle(avg_rot, target_rot).item() < 0.25


def test_transform_mhr_params_composes_root_rotation_with_row_vector_convention():
    source_rot = euler_angles_to_matrix(
        torch.tensor([[0.3, -0.2, 0.1]], dtype=torch.float64),
        "ZYX",
    )[0]
    world_rot = euler_angles_to_matrix(
        torch.tensor([[0.4, 0.2, -0.3]], dtype=torch.float64),
        "ZYX",
    )[0]
    transform = torch.eye(4, dtype=torch.float64)
    transform[:3, :3] = world_rot

    mhr_inputs = _mhr_inputs_with_root_angle(0.0)
    mhr_inputs["global_rot"] = matrix_to_euler_angles(source_rot[None], "ZYX")
    result = mv_opt.transform_mhr_params(
        {k: v.clone() for k, v in mhr_inputs.items()},
        transform,
    )

    F = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))
    expected = source_rot @ (F @ world_rot @ F).T
    actual = euler_angles_to_matrix(result["global_rot"], "ZYX")
    assert torch.allclose(actual, expected[None], atol=1e-6)


def _fake_mhr_inputs(value: float, n_frames: int = 3) -> dict:
    return {
        "global_trans": torch.full((n_frames, 3), value),
        "global_rot": torch.full((n_frames, 3), value),
        "body_pose_params": torch.full((n_frames, 133), value),
        "hand_pose_params": torch.full((n_frames, 108), value),
        "scale_params": torch.full((n_frames, 28), value),
        "shape_params": torch.full((n_frames, 45), value),
    }


def _source_keypoints_from_x_values(x_values: list[list[float]]) -> torch.Tensor:
    x = torch.tensor(x_values, dtype=torch.float64)
    source_keypoints = torch.zeros(x.shape[0], x.shape[1], 70, 3, dtype=torch.float64)
    stable_idxs = mv_opt.INIT_CONSENSUS_KEYPOINT_IDXS
    source_keypoints[:, :, stable_idxs, 0] = x[:, :, None]
    source_keypoints[:, :, 0, 0] = 1000.0
    return source_keypoints


def test_consensus_scores_pick_medoid_with_outlier():
    source_keypoints = _source_keypoints_from_x_values([
        [0.0, -5.0],
        [0.05, 0.0],
        [0.1, 0.05],
        [5.0, 0.1],
    ])
    avg_keypoints = torch.zeros(2, 70, 3, dtype=torch.float64)

    _, source_scores = mv_opt.score_keypoint_consensus_per_frame(
        avg_keypoints_3d=avg_keypoints,
        source_keypoints_3d=source_keypoints,
    )

    assert source_scores.argmin(dim=0).tolist() == [1, 2]


def test_consensus_keeps_average_when_gate_does_not_fail():
    avg_inputs = _fake_mhr_inputs(0.0, n_frames=2)
    source_inputs = [_fake_mhr_inputs(10.0, n_frames=2), _fake_mhr_inputs(20.0, n_frames=2)]
    avg_scores = torch.tensor([0.11, 0.13])
    source_scores = torch.tensor([
        [0.08, 0.09],
        [0.10, 0.10],
    ])

    robust_inputs, diagnostics = mv_opt.select_robust_mhr_inputs_by_consensus_scores(
        mhr_inputs_avg=avg_inputs,
        mhr_inputs_all=source_inputs,
        avg_scores=avg_scores,
        source_scores=source_scores,
    )

    assert diagnostics["fallback_count"] == 0
    assert diagnostics["avg_init_kept"] is True
    assert diagnostics["medoid_init"] is False
    assert diagnostics["chosen_counts"] == [0, 0]
    assert torch.all(robust_inputs["global_trans"] == 0.0)
    assert torch.all(robust_inputs["body_pose_params"] == 0.0)
    assert torch.all(robust_inputs["hand_pose_params"] == 0.0)
    assert torch.all(robust_inputs["scale_params"] == 0.0)
    assert torch.all(robust_inputs["shape_params"] == 0.0)


def test_consensus_fallback_replaces_only_failing_pose_frames():
    avg_inputs = _fake_mhr_inputs(0.0, n_frames=3)
    source_inputs = [_fake_mhr_inputs(10.0, n_frames=3), _fake_mhr_inputs(20.0, n_frames=3)]
    avg_scores = torch.tensor([0.11, 0.50, 0.11])
    source_scores = torch.tensor([
        [0.08, 0.09, 0.08],
        [0.10, 0.10, 0.10],
    ])

    robust_inputs, diagnostics = mv_opt.select_robust_mhr_inputs_by_consensus_scores(
        mhr_inputs_avg=avg_inputs,
        mhr_inputs_all=source_inputs,
        avg_scores=avg_scores,
        source_scores=source_scores,
    )

    assert diagnostics["fallback_count"] == 1
    assert diagnostics["avg_init_kept"] is False
    assert diagnostics["medoid_init"] is True
    assert diagnostics["chosen_counts"] == [1, 0]
    assert torch.all(robust_inputs["global_trans"][0] == 0.0)
    assert torch.all(robust_inputs["global_trans"][1] == 10.0)
    assert torch.all(robust_inputs["global_trans"][2] == 0.0)
    assert torch.all(robust_inputs["body_pose_params"][1] == 10.0)
    assert torch.all(robust_inputs["hand_pose_params"][1] == 10.0)
    assert torch.all(robust_inputs["scale_params"] == 0.0)
    assert torch.all(robust_inputs["shape_params"] == 0.0)


def test_hysteresis_ignores_short_one_frame_source_spike():
    source_scores = torch.full((2, 12), 0.20)
    source_scores[0] = 0.10
    source_scores[0, 5] = 0.30
    source_scores[1, 5] = 0.01

    smoothed_idxs = mv_opt.smooth_medoid_sources_with_hysteresis(source_scores)

    assert smoothed_idxs.tolist() == [0] * 12


def test_hysteresis_switches_after_sustained_better_source():
    source_scores = torch.full((2, 30), 0.20)
    source_scores[0] = 0.10
    source_scores[0, 5:15] = 0.30
    source_scores[1, 5:15] = 0.01

    smoothed_idxs = mv_opt.smooth_medoid_sources_with_hysteresis(source_scores)

    assert smoothed_idxs[:14].tolist() == [0] * 14
    assert smoothed_idxs[14:24].tolist() == [1] * 10
    assert smoothed_idxs[24:].tolist() == [0] * 6


def test_consensus_fallback_can_use_different_sources_after_hysteresis():
    avg_inputs = _fake_mhr_inputs(0.0, n_frames=30)
    source_inputs = [
        _fake_mhr_inputs(10.0, n_frames=30),
        _fake_mhr_inputs(20.0, n_frames=30),
    ]
    avg_scores = torch.ones(30)
    source_scores = torch.full((2, 30), 0.20)
    source_scores[0] = 0.10
    source_scores[0, 5:15] = 0.30
    source_scores[1, 5:15] = 0.01

    robust_inputs, diagnostics = mv_opt.select_robust_mhr_inputs_by_consensus_scores(
        mhr_inputs_avg=avg_inputs,
        mhr_inputs_all=source_inputs,
        avg_scores=avg_scores,
        source_scores=source_scores,
    )

    assert diagnostics["fallback_count"] == 30
    assert diagnostics["avg_init_kept"] is False
    assert diagnostics["medoid_init"] is True
    assert diagnostics["chosen_counts"] == [20, 10]
    assert diagnostics["medoid_switches"] == 2
    assert torch.all(robust_inputs["global_trans"][0] == 10.0)
    assert torch.all(robust_inputs["global_trans"][14] == 20.0)
    assert torch.all(robust_inputs["global_trans"][24] == 10.0)
    assert torch.all(robust_inputs["body_pose_params"][14] == 20.0)
    assert torch.all(robust_inputs["hand_pose_params"][24] == 10.0)
    assert torch.all(robust_inputs["scale_params"] == 0.0)
    assert torch.all(robust_inputs["shape_params"] == 0.0)
