"""CPU tests for MECKA contracts and retarget geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inpainting.adapters.mecka import build_tracking_arrays
from inpainting.adapters.mecka_parallel_jaw import retarget_tracking_arrays
from inpainting.composite_robot import depth_visible_robot_mask
from inpainting.contracts import (
    ContractError,
    validate_parallel_jaw_arrays,
    validate_tracking_arrays,
)
from inpainting.panda_renderer.kinematics import PandaIK, build_panda_model
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


def test_panda_dls_smoke() -> None:
    if not (DEFAULT_PANDA_DIR / "panda.xml").is_file():
        pytest.skip("local MuJoCo Menagerie Panda assets are unavailable")
    model = build_panda_model(DEFAULT_PANDA_DIR, 60.0, 64, 64)
    solver = PandaIK(model, np.array([0.0, 0.4, 0.4]), np.array([0.0, 0.0, 0.6]))
    target = solver.fingertip_center().copy()
    hand_rotation = solver.data.body(solver.hand_id).xmat.reshape(3, 3).copy()
    semantic_rotation = hand_rotation @ solver.semantic_to_hand
    residual = solver.solve_dls(
        target,
        semantic_rotation,
        0.05,
        previous_q=None,
        elbow_outward=np.array([1.0, 0.0, 0.0]),
        iterations=4,
    )
    assert residual < 1e-6

