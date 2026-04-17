# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for the replay data adapter (schema detection, trajectory loading, joint reorder).

These tests run without Isaac Lab / Omniverse — they only exercise the Parquet
adapter and joint-reorder logic.

Usage (pytest):
    pytest tests/test_replay_data.py -v

Usage (direct):
    python tests/test_replay_data.py
"""

import importlib.util
import inspect
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from robotic_grounding.retarget.data_logger import (
    ManoSharpaData,
    NvhumanG1Data,
)
from robotic_grounding.retarget.params import G1_WHOLEBODY_TO_NVHUMAN_MAPPING

# replay_data lives under robotic_grounding.tasks which transitively imports
# Isaac Lab / Omniverse.  Use the same importlib fallback as the existing
# test_nvhuman_g1_parquet_integration.py so the test runs without Omniverse.
_SCENE_UTILS_DIR = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "robotic_grounding"
    / "robotic_grounding"
    / "tasks"
    / "scene_utils"
)


def _load_module_directly(name: str, path: Path) -> Any:
    """Load a single .py module without triggering __init__ chains."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# scene_config must be loaded first because replay_data imports it.
_scene_config_mod = _load_module_directly(
    "robotic_grounding.tasks.scene_utils.scene_config",
    _SCENE_UTILS_DIR / "scene_config.py",
)
_replay_data_mod = _load_module_directly(
    "robotic_grounding.tasks.scene_utils.replay_data",
    _SCENE_UTILS_DIR / "replay_data.py",
)

DualHandTrajectory = _replay_data_mod.DualHandTrajectory
SingleRobotTrajectory = _replay_data_mod.SingleRobotTrajectory
_is_dex3_schema = _replay_data_mod._is_dex3_schema
_is_g1_schema = _replay_data_mod._is_g1_schema
_is_sharpa_schema = _replay_data_mod._is_sharpa_schema
load_replay_trajectory = _replay_data_mod.load_replay_trajectory


def _build_joint_reorder(
    parquet_names: list[str], sim_names: list[str]
) -> torch.Tensor | None:
    """Inline copy of the reorder helper (avoids importing replay_motion which needs Isaac Lab)."""
    if parquet_names == sim_names:
        return None
    sim_name_to_idx = {n: i for i, n in enumerate(sim_names)}
    indices: list[int] = []
    for pq_name in parquet_names:
        if pq_name not in sim_name_to_idx:
            raise ValueError(
                f"Parquet joint '{pq_name}' not found in spawned robot joints: "
                f"{sim_names}"
            )
        indices.append(sim_name_to_idx[pq_name])
    return torch.tensor(indices, dtype=torch.long)


# ============================================================
# Schema detection
# ============================================================


def test_schema_detection_g1() -> None:
    """G1 schema detected from required columns."""
    cols = {
        "robot_root_position",
        "robot_root_wxyz",
        "robot_joint_positions",
        "fps",
    }
    assert _is_g1_schema(cols)
    assert not _is_sharpa_schema(cols)
    assert not _is_dex3_schema(cols)


def test_schema_detection_sharpa() -> None:
    """Sharpa schema detected from required columns."""
    cols = {
        "robot_right_wrist_position",
        "robot_right_wrist_wxyz",
        "robot_right_finger_joints",
        "robot_left_wrist_position",
        "robot_left_wrist_wxyz",
        "robot_left_finger_joints",
    }
    assert _is_sharpa_schema(cols)
    assert not _is_g1_schema(cols)
    assert not _is_dex3_schema(cols)


def test_schema_detection_dex3() -> None:
    """Dex3 schema detected from required columns."""
    cols = {
        "robot_right_wrist_position",
        "robot_right_wrist_euler_xyz",
        "robot_right_finger_joints",
        "robot_left_wrist_position",
        "robot_left_wrist_euler_xyz",
        "robot_left_finger_joints",
    }
    assert _is_dex3_schema(cols)
    assert not _is_g1_schema(cols)
    assert not _is_sharpa_schema(cols)


# ============================================================
# G1 trajectory round-trip
# ============================================================


def _write_g1_parquet(output_dir: Path) -> Path:
    """Write a minimal NvhumanG1Data parquet and return the partition dir."""
    seq_id = "replay_test_seq"
    robot_name = "g1"
    frame_task_errors = [0.0] * len(G1_WHOLEBODY_TO_NVHUMAN_MAPPING)
    joint_names = ["joint_a", "joint_b"]

    data = NvhumanG1Data(
        sequence_id=seq_id,
        raw_motion_file="fake.pt",
        robot_name=robot_name,
        fps=30.0,
        nvhuman_betas=[0.0] * 10,
        robot_joint_names=joint_names,
        robot_frame_names=["pelvis"],
        robot_frame_task_names=list(G1_WHOLEBODY_TO_NVHUMAN_MAPPING.keys()),
        source_to_robot_scale=1.0,
        object_name="test_obj",
        safe_object_name="test_obj",
        object_body_names=[],
        safe_object_body_names=[],
        object_mesh_paths=[],
        object_urdf_paths=[],
        object_mesh_radius=[],
    )

    for i in range(5):
        data.log_timestep(
            nvhuman_joints=[[0.0, 0.0, 0.0] for _ in range(93)],
            nvhuman_joints_wxyz=[[1.0, 0.0, 0.0, 0.0] for _ in range(93)],
            nvhuman_head_translation=[0.0, 0.0, 1.0 + i * 0.01],
            nvhuman_head_wxyz=[1.0, 0.0, 0.0, 0.0],
            nvhuman_root_translation=[0.0, 0.0, 0.8 + i * 0.01],
            nvhuman_root_wxyz=[1.0, 0.0, 0.0, 0.0],
            robot_root_position=[0.0, 0.0, 0.1 * i],
            robot_root_wxyz=[1.0, 0.0, 0.0, 0.0],
            robot_joint_positions=[0.1 * i, -0.1 * i],
            robot_frames=[[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]],
            robot_frame_task_errors=frame_task_errors,
            robot_ik_error=0.0,
            robot_num_optimization_iterations=1,
            object_articulation=0.0,
            object_root_axis_angle=[0.0, 0.0, 0.1 * i],
            object_root_position=[0.2, 0.3, 0.4 + 0.01 * i],
            object_body_position=[[0.2, 0.3, 0.4 + 0.01 * i]],
            object_body_wxyz=[[1.0, 0.0, 0.0, 0.0]],
        )

    data.save_to_parquet(str(output_dir), partition_cols=["sequence_id", "robot_name"])
    return output_dir / f"sequence_id={seq_id}" / f"robot_name={robot_name}"


def test_load_g1_trajectory(tmp_path: Path) -> None:
    """Load G1 parquet via replay adapter and verify canonical structure."""
    partition_dir = _write_g1_parquet(tmp_path / "g1_data")
    traj = load_replay_trajectory(str(partition_dir))

    assert isinstance(traj, SingleRobotTrajectory)
    assert traj.schema == "nvhuman_g1"
    assert traj.robot_layout == "single_robot"
    assert traj.num_frames == 5
    assert traj.fps == 30.0
    assert traj.robot_joint_names == ["joint_a", "joint_b"]
    assert traj.robot_root_position.shape == (5, 3)
    assert traj.robot_root_wxyz.shape == (5, 4)
    assert traj.robot_joint_positions.shape == (5, 2)
    assert traj.object_traj is not None
    assert traj.object_traj.root_position.shape == (5, 3)
    assert traj.object_traj.root_wxyz.shape == (5, 4)
    np.testing.assert_allclose(traj.robot_root_position[0, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(traj.robot_root_position[4, 2], 0.4, atol=1e-6)


# ============================================================
# Sharpa trajectory round-trip
# ============================================================


def _write_sharpa_parquet(output_dir: Path) -> Path:
    """Write a minimal ManoSharpaData parquet and return the partition dir."""
    seq_id = "sharpa_test_seq"
    robot_name = "sharpa_wave"

    data = ManoSharpaData(
        sequence_id=seq_id,
        raw_motion_file="fake.pt",
        robot_name=robot_name,
        fps=120.0,
        mano_flat_hand_mean=True,
        mano_center_idx=0,
        mano_to_robot_scale=1.0,
        mano_right_betas=[0.0] * 10,
        mano_left_betas=[0.0] * 10,
        mano_link_names=["palm"],
        right_robot_finger_joint_names=["r_j0", "r_j1"],
        right_robot_frame_names=["r_frame"],
        right_robot_frame_task_names=["r_task"],
        left_robot_finger_joint_names=["l_j0", "l_j1"],
        left_robot_frame_names=["l_frame"],
        left_robot_frame_task_names=["l_task"],
        object_name="sharpa_obj",
        safe_object_name="sharpa_obj",
        object_body_names=[],
        safe_object_body_names=[],
        object_mesh_paths=[],
        object_urdf_paths=[],
        object_mesh_radius=[],
    )

    for i in range(3):
        data.log_timestep(
            mano_right_trans=[0.1 * i, 0.0, 0.0],
            mano_right_global_orient=[0.0, 0.0, 0.0],
            mano_right_finger_pose=[0.0] * 45,
            mano_right_joints=[[0.0, 0.0, 0.0]] * 21,
            mano_right_joints_wxyz=[[1.0, 0.0, 0.0, 0.0]] * 21,
            mano_left_trans=[-0.1 * i, 0.0, 0.0],
            mano_left_global_orient=[0.0, 0.0, 0.0],
            mano_left_finger_pose=[0.0] * 45,
            mano_left_joints=[[0.0, 0.0, 0.0]] * 21,
            mano_left_joints_wxyz=[[1.0, 0.0, 0.0, 0.0]] * 21,
            robot_right_wrist_position=[0.1 * i, 0.0, 0.3],
            robot_right_wrist_wxyz=[1.0, 0.0, 0.0, 0.0],
            robot_right_finger_joints=[0.0] * 22,
            robot_right_frames=[[0.0] * 7] * 67,
            robot_right_frame_task_errors=[0.0] * 11,
            robot_right_num_optimization_iterations=1,
            robot_left_wrist_position=[-0.1 * i, 0.0, 0.3],
            robot_left_wrist_wxyz=[1.0, 0.0, 0.0, 0.0],
            robot_left_finger_joints=[0.0] * 22,
            robot_left_frames=[[0.0] * 7] * 67,
            robot_left_frame_task_errors=[0.0] * 11,
            robot_left_num_optimization_iterations=1,
            object_articulation=0.0,
            object_root_axis_angle=[0.0, 0.0, 0.0],
            object_root_position=[0.0, 0.0, 0.5],
        )

    data.save_to_parquet(str(output_dir), partition_cols=["sequence_id", "robot_name"])
    return output_dir / f"sequence_id={seq_id}" / f"robot_name={robot_name}"


def test_load_sharpa_trajectory(tmp_path: Path) -> None:
    """Load Sharpa parquet via replay adapter and verify dual-hand structure."""
    partition_dir = _write_sharpa_parquet(tmp_path / "sharpa_data")
    traj = load_replay_trajectory(str(partition_dir))

    assert isinstance(traj, DualHandTrajectory)
    assert traj.schema == "mano_sharpa"
    assert traj.robot_layout == "dual_hand"
    assert traj.num_frames == 3
    assert traj.fps == 120.0
    assert traj.wrist_orientation_format == "wxyz"
    assert traj.right_joint_names == ["r_j0", "r_j1"]
    assert traj.left_joint_names == ["l_j0", "l_j1"]
    assert traj.right_wrist_position.shape == (3, 3)
    assert traj.left_wrist_position.shape == (3, 3)
    assert traj.right_wrist_orientation.shape == (3, 4)
    assert traj.right_finger_joints.shape == (3, 22)
    assert traj.object_traj is not None
    assert traj.object_traj.root_position.shape == (3, 3)


# ============================================================
# Joint reorder
# ============================================================


def test_build_joint_reorder_identity() -> None:
    """Identical ordering returns None (no reorder needed)."""
    names = ["a", "b", "c"]
    assert _build_joint_reorder(names, names) is None


def test_build_joint_reorder_permutation() -> None:
    """Permuted ordering returns correct index mapping."""
    parquet = ["c", "a", "b"]
    sim = ["a", "b", "c"]
    reorder = _build_joint_reorder(parquet, sim)
    assert reorder is not None
    expected = torch.tensor([2, 0, 1], dtype=torch.long)
    assert torch.equal(reorder, expected)


# ============================================================
# Script runner (no pytest)
# ============================================================

_ALL_TESTS: list[Callable[..., Any]] = [
    test_schema_detection_g1,
    test_schema_detection_sharpa,
    test_schema_detection_dex3,
    test_load_g1_trajectory,
    test_load_sharpa_trajectory,
    test_build_joint_reorder_identity,
    test_build_joint_reorder_permutation,
]


def _run_as_script() -> int:
    """Run all tests directly without pytest."""
    print("=" * 72, flush=True)
    print("Running replay_data adapter tests", flush=True)
    print("=" * 72, flush=True)
    passed = 0
    failed = 0
    for test_fn in _ALL_TESTS:
        name = test_fn.__name__
        try:
            with tempfile.TemporaryDirectory(prefix="replay_test_") as tmp_dir:
                sig = inspect.signature(test_fn)
                if "tmp_path" in sig.parameters:
                    test_fn(tmp_path=Path(tmp_dir))
                else:
                    test_fn()
            print(f"  [PASS] {name}", flush=True)
            passed += 1
        except Exception:
            print(f"  [FAIL] {name}", flush=True)
            traceback.print_exc()
            failed += 1
    print("-" * 72, flush=True)
    print(f"Results: {passed} passed, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
