# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for contract-driven semantic recording conversion."""

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

from groot_finetune.closed_loop.embodiment import (
    Q_REF_LEFT_WRIST,
    Q_REF_RIGHT_WRIST,
    align_quat_to_ref,
)
from groot_finetune.contracts import (
    SHARPA_DUAL_HAND_THREE_CAMERA,
    VEGA_SHARPA_JOINT,
)
from groot_finetune.convert_to_gr00t import _canonicalize_quat, convert
from groot_finetune.task_profile import (
    LiftHoldEvaluator,
    TargetObject,
    TaskProfile,
)

PROFILE = TaskProfile(
    task_id="object_lift_hold",
    object_prompt="an object",
    instruction="lift the object and hold it",
    target_object=TargetObject("primary"),
    evaluator=LiftHoldEvaluator(0.1, 0.05, 3),
)


def _write_recording(path: Path, *, frames: int = 5) -> Path:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        data.attrs.update(
            {
                "schema_version": 1,
                "embodiment_contract": VEGA_SHARPA_JOINT.contract_id,
                "embodiment_contract_sha256": VEGA_SHARPA_JOINT.sha256,
                "task_profile": PROFILE.task_id,
                "task_profile_sha256": PROFILE.sha256,
                "fps": VEGA_SHARPA_JOINT.fps,
            }
        )
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset(
            "arm_joint_pos", data=rng.random((frames, 14), dtype=np.float32)
        )
        obs.create_dataset(
            "finger_joint_pos", data=rng.random((frames, 44), dtype=np.float32)
        )
        for camera in VEGA_SHARPA_JOINT.cameras:
            obs.create_dataset(
                camera.observation_term,
                data=rng.integers(0, 255, size=(frames, 16, 16, 3), dtype=np.uint8),
            )
        demo.create_dataset(
            "actions",
            data=rng.random((frames, VEGA_SHARPA_JOINT.action_dim), dtype=np.float32),
        )
    return path


def test_converts_exact_contract(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    convert(
        _write_recording(tmp_path / "data.h5"),
        output,
        VEGA_SHARPA_JOINT,
        PROFILE,
    )
    modality = json.loads((output / "meta/modality.json").read_text())
    assert list(modality["state"]) == [
        "right_arm",
        "left_arm",
        "right_finger",
        "left_finger",
    ]
    assert list(modality["action"]) == [
        "right_arm",
        "right_finger",
        "left_arm",
        "left_finger",
    ]
    assert list(modality["video"]) == [
        "front",
        "right_wrist_view",
        "left_wrist_view",
    ]
    info = json.loads((output / "meta/info.json").read_text())
    assert info["robot_type"] == "vega_sharpa_joint"
    assert info["task_profile"] == PROFILE.task_id
    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    assert table.num_rows == 5


def test_rejects_provenance_mismatch(tmp_path: Path) -> None:
    path = _write_recording(tmp_path / "data.h5")
    with h5py.File(path, "a") as recording:
        recording["data"].attrs["embodiment_contract_sha256"] = "invalid"
    with pytest.raises(ValueError, match="provenance"):
        convert(path, tmp_path / "out", VEGA_SHARPA_JOINT, PROFILE)


def test_rejects_missing_required_camera(tmp_path: Path) -> None:
    path = _write_recording(tmp_path / "data.h5")
    with h5py.File(path, "a") as recording:
        del recording["data/demo_0/obs/image_left_wrist"]
    with pytest.raises(ValueError, match="missing required camera"):
        convert(path, tmp_path / "out", VEGA_SHARPA_JOINT, PROFILE)


def test_rejects_action_dimension_mismatch(tmp_path: Path) -> None:
    path = _write_recording(tmp_path / "data.h5")
    with h5py.File(path, "a") as recording:
        demo = recording["data/demo_0"]
        del demo["actions"]
        demo.create_dataset("actions", data=np.zeros((5, 57), dtype=np.float32))
    with pytest.raises(ValueError, match="actions must be"):
        convert(path, tmp_path / "out", VEGA_SHARPA_JOINT, PROFILE)


def test_rejects_nonfinite_semantic_values(tmp_path: Path) -> None:
    path = _write_recording(tmp_path / "data.h5")
    with h5py.File(path, "a") as recording:
        recording["data/demo_0/obs/arm_joint_pos"][2, 3] = np.nan
    with pytest.raises(ValueError, match="finite numeric"):
        convert(path, tmp_path / "out", VEGA_SHARPA_JOINT, PROFILE)


def test_rejects_frequency_mismatch(tmp_path: Path) -> None:
    path = _write_recording(tmp_path / "data.h5")
    with h5py.File(path, "a") as recording:
        recording["data"].attrs["fps"] = 10
    with pytest.raises(ValueError, match="provenance"):
        convert(path, tmp_path / "out", VEGA_SHARPA_JOINT, PROFILE)


def test_quaternion_transform_is_continuous_and_matches_inference() -> None:
    rng = np.random.default_rng(7)
    trajectory = np.tile(Q_REF_RIGHT_WRIST, (50, 1)).astype(np.float32)
    trajectory += np.linspace(0.0, 0.01, 50, dtype=np.float32)[:, None]
    trajectory /= np.linalg.norm(trajectory, axis=1, keepdims=True)
    trajectory *= rng.choice((-1.0, 1.0), size=(50, 1)).astype(np.float32)

    _canonicalize_quat(trajectory, 0, Q_REF_RIGHT_WRIST)

    assert np.linalg.norm(np.diff(trajectory, axis=0), axis=1).max() < 0.1
    np.testing.assert_allclose(
        trajectory,
        align_quat_to_ref(trajectory, Q_REF_RIGHT_WRIST),
        atol=1e-6,
    )


def test_floating_contract_rejects_missing_required_camera(tmp_path: Path) -> None:
    path = tmp_path / "floating.h5"
    frames = 3
    rng = np.random.default_rng(3)
    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        data.attrs.update(
            {
                "schema_version": 1,
                "embodiment_contract": SHARPA_DUAL_HAND_THREE_CAMERA.contract_id,
                "embodiment_contract_sha256": SHARPA_DUAL_HAND_THREE_CAMERA.sha256,
                "task_profile": PROFILE.task_id,
                "task_profile_sha256": PROFILE.sha256,
                "fps": SHARPA_DUAL_HAND_THREE_CAMERA.fps,
            }
        )
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset(
            "wrist_position_e", data=rng.random((frames, 6), dtype=np.float32)
        )
        obs.create_dataset(
            "wrist_orientation_e",
            data=np.tile(
                np.concatenate([Q_REF_RIGHT_WRIST, Q_REF_LEFT_WRIST]),
                (frames, 1),
            ),
        )
        obs.create_dataset(
            "finger_joint_pos", data=rng.random((frames, 44), dtype=np.float32)
        )
        obs.create_dataset(
            "image",
            data=rng.integers(0, 255, (frames, 16, 16, 3), dtype=np.uint8),
        )
        actions = rng.random(
            (frames, SHARPA_DUAL_HAND_THREE_CAMERA.action_dim), dtype=np.float32
        )
        actions[:, 3:7] = Q_REF_RIGHT_WRIST
        actions[:, 32:36] = Q_REF_LEFT_WRIST
        demo.create_dataset("actions", data=actions)

    with pytest.raises(ValueError, match="missing required camera"):
        convert(
            path, tmp_path / "floating_dataset", SHARPA_DUAL_HAND_THREE_CAMERA, PROFILE
        )
