# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for strict joint-rollout validation."""

import json
from pathlib import Path

import numpy as np
import pytest

from groot_finetune.contracts import VEGA_SHARPA_JOINT
from groot_finetune.replay_export import load_replay_export

OBJECTS = ("object", "tray")


def _write_episode(root: Path, index: int, *, source_success: bool) -> None:
    frames = 4
    np.savez_compressed(
        root / f"episode_{index:05d}.npz",
        joint_pos=np.full(
            (frames, VEGA_SHARPA_JOINT.action_dim), index, dtype=np.float32
        ),
        action_target=np.full(
            (frames, VEGA_SHARPA_JOINT.action_dim), index + 1, dtype=np.float32
        ),
        object_pose=np.zeros((frames, len(OBJECTS), 7), dtype=np.float32),
        source_success=np.asarray(source_success),
        termination_reasons=np.asarray(
            ["timeout"] if source_success else ["robot_state_diverged"]
        ),
    )


def _write_export(
    root: Path,
    *,
    successes: tuple[bool, ...] = (True,),
    object_names: tuple[str, ...] = OBJECTS,
) -> Path:
    root.mkdir()
    manifest = {
        "format": "joint_rollout",
        "schema_version": 1,
        "embodiment_contract": VEGA_SHARPA_JOINT.contract_id,
        "embodiment_contract_sha256": VEGA_SHARPA_JOINT.sha256,
        "fps": VEGA_SHARPA_JOINT.fps,
        "joint_names": list(VEGA_SHARPA_JOINT.joint_names),
        "object_names": list(object_names),
        "timeout_termination": "timeout",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for index, success in enumerate(successes):
        _write_episode(root, index, source_success=success)
    return root


def test_loads_multi_object_rollout(tmp_path: Path) -> None:
    export = load_replay_export(
        _write_export(tmp_path / "export"), VEGA_SHARPA_JOINT, OBJECTS
    )
    assert export.episodes[0].object_pose.shape == (4, 2, 7)
    assert export.source_successful_episode_count == 1


def test_filters_source_failures(tmp_path: Path) -> None:
    export = load_replay_export(
        _write_export(tmp_path / "export", successes=(False, True, False)),
        VEGA_SHARPA_JOINT,
        OBJECTS,
    )
    assert [episode.path.name for episode in export.episodes] == ["episode_00001.npz"]
    assert export.source_unsuccessful_episode_count == 2


def test_all_mode_is_explicit(tmp_path: Path) -> None:
    export = load_replay_export(
        _write_export(tmp_path / "export", successes=(False, True)),
        VEGA_SHARPA_JOINT,
        OBJECTS,
        source_success_only=False,
    )
    assert [episode.source_success for episode in export.episodes] == [False, True]


def test_rejects_contract_hash_mismatch(tmp_path: Path) -> None:
    root = _write_export(tmp_path / "export")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["embodiment_contract_sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash"):
        load_replay_export(root, VEGA_SHARPA_JOINT, OBJECTS)


def test_rejects_object_order_mismatch(tmp_path: Path) -> None:
    root = _write_export(tmp_path / "export", object_names=tuple(reversed(OBJECTS)))
    with pytest.raises(ValueError, match="object names or order"):
        load_replay_export(root, VEGA_SHARPA_JOINT, OBJECTS)


def test_rejects_nonfinite_actions(tmp_path: Path) -> None:
    root = _write_export(tmp_path / "export")
    path = root / "episode_00000.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["action_target"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="non-finite"):
        load_replay_export(root, VEGA_SHARPA_JOINT, OBJECTS)
