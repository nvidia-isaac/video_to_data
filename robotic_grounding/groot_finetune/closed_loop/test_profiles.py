# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the closed-loop embodiment profiles (no server, no IsaacLab).

Proves: both built-in profiles register; each profile emits every video key with
the right dtype and shape through ``Gr00tActionAdapter.build_observation``; the adapter
rejects a checkpoint modality config missing a key the profile needs. Floating-hand and
joint-space embodiments both use front and wrist cameras.

Run from the repo root:

    python -m pytest groot_finetune/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

from groot_finetune.closed_loop.action_adapter import Gr00tActionAdapter
from groot_finetune.closed_loop.embodiment import Q_REF_RIGHT_WRIST, get_profile

_ALL_VIEWS = ["front", "right_wrist_view", "left_wrist_view"]

_SHARPA_STATE_KEYS = [
    "right_wrist_pos",
    "left_wrist_pos",
    "right_wrist_quat",
    "left_wrist_quat",
    "right_finger",
    "left_finger",
]
_SHARPA_ACTION_KEYS = [
    "right_wrist_pos",
    "right_wrist_quat",
    "right_finger",
    "left_wrist_pos",
    "left_wrist_quat",
    "left_finger",
]
_VEGA_STATE_KEYS = ["right_arm", "left_arm", "right_finger", "left_finger"]
_VEGA_ACTION_KEYS = ["right_arm", "right_finger", "left_arm", "left_finger"]


class _FakeClient:
    """Stands in for the ZMQ policy client: serves a plain-dict modality config."""

    def __init__(
        self,
        video_keys: list[str],
        state_keys: list[str] | None = None,
        action_keys: list[str] | None = None,
    ):
        self._config = {
            "video": {"delta_indices": [0], "modality_keys": video_keys},
            "state": {
                "delta_indices": [0],
                "modality_keys": state_keys or _SHARPA_STATE_KEYS,
            },
            "action": {
                "delta_indices": list(range(16)),
                "modality_keys": action_keys or _SHARPA_ACTION_KEYS,
            },
            "language": {
                "delta_indices": [0],
                "modality_keys": ["annotation.human.task_description"],
            },
        }

    def get_modality_config(self):
        return self._config


class _ChunkingClient(_FakeClient):
    """Return deterministic chunks that expose query, timestep, and field order."""

    _WIDTHS = {
        "right_arm": 7,
        "right_finger": 22,
        "left_arm": 7,
        "left_finger": 22,
    }

    def __init__(self, *, response_batch: int | None = None) -> None:
        super().__init__(_ALL_VIEWS, _VEGA_STATE_KEYS, _VEGA_ACTION_KEYS)
        self.query_count = 0
        self.response_batch = response_batch

    def get_action(self, observation):
        batch = next(iter(observation["video"].values())).shape[0]
        if self.response_batch is not None:
            batch = self.response_batch
        query = self.query_count
        self.query_count += 1
        result = {}
        for field_index, key in enumerate(_VEGA_ACTION_KEYS):
            values = np.empty((batch, 16, self._WIDTHS[key]), dtype=np.float32)
            for timestep in range(16):
                values[:, timestep, :] = query * 1000 + timestep * 10 + field_index
            result[key] = values
        return result, {"query": query}


def _expected_chunk_action(query: int, timestep: int, batch: int = 2) -> np.ndarray:
    parts = [
        np.full(
            (batch, _ChunkingClient._WIDTHS[key]),
            query * 1000 + timestep * 10 + field_index,
            dtype=np.float32,
        )
        for field_index, key in enumerate(_VEGA_ACTION_KEYS)
    ]
    return np.concatenate(parts, axis=1)


def _vega_client(video_keys: list[str]) -> _FakeClient:
    return _FakeClient(video_keys, _VEGA_STATE_KEYS, _VEGA_ACTION_KEYS)


def _images(batch: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        term: rng.integers(0, 255, (batch, 16, 16, 3)).astype(np.uint8)
        for term in ("image", "image_right_wrist", "image_left_wrist")
    }


def _sharpa_record_obs(batch: int = 2) -> dict[str, np.ndarray]:
    """Task-space record group: wrist pose + scaled finger joints."""
    rng = np.random.default_rng(0)
    quats = np.zeros((batch, 8), dtype=np.float32)
    quats[:, 0] = 1.0
    quats[:, 4] = 1.0
    return {
        **_images(batch, rng),
        "wrist_position_e": rng.random((batch, 6), dtype=np.float32),
        "wrist_orientation_e": quats,
        "finger_joint_pos": rng.random((batch, 44), dtype=np.float32),
    }


def _vega_record_obs(batch: int = 2) -> dict[str, np.ndarray]:
    """Joint-space record group: raw arm + finger joint positions."""
    rng = np.random.default_rng(0)
    return {
        **_images(batch, rng),
        "arm_joint_pos": rng.random((batch, 14), dtype=np.float32),
        "finger_joint_pos": rng.random((batch, 44), dtype=np.float32),
    }


def test_both_profiles_registered() -> None:
    assert list(get_profile("sharpa_dual_hand_three_camera").video_keys) == _ALL_VIEWS
    assert list(get_profile("vega_sharpa_joint").video_keys) == _ALL_VIEWS


def test_multiview_observation_has_three_views() -> None:
    adapter = Gr00tActionAdapter(
        _vega_client(_ALL_VIEWS),
        get_profile("vega_sharpa_joint"),
        task="move the object",
    )
    obs = adapter.build_observation(_vega_record_obs())
    assert sorted(obs["video"]) == sorted(_ALL_VIEWS)
    for key in _ALL_VIEWS:
        arr = obs["video"][key]
        assert arr.dtype == np.uint8
        assert arr.shape[0] == 2 and arr.shape[-3:] == (16, 16, 3), arr.shape


def test_build_observation_uses_mapping_keys_for_tensordict_compatibility() -> None:
    class _TensorDictLike:
        def __init__(self, values: dict[str, np.ndarray]) -> None:
            self._values = values

        def __iter__(self):
            # TensorDict iteration does not yield field names.
            return iter([{"batch": 0}])

        def keys(self):
            return self._values.keys()

        def __getitem__(self, key: str) -> np.ndarray:
            return self._values[key]

    adapter = Gr00tActionAdapter(
        _vega_client(_ALL_VIEWS),
        get_profile("vega_sharpa_joint"),
        task="move the object",
    )
    obs = adapter.build_observation(_TensorDictLike(_vega_record_obs()))
    assert list(obs["state"]) == _VEGA_STATE_KEYS


def test_validate_rejects_missing_wrist_key() -> None:
    with pytest.raises(KeyError, match="video keys/order"):
        Gr00tActionAdapter(
            _vega_client(["front"]),
            get_profile("vega_sharpa_joint"),
            task="move the object",
        )


def test_validate_rejects_reordered_action_keys() -> None:
    client = _FakeClient(
        _ALL_VIEWS,
        _VEGA_STATE_KEYS,
        list(reversed(_VEGA_ACTION_KEYS)),
    )
    with pytest.raises(KeyError, match="action keys/order"):
        Gr00tActionAdapter(
            client,
            get_profile("vega_sharpa_joint"),
            task="move the object",
        )


def test_validate_rejects_observation_history() -> None:
    client = _vega_client(_ALL_VIEWS)
    client._config["state"]["delta_indices"] = [-1, 0]
    with pytest.raises(ValueError, match="state delta indices"):
        Gr00tActionAdapter(
            client,
            get_profile("vega_sharpa_joint"),
            task="move the object",
        )


def test_reassemble_rejects_wrong_field_width_and_nonfinite_action() -> None:
    adapter = Gr00tActionAdapter(
        _vega_client(_ALL_VIEWS),
        get_profile("vega_sharpa_joint"),
        task="move the object",
    )
    action = {
        "right_arm": np.zeros((2, 16, 7), dtype=np.float32),
        "right_finger": np.zeros((2, 16, 22), dtype=np.float32),
        "left_arm": np.zeros((2, 16, 7), dtype=np.float32),
        "left_finger": np.zeros((2, 16, 22), dtype=np.float32),
    }
    action["left_arm"] = np.zeros((2, 16, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="left_arm"):
        adapter._reassemble(action)
    action["left_arm"] = np.zeros((2, 16, 7), dtype=np.float32)
    action["left_arm"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite numeric"):
        adapter._reassemble(action)


def test_build_observation_rejects_float_camera() -> None:
    adapter = Gr00tActionAdapter(
        _vega_client(_ALL_VIEWS),
        get_profile("vega_sharpa_joint"),
        task="move the object",
    )
    record = _vega_record_obs()
    record["image"] = record["image"].astype(np.float32)
    with pytest.raises(ValueError, match="must be uint8"):
        adapter.build_observation(record)


def test_action_buffer_requeries_after_execution_length() -> None:
    client = _ChunkingClient()
    adapter = Gr00tActionAdapter(
        client,
        get_profile("vega_sharpa_joint"),
        execution_length=4,
        task="move the object",
    )
    record = _vega_record_obs()

    for timestep in range(4):
        action = adapter.act(record)
        np.testing.assert_array_equal(action, _expected_chunk_action(0, timestep))
        assert action.dtype == np.float32
        assert action.flags.c_contiguous
    assert client.query_count == 1

    np.testing.assert_array_equal(adapter.act(record), _expected_chunk_action(1, 0))
    assert client.query_count == 2


def test_done_and_reset_invalidate_the_buffered_chunk() -> None:
    client = _ChunkingClient()
    adapter = Gr00tActionAdapter(
        client,
        get_profile("vega_sharpa_joint"),
        execution_length=4,
        task="move the object",
    )
    record = _vega_record_obs()

    np.testing.assert_array_equal(adapter.act(record), _expected_chunk_action(0, 0))
    np.testing.assert_array_equal(
        adapter.act(record, dones=np.asarray([False, True])),
        _expected_chunk_action(1, 0),
    )
    assert client.query_count == 2

    adapter.reset()
    np.testing.assert_array_equal(adapter.act(record), _expected_chunk_action(2, 0))
    assert client.query_count == 3


def test_action_buffer_rejects_response_batch_mismatch() -> None:
    client = _ChunkingClient(response_batch=1)
    adapter = Gr00tActionAdapter(
        client,
        get_profile("vega_sharpa_joint"),
        task="move the object",
    )

    for _ in range(2):
        with pytest.raises(ValueError, match="action batch does not match"):
            adapter.act(_vega_record_obs(batch=2))
    assert client.query_count == 2


def test_sharpa_profile_builds_all_required_views() -> None:
    adapter = Gr00tActionAdapter(
        _FakeClient(_ALL_VIEWS),
        get_profile("sharpa_dual_hand_three_camera"),
        task="move the object",
    )
    obs = adapter.build_observation(_sharpa_record_obs())
    assert list(obs["video"]) == _ALL_VIEWS


def test_state_fields_are_sliced_by_spec() -> None:
    """Every state key must carry its own slice of the source term, in order."""
    adapter = Gr00tActionAdapter(
        _FakeClient(_ALL_VIEWS),
        get_profile("sharpa_dual_hand_three_camera"),
        task="move the object",
    )
    record = _sharpa_record_obs(batch=2)
    obs = adapter.build_observation(record)
    state = obs["state"]
    assert set(state) == set(_SHARPA_STATE_KEYS)
    # (B, T_state, D) with the dims the profile declares.
    assert state["right_wrist_pos"].shape == (2, 1, 3)
    assert state["right_finger"].shape == (2, 1, 22)
    # right/left finger must come from opposite halves of finger_joint_pos, not the same one.
    fingers = record["finger_joint_pos"]
    np.testing.assert_allclose(state["right_finger"][:, 0, :], fingers[:, 0:22])
    np.testing.assert_allclose(state["left_finger"][:, 0, :], fingers[:, 22:44])
    np.testing.assert_allclose(
        state["right_wrist_pos"][:, 0, :], record["wrist_position_e"][:, 0:3]
    )
    np.testing.assert_allclose(
        state["left_wrist_pos"][:, 0, :], record["wrist_position_e"][:, 3:6]
    )


def test_sharpa_profile_applies_reference_hemisphere() -> None:
    """Inference uses the same reference hemisphere as conversion."""
    q = Q_REF_RIGHT_WRIST.copy()
    q[0] = -q[0]
    assert float(q @ Q_REF_RIGHT_WRIST) > 0

    record = _sharpa_record_obs(batch=1)
    record["wrist_orientation_e"] = np.concatenate(
        [q[None, :], q[None, :]], axis=1
    ).astype(np.float32)

    adapter = Gr00tActionAdapter(
        _FakeClient(_ALL_VIEWS),
        get_profile("sharpa_dual_hand_three_camera"),
        task="move the object",
    )
    out = adapter.build_observation(record)["state"]["right_wrist_quat"][0, 0]
    np.testing.assert_allclose(out, q, atol=1e-6)
