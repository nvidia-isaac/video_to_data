# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for inference-only source-policy adapters."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from groot_finetune.source_policy import OnnxSourcePolicy


class _TensorInfo:
    def __init__(
        self,
        name: str,
        *,
        shape: tuple[object, ...],
        tensor_type: str = "tensor(float)",
    ) -> None:
        self.name = name
        self.shape = shape
        self.type = tensor_type


class _Session:
    def __init__(self, *, batch: object = "batch") -> None:
        self.last_feed: dict[str, np.ndarray] | None = None
        self.feeds: list[np.ndarray] = []
        self.batch = batch

    def get_inputs(self) -> list[_TensorInfo]:
        return [_TensorInfo("observations", shape=(self.batch, 4))]

    def get_outputs(self) -> list[_TensorInfo]:
        return [_TensorInfo("actions", shape=("batch", 2))]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(
        self, output_names: list[str], feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names == ["actions"]
        self.last_feed = feed
        self.feeds.append(feed["observations"])
        return [feed["observations"][:, :2] * 2.0]


class _MultiInputSession(_Session):
    def get_inputs(self) -> list[_TensorInfo]:
        return [
            _TensorInfo("first", shape=("batch", 4)),
            _TensorInfo("second", shape=("batch", 4)),
        ]


def test_onnx_source_policy_preserves_batch_and_device() -> None:
    session = _Session()
    policy = OnnxSourcePolicy(session, device="cpu")
    observations = torch.arange(8, dtype=torch.float64).reshape(2, 4)

    actions = policy(observations)

    assert session.last_feed is not None
    assert session.last_feed["observations"].dtype == np.float32
    assert actions.dtype == torch.float32
    assert actions.device.type == "cpu"
    torch.testing.assert_close(
        actions, torch.tensor([[0.0, 2.0], [8.0, 10.0]], dtype=torch.float32)
    )
    assert policy.runtime == "onnxruntime:CPUExecutionProvider"
    assert policy.reset(torch.tensor([False, True])) is None


def test_onnx_source_policy_extracts_policy_observation_group() -> None:
    session = _Session()
    policy = OnnxSourcePolicy(session, device="cpu")

    actions = policy(
        {
            "critic": torch.full((2, 7), -1.0),
            "policy": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        }
    )

    torch.testing.assert_close(actions, torch.tensor([[0.0, 2.0], [8.0, 10.0]]))


def test_onnx_source_policy_requires_policy_observation_group() -> None:
    policy = OnnxSourcePolicy(_Session(), device="cpu")

    with pytest.raises(KeyError, match="'policy' observation group"):
        policy({"critic": torch.zeros((2, 4))})


def test_onnx_source_policy_rejects_incompatible_observations() -> None:
    policy = OnnxSourcePolicy(_Session(), device="cpu")
    with pytest.raises(ValueError, match="observation width"):
        policy(torch.zeros((2, 5)))


def test_onnx_source_policy_expands_a_fixed_single_item_batch() -> None:
    session = _Session(batch=1)
    policy = OnnxSourcePolicy(session, device="cpu")

    actions = policy(torch.arange(12, dtype=torch.float32).reshape(3, 4))

    assert [feed.shape for feed in session.feeds] == [(1, 4), (1, 4), (1, 4)]
    torch.testing.assert_close(
        actions,
        torch.tensor([[0.0, 2.0], [8.0, 10.0], [16.0, 18.0]]),
    )


def test_onnx_source_policy_requires_one_input() -> None:
    with pytest.raises(ValueError, match="exactly one input"):
        OnnxSourcePolicy(_MultiInputSession(), device="cpu")
