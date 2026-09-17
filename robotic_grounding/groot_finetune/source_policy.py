# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference-only source-policy adapters used during rollout collection."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


class OnnxSourcePolicy:
    """Run a single-input, single-action-output ONNX expert as a Torch callable."""

    def __init__(self, session: Any, *, device: str | torch.device) -> None:
        """Validate the session contract and select its observation/action tensors."""
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(
                f"source-policy ONNX must have exactly one input; got {len(inputs)}"
            )
        if not outputs:
            raise ValueError("source-policy ONNX must have at least one output")

        action_outputs = [output for output in outputs if output.name == "actions"]
        output = action_outputs[0] if action_outputs else outputs[0]
        if getattr(inputs[0], "type", "tensor(float)") != "tensor(float)":
            raise ValueError(
                "source-policy ONNX input must be float32; "
                f"got {getattr(inputs[0], 'type', None)!r}"
            )

        self._session = session
        self._device = torch.device(device)
        self._input_name = inputs[0].name
        self._input_shape = tuple(getattr(inputs[0], "shape", ()))
        self._output_name = output.name

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, *, device: str | torch.device
    ) -> "OnnxSourcePolicy":
        """Create an ONNX Runtime session using the simulator device when available."""
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:  # pragma: no cover - exercised in the IsaacLab image
            raise RuntimeError(
                "ONNX source experts require onnxruntime; rebuild the robotic_grounding image"
            ) from exc

        torch_device = torch.device(device)
        available = set(ort.get_available_providers())
        providers: list[str] = []
        if torch_device.type == "cuda" and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        if not providers:
            raise RuntimeError(
                "ONNX Runtime has no supported execution provider; "
                f"available={sorted(available)}"
            )

        session = ort.InferenceSession(str(checkpoint), providers=providers)
        return cls(session, device=torch_device)

    @property
    def runtime(self) -> str:
        """Return the active provider for rollout provenance."""
        return f"onnxruntime:{self._session.get_providers()[0]}"

    def __call__(self, observations: torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
        """Evaluate a simulator observation batch and return actions on its device."""
        if isinstance(observations, Mapping):
            if "policy" not in observations:
                raise KeyError(
                    "ONNX source policy requires the RSL-RL 'policy' observation group"
                )
            observations = observations["policy"]
        if not isinstance(observations, torch.Tensor):
            raise TypeError(
                "ONNX source policy expects the RSL-RL observation tensor; "
                f"got {type(observations).__name__}"
            )
        if observations.ndim != 2:
            raise ValueError(
                f"ONNX source policy expects [batch, features]; got {tuple(observations.shape)}"
            )
        if self._input_shape and isinstance(self._input_shape[-1], int):
            expected = self._input_shape[-1]
            if observations.shape[-1] != expected:
                raise ValueError(
                    "ONNX observation width does not match the exported expert: "
                    f"{observations.shape[-1]} != {expected}"
                )

        observation_array = (
            observations.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
        )
        fixed_batch = (
            self._input_shape[0]
            if self._input_shape and isinstance(self._input_shape[0], int)
            else None
        )
        if fixed_batch is not None and fixed_batch != observation_array.shape[0]:
            if fixed_batch != 1:
                raise ValueError(
                    "ONNX source policy has an unsupported fixed batch size: "
                    f"model={fixed_batch}, observations={observation_array.shape[0]}"
                )
            results = [
                self._session.run(
                    [self._output_name],
                    {self._input_name: observation_array[index : index + 1]},
                )[0]
                for index in range(observation_array.shape[0])
            ]
            action_array = np.concatenate(results, axis=0).astype(
                np.float32, copy=False
            )
        else:
            result = self._session.run(
                [self._output_name], {self._input_name: observation_array}
            )[0]
            action_array = np.asarray(result, dtype=np.float32)
        if action_array.ndim != 2 or action_array.shape[0] != observations.shape[0]:
            raise ValueError(
                "ONNX source policy returned an invalid action batch: "
                f"{action_array.shape} for observations {tuple(observations.shape)}"
            )
        return torch.from_numpy(action_array.copy()).to(self._device)

    def reset(self, _dones: torch.Tensor) -> None:
        """Match the RSL-RL policy interface; the exported expert is feed-forward."""
