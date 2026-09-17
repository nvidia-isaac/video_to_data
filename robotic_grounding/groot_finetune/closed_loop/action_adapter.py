# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic closed-loop action adapter: sim ``record`` obs -> GR00T obs -> flat action.

Embodiment-agnostic. Given a :class:`~.policy_client.PolicyClient` (or any object with
``get_action`` / ``get_modality_config``) and an :class:`~.embodiment.EmbodimentProfile`,
it (1) builds the nested ``{video, state, language}`` GR00T observation from a step's
``record`` obs group, (2) queries the server for a T-step action chunk, (3) reassembles
the per-key chunks into the flat env action vector in the **server-reported** action-key
order, and (4) serves it one step at a time with a receding-horizon buffer.

Pure numpy — **no IsaacLab / gr00t / torch import** (torch tensors are duck-typed to numpy
so it stays unit-testable with a mock client and fake numpy arrays).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .embodiment import EmbodimentProfile


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a torch tensor (duck-typed) or array-like to a numpy array (no torch import)."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


class Gr00tActionAdapter:
    """Translate ``record`` obs to GR00T and serve flat actions with a receding horizon.

    Args:
        client: a ``PolicyClient``-like object (``get_action``, ``get_modality_config``).
        profile: the embodiment obs-mapping profile.
        execution_length: number of actions to execute from each predicted chunk before
            re-querying the policy (the receding-horizon "open-loop horizon"). The server
            predicts ``action_chunk_size`` (e.g. 16) actions per query but we need not
            execute them all; ``1 <= execution_length <= action_chunk_size``.
        task: non-empty language instruction from the task profile.
    """

    def __init__(
        self,
        client: Any,
        profile: EmbodimentProfile,
        execution_length: int = 8,
        task: str | None = None,
    ) -> None:
        """Bind a policy client to an embodiment profile and size the action buffer."""
        self.client = client
        self.profile = profile
        if task is None or not task.strip():
            raise ValueError("a non-empty task-profile instruction is required")
        self.task = task

        mc = client.get_modality_config()
        self._mc = mc
        # Server-driven action layout: reassembly order + chunk size come from the server.
        self.action_keys: list[str] = list(_cfg(mc, "action")["modality_keys"])
        action_delta_indices = list(_cfg(mc, "action")["delta_indices"])
        self.action_chunk_size = len(action_delta_indices)
        if action_delta_indices != list(range(self.action_chunk_size)):
            raise ValueError(
                f"action delta indices must be consecutive from zero: {action_delta_indices}"
            )
        for section in ("video", "state", "language"):
            delta_indices = list(_cfg(mc, section)["delta_indices"])
            if delta_indices != [0]:
                raise ValueError(
                    f"{section} delta indices must be exactly [0], got {delta_indices}"
                )

        if not (1 <= execution_length <= self.action_chunk_size):
            raise ValueError(
                f"execution_length={execution_length} must satisfy "
                f"1 <= n <= action_chunk_size={self.action_chunk_size}."
            )
        self.execution_length = execution_length

        self._validate_keys()

        self._chunk: np.ndarray | None = None  # (B, action_chunk_size, action_dim)
        self._cursor: int = 0

    def _validate_keys(self) -> None:
        """Assert the profile and served checkpoint expose one exact key contract."""
        server_state = list(_cfg(self._mc, "state")["modality_keys"])
        if server_state != self.profile.state_keys:
            raise KeyError(
                "Profile state keys/order do not match the server config: "
                f"profile={self.profile.state_keys}, server={server_state}."
            )
        server_video = list(_cfg(self._mc, "video")["modality_keys"])
        if server_video != self.profile.video_keys:
            raise KeyError(
                "Profile video keys/order do not match the server config: "
                f"profile={self.profile.video_keys}, server={server_video}."
            )
        expected_action_keys = [field.key for field in self.profile.action_fields]
        if self.action_keys != expected_action_keys:
            raise KeyError(
                "Profile action keys/order do not match the server config: "
                f"profile={expected_action_keys}, server={self.action_keys}."
            )
        server_lang = set(_cfg(self._mc, "language")["modality_keys"])
        if self.profile.language_key not in server_lang:
            raise KeyError(
                f"Profile language key {self.profile.language_key!r} not in server config {sorted(server_lang)}."
            )

    def build_observation(self, record_obs: Any) -> dict[str, dict]:
        """Build the nested GR00T obs dict from a step's ``record`` obs group.

        ``record_obs`` maps obs-term name -> array ``(B, ...)`` (torch or numpy).
        Returns ``{"video": {...}, "state": {...}, "language": {...}}`` with a leading
        batch dim and a time dim (``T`` from the model's delta_indices).
        """
        required_terms = {
            *(field.source_term for field in self.profile.video_fields),
            *(field.source_term for field in self.profile.state_fields),
        }
        if not hasattr(record_obs, "keys"):
            raise TypeError("record observations must provide a keys() method")
        # TensorDict iteration walks its batch entries rather than its field names.
        # Query keys explicitly so both plain mappings and Isaac Lab TensorDicts use
        # the same observation boundary contract.
        missing = sorted(required_terms - set(record_obs.keys()))
        if missing:
            raise KeyError(f"record observations are missing required terms: {missing}")

        # Batch size from the first video source term.
        first_video_term = self.profile.video_fields[0].source_term
        batch = _to_numpy(record_obs[first_video_term]).shape[0]

        video: dict[str, np.ndarray] = {}
        for vf in self.profile.video_fields:
            arr = _to_numpy(record_obs[vf.source_term])
            if arr.ndim != 4 or arr.shape[0] != batch or arr.shape[-1] != 3:
                raise ValueError(
                    f"record camera {vf.source_term!r} must be (B, H, W, 3) with B={batch}, got {arr.shape}"
                )
            if arr.dtype != np.uint8:
                raise ValueError(
                    f"record camera {vf.source_term!r} must be uint8, got {arr.dtype}"
                )
            video[vf.key] = np.ascontiguousarray(arr[:, None, ...])

        state: dict[str, np.ndarray] = {}
        for sf in self.profile.state_fields:
            term = _to_numpy(record_obs[sf.source_term])
            if term.ndim != 2 or term.shape[0] != batch or term.shape[1] < sf.end:
                raise ValueError(
                    f"record state {sf.source_term!r} must be a 2-D batch with "
                    f"at least {sf.end} columns, got {term.shape}"
                )
            if not np.issubdtype(term.dtype, np.number) or not np.isfinite(term).all():
                raise ValueError(
                    f"record state {sf.source_term!r} must contain finite numeric values"
                )
            term = term.astype(np.float32)
            sl = term[:, sf.start : sf.end]
            if sf.transform is not None:
                sl = sf.transform(sl)
            state[sf.key] = np.ascontiguousarray(sl[:, None, :].astype(np.float32))

        language = {self.profile.language_key: [[self.task] for _ in range(batch)]}
        return {"video": video, "state": state, "language": language}

    def _reassemble(self, action_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate per-key action chunks in server order -> ``(B, chunk, action_dim)``."""
        if set(action_dict) != set(self.action_keys):
            raise KeyError(
                "policy action keys do not match the served modality config: "
                f"expected={self.action_keys}, received={sorted(action_dict)}"
            )
        parts = []
        batch: int | None = None
        field_by_key = {field.key: field for field in self.profile.action_fields}
        for key in self.action_keys:
            part = np.asarray(action_dict[key])
            expected_width = field_by_key[key].end - field_by_key[key].start
            if (
                part.ndim != 3
                or part.shape[1] != self.action_chunk_size
                or part.shape[2] != expected_width
                or (batch is not None and part.shape[0] != batch)
            ):
                raise ValueError(
                    f"policy action {key!r} must be (B, {self.action_chunk_size}, "
                    f"{expected_width}) with one common B, got {part.shape}"
                )
            if not np.issubdtype(part.dtype, np.number) or not np.isfinite(part).all():
                raise ValueError(
                    f"policy action {key!r} must contain finite numeric values"
                )
            batch = int(part.shape[0])
            parts.append(np.asarray(part, dtype=np.float32))
        return np.concatenate(parts, axis=-1)

    def act(self, record_obs: Any, dones: Any = None) -> np.ndarray:
        """Return the flat action ``(B, action_dim)`` (numpy) for this step.

        Re-queries the server when ``execution_length`` actions have been executed from
        the current chunk, or any env reset (``dones`` any True) — a done env's buffered
        chunk is stale, so we re-plan the whole batch from the current observation.
        """
        need_requery = self._chunk is None or self._cursor >= self.execution_length
        if dones is not None:
            dones_np = _to_numpy(dones).astype(bool)
            if dones_np.any():
                need_requery = True

        if need_requery:
            obs = self.build_observation(record_obs)
            action_dict, _info = self.client.get_action(obs)
            candidate_chunk = self._reassemble(action_dict)
            expected_batch = next(iter(obs["video"].values())).shape[0]
            if candidate_chunk.shape[0] != expected_batch:
                raise ValueError(
                    "policy action batch does not match the observation batch: "
                    f"actions={candidate_chunk.shape[0]}, observations={expected_batch}"
                )
            self._chunk = candidate_chunk
            self._cursor = 0

        chunk = self._chunk
        if chunk is None:  # unreachable: need_requery is True whenever _chunk is None
            raise RuntimeError("action chunk unavailable; call act() after a requery")
        action = chunk[:, self._cursor, :]  # (B, action_dim)
        self._cursor += 1
        return np.ascontiguousarray(action.astype(np.float32))

    def reset(self) -> None:
        """Drop the buffered chunk (forces a re-query on the next ``act``)."""
        self._chunk = None
        self._cursor = 0


def _cfg(mc: dict, name: str) -> dict:
    """Fetch a modality section (``video``/``state``/``action``/``language``) as a dict.

    The vendored ``PolicyClient`` returns plain dicts; tolerate objects with attributes
    too (so a real gr00t ``ModalityConfig`` also works in tests).
    """
    if name not in mc:
        raise KeyError(f"Modality config missing {name!r} section; keys={sorted(mc)}")
    sec = mc[name]
    if isinstance(sec, dict):
        return sec
    return {
        "modality_keys": list(sec.modality_keys),
        "delta_indices": list(sec.delta_indices),
    }
