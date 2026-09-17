# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declarative embodiment profile mapping a sim ``record`` obs group to GR00T modalities.

Describes how a sim env's ``record`` observation group maps to GR00T state/video/language
modality keys. This is the *per-embodiment* seam of the closed-loop pipeline. The generic adapter
(:mod:`action_adapter`) consumes an :class:`EmbodimentProfile` to build the GR00T
observation dict; action reassembly is server-driven (from ``get_modality_config()``) and
lives in the adapter, not here. Adding a new robot = write a new profile (see
``profiles/sharpa_dual_hand_three_camera.py``) — no change to the adapter, transport,
or runner.

Pure numpy + stdlib; **no IsaacLab / gr00t import** (unit-testable with fake arrays).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Fixed wrist references (wxyz) select one continuous quaternion hemisphere. The
# converter and closed-loop adapter must apply the same convention.
Q_REF_RIGHT_WRIST = np.array([0.0788, -0.0713, -0.7000, -0.7062], dtype=np.float32)
Q_REF_LEFT_WRIST = np.array([0.6564, -0.7273, -0.0419, -0.1960], dtype=np.float32)


def align_quat_to_ref(arr: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Sign-align wxyz quaternions of shape ``(..., 4)`` to a reference quaternion.

    Flips ``q -> -q`` wherever ``dot(q, q_ref) < 0`` so all quats live on the same sheet
    of the double cover as ``q_ref``. Must match the convention used when the training
    dataset was built (``groot_finetune/convert_to_gr00t.py``) so the state distribution
    the policy sees at inference matches training. Operates on the last axis; returns a
    new array.
    """
    arr = np.asarray(arr, dtype=np.float32).copy()
    flip = arr @ np.asarray(q_ref, dtype=np.float32) < 0.0
    arr[flip] *= -1.0
    return arr


@dataclass(frozen=True)
class StateFieldSpec:
    """One GR00T ``state`` modality key sourced from a slice of a ``record`` obs term.

    Args:
        key: the GR00T state modality key (must match the training modality config).
        source_term: the ``record`` obs term name to slice (e.g. ``"wrist_position_e"``).
        start, end: column slice ``[start:end]`` of that term (last axis).
        transform: optional per-field transform applied after slicing.
    """

    key: str
    source_term: str
    start: int
    end: int
    transform: Callable[[np.ndarray], np.ndarray] | None = None


@dataclass(frozen=True)
class VideoFieldSpec:
    """One GR00T ``video`` modality key sourced from a ``record`` image term."""

    key: str
    source_term: str


@dataclass(frozen=True)
class EmbodimentProfile:
    """Maps a sim env's ``record`` obs group to the GR00T modality keys for one robot.

    Args:
        name: contract identifier used to resolve this profile.
        state_fields: ordered state modality fields.
        action_fields: ordered action modality fields.
        video_fields: video modality fields.
        language_key: the language modality key (e.g. ``annotation.human.task_description``).
    """

    name: str
    state_fields: tuple[StateFieldSpec, ...]
    action_fields: tuple[StateFieldSpec, ...]
    video_fields: tuple[VideoFieldSpec, ...]
    language_key: str

    @property
    def state_keys(self) -> list[str]:
        """GR00T ``state.*`` modality keys, in wire order."""
        return [f.key for f in self.state_fields]

    @property
    def video_keys(self) -> list[str]:
        """GR00T ``video.*`` modality keys, in wire order."""
        return [f.key for f in self.video_fields]


# --- profile registry (populated by importing profile modules) ---
_REGISTRY: dict[str, EmbodimentProfile] = {}


def register_profile(profile: EmbodimentProfile) -> EmbodimentProfile:
    """Register a profile by name (idempotent for the same object)."""
    _REGISTRY[profile.name] = profile
    return profile


def get_profile(name: str) -> EmbodimentProfile:
    """Look up a registered profile, importing the built-in profiles on first use."""
    if not _REGISTRY:
        # Lazy import so the registry is populated without an explicit import in callers.
        # Must stay inside the function: profiles/ imports this module at import time.
        from . import profiles  # noqa: F401, PLC0415
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown embodiment profile {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]
