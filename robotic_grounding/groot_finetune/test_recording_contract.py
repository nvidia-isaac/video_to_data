# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for embodiment-neutral recording contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from groot_finetune.recording_contract import (
    contract_from_env_cfg,
    joint_reorder_indices,
)


def test_recording_contract_must_be_explicit() -> None:
    with pytest.raises(ValueError, match="explicit GR00T recording contract"):
        contract_from_env_cfg(SimpleNamespace())


def test_vega_contract_is_explicit() -> None:
    cfg = SimpleNamespace(
        gr00t_record_action_terms=("joint_pos",),
        gr00t_record_action_joint_order=("R", "RF", "L", "LF"),
        gr00t_record_camera_terms=(("image", "ego"),),
    )
    contract = contract_from_env_cfg(cfg)
    assert contract.action_terms == ("joint_pos",)
    assert contract.action_joint_order == ("R", "RF", "L", "LF")
    assert contract.camera_terms == (("image", "ego"),)


def test_joint_reorder_uses_names_not_position() -> None:
    assert joint_reorder_indices(["L", "LF", "R", "RF"], ("R", "RF", "L", "LF")) == [
        2,
        3,
        0,
        1,
    ]


def test_joint_reorder_rejects_cross_embodiment_layout() -> None:
    with pytest.raises(ValueError, match="missing=.*RF.*extra=.*wrist"):
        joint_reorder_indices(["R", "wrist", "L", "LF"], ("R", "RF", "L", "LF"))
