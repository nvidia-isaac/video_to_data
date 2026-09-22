# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the reproducible SOMA body-model asset contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from robotic_grounding.retarget.read_soma import (
    _SOMA_NEUTRAL_RIG_KEYS,
    _resolve_data_root,
)


def _write_minimal_asset_bundle(root: Path) -> None:
    root.mkdir(parents=True)
    np.savez(
        root / "SOMA_neutral.npz",
        **{key: np.empty(0) for key in _SOMA_NEUTRAL_RIG_KEYS},
    )
    (root / "correctives_model.pt").touch()
    for relative in (
        "MHR/mhr_model_lod1.pt",
        "MHR/base_body_lod1.obj",
        "MHR/SOMA_wrap_lod1.obj",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_resolve_data_root_accepts_complete_explicit_bundle(tmp_path: Path) -> None:
    root = tmp_path / "soma"
    _write_minimal_asset_bundle(root)

    assert _resolve_data_root(root, "mhr") == root


def test_resolve_data_root_rejects_mutable_upstream_fallback(tmp_path: Path) -> None:
    root = tmp_path / "missing-soma"

    with pytest.raises(FileNotFoundError, match="setup_soma_assets.py") as exc_info:
        _resolve_data_root(root, "mhr")

    assert "mutable SOMA-X default snapshot is intentionally not used" in str(
        exc_info.value
    )
