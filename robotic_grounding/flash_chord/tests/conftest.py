# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures and optional-dependency gating.

Tests are tiered: pure-function and CPU-device Warp-kernel tests run anywhere;
tests marked ``@pytest.mark.gpu`` need a CUDA device and are skipped when absent.
Tests marked ``@pytest.mark.sequence_data`` need the optional reference corpora.
Warp is imported lazily so the suite collects even where Warp is not installed.
"""

import pytest

from flash_chord.assets import ASSETS_DIR


def _cuda_available() -> bool:
    try:
        import warp as wp

        wp.init()
        return bool(wp.is_cuda_available())
    except Exception:  # noqa: BLE001  # an unavailable/misconfigured Warp install means no CUDA tests
        return False


CUDA_AVAILABLE = _cuda_available()
SEQUENCE_DATA_AVAILABLE = (ASSETS_DIR / "human_motion_data").is_dir()


def pytest_collection_modifyitems(config, items):
    """Skip tests whose explicitly marked runtime prerequisites are absent."""
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    skip_sequences = pytest.mark.skip(reason="optional local reference sequences are not installed")
    for item in items:
        if not CUDA_AVAILABLE and "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if not SEQUENCE_DATA_AVAILABLE and "sequence_data" in item.keywords:
            item.add_marker(skip_sequences)
