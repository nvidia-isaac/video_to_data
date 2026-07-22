# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference-only checkpoint loader needed by E2FGVI's SPyNet constructor."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_checkpoint(
    model: Any,
    filename: str,
    map_location: str = "cpu",
    strict: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """Skip E2FGVI's constructor-time URL fetch or load a local state dict.

    The complete E2FGVI checkpoint is loaded strictly immediately after model
    construction, including the SPyNet parameters. Avoiding the upstream URL
    here makes container inference offline and reproducible.
    """
    if filename.startswith(("http://", "https://")):
        return {}

    import torch

    path = Path(filename)
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=strict)
    return checkpoint
