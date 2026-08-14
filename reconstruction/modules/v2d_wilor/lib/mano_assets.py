# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility layout for WiLoR and manotorch MANO consumers."""

from __future__ import annotations

import shutil
from pathlib import Path


def prepare_manotorch_mano(weights_dir: str | Path) -> Path:
    """Copy WiLoR's root MANO asset into manotorch's models/ layout."""
    pretrained = Path(weights_dir) / "pretrained_models"
    source = pretrained / "MANO_RIGHT.pkl"
    if not source.is_file():
        raise FileNotFoundError(
            f"WiLoR MANO asset not found at {source}. Run "
            "./scripts/download_ego_reconstruction_weights.sh first."
        )
    destination = pretrained / "models/MANO_RIGHT.pkl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination
