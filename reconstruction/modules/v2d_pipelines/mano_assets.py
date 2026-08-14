# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side MANO asset preparation for the ego reconstruction pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path


def prepare_hamer_mano_assets(weights_dir: str | Path, source_mano: str | Path) -> Path:
    """Migrate HaMeR to manotorch's models/ layout and stage MANO_RIGHT."""
    weights_dir = Path(weights_dir)
    config_path = weights_dir / "_DATA/hamer_ckpts/model_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"HaMeR model config not found at {config_path}. Run "
            "./scripts/download_ego_reconstruction_weights.sh first."
        )
    config = config_path.read_text()
    old_path = "MODEL_PATH: data/mano"
    new_path = "MODEL_PATH: data/models"
    if old_path in config:
        config_path.write_text(config.replace(old_path, new_path, 1))
    elif new_path not in config:
        raise RuntimeError(
            f"Unsupported HaMeR MANO MODEL_PATH in {config_path}; expected "
            f"{old_path!r} or {new_path!r}."
        )

    source_mano = Path(source_mano)
    if not source_mano.is_file():
        raise FileNotFoundError(
            f"MANO_RIGHT.pkl is required for --hand_tracking hamer. Expected "
            f"the licensed MANO asset at {source_mano}."
        )
    destination = weights_dir / "_DATA/data/models/MANO_RIGHT.pkl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_mano.resolve() != destination.resolve():
        shutil.copy2(source_mano, destination)
    return destination


def prepare_wilor_manotorch_mano(weights_dir: str | Path) -> Path:
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
