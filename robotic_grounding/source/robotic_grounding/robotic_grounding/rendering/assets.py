# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render-asset discovery without importing Isaac Sim."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_asset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "assets"


@dataclass(frozen=True)
class RenderAssetRegistry:
    """Resolve external rendering assets from an injected or conventional root.

    ``V2D_RENDER_ASSET_ROOT`` allows another task, image, or checkout to own the large
    asset bundles while reusing this source module.
    """

    root: Path | str | None = None

    def __post_init__(self) -> None:
        """Resolve the explicit, environment-provided, or conventional root."""
        configured = self.root or os.environ.get("V2D_RENDER_ASSET_ROOT")
        root = Path(configured) if configured else _default_asset_root()
        object.__setattr__(self, "root", root.expanduser().resolve())

    @property
    def vmaterials_mdl_root(self) -> Path:
        """Return the NVIDIA vMaterials MDL module root."""
        return self.resolve("vMaterials_2/opt/nvidia/mdl")

    @property
    def texture_root(self) -> Path:
        """Return the CC0 rendering texture-library root."""
        return self.resolve("textures")

    def resolve(self, relative_path: str | Path, *, required: bool = False) -> Path:
        """Resolve a path beneath the registry root and optionally require it to exist."""
        root = Path(self.root) if self.root is not None else _default_asset_root()
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Render asset path escapes registry root: {relative_path}"
            ) from exc
        if required and not path.exists():
            raise FileNotFoundError(path)
        return path

    def missing(self, relative_paths: list[str] | tuple[str, ...]) -> list[str]:
        """Return required relative paths absent from this registry."""
        return [path for path in relative_paths if not self.resolve(path).exists()]
