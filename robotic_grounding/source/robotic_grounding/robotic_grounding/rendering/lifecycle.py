# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderer lifecycle helpers.

Call :func:`prepare_mdl_environment` before ``AppLauncher``. Call
:func:`register_mdl_search_paths` immediately after the application starts and before
materials or the stage are created.
"""

# ``carb`` is intentionally imported only after AppLauncher.
# ruff: noqa: PLC0415

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping

from .assets import RenderAssetRegistry

MDL_ENVIRONMENT_KEYS = ("MDL_PATHS", "MDL_USER_PATH", "MDL_SYSTEM_PATH")
MDL_SETTING_KEYS = (
    "/renderer/mdl/searchPaths/custom",
    "/renderer/mdl/searchPaths/templates",
)


def _prepend_unique(value: str, current: str, separator: str) -> str:
    parts = [part for part in current.split(separator) if part]
    return separator.join([value, *[part for part in parts if part != value]])


def prepare_mdl_environment(
    registry: RenderAssetRegistry | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
    required: bool = False,
) -> Path | None:
    """Prepend the vMaterials MDL root to process environment search paths.

    This function is host-safe and idempotent. It returns ``None`` when the optional
    bundle is absent, or raises when ``required=True``.
    """
    registry = registry or RenderAssetRegistry()
    root = registry.vmaterials_mdl_root
    if not root.is_dir():
        if required:
            raise FileNotFoundError(root)
        return None
    target = environ if environ is not None else os.environ
    root_text = str(root)
    for key in MDL_ENVIRONMENT_KEYS:
        target[key] = _prepend_unique(root_text, target.get(key, ""), os.pathsep)
    return root


def register_mdl_search_paths(
    registry: RenderAssetRegistry | None = None,
    *,
    required: bool = False,
) -> Path | None:
    """Register the MDL root with RTX after ``AppLauncher`` starts.

    ``carb`` is imported only when the function is called inside an Isaac Sim process.
    """
    root = prepare_mdl_environment(registry, required=required)
    if root is None:
        return None

    import carb

    settings = carb.settings.get_settings()
    root_text = str(root)
    for key in MDL_SETTING_KEYS:
        current = settings.get(key) or ""
        if isinstance(current, str):
            settings.set(key, _prepend_unique(root_text, current, ";"))
        else:
            values = list(current)
            settings.set(
                key, [root_text, *[value for value in values if value != root_text]]
            )
    return root
