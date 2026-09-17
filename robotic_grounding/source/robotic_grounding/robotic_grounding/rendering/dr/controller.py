# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for composing task-owned Isaac Lab visual event configurations."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def iter_visual_event_terms(
    events_cfg: Any,
    prefix: str = "visual_",
) -> Iterator[tuple[str, object]]:
    """Yield configured visual event terms using a stable naming convention."""
    if events_cfg is None:
        return
    for name in dir(events_cfg):
        if not name.startswith(prefix):
            continue
        term = getattr(events_cfg, name)
        if term is not None:
            yield name, term


def disable_visual_event_terms(events_cfg: Any, prefix: str = "visual_") -> list[str]:
    """Disable all matching event terms and return the attributes changed."""
    changed = []
    for name, _term in list(iter_visual_event_terms(events_cfg, prefix)):
        setattr(events_cfg, name, None)
        changed.append(name)
    return changed


def set_visual_event_mode(
    events_cfg: Any,
    mode: str,
    prefix: str = "visual_",
    *,
    exclude_suffixes: tuple[str, ...] = (),
) -> list[str]:
    """Set ``mode`` on each matching term and return the attributes changed."""
    changed = []
    for name, term in iter_visual_event_terms(events_cfg, prefix):
        if name.endswith(exclude_suffixes):
            continue
        if hasattr(term, "mode"):
            term.mode = mode
            changed.append(name)
    return changed
