# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject arctic sequences with more than one reconstructed support surface.

Arctic objects are articulated (root + lid/lever) but should only produce a
single support disk — the root body's footprint.  Multiple disks indicate a
regression in ``scripts/reconstruct_support_surfaces.py`` (the ``top`` body
leaking through, or cross-body merge failing to collapse stacked footprints).

Non-arctic sequences skip this check vacuously.
"""

from __future__ import annotations

from pathlib import Path

MAX_SUPPORT_DISKS = 1
CYLINDER_TOKEN = "def Cylinder "


def _dataset_from_seq_dir(seq_dir: Path) -> str | None:
    """Infer dataset from the ``<dataset>_processed`` grandparent of ``seq_dir``.

    Expected layout: ``<root>/<dataset>_processed/sequence_id=<id>/robot_name=<robot>``.
    """
    try:
        processed = seq_dir.parent.parent
    except AttributeError:
        return None
    suffix = "_processed"
    return processed.name[: -len(suffix)] if processed.name.endswith(suffix) else None


def _support_usda(seq_dir: Path) -> Path | None:
    """Return the path to ``reconstructed_stage/<seq_id>_support.usda`` or None."""
    try:
        seq_id = seq_dir.parent.name.split("=", 1)[1]
        root = seq_dir.parent.parent.parent
    except IndexError:
        return None
    return root / "reconstructed_stage" / f"{seq_id}_support.usda"


def check(data: dict, seq_dir: Path | None = None) -> dict:
    """Pass unless an arctic sequence has ≥ 2 ``Cylinder`` prims in its support USD."""
    if seq_dir is None:
        return {"pass": True, "score": 0.0, "reason": "no seq_dir provided"}
    seq_dir = Path(seq_dir)
    dataset = _dataset_from_seq_dir(seq_dir)
    if dataset != "arctic":
        return {"pass": True, "score": 0.0, "reason": f"skipped (dataset={dataset})"}

    usda = _support_usda(seq_dir)
    if usda is None or not usda.exists():
        return {"pass": True, "score": 0.0, "reason": "no support USD"}

    try:
        n = usda.read_text().count(CYLINDER_TOKEN)
    except OSError as exc:
        return {
            "pass": True,
            "score": 0.0,
            "reason": f"could not read {usda.name}: {exc}",
        }

    return {
        "pass": n <= MAX_SUPPORT_DISKS,
        "score": float(n),
        "reason": f"support_disks={n} (max={MAX_SUPPORT_DISKS})",
    }
