# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Path helpers for materialized per-dataset OSMO bundles.

Bundles are materialized under ``HUMAN_MOTION_DATA_DIR/{dataset}`` and parquet
asset references are stored relative to that bundle root, e.g.
``assets/meshes/taco/023_cm.obj``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PACKAGE_ROOT / "assets"
DEFAULT_HUMAN_MOTION_DATA_DIR = ASSETS_DIR / "human_motion_data"

ASSET_PREFIX = "assets"
MESHES_PREFIX = Path(ASSET_PREFIX) / "meshes"
URDFS_PREFIX = Path(ASSET_PREFIX) / "urdfs"


def get_human_motion_data_dir() -> Path:
    """Return the local root where dataset bundles are materialized."""
    return Path(
        os.environ.get("HUMAN_MOTION_DATA_DIR", str(DEFAULT_HUMAN_MOTION_DATA_DIR))
    ).expanduser()


def get_dataset_bundle_root(dataset: str) -> Path:
    """Return the expected materialized bundle root for a dataset."""
    return get_human_motion_data_dir() / dataset


def infer_bundle_root(path: str | Path) -> Path | None:
    """Infer a dataset bundle root from a path inside a materialized bundle."""
    current = Path(path)
    if current.suffix == ".parquet" or not current.is_dir():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "manifest.json").exists():
            return candidate
        name = candidate.name
        if (
            name.endswith("_loaded")
            or name.endswith("_processed")
            or name.endswith("_html")
            or name.endswith("_videos")
            or name in {"reconstructed_stage"}
        ):
            return candidate.parent
    return None


def is_bundle_relative(path: str | Path) -> bool:
    """Whether *path* is a relative path intended to be resolved in a bundle."""
    p = Path(path)
    return not p.is_absolute() and p.parts[:1] == (ASSET_PREFIX,)


def resolve_bundle_path(
    path: str | Path | None,
    *,
    bundle_root: str | Path | None = None,
    dataset: str | None = None,
) -> Path | None:
    """Resolve an absolute or bundle-relative path.

    Absolute paths are returned unchanged. Relative paths are resolved against
    ``bundle_root`` when available, otherwise against
    ``HUMAN_MOTION_DATA_DIR/{dataset}``.
    """
    if path is None or path == "":
        return None

    p = Path(path)
    if p.is_absolute():
        return p

    root: Path | None = Path(bundle_root) if bundle_root is not None else None
    if root is None and dataset is not None:
        root = get_dataset_bundle_root(dataset)
    if root is None:
        return p
    return root / p


def to_bundle_relative_path(
    path: str | Path | None,
    *,
    bundle_root: str | Path | None = None,
) -> str:
    """Convert a local asset path to a portable bundle-relative path.

    Paths already relative are preserved. Paths under the package's committed
    ``assets`` tree are rewritten below ``assets/`` so newly written parquets do
    not persist repo-absolute locations. Paths under ``bundle_root`` are also
    made relative to that root.
    """
    if path is None or path == "":
        return ""

    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()

    roots: list[Path] = []
    if bundle_root is not None:
        roots.append(Path(bundle_root))
    roots.append(ASSETS_DIR.parent)

    for root in roots:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(p)


def localize_mesh_path(
    path: str | Path | None,
    *,
    bundle_root: str | Path,
    dataset: str,
) -> str:
    """Return a bundle-relative mesh path, copying raw fallback meshes if needed."""
    if path is None or path == "":
        return ""

    p = Path(path)
    if not p.is_absolute() or not p.exists():
        return to_bundle_relative_path(p, bundle_root=bundle_root)

    # Committed package assets are already portable once made relative to
    # ``robotic_grounding/``.
    try:
        return p.relative_to(ASSETS_DIR.parent).as_posix()
    except ValueError:
        pass

    root = Path(bundle_root)
    try:
        rel = p.relative_to(root)
        if rel.parts[:1] == (ASSET_PREFIX,):
            return rel.as_posix()
    except ValueError:
        pass

    raw_suffix = _mesh_suffix_for_bundle(p, root)
    dest = root / MESHES_PREFIX / dataset / raw_suffix
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or p.stat().st_mtime > dest.stat().st_mtime:
        shutil.copy2(p, dest)
    return dest.relative_to(root).as_posix()


def _mesh_suffix_for_bundle(path: Path, bundle_root: Path) -> Path:
    """Choose a stable per-dataset mesh suffix for raw provider paths."""
    try:
        suffix = path.relative_to(bundle_root / "dataset")
    except ValueError:
        return Path(path.name)

    parts = suffix.parts
    if parts[:1] in {("object",), ("models",)} and len(parts) > 1:
        return Path(*parts[1:])
    if parts[:3] == ("tools", "object_meshes", "contact_meshes"):
        return Path(parts[-1])
    return suffix


def resolve_paths(
    paths: list[str],
    *,
    bundle_root: str | Path | None = None,
    dataset: str | None = None,
) -> list[str]:
    """Resolve a list of asset references and return string paths."""
    resolved: list[str] = []
    for path in paths:
        resolved_path = resolve_bundle_path(
            path, bundle_root=bundle_root, dataset=dataset
        )
        resolved.append(str(resolved_path) if resolved_path is not None else "")
    return resolved
