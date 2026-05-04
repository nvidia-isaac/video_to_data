# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""One-time downloader for SOMA-X body-model assets.

Drives ``soma.SOMALayer`` once with ``data_root=None`` so SOMA-X downloads
the asset bundle into its built-in HuggingFace cache, then copies the
specific files that ``robotic_grounding.retarget.read_soma`` requires
into the canonical repo path (``assets/body_models/soma/`` by default).

We cannot pass ``--data-root`` straight through to ``SOMALayer``: SOMA-X
treats an existing-but-incomplete directory as "assets are supposed to
be here already" and crashes with ``FileNotFoundError`` for
``SOMA_neutral.npz`` instead of falling back to HuggingFace. The HF
cache path is also ephemeral inside the retarget Docker container
(``HOME=/tmp`` is tmpfs), so the bind-mounted repo path is the only
location that survives container restarts.

The required asset list is the source of truth in
``robotic_grounding.retarget.read_soma._SOMA_REQUIRED_ASSETS`` /
``_SOMA_REQUIRED_BY_IDENTITY``; this script imports those constants via
``_missing_assets`` so it stays in sync when the manifest changes.

Usage:
    python scripts/setup_soma_assets.py
    python scripts/setup_soma_assets.py --identity-model-type mhr --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from robotic_grounding.retarget import BODY_MODELS_DIR
from robotic_grounding.retarget.read_soma import (
    _SOMA_REQUIRED_ASSETS,
    _SOMA_REQUIRED_BY_IDENTITY,
    _missing_assets,
)

DEFAULT_ROOT = BODY_MODELS_DIR / "soma"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Populate assets/body_models/soma/ with the SOMA-X assets needed by scripts/retarget/soma_to_g1.py."
        )
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Destination dir. Defaults to the canonical repo location "
            f"({DEFAULT_ROOT}) so read_soma picks it up without needing "
            "--soma-data-root at retarget time."
        ),
    )
    p.add_argument(
        "--identity-model-type",
        default="mhr",
        choices=("mhr",),
        help=(
            "SOMA identity model variant. Only 'mhr' is exercised by the "
            "current retarget scripts; passed straight through to SOMALayer."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Wipe the destination directory before downloading. Use when "
            "SOMA-X publishes new asset versions and you want to refresh "
            "the on-disk cache. Does NOT clear the HuggingFace cache "
            "itself; that is owned by SOMA-X / huggingface_hub."
        ),
    )
    return p.parse_args()


def _required_relpaths(identity_model_type: str) -> list[str]:
    """Files that ``read_soma._missing_assets`` will check for."""
    return list(_SOMA_REQUIRED_ASSETS) + list(
        _SOMA_REQUIRED_BY_IDENTITY.get(identity_model_type.lower(), ())
    )


def _copy_required_assets(
    src_root: Path, dst_root: Path, identity_model_type: str
) -> list[str]:
    """Copy the required SOMA assets from ``src_root`` to ``dst_root``.

    Returns the list of files that could not be located under ``src_root``
    (typically zero; non-empty means the SOMA-X HF download did not match
    the file layout this repo expects).
    """
    not_found: list[str] = []
    for relpath in _required_relpaths(identity_model_type):
        src = src_root / relpath
        if not src.is_file():
            not_found.append(relpath)
            continue
        dst = dst_root / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        # ``copy2`` follows symlinks so the destination becomes a real
        # file (the HF cache stores blobs and exposes them via symlinks).
        # This matters because the bind-mounted repo path may outlive
        # the container's HF cache.
        shutil.copy2(src, dst)
        print(f"  copied {relpath}  ({src.stat().st_size / (1024 * 1024):.1f} MB)")
    return not_found


def main() -> int:
    """Download SOMA-X assets and stage them into ``--data-root``."""
    args = parse_args()
    dst_root: Path = args.data_root.expanduser().resolve()

    if args.force and dst_root.is_dir():
        print(f"[setup_soma_assets] --force: removing {dst_root}")
        shutil.rmtree(dst_root)

    if dst_root.is_dir():
        missing = _missing_assets(dst_root, args.identity_model_type)
        if not missing:
            print(
                f"[setup_soma_assets] {dst_root} already populated with "
                f"identity_model_type={args.identity_model_type!r}; nothing to do."
            )
            return 0
        print(
            f"[setup_soma_assets] {dst_root} exists but is missing: {missing}; downloading."
        )
    else:
        print(f"[setup_soma_assets] {dst_root} does not exist yet; will create.")

    try:
        from soma import SOMALayer  # noqa: PLC0415
    except ImportError as exc:
        print(
            "ERROR: py-soma-x is not installed. Install via "
            "`pip install py-soma-x` (already baked into the retarget "
            "Docker image at workflow/Dockerfile).",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    # Pass ``data_root=None`` so SOMA-X downloads into its HuggingFace
    # cache (``get_assets_dir()``). Passing the destination directly
    # crashes when the directory exists but is empty -- SOMA-X assumes
    # the assets are present and skips its own HF fallback.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        "[setup_soma_assets] running SOMALayer(data_root=None) to trigger "
        f"HuggingFace download (identity_model_type={args.identity_model_type}, "
        f"device={device})."
    )
    layer = SOMALayer(
        data_root=None,
        identity_model_type=args.identity_model_type,
        device=device,
    )
    src_root = Path(layer.data_root).resolve()
    print(f"[setup_soma_assets] SOMA-X downloaded to: {src_root}")

    if src_root == dst_root:
        # Unlikely (HF cache lives under HOME, not the repo) but handled
        # for completeness so we do not no-op silently when the source
        # and destination coincide.
        print(
            "[setup_soma_assets] SOMA-X resolved to the destination directly; no copy needed."
        )
    else:
        print(f"[setup_soma_assets] copying required assets -> {dst_root}")
        dst_root.mkdir(parents=True, exist_ok=True)
        not_found = _copy_required_assets(src_root, dst_root, args.identity_model_type)
        if not_found:
            print(
                f"ERROR: the SOMA-X snapshot at {src_root} is missing "
                f"these expected files: {not_found}. The bundle layout "
                "may have changed; update _SOMA_REQUIRED_BY_IDENTITY in "
                "read_soma.py to match the current SOMA-X release.",
                file=sys.stderr,
            )
            return 1

    missing_after = _missing_assets(dst_root, args.identity_model_type)
    if missing_after:
        print(
            f"ERROR: {dst_root} is still missing required assets after "
            f"copy: {missing_after}. Inspect {src_root} to see what was "
            "actually downloaded.",
            file=sys.stderr,
        )
        return 1

    print(f"[setup_soma_assets] done. {dst_root} now satisfies read_soma.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
