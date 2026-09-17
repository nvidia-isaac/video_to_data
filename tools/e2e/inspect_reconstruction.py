"""Validate and report a reconstruction bundle before retargeting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import RECONSTRUCTION_BUNDLE_FILES, validate_input_path


def inspect_bundle(bundle: Path) -> tuple[Path, ...]:
    """Return the validated required files in ``bundle``."""
    validated = validate_input_path("reconstruction_bundle", bundle)
    return tuple(validated / relative for relative in RECONSTRUCTION_BUNDLE_FILES)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a bundle and print its release-relevant artifacts."""
    args = build_parser().parse_args(argv)
    files = inspect_bundle(args.bundle)
    print(f"Reconstruction bundle OK: {args.bundle.resolve()}")
    for path in files:
        print(
            f"  {path.relative_to(args.bundle.resolve()).as_posix()}: {path.stat().st_size} bytes"
        )
    print(f"Three.js scene: {(args.bundle.resolve() / 'threejs_scene/index.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
