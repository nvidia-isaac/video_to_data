"""Copy a validated reconstruction bundle into an E2E run."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from .config import validate_input_path


class BundleImportError(ValueError):
    """Raised when a reconstruction bundle cannot be imported safely."""


def _content_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def import_bundle(source: Path, destination: Path) -> str:
    """Validate and atomically copy ``source`` to ``destination``."""
    source = validate_input_path("reconstruction_bundle", source)
    destination = destination.expanduser().resolve()

    if destination.exists():
        validate_input_path("reconstruction_bundle", destination)
        if _content_fingerprint(source) == _content_fingerprint(destination):
            return "already-imported"
        raise BundleImportError(
            f"reconstruction destination already exists with different contents: "
            f"{destination}; use a new run root"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    temporary.rmdir()
    try:
        shutil.copytree(source, temporary)
        validate_input_path("reconstruction_bundle", temporary)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "imported"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = import_bundle(args.source, args.destination)
    print(f"Reconstruction bundle {result}: {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
