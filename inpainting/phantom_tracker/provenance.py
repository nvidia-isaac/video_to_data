"""Small provenance helpers shared by the Phantom host and container paths."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> tuple[str, list[dict[str, str | int]]]:
    """Hash a tree from sorted relative paths, byte sizes, and file hashes."""

    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(root)
    digest = hashlib.sha256()
    entries: list[dict[str, str | int]] = []
    for candidate in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = candidate.relative_to(root).as_posix()
        file_hash = sha256_file(candidate)
        size = candidate.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        entries.append({"path": relative, "bytes": size, "sha256": file_hash})
    return digest.hexdigest(), entries
