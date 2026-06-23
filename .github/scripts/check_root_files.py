#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Enforce that root-level shared files don't carry package-specific rules.

The monorepo intentionally keeps two subprojects (video_ingestion_agent,
reconstruction) self-contained. Anything anchored to a
specific subproject must live inside that subproject's own version of the
shared file (.gitignore, .gitattributes, .pre-commit-config.yaml, .envrc),
not at the repo root. This check enforces that rule on every PR.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGES = ("video_ingestion_agent", "reconstruction")
REPO_ROOT = Path(__file__).resolve().parents[2]

# Files that must not exist at repo root because their content is
# package-specific. Move them under the relevant subproject.
FORBIDDEN_AT_ROOT = (".pre-commit-config.yaml", ".envrc")


def _iter_pattern_lines(path: Path):
    """Yield (lineno, raw_line, pattern_token) for non-comment, non-blank lines.

    gitignore patterns may be negated (leading '!') and/or anchored (leading
    '/'); gitattributes lines are '<pattern> <attr>...' where the pattern is
    the first whitespace-separated token. Both are normalized to a bare
    pattern token here.
    """
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        token = stripped.split()[0].lstrip("!").lstrip("/")
        yield lineno, stripped, token


def _check_path_patterns(filename: str) -> list[str]:
    path = REPO_ROOT / filename
    if not path.exists():
        return []
    errors: list[str] = []
    for lineno, raw_line, token in _iter_pattern_lines(path):
        for pkg in PACKAGES:
            if token == pkg or token.startswith(pkg + "/"):
                errors.append(
                    f"{filename}:{lineno}: '{raw_line}' is anchored to "
                    f"'{pkg}/'. Move it into {pkg}/{filename} and drop the "
                    f"'{pkg}/' prefix (the pattern is then relative to that "
                    f"file's location)."
                )
                break
    return errors


def _check_forbidden_files() -> list[str]:
    errors: list[str] = []
    for name in FORBIDDEN_AT_ROOT:
        if (REPO_ROOT / name).exists():
            errors.append(
                f"{name} exists at repo root but is package-specific. "
                f"Move it under one of: "
                f"{', '.join(p + '/' for p in PACKAGES)}."
            )
    return errors


def main() -> int:
    errors = (
        _check_path_patterns(".gitignore")
        + _check_path_patterns(".gitattributes")
        + _check_forbidden_files()
    )
    if errors:
        print("Monorepo hygiene check FAILED:\n", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            "\nRoot-level shared files must not contain package-specific "
            "rules. Move each flagged rule into the relevant subproject's "
            "own version of the file.",
            file=sys.stderr,
        )
        return 1
    print("Monorepo hygiene check OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
