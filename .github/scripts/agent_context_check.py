#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Ensure agent-context files (CLAUDE.md, AGENTS.md) stay committable.

Past incident: a top-level .gitignore rule for `CLAUDE.md` silently kept
per-package CLAUDE.md files out of git, so freshly-cloned teammates had no
agent guidance for the video_ingestion_agent package. This check scans the
ROOT .gitignore (the only one that affects every package) for rules that
would exclude these files and fails CI if any are found. Per-package
gitignores are intentionally out of scope — they only affect their own
subtree and may have legitimate local reasons.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_GITIGNORE = REPO_ROOT / ".gitignore"

FORBIDDEN_PATTERNS = {"CLAUDE.md", "AGENTS.md"}


def _check_root_gitignore() -> list[str]:
    if not ROOT_GITIGNORE.exists():
        return []
    rel = ROOT_GITIGNORE.relative_to(REPO_ROOT)
    errors: list[str] = []
    for lineno, raw in enumerate(ROOT_GITIGNORE.read_text().splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # A '!' prefix is a negation — it ALLOWS the file, which is fine.
        if stripped.startswith("!"):
            continue
        # First whitespace token, then strip an anchored '/' prefix.
        token = stripped.split()[0].lstrip("/")
        if token in FORBIDDEN_PATTERNS:
            errors.append(
                f"{rel}:{lineno}: '{stripped}' ignores an agent-context "
                f"file. CLAUDE.md and AGENTS.md must remain committable; "
                f"remove this rule (or replace with a more specific pattern "
                f"that doesn't match these files)."
            )
    return errors


def main() -> int:
    all_errors = _check_root_gitignore()
    if all_errors:
        print("agent_context_check FAILED:\n", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        print(
            "\nAgent-facing context files (CLAUDE.md, AGENTS.md) must be "
            "trackable everywhere in the monorepo. Remove any .gitignore "
            "rule that excludes them.",
            file=sys.stderr,
        )
        return 1
    print("agent_context_check OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
