#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-commit hook: enforce the NVIDIA Apache-2.0 SPDX header on Python files.

Recognizes NVIDIA's own license headers in any of their historical forms (the
legacy proprietary "all rights reserved / strictly prohibited" block, the
verbose Apache-2.0 boilerplate, or the canonical 2-line SPDX header) and
normalizes them to the concise 2-line SPDX header used across the
``reconstruction/`` tree::

    # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    # SPDX-License-Identifier: Apache-2.0

Files that carry no header get one inserted. Files whose leading comment block
carries a *non-NVIDIA* copyright (e.g. third-party Isaac Lab code) are left
completely untouched so we never relicense or strip upstream attribution.

Run by pre-commit; modifies files in place and exits non-zero when it changed
anything (so the commit aborts and the change gets staged).
"""
from __future__ import annotations

import sys

HEADER_LINES = [
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n",
    "# SPDX-License-Identifier: Apache-2.0\n",
]

# A comment is part of a license header if it is an empty `#` spacer or mentions
# one of these tokens. Any other comment (a tool directive, a real code comment)
# ends the header run so it is never consumed.
LICENSE_TOKENS = (
    "copyright",
    "spdx-",
    "nvidia",
    "all rights reserved",
    "licensed under",
    "license at",
    "license for",
    "license is",
    "the license",
    "apache license",
    "version 2.0",
    "warranties",
    "obtain a copy",
    "compliance with",
    "intellectual property",
    "proprietary rights",
    "strictly prohibited",
    "related documentation",
    "modifications thereto",
    "applicable law",
    "apache.org/licenses",
    "express or implied",
    "as is",
    "permissions and",
    "limitations under",
)

# If the header run carries any of these, it is third-party — leave it alone.
THIRD_PARTY_TOKENS = ("isaac lab project developers",)


def _is_license_comment(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return False
    body = stripped[1:].strip()
    if body == "":  # bare `#` spacer line inside a header block
        return True
    low = body.lower()
    return any(tok in low for tok in LICENSE_TOKENS)


def normalize(text: str) -> str:
    """Return ``text`` with the canonical SPDX header, or unchanged if third-party."""
    lines = text.splitlines(keepends=True)

    head: list[str] = []
    idx = 0
    if lines and lines[idx].startswith("#!"):
        head.append(lines[idx])
        idx += 1

    # Skip blank lines between an optional shebang and the header/body.
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1

    # Consume the contiguous run of license-like comment lines.
    run_end = idx
    while run_end < len(lines) and _is_license_comment(lines[run_end]):
        run_end += 1
    run = "".join(lines[idx:run_end]).lower()

    if any(tok in run for tok in THIRD_PARTY_TOKENS):
        return text  # third-party header — never touch
    if "copyright" in run and "nvidia" not in run:
        return text  # some other party's copyright — never touch

    is_nvidia_header = "nvidia" in run and ("copyright" in run or "spdx-" in run)
    body_start = run_end if is_nvidia_header else idx

    body = lines[body_start:]
    while body and body[0].strip() == "":  # no blank line between header and body
        body.pop(0)

    return "".join(head + HEADER_LINES + body)


def main(argv: list[str]) -> int:
    """Normalize each path in ``argv``; exit non-zero if any file changed."""
    changed = False
    for path in argv:
        try:
            original = open(path, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        updated = normalize(original)
        if updated != original:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(updated)
            print(f"license header fixed: {path}")
            changed = True
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
