#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PreToolUse and CLI guard preventing autonomous/RSI agents from executing live Kaggle uploads.

Many V2D Challenge participants use autonomous coding agents or Recursive
Self-Improvement (RSI) harnesses (Claude Code, Codex, Cursor, Antigravity,
SWE-agent, OpenHands, etc.). When optimizing a task, an RSI model will often
attempt to "test whether the submission can be sent" by invoking `v2d_submit.py`
without `--dry-run` or calling Kaggle's CLI/SDK submission endpoints directly,
burning the 5-per-week submission limit, uploading across unauthorized tracks,
and triggering challenge suspension or disqualification.

This script operates in two modes:
1. Agent PreToolUse Hook Mode (default when JSON is piped on stdin):
   Compatible with Claude Code / Codex / Cursor `PreToolUse` hooks. Inspects
   shell commands and file edits; denies any command that invokes a live Kaggle
   upload or any `v2d_submit.py` invocation lacking `--dry-run`, `--check-entry`,
   or `--poll-receipt`.
2. CLI Audit Mode (`--check-command "<cmd>"` or `--scan-scripts <dir>`):
   Scans shell commands or Python scripts in CI to ensure no training,
   reconstruction, or evaluation script embeds automated Kaggle upload calls.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

SHELL_TOOL_MARKERS = (
    "bash",
    "exec_command",
    "functions.exec",
    "shell",
    "terminal",
    "unified_exec",
    "write_stdin",
    "run_command",
)
MUTATING_TOOL_MARKERS = (
    "apply_patch",
    "create_file",
    "edit_file",
    "write_file",
    "writefile",
)

# Built from token tuples so static scanners never false-trigger on this guard file itself.
_CLI_SUB_PATTERN = r"\bkaggle\s+(?:competitions|c)\s+" + "submit" + r"\b"
_SDK_SUB_PATTERN = r"\b" + "_".join(("competition", "submit")) + r"(?:_cli|_code)?\s*\("
_SDK_START_UPLOAD = r"\b" + "_".join(("start", "submission", "upload")) + r"\s*\("
_SDK_CREATE_SUB = r"\b" + "_".join(("create", "submission")) + r"\s*\("
_SDK_CREATE_CODE = r"\b" + "_".join(("create", "code", "submission")) + r"\s*\("
_REST_URL_PATTERN = "/".join(("competitions", "submissions", "url"))
_RPC_SERVICE_PATTERN = "/".join(("CompetitionsService", "CreateSubmission"))

DIRECT_UPLOAD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("kaggle_cli_direct_submit", re.compile(_CLI_SUB_PATTERN, re.IGNORECASE)),
    ("kaggle_sdk_submit", re.compile(_SDK_SUB_PATTERN, re.IGNORECASE)),
    ("kaggle_sdk_start_upload", re.compile(_SDK_START_UPLOAD, re.IGNORECASE)),
    ("kaggle_sdk_create_submission", re.compile(_SDK_CREATE_SUB, re.IGNORECASE)),
    ("kaggle_sdk_create_code_submission", re.compile(_SDK_CREATE_CODE, re.IGNORECASE)),
    ("kaggle_rest_submission_url", re.compile(_REST_URL_PATTERN, re.IGNORECASE)),
    ("kaggle_rpc_create_submission", re.compile(_RPC_SERVICE_PATTERN, re.IGNORECASE)),
)

PROTECTED_GUARD_FILENAMES = (
    "agent_submission_guard.py",
    "compliance_gate.py",
    ".v2d_policy.json",
)

SAFE_V2D_SUBMIT_FLAGS = frozenset({"--dry-run", "--check-entry", "--poll-receipt", "--help", "-h"})


def _extract_command_strings(value: Any, *, command_context: bool = False) -> list[str]:
    if isinstance(value, dict):
        found: list[str] = []
        for key, child in value.items():
            is_cmd = command_context or str(key).casefold() in {
                "cmd",
                "command",
                "commandline",
                "commands",
                "script",
                "chars",
            }
            found.extend(_extract_command_strings(child, command_context=is_cmd))
        return found
    if isinstance(value, list):
        found = []
        for child in value:
            found.extend(_extract_command_strings(child, command_context=command_context))
        return found
    if command_context and isinstance(value, str):
        return [value]
    return []


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [text for child in value.values() for text in _all_strings(child)]
    if isinstance(value, list):
        return [text for child in value for text in _all_strings(child)]
    return [value] if isinstance(value, str) else []


def check_v2d_submit_command(cmd: str) -> str | None:
    """If `v2d_submit.py` is invoked in `cmd`, require a read-only flag (`--dry-run`, etc.)."""
    if "v2d_submit.py" not in cmd:
        return None
    # Split compound shell commands on ;, &&, ||, |, newline
    segments = re.split(r"(?:&&|\|\||[;|\n])", cmd)
    for seg in segments:
        if "v2d_submit.py" not in seg:
            continue
        try:
            tokens = shlex.split(seg, comments=True)
        except ValueError:
            tokens = seg.split()
        if not any("v2d_submit.py" in tok for tok in tokens):
            continue
        if "--only" in tokens and "--retry-receipt" not in tokens:
            return "v2d_submit_only_without_retry_receipt"
        if not any(flag in SAFE_V2D_SUBMIT_FLAGS for flag in tokens):
            return "v2d_submit_without_dry_run"
    return None


def inspect_command_string(cmd: str) -> str | None:
    """Return a violation code if `cmd` attempts a live Kaggle upload, else `None`."""
    for name, pat in DIRECT_UPLOAD_PATTERNS:
        if pat.search(cmd):
            return f"prohibited_upload_command:{name}"
    v2d_issue = check_v2d_submit_command(cmd)
    if v2d_issue is not None:
        return f"prohibited_upload_command:{v2d_issue}"
    return None


def inspect_hook_payload(payload: dict[str, Any]) -> str | None:
    """Inspect a PreToolUse JSON payload and return a violation code if blocked."""
    tool_name = str(payload.get("tool_name") or payload.get("toolName") or "")
    folded = tool_name.casefold()
    is_shell = any(marker in folded for marker in SHELL_TOOL_MARKERS)
    is_mutating = any(marker in folded for marker in MUTATING_TOOL_MARKERS)
    if not is_shell and not is_mutating:
        return None

    tool_input = payload.get("tool_input")
    if tool_input is None:
        tool_input = payload.get("toolInput")
    if tool_input is None:
        tool_input = payload.get("input")

    if is_mutating:
        for text in _all_strings(tool_input):
            for protected in PROTECTED_GUARD_FILENAMES:
                if protected in text:
                    return f"protected_compliance_file:{protected}"
            for name, pat in DIRECT_UPLOAD_PATTERNS:
                if pat.search(text):
                    return f"prohibited_upload_in_code:{name}"
        return None

    for cmd in _extract_command_strings(tool_input, command_context=isinstance(tool_input, str)):
        issue = inspect_command_string(cmd)
        if issue is not None:
            return issue
    return None


def scan_directory_for_upload_calls(
    root: Path,
    *,
    allowlist_relpaths: set[str] | None = None,
) -> list[str]:
    """Scan `.py` and `.sh` files under `root` for embedded Kaggle upload calls."""
    allowed = allowlist_relpaths or {
        "submit/v2d_submit.py",
        "tools/agent_submission_guard.py",
        "tools/v2d_compliance/agent_submission_guard.py",
        "tools/v2d_compliance/tests/test_compliance_gate.py",
    }
    violations: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".py", ".sh"):
            continue
        rel = path.relative_to(root).as_posix()
        if rel in allowed or any(rel.endswith(a) for a in allowed):
            continue
        try:
            content = path.read_text(errors="ignore")
        except OSError:
            continue
        for name, pat in DIRECT_UPLOAD_PATTERNS:
            if pat.search(content):
                violations.append(f"{rel}: contains prohibited automated upload call ({name})")
    return violations


def _deny_hook(marker: str) -> int:
    reason = (
        "V2D-AGENT-SUBMISSION-GUARD BLOCKED: Autonomous agents and RSI models are prohibited "
        "from executing live Kaggle uploads or modifying compliance guards. Always use "
        "`python3 submit/v2d_submit.py <FILE> --track <TRACK> -m '<MSG>' --dry-run` and "
        "`python3 tools/compliance_gate.py audit <FILE> --track <TRACK>` to verify "
        "submissions locally without spending weekly submission quota."
    )
    print(reason, file=sys.stderr)
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                },
                "matchedPolicyMarker": marker,
            },
            separators=(",", ":"),
        )
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-command",
        default=None,
        help="Check a shell command string and exit non-zero if it triggers a live upload",
    )
    parser.add_argument(
        "--scan-scripts",
        type=Path,
        default=None,
        help="Scan a directory of .py/.sh scripts for prohibited embedded Kaggle upload calls",
    )
    args = parser.parse_args(argv)

    if args.check_command is not None:
        marker = inspect_command_string(args.check_command)
        if marker is not None:
            print(f"BLOCKED ({marker}): {args.check_command}", file=sys.stderr)
            return 2
        print("OK: command does not trigger a live Kaggle upload.")
        return 0

    if args.scan_scripts is not None:
        violations = scan_directory_for_upload_calls(args.scan_scripts)
        if violations:
            print("Script scan FAILED — prohibited automated upload calls detected:", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)
            return 1
        print(f"OK: scanned {args.scan_scripts} with zero prohibited upload calls.")
        return 0

    # Default: PreToolUse JSON hook over stdin
    try:
        payload = json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        return 0
    if not isinstance(payload, dict):
        return 0
    marker = inspect_hook_payload(payload)
    if marker is not None:
        return _deny_hook(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
