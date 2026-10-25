#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V2D Challenge compliance, weekly quota, and anti-reward-hacking gate.

Enforces the stated V2D Challenge rules in local preflight, CI/CD pipelines, and
the uploader before any file can be sent to Kaggle:

1. Weekly Quota Enforcement (5/week vs Kaggle's 5/day UI limit):
   Kaggle's competition configuration only supports a daily limit (5/day), while
   the stated V2D Challenge rule limits teams to five (5) submissions per rolling
   7-day week (with unlimited submissions only during the final 3 days before the
   November 4, 2026 5:00 p.m. EST freeze). Autonomous/RSI models that rely on
   Kaggle's API quota response alone will upload up to 35 times per week and trigger
   disqualification or suspension. This gate enforces the rolling 7-day cap across
   both live Kaggle submission history and local receipt ledgers.

2. Competition Scope & Atomic Track Bundle Enforcement:
   Prevents uploading to more competitions/tracks than declared in the team's policy
   file (`.v2d_policy.json`), blocks standalone `track_3_mesh` split uploads, and
   blocks `--only` single-leaderboard probing unless paired with a verified failed
   receipt (`--retry-receipt`) for the exact same file SHA-256.

3. Autonomous Agent / RSI Live-Upload Lock:
   Detects autonomous coding agent environments (Claude Code, Codex, Cursor,
   Antigravity, Aider, SWE-agent, Devin, OpenHands, etc.) and blocks non-dry-run
   uploads so an RSI model cannot "test whether a submission can be sent" against
   the live Kaggle endpoint.

4. Anti-Reward-Hacking & Data-Leakage Audit:
   - Verifies `code_commit_url` is a real, non-placeholder GitHub commit SHA (and
     matches clean `HEAD` in CI).
   - Verifies `eval_script_sha256`, `eval_code_sha256`, and per-episode
     `checkpoint_sha256` digests against real checkpoint files and released
     evaluator hashes, catching synthetic string hashes.
   - Screens object meshes for `< 1.0 mm` near-duplicate copying of public/Tier 1
     scans and `0 < max_diff < 0.1 mm` (`1e-7 m`) microscopic Gaussian noise evasion.
   - Screens Track 2 Tier 2/Tier 3 trajectories for leaked Tier 1 ground-truth poses
     and Track 1 `mhr_shape` for leaked Track 2 `identity_coeffs`.
   - Screens Track 1 reconstructions for degenerate metric-gaming signatures
     (frozen root orientation, frozen object orientation, or static body pose).
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Canonical 23-competition mapping across the 5 participant-submittable tracks.
CANONICAL_TRACK_COMPETITIONS: dict[str, dict[str, str]] = {
    "track_1": {
        "cd_h_cm": "v2d-challenge-track1-cd-h",
        "cd_o_cm": "v2d-challenge-track1-cd-o",
        "acc_h_cm": "v2d-challenge-track1-acc-h",
        "acc_o_cm": "v2d-challenge-track1-acc-o",
        "interpenetration_cm": "v2d-challenge-track1-pen",
    },
    "track_2_tier1": {
        "add_auc": "v2d-challenge-track2-tier1-auc",
        "spider_sr": "v2d-challenge-track2-tier1-sp-sr",
        "maniptrans_sr": "v2d-challenge-track2-tier1-mp-sr",
        "mppe_cm": "v2d-challenge-track2-tier1-mppe",
    },
    "track_2_tier2": {
        "add_auc": "v2d-challenge-track2-tier2-auc",
        "spider_sr": "v2d-challenge-track2-tier2-sp-sr",
        "maniptrans_sr": "v2d-challenge-track2-tier2-mp-sr",
        "mppe_cm": "v2d-challenge-track2-tier2-mppe",
    },
    "track_2_tier3": {
        "add_auc": "v2d-challenge-track2-tier3-auc",
        "spider_sr": "v2d-challenge-track2-tier3-sp-sr",
        "maniptrans_sr": "v2d-challenge-track2-tier3-mp-sr",
        "mppe_cm": "v2d-challenge-track2-tier3-mppe",
    },
    "track_3": {
        "add_auc": "v2d-challenge-track3-auc",
        "spider_sr": "v2d-challenge-track3-sp-sr",
        "maniptrans_sr": "v2d-challenge-track3-mp-sr",
        "rpe_cm": "v2d-challenge-track3-rpe",
        "mppe_cm": "v2d-challenge-track3-mppe",
        "cd_o_cm": "v2d-challenge-track3-cd-o",
    },
}

# Official timeline bounds (November 4, 2026 at 5:00 p.m. EST = 22:00:00 UTC).
DEFAULT_LAUNCH_UTC = dt.datetime(2026, 9, 21, 0, 0, 0, tzinfo=dt.timezone.utc)
DEFAULT_UNLIMITED_START_UTC = dt.datetime(2026, 11, 1, 22, 0, 0, tzinfo=dt.timezone.utc)
DEFAULT_FREEZE_UTC = dt.datetime(2026, 11, 4, 22, 0, 0, tzinfo=dt.timezone.utc)
DEFAULT_MAX_WEEKLY_SUBMISSIONS = 5
ROLLING_WEEK_WINDOW = dt.timedelta(days=7)

# Released official evaluate_policy.py SHA-256 from KIT_MANIFEST.json.
RELEASED_EVAL_SCRIPT_SHA256 = {
    "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1",
}

CODE_COMMIT_RE = re.compile(
    r"^https://(?P<host>[\w.-]*github[\w.-]*)/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/commit/(?P<sha>[0-9a-fA-F]{7,40})/?$"
)
SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")

PROHIBITED_UPSTREAM_REPOS = {
    "nvidia-isaac/video_to_data",
    "nvidia-v2d-challenge/v2d_submission_kit",
    "nvidia/video_to_data",
    "owner/repo",
    "your-org/your-repo",
    "example/repo",
    "user/repo",
}

PROHIBITED_COMMIT_SHAS = {
    "04abe86f",
    "8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef",
    "09a1e3f27b481029384756102938475610293847",
}

# Environment variables set by autonomous coding agents / RSI harnesses.
AGENT_ENV_MARKERS = (
    "V2D_AGENT_MODE",
    "CLAUDECODE",
    "CLAUDE_CODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CODEX_SANDBOX",
    "CODEX_ENV",
    "OPENAI_CODEX",
    "ANTIGRAVITY_AGENT",
    "ANTIGRAVITY_CLI",
    "CURSOR_TRACE_ID",
    "CURSOR_AGENT",
    "AIDER",
    "SWE_AGENT",
    "DEVIN_SESSION",
    "OPENHANDS",
)

POSE_COLUMNS = ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")
PROVENANCE_COLUMNS = (
    "eval_script_sha256",
    "eval_code_sha256",
    "checkpoint_sha256",
    "packer_script_sha256",
    "uploader_script_sha256",
    "code_commit_url",
)
UNIFORM_PROVENANCE_COLUMNS = (
    "eval_script_sha256",
    "eval_code_sha256",
    "packer_script_sha256",
    "uploader_script_sha256",
    "code_commit_url",
)


class ComplianceError(ValueError):
    """Raised when a candidate submission or upload request violates V2D rules."""


@dataclass
class QuotaStatus:
    track: str
    now_utc: str
    in_unlimited_window: bool
    after_freeze: bool
    max_weekly_allowed: int
    weekly_used_by_competition: dict[str, int] = field(default_factory=dict)
    weekly_used_max: int = 0
    weekly_remaining: int = DEFAULT_MAX_WEEKLY_SUBMISSIONS
    next_slot_opens_utc: str | None = None
    already_uploaded_sha256: bool = False
    cross_board_divergence: bool = False
    blocking_reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.blocking_reasons) == 0


@dataclass
class ComplianceReport:
    track: str
    submission_path: str
    submission_sha256: str
    rows: int
    episodes: int
    code_commit_url: str = ""
    derived_cd_o_sha256: str | None = None
    checks_passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    quota: QuotaStatus | None = None

    @property
    def ok(self) -> bool:
        quota_ok = self.quota.ok if self.quota is not None else True
        return len(self.failures) == 0 and quota_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "track": self.track,
            "submission_path": self.submission_path,
            "submission_sha256": self.submission_sha256,
            "derived_cd_o_sha256": self.derived_cd_o_sha256,
            "rows": self.rows,
            "episodes": self.episodes,
            "code_commit_url": self.code_commit_url,
            "checks_passed": self.checks_passed,
            "warnings": self.warnings,
            "failures": self.failures,
            "quota": (
                {
                    "ok": self.quota.ok,
                    "now_utc": self.quota.now_utc,
                    "in_unlimited_window": self.quota.in_unlimited_window,
                    "after_freeze": self.quota.after_freeze,
                    "max_weekly_allowed": self.quota.max_weekly_allowed,
                    "weekly_used_max": self.quota.weekly_used_max,
                    "weekly_remaining": self.quota.weekly_remaining,
                    "weekly_used_by_competition": self.quota.weekly_used_by_competition,
                    "next_slot_opens_utc": self.quota.next_slot_opens_utc,
                    "already_uploaded_sha256": self.quota.already_uploaded_sha256,
                    "cross_board_divergence": self.quota.cross_board_divergence,
                    "blocking_reasons": self.quota.blocking_reasons,
                }
                if self.quota is not None
                else None
            ),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(policy_path: Path | None = None) -> dict[str, Any]:
    """Load participant `.v2d_policy.json` if present."""
    candidates = (
        [policy_path]
        if policy_path is not None
        else [
            Path(".v2d_policy.json"),
            Path("config/v2d_policy.json"),
        ]
    )
    for path in candidates:
        if path is not None and path.exists():
            return json.loads(path.read_text())
    return {}


def detect_agent_environment(env: dict[str, str] | None = None) -> list[str]:
    """Return active autonomous agent / RSI environment markers."""
    environ = os.environ if env is None else env
    detected: list[str] = []
    for key in AGENT_ENV_MARKERS:
        val = str(environ.get(key, "")).strip()
        if val and val.lower() not in ("0", "false", "no", "off"):
            detected.append(f"{key}={val}")
    return detected


def check_upload_authorization(
    *,
    dry_run: bool,
    confirm_human_upload: bool = False,
    env: dict[str, str] | None = None,
    stdin_isatty: bool | None = None,
) -> list[str]:
    """Verify that a live Kaggle upload is not being triggered by an autonomous agent.

    Returns a list of blocking violation messages (empty if authorized).
    """
    if dry_run:
        return []

    environ = os.environ if env is None else env
    isatty = sys.stdin.isatty() if stdin_isatty is None else stdin_isatty

    agent_markers = detect_agent_environment(environ)
    if agent_markers:
        return [
            "AGENT_LIVE_UPLOAD_PROHIBITED: Live Kaggle uploads are blocked inside autonomous "
            f"agent / RSI environments ({', '.join(agent_markers)}). Agents must never test "
            "whether a submission can be sent against the live Kaggle endpoint. Use "
            "`--dry-run` or `tools/compliance_gate.py audit` instead, and perform live "
            "uploads only via human execution or the manual-approval CD workflow."
        ]

    # Check if running inside GitHub Actions or generic CI
    in_ci = str(environ.get("CI", "")).lower() in ("1", "true", "yes") or bool(
        environ.get("GITHUB_ACTIONS")
    )
    if in_ci:
        gh_event = str(environ.get("GITHUB_EVENT_NAME", ""))
        ci_approved = str(environ.get("V2D_CI_APPROVED_UPLOAD", "")) == "1"
        if gh_event != "workflow_dispatch" or not ci_approved:
            return [
                "CI_UPLOAD_LOCKED: Automated CI runs on `push` or `pull_request` must never "
                "upload to Kaggle. Live CI/CD uploads are restricted to manual "
                "`workflow_dispatch` runs with `V2D_CI_APPROVED_UPLOAD=1` inside the "
                "protected `v2d-kaggle-release` environment."
            ]
        return []

    if not confirm_human_upload and not isatty:
        return [
            "NON_INTERACTIVE_UPLOAD_BLOCKED: Refusing live Kaggle upload from a non-interactive "
            "shell without `--confirm-human-upload`. Use `--dry-run` to test validation."
        ]

    if not confirm_human_upload:
        return [
            "HUMAN_CONFIRMATION_REQUIRED: Pass `--confirm-human-upload` (or use `--dry-run` "
            "for validation) to confirm this upload will spend 1 of your 5 weekly submissions."
        ]

    return []


def check_track_scope(
    track: str,
    *,
    policy: dict[str, Any] | None = None,
    allowed_tracks_override: list[str] | None = None,
    allow_internal_track3_mesh: bool = False,
) -> tuple[dict[str, str], list[str]]:
    """Verify the track is canonical and permitted by the participant's declared scope."""
    errors: list[str] = []
    if track == "track_3_mesh" and not allow_internal_track3_mesh:
        errors.append(
            "STANDALONE_TRACK3_MESH_BLOCKED: `track_3_mesh` (CD-O) cannot be uploaded as a "
            "standalone track. Competition rules require submitting one combined Track 3 "
            "Parquet (`--track track_3`) containing both rollout trajectories and "
            "`object_mesh_b64` payloads; the uploader extracts `.cd-o.parquet` automatically."
        )
        return {"cd_o_cm": "v2d-challenge-track3-cd-o"}, errors

    if track not in CANONICAL_TRACK_COMPETITIONS:
        errors.append(
            f"UNKNOWN_TRACK: {track!r} is not a valid V2D Challenge submission track. "
            f"Valid tracks: {sorted(CANONICAL_TRACK_COMPETITIONS)}."
        )
        return {}, errors

    active_policy = policy or {}
    allowed = allowed_tracks_override or active_policy.get("allowed_tracks")
    if allowed is not None:
        allowed_set = {str(t).strip() for t in allowed if str(t).strip()}
        if track not in allowed_set:
            errors.append(
                f"TRACK_SCOPE_VIOLATION: Track {track!r} is not in the declared "
                f"`allowed_tracks` ({sorted(allowed_set)}). Refusing to upload to more "
                "competitions/tracks than authorized by the team policy."
            )

    return dict(CANONICAL_TRACK_COMPETITIONS[track]), errors


def check_bundle_atomicity(
    track: str,
    submission_path: Path,
    *,
    only_metrics: list[str] | None = None,
    retry_receipt_path: Path | None = None,
) -> list[str]:
    """Enforce that all metric leaderboards of a track receive the exact same submission.

    Blocks `--only` single-leaderboard probing unless `--retry-receipt` proves that
    the exact same file SHA-256 already succeeded on sibling metrics of the track and
    failed only on the retried metric(s).
    """
    errors: list[str] = []
    if not only_metrics:
        return errors

    if track not in CANONICAL_TRACK_COMPETITIONS:
        errors.append(f"Cannot validate `--only` for unknown track {track!r}.")
        return errors

    canonical = CANONICAL_TRACK_COMPETITIONS[track]
    unknown = [m for m in only_metrics if m not in canonical]
    if unknown:
        errors.append(
            f"UNKNOWN_METRIC_IN_ONLY: {unknown} not in {track} metrics {sorted(canonical)}."
        )
        return errors

    if set(only_metrics) == set(canonical):
        return errors

    if retry_receipt_path is None:
        errors.append(
            "SPLIT_LEADERBOARD_PROBING_BLOCKED: Submitting to a subset of a track's metric "
            f"competitions (`--only {' '.join(only_metrics)}`) without `--retry-receipt` is "
            "prohibited by V2D Challenge rules. Every metric leaderboard in a track must "
            "be scored on the exact same submission file. To recover a partial network "
            "failure from a prior full-track upload, pass `--retry-receipt <receipt.json>`."
        )
        return errors

    if not retry_receipt_path.exists():
        errors.append(f"RETRY_RECEIPT_MISSING: `{retry_receipt_path}` does not exist.")
        return errors

    try:
        receipt = json.loads(retry_receipt_path.read_text())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"RETRY_RECEIPT_INVALID: Could not parse `{retry_receipt_path}` ({exc}).")
        return errors

    if receipt.get("track") != track:
        errors.append(
            f"RETRY_RECEIPT_TRACK_MISMATCH: Receipt is for track {receipt.get('track')!r}, "
            f"not {track!r}."
        )

    current_sha = sha256_file(submission_path)
    receipt_sha = receipt.get("file_sha256")
    if not receipt_sha and receipt.get("file"):
        prev_file = Path(str(receipt["file"]))
        if prev_file.exists() and prev_file.resolve() == submission_path.resolve():
            receipt_sha = current_sha
    if receipt_sha != current_sha:
        errors.append(
            "RETRY_RECEIPT_HASH_MISMATCH: `--only` retry must use the exact same file "
            f"(SHA-256 {current_sha[:16]}...) as the original receipt "
            f"({str(receipt_sha)[:16]}...). Submitting different files to different metric "
            "leaderboards of the same track is prohibited."
        )

    recorded_subs = receipt.get("submissions") or {}
    succeeded_siblings = [
        m
        for m, entry in recorded_subs.items()
        if isinstance(entry, dict) and entry.get("state") in ("PENDING", "COMPLETE", "ACCEPTED")
    ]
    if not succeeded_siblings:
        errors.append(
            "RETRY_RECEIPT_NO_PRIOR_SUCCESS: Receipt has no previously accepted sibling "
            "submissions; run a full-track upload without `--only` instead."
        )

    for metric in only_metrics:
        entry = recorded_subs.get(metric)
        state = entry.get("state") if isinstance(entry, dict) else None
        if state not in ("UPLOAD_FAILED", "ERROR"):
            errors.append(
                f"RETRY_METRIC_NOT_FAILED: Metric {metric!r} has state {state!r} in "
                f"`{retry_receipt_path.name}` (only `UPLOAD_FAILED` metrics may be retried)."
            )

    return errors


def _parse_utc_timestamp(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    return None


def check_weekly_quota(
    track: str,
    *,
    submission_sha256: str | None = None,
    receipt_dirs: list[Path] | None = None,
    ledger_path: Path | None = None,
    kaggle_submissions_by_slug: dict[str, list[Any]] | None = None,
    now_utc: dt.datetime | None = None,
    max_weekly_allowed: int = DEFAULT_MAX_WEEKLY_SUBMISSIONS,
    unlimited_start_utc: dt.datetime = DEFAULT_UNLIMITED_START_UTC,
    freeze_utc: dt.datetime = DEFAULT_FREEZE_UTC,
) -> QuotaStatus:
    """Enforce the stated 5-submissions-per-week V2D rule across all boards of a track.

    Combines timestamps from:
    1. Live Kaggle API submission records (`kaggle_submissions_by_slug`), and
    2. Local `*.kaggle-receipt.json` files and `.v2d_submission_ledger.json`.
    """
    now = (now_utc or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    window_start = now - ROLLING_WEEK_WINDOW
    in_unlimited = unlimited_start_utc <= now <= freeze_utc
    after_freeze = now > freeze_utc

    competitions = CANONICAL_TRACK_COMPETITIONS.get(track, {})
    status = QuotaStatus(
        track=track,
        now_utc=now.isoformat(),
        in_unlimited_window=in_unlimited,
        after_freeze=after_freeze,
        max_weekly_allowed=max_weekly_allowed,
    )

    if after_freeze:
        status.blocking_reasons.append(
            f"COMPETITION_FROZEN: Current time ({now.isoformat()}) is past the official "
            f"V2D Challenge deadline ({freeze_utc.isoformat()}). No further uploads are permitted."
        )

    # Track timestamps per competition slug, deduplicated by submission ref or minute bucket.
    events_by_slug: dict[str, list[dt.datetime]] = {slug: [] for slug in competitions.values()}
    seen_refs_by_slug: dict[str, set[str]] = {slug: set() for slug in competitions.values()}
    uploaded_shas_for_track: set[str] = set()

    # 1. Inspect local ledger if present
    if ledger_path is not None and ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text())
            for entry in ledger.get("uploads", []):
                if entry.get("track") != track:
                    continue
                if entry.get("file_sha256"):
                    uploaded_shas_for_track.add(str(entry["file_sha256"]))
                ts = _parse_utc_timestamp(entry.get("timestamp_utc"))
                if ts is not None and window_start <= ts <= now:
                    for slug in entry.get("competitions", list(competitions.values())):
                        if slug in events_by_slug:
                            ref_key = f"ledger:{entry.get('id') or ts.isoformat()}"
                            if ref_key not in seen_refs_by_slug[slug]:
                                seen_refs_by_slug[slug].add(ref_key)
                                events_by_slug[slug].append(ts)
        except Exception:  # noqa: BLE001
            pass

    # 2. Inspect local *.kaggle-receipt.json files
    for rdir in receipt_dirs or []:
        if not rdir.exists():
            continue
        for rpath in sorted(rdir.glob("*.kaggle-receipt.json")):
            try:
                receipt = json.loads(rpath.read_text())
            except Exception:  # noqa: BLE001
                continue
            if receipt.get("track") != track:
                continue
            r_sha = receipt.get("file_sha256")
            r_ts = _parse_utc_timestamp(receipt.get("timestamp_utc"))
            if r_ts is None:
                r_ts = dt.datetime.fromtimestamp(rpath.stat().st_mtime, tz=dt.timezone.utc)
            subs = receipt.get("submissions") or {}
            any_accepted = False
            for entry in subs.values():
                if not isinstance(entry, dict):
                    continue
                slug = str(entry.get("slug", ""))
                state = str(entry.get("state", ""))
                if state in ("PENDING", "COMPLETE", "ACCEPTED", "ERROR"):
                    any_accepted = True
                    if slug in events_by_slug and window_start <= r_ts <= now:
                        ref_id = str(entry.get("ref") or f"file:{rpath.name}")
                        if ref_id not in seen_refs_by_slug[slug]:
                            seen_refs_by_slug[slug].add(ref_id)
                            events_by_slug[slug].append(r_ts)
            if any_accepted and r_sha:
                uploaded_shas_for_track.add(str(r_sha))

    # 3. Inspect live Kaggle API submissions per competition slug
    if kaggle_submissions_by_slug:
        for slug, sub_list in kaggle_submissions_by_slug.items():
            if slug not in events_by_slug:
                continue
            for sub in sub_list or []:
                ref_val = getattr(sub, "ref", None) or (
                    sub.get("ref") if isinstance(sub, dict) else None
                )
                date_val = getattr(sub, "date", None) or (
                    sub.get("date") if isinstance(sub, dict) else None
                )
                ts = _parse_utc_timestamp(date_val)
                if ts is None or not (window_start <= ts <= now):
                    continue
                ref_key = str(ref_val) if ref_val is not None else f"ts:{ts.isoformat()}"
                if ref_key not in seen_refs_by_slug[slug]:
                    seen_refs_by_slug[slug].add(ref_key)
                    events_by_slug[slug].append(ts)

    counts = {slug: len(events) for slug, events in events_by_slug.items()}
    status.weekly_used_by_competition = counts
    status.weekly_used_max = max(counts.values()) if counts else 0
    status.weekly_remaining = (
        999999 if in_unlimited else max(0, max_weekly_allowed - status.weekly_used_max)
    )

    if counts and len(set(counts.values())) > 1:
        status.cross_board_divergence = True
        status.blocking_reasons.append(
            "CROSS_BOARD_COUNT_DIVERGENCE: The metric leaderboards for "
            f"{track} have unequal submission counts in the last 7 days ({counts}). "
            "This indicates a prior partial or split-leaderboard upload. Reconcile failed "
            "metrics with `--only <metric> --retry-receipt <receipt.json>` before starting "
            "a new full-track upload."
        )

    if submission_sha256 and submission_sha256 in uploaded_shas_for_track:
        status.already_uploaded_sha256 = True
        status.blocking_reasons.append(
            f"DUPLICATE_SUBMISSION_SHA256: File with SHA-256 {submission_sha256[:16]}... "
            f"has already been uploaded to {track}. Refusing to burn weekly quota on an "
            "identical re-upload."
        )

    if not in_unlimited and status.weekly_used_max >= max_weekly_allowed:
        # Compute when the oldest submission in the 7-day window expires
        oldest_across_max: list[dt.datetime] = []
        for slug, events in events_by_slug.items():
            if len(events) >= max_weekly_allowed:
                sorted_ev = sorted(events)
                # The (len - max_weekly_allowed)-th event is the one that must expire
                idx = len(sorted_ev) - max_weekly_allowed
                oldest_across_max.append(sorted_ev[idx] + ROLLING_WEEK_WINDOW)
        if oldest_across_max:
            next_open = max(oldest_across_max)
            status.next_slot_opens_utc = next_open.isoformat()
        status.blocking_reasons.append(
            f"WEEKLY_QUOTA_EXCEEDED: {status.weekly_used_max}/{max_weekly_allowed} submissions "
            f"used in the rolling 7-day window on {track} ({counts}). "
            "NOTE: Kaggle's API allows 5/day, but the official V2D Challenge rules strictly "
            "limit teams to 5 submissions per week (except the final 3 days: "
            f"{unlimited_start_utc.isoformat()} to {freeze_utc.isoformat()}). "
            + (
                f"Next weekly slot opens at {status.next_slot_opens_utc}."
                if status.next_slot_opens_utc
                else ""
            )
        )

    return status


def append_submission_ledger(
    ledger_path: Path,
    *,
    track: str,
    submission_path: Path,
    submission_sha256: str,
    code_commit_url: str,
    competitions: list[str],
    receipt_path: Path,
    now_utc: dt.datetime | None = None,
) -> None:
    """Record an upload in the local append-only submission ledger."""
    now = (now_utc or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    data: dict[str, Any] = {"uploads": []}
    if ledger_path.exists():
        try:
            data = json.loads(ledger_path.read_text())
            if not isinstance(data.get("uploads"), list):
                data["uploads"] = []
        except Exception:  # noqa: BLE001
            data = {"uploads": []}

    data["uploads"].append(
        {
            "id": f"{track}:{submission_sha256[:12]}:{int(now.timestamp())}",
            "timestamp_utc": now.isoformat(),
            "track": track,
            "file": str(submission_path),
            "file_sha256": submission_sha256,
            "code_commit_url": code_commit_url,
            "competitions": competitions,
            "receipt": str(receipt_path),
        }
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger_path.with_name(ledger_path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(ledger_path)


def _decode_track3_mesh_b64(payload: str) -> tuple[np.ndarray, np.ndarray]:
    """Decode a Track 3 `object_mesh_b64` compressed payload into `(vertices, faces)`."""
    raw = zlib.decompress(base64.b64decode(payload.encode("ascii")))
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
    return vertices, faces


def _active_surface_vertices(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return distinct vertices belonging to positive-area triangles (excluding padding)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(v) == 0 or len(f) == 0:
        return np.empty((0, 3), dtype=np.float64)
    valid_idx = (f >= 0).all(axis=1) & (f < len(v)).all(axis=1)
    f = f[valid_idx]
    if len(f) == 0:
        return np.empty((0, 3), dtype=np.float64)
    tris = v[f]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    areas = np.linalg.norm(normals, axis=1)
    active_faces = f[areas > 0]
    if len(active_faces) == 0:
        return np.empty((0, 3), dtype=np.float64)
    used_v = v[np.unique(active_faces)]
    return np.unique(used_v, axis=0)


def _symmetric_chamfer_m(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """Compute centered symmetric nearest-neighbor Chamfer distance (in metres).

    Uses `scipy.spatial.cKDTree` when installed, with a pure-NumPy chunked fallback
    so lightweight CI environments need only `numpy`.
    """
    if len(v_a) == 0 or len(v_b) == 0:
        return float("inf")
    ca = v_a - v_a.mean(axis=0, keepdims=True)
    cb = v_b - v_b.mean(axis=0, keepdims=True)
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-untyped]

        tree_a = cKDTree(ca)
        tree_b = cKDTree(cb)
        d_a2b, _ = tree_b.query(ca, k=1)
        d_b2a, _ = tree_a.query(cb, k=1)
        return 0.5 * (float(np.mean(d_a2b)) + float(np.mean(d_b2a)))
    except ImportError:
        # Subsample to at most 1024 points for fast O(N^2) pure-NumPy fallback
        if len(ca) > 1024:
            idx_a = np.linspace(0, len(ca) - 1, 1024, dtype=int)
            ca = ca[idx_a]
        if len(cb) > 1024:
            idx_b = np.linspace(0, len(cb) - 1, 1024, dtype=int)
            cb = cb[idx_b]
        diff = ca[:, None, :] - cb[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)
        return 0.5 * (float(np.mean(dist_mat.min(axis=1))) + float(np.mean(dist_mat.min(axis=0))))


def _is_synthetic_sha256(digest: str, episode: int | None = None) -> str | None:
    """Detect synthetic or fabricated SHA-256 strings generated by reward-hacking scripts."""
    clean = digest.strip().lower()
    if not SHA256_HEX_RE.match(clean):
        return f"invalid SHA-256 hex format ({digest!r})"
    if len(set(clean)) <= 4:
        return f"low-entropy placeholder SHA-256 ({clean[:16]}...)"
    if clean.endswith("00000000") or clean.endswith("00000001") or clean.startswith("00000000"):
        return f"padded placeholder SHA-256 ({clean})"

    # Check common string-derived synthetic hashes (e.g. sha256("episode_000000"))
    probes = [
        "",
        "none",
        "null",
        "default",
        "baseline",
        "checkpoint",
        "policy",
        "placeholder",
        "fake",
        "test",
    ]
    if episode is not None:
        probes.extend(
            [
                str(episode),
                f"ep{episode}",
                f"ep_{episode}",
                f"episode_{episode}",
                f"episode_{episode:06d}",
                f"policy_{episode}",
                f"policy_{episode:06d}",
                f"track2_ep{episode}",
                f"track3_ep{episode}",
            ]
        )
    for probe in probes:
        if hashlib.sha256(probe.encode("utf-8")).hexdigest() == clean:
            return f"synthetic hash derived from literal string {probe!r}"
    return None


def _verify_commit_provenance(
    commit_url: str,
    *,
    repo_root: Path | None = None,
    require_clean_git: bool = False,
    require_head_match: bool = False,
) -> tuple[list[str], list[str]]:
    """Validate `code_commit_url` under Section 2.9.a of the V2D Challenge rules."""
    failures: list[str] = []
    warnings: list[str] = []

    match = CODE_COMMIT_RE.match(commit_url.strip())
    if not match:
        failures.append(
            f"INVALID_COMMIT_URL: `code_commit_url` ({commit_url!r}) must match "
            "`https://github.com/<owner>/<repo>/commit/<sha>`."
        )
        return failures, warnings

    owner = match.group("owner").lower()
    repo = match.group("repo").lower()
    if repo.endswith(".git"):
        repo = repo[:-4]
    slug = f"{owner}/{repo}"
    sha = match.group("sha").lower()

    if slug in PROHIBITED_UPSTREAM_REPOS:
        failures.append(
            f"PROHIBITED_UPSTREAM_COMMIT_URL: `code_commit_url` points to upstream or "
            f"placeholder repository `{slug}` ({commit_url}). Section 2.9.a requires a "
            "commit in YOUR team's repository containing the code that produced the submission."
        )

    if sha in PROHIBITED_COMMIT_SHAS or any(sha.startswith(p) for p in PROHIBITED_COMMIT_SHAS):
        failures.append(
            f"PROHIBITED_PLACEHOLDER_COMMIT_SHA: Commit SHA `{sha}` is a known upstream or "
            "template placeholder hash."
        )

    if len(sha) >= 16 and (
        len(set(sha)) <= 4
        or sha.endswith("00000001")
        or sha.endswith("00000000")
        or re.search(r"(.)\1{9,}", sha)
    ):
        failures.append(
            f"SYNTHETIC_COMMIT_SHA: Commit SHA `{sha}` exhibits a synthetic or padded-UUID "
            "pattern. Provide the real `git rev-parse HEAD` commit SHA."
        )

    if repo_root is not None and (repo_root / ".git").exists():
        try:
            cat = subprocess.run(
                ["git", "-C", str(repo_root), "cat-file", "-t", sha],
                capture_output=True,
                text=True,
                check=False,
            )
            if cat.returncode != 0 or cat.stdout.strip() != "commit":
                failures.append(
                    f"UNVERIFIED_GIT_COMMIT: Commit `{sha}` does not exist in `{repo_root}`."
                )
            else:
                head = subprocess.run(
                    ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip().lower()
                if require_head_match and not head.startswith(sha) and not sha.startswith(head):
                    failures.append(
                        f"COMMIT_HEAD_MISMATCH: Submission records commit `{sha}`, but "
                        f"checked-out `HEAD` is `{head}`."
                    )
            if require_clean_git:
                status = subprocess.run(
                    ["git", "-C", str(repo_root), "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if status.stdout.strip():
                    failures.append(
                        "DIRTY_GIT_WORKTREE: Repository has uncommitted modifications. "
                        "Commit and push all producing code before generating a submission."
                    )
        except OSError as exc:
            warnings.append(f"Could not run git verification in {repo_root}: {exc}")

    return failures, warnings


def _audit_meshes(
    meshes: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    reference_meshes: dict[str, np.ndarray] | None = None,
) -> tuple[list[str], list[str]]:
    """Screen decoded meshes for degeneracies, reference scan copying, and `1e-7 m` noise evasion."""
    failures: list[str] = []
    warnings: list[str] = []

    active_by_id: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for label, v_raw, f_raw in meshes:
        if not np.all(np.isfinite(v_raw)):
            failures.append(f"NON_FINITE_MESH_VERTICES: {label} has NaN or infinite vertices.")
            continue
        surf_v = _active_surface_vertices(v_raw, f_raw)
        if len(surf_v) == 0:
            failures.append(f"ZERO_AREA_MESH: {label} has no positive-area triangles.")
            continue
        if len(surf_v) < 32:
            warnings.append(
                f"LOW_DETAIL_MESH: {label} has only {len(surf_v)} active surface vertices."
            )
        digest = hashlib.sha256(
            np.asarray(v_raw, dtype="<f8").tobytes() + np.asarray(f_raw, dtype="<i8").tobytes()
        ).hexdigest()
        active_by_id.append((label, np.asarray(v_raw, dtype=np.float64), surf_v, digest))

        if reference_meshes:
            best_dist = float("inf")
            best_ref = ""
            for ref_name, ref_v in reference_meshes.items():
                dist = _symmetric_chamfer_m(surf_v, ref_v)
                if dist < best_dist:
                    best_dist = dist
                    best_ref = ref_name
            if best_dist < 1e-3:
                failures.append(
                    f"PROHIBITED_REFERENCE_MESH_COPY: {label} is a near-duplicate of "
                    f"prohibited scan `{best_ref}` (mean symmetric Chamfer = {best_dist:.3e} m "
                    "< 1.0e-3 m)."
                )

    # Pairwise check for microscopic noise evasion (0 < max_diff < 1e-4 m)
    for i in range(len(active_by_id)):
        label_i, vi_raw, vi_surf, dig_i = active_by_id[i]
        for j in range(i + 1, len(active_by_id)):
            label_j, vj_raw, vj_surf, dig_j = active_by_id[j]
            if dig_i == dig_j:
                continue
            if vi_raw.shape == vj_raw.shape:
                vi_c = vi_raw - vi_raw.mean(axis=0, keepdims=True)
                vj_c = vj_raw - vj_raw.mean(axis=0, keepdims=True)
                max_diff = float(np.max(np.abs(vi_c - vj_c)))
                if 0.0 < max_diff < 1e-4:
                    failures.append(
                        f"MICROSCOPIC_MESH_NOISE_EVASION: {label_i} and {label_j} differ by "
                        f"only {max_diff:.3e} m (< 1.0e-4 m). Adding microscopic noise to "
                        "bypass duplicate-mesh detection is prohibited."
                    )
                    break
            elif vi_surf.shape == vj_surf.shape and len(vi_surf) >= 32:
                vi_c = vi_surf - vi_surf.mean(axis=0, keepdims=True)
                vj_c = vj_surf - vj_surf.mean(axis=0, keepdims=True)
                max_diff = float(np.max(np.abs(vi_c - vj_c)))
                if 0.0 < max_diff < 1e-4:
                    failures.append(
                        f"MICROSCOPIC_MESH_NOISE_EVASION: Active surfaces of {label_i} and "
                        f"{label_j} differ by only {max_diff:.3e} m (< 1.0e-4 m)."
                    )
                    break

    return failures, warnings


def audit_submission_artifact(
    submission_path: Path,
    track: str,
    *,
    policy: dict[str, Any] | None = None,
    allowed_tracks_override: list[str] | None = None,
    checkpoint_dir: Path | None = None,
    tier1_reference_dir: Path | None = None,
    reference_meshes: dict[str, np.ndarray] | None = None,
    repo_root: Path | None = None,
    require_clean_git: bool = False,
    require_head_match: bool = False,
    derived_cd_o_path: Path | None = None,
) -> ComplianceReport:
    """Run offline schema, provenance, leakage, and anti-reward-hacking checks on a candidate file."""
    if not submission_path.exists():
        raise ComplianceError(f"Submission file does not exist: {submission_path}")

    file_sha = sha256_file(submission_path)
    report = ComplianceReport(
        track=track,
        submission_path=str(submission_path),
        submission_sha256=file_sha,
        rows=0,
        episodes=0,
    )

    _, scope_errors = check_track_scope(
        track,
        policy=policy,
        allowed_tracks_override=allowed_tracks_override,
    )
    report.failures.extend(scope_errors)
    if scope_errors:
        return report
    report.checks_passed.append("track_scope_allowed")

    # 1. Track 1 Reconstruction Parquet Audit
    if track == "track_1":
        if submission_path.suffix.lower() != ".parquet":
            report.failures.append("INVALID_FORMAT: `track_1` requires a `.parquet` file.")
            return report
        try:
            df = pd.read_parquet(submission_path)
        except Exception as exc:  # noqa: BLE001
            report.failures.append(f"CORRUPT_PARQUET: Could not read {submission_path} ({exc}).")
            return report

        report.rows = len(df)
        expected_cols = ["row_id", "x", "y", "z", "code_commit_url"]
        if list(df.columns) != expected_cols:
            report.failures.append(
                f"INVALID_COLUMNS: Expected {expected_cols}, got {list(df.columns)}."
            )
            return report

        coords = df[["x", "y", "z"]].to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.all(np.isfinite(coords)):
            report.failures.append("NON_FINITE_COORDINATES: Submission contains NaN or Inf values.")

        commits = df["code_commit_url"].dropna().astype(str).str.strip().unique()
        if len(commits) != 1:
            report.failures.append(
                f"NON_UNIFORM_COMMIT_URL: Expected exactly 1 unique `code_commit_url`, got {len(commits)}."
            )
        else:
            report.code_commit_url = str(commits[0])
            c_fail, c_warn = _verify_commit_provenance(
                report.code_commit_url,
                repo_root=repo_root,
                require_clean_git=require_clean_git,
                require_head_match=require_head_match,
            )
            report.failures.extend(c_fail)
            report.warnings.extend(c_warn)
            if not c_fail:
                report.checks_passed.append("commit_provenance_valid")

        parts = df["row_id"].astype(str).str.split("_", expand=True)
        if parts.shape[1] != 5 or not (parts[0] == "t1").all():
            report.failures.append("INVALID_ROW_IDS: Track 1 row_ids must have format `t1_<ep>_<frame>_<array>_<pt>`.")
            return report

        ep_col = parts[1].astype(int)
        frame_col = parts[2].astype(int)
        arr_col = parts[3].astype(int)
        pt_col = parts[4].astype(int)
        episodes = sorted(ep_col.unique().tolist())
        report.episodes = len(episodes)

        # Extract per-episode meshes (frame=999999, array=5 vertices, array=6 faces) and MHR shape (array=1)
        work = pd.DataFrame(
            {
                "ep": ep_col,
                "frame": frame_col,
                "arr": arr_col,
                "pt": pt_col,
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
            }
        )
        const_rows = work[work["frame"] == 999999]
        dyn_rows = work[work["frame"] != 999999]

        t1_meshes: list[tuple[str, np.ndarray, np.ndarray]] = []
        t1_shapes: dict[int, np.ndarray] = {}
        for ep in episodes:
            ep_const = const_rows[const_rows["ep"] == ep]
            v_df = ep_const[ep_const["arr"] == 5].sort_values("pt")
            f_df = ep_const[ep_const["arr"] == 6].sort_values("pt")
            sh_df = ep_const[ep_const["arr"] == 1].sort_values("pt")
            if len(v_df) > 0 and len(f_df) > 0:
                v_arr = v_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
                f_arr = np.rint(f_df[["x", "y", "z"]].to_numpy(dtype=np.float64)).astype(np.int64)
                t1_meshes.append((f"track_1/ep_{ep:06d}", v_arr, f_arr))
            if len(sh_df) == 15:
                t1_shapes[ep] = sh_df[["x", "y", "z"]].to_numpy(dtype=np.float64).reshape(-1)[:45]

        m_fail, m_warn = _audit_meshes(t1_meshes, reference_meshes=reference_meshes)
        report.failures.extend(m_fail)
        report.warnings.extend(m_warn)
        if not m_fail:
            report.checks_passed.append("mesh_integrity_and_noise_screen")

        # Check Track 1 -> Track 2 identity_coeffs leakage if tier1_reference_dir provided
        if tier1_reference_dir is not None and tier1_reference_dir.exists() and t1_shapes:
            t2_ic_list: list[tuple[str, np.ndarray]] = []
            for pq_path in sorted(tier1_reference_dir.rglob("episode_*.parquet")):
                try:
                    t2_df = pd.read_parquet(pq_path, columns=["observation.human.identity_coeffs"])
                    ic = np.asarray(t2_df.iloc[0]["observation.human.identity_coeffs"], dtype=np.float64)[:45]
                    t2_ic_list.append((pq_path.name, ic))
                except Exception:  # noqa: BLE001
                    continue
            for ep, sh in t1_shapes.items():
                for t2_name, t2_ic in t2_ic_list:
                    if float(np.max(np.abs(sh - t2_ic))) < 1e-4:
                        report.failures.append(
                            f"CROSS_TRACK_IDENTITY_LEAKAGE: Track 1 episode {ep} `mhr_shape` "
                            f"copies Track 2 `identity_coeffs` from `{t2_name}` "
                            f"(L_inf diff < 1e-4)."
                        )
                        break

        # Check Track 1 degenerate motion / metric gaming (frozen root rotation, frozen object rotation, static body)
        frozen_root_eps = 0
        frozen_obj_rot_eps = 0
        static_body_eps = 0
        for ep, ep_dyn in dyn_rows.groupby("ep"):
            # mhr_pose is arr == 0; pt == 1 holds pose[3:6] (root rotation), pt >= 2 holds body joints
            pose_rows = ep_dyn[ep_dyn["arr"] == 0]
            root_rot_rows = pose_rows[pose_rows["pt"] == 1].sort_values("frame")
            if len(root_rot_rows) >= 2:
                rr = root_rot_rows[["x", "y", "z"]].to_numpy(dtype=np.float64)
                rr_span = float(np.max(np.ptp(rr, axis=0)))
                if rr_span < 1e-4:
                    frozen_root_eps += 1

            body_rows = pose_rows[pose_rows["pt"] >= 2]
            if len(body_rows) >= 2:
                pt_spans = body_rows.groupby("pt")[["x", "y", "z"]].agg(lambda s: float(np.ptp(s.to_numpy())))
                if float(pt_spans.to_numpy().max()) < 1e-4:
                    static_body_eps += 1

            obj_rot_rows = ep_dyn[ep_dyn["arr"] == 3]
            if len(obj_rot_rows) >= 6:
                rot_spans = obj_rot_rows.groupby("pt")[["x", "y", "z"]].agg(
                    lambda s: float(np.ptp(s.to_numpy()))
                )
                if float(rot_spans.to_numpy().max()) < 1e-4:
                    frozen_obj_rot_eps += 1

        if episodes:
            if frozen_root_eps >= max(1, int(0.8 * len(episodes))):
                report.failures.append(
                    f"DEGENERATE_FROZEN_ROOT_ROTATION: Root rotation (`mhr_pose[3:6]`) is "
                    f"constant/zero in {frozen_root_eps}/{len(episodes)} episodes. "
                    "Metric gaming via frozen root orientation is prohibited."
                )
            if frozen_obj_rot_eps >= max(1, int(0.8 * len(episodes))):
                report.failures.append(
                    f"DEGENERATE_FROZEN_OBJECT_ROTATION: Object rotation (`object_rotation`) is "
                    f"constant in {frozen_obj_rot_eps}/{len(episodes)} episodes."
                )
            if static_body_eps >= max(1, int(0.8 * len(episodes))):
                report.failures.append(
                    f"DEGENERATE_STATIC_BODY_POSE: Articulated body joints (`mhr_pose[6:136]`) "
                    f"are static across time in {static_body_eps}/{len(episodes)} episodes."
                )
            if (
                frozen_root_eps < max(1, int(0.8 * len(episodes)))
                and frozen_obj_rot_eps < max(1, int(0.8 * len(episodes)))
                and static_body_eps < max(1, int(0.8 * len(episodes)))
            ):
                report.checks_passed.append("track1_kinematic_non_degeneracy")

        return report

    # 2. Track 2 (CSV) and Track 3 (Parquet) Object-Pose Rollout Audit
    is_track3 = track == "track_3"
    expected_ext = ".parquet" if is_track3 else ".csv"
    if submission_path.suffix.lower() != expected_ext:
        report.failures.append(f"INVALID_FORMAT: `{track}` requires a `{expected_ext}` submission.")
        return report

    try:
        df = pd.read_parquet(submission_path) if is_track3 else pd.read_csv(submission_path)
    except Exception as exc:  # noqa: BLE001
        report.failures.append(f"CORRUPT_SUBMISSION: Could not read {submission_path} ({exc}).")
        return report

    report.rows = len(df)
    required_cols = ["row_id", *POSE_COLUMNS, *PROVENANCE_COLUMNS]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        report.failures.append(f"MISSING_COLUMNS: Missing required columns {missing_cols}.")
        return report

    if df["row_id"].duplicated().any():
        report.failures.append("DUPLICATE_ROW_IDS: Submission contains duplicate `row_id` entries.")

    expected_prefix = "track_3/" if is_track3 else "track_2/"
    if not df["row_id"].astype(str).str.startswith(expected_prefix).all():
        report.failures.append(
            f"WRONG_TRACK_ROW_IDS: Not all `row_id` values start with `{expected_prefix}`."
        )
        return report

    extracted = df["row_id"].astype(str).str.extract(r"^track_[23]/e(\d+)/w(\d+)/f(\d+)/o(\d+)$")
    if extracted.isna().any().any():
        report.failures.append("MALFORMED_ROW_IDS: Row IDs do not match `track_<N>/e<ep>/w<w>/f<f>/o<o>`.")
        return report

    ep_series = extracted[0].astype(int)
    frame_series = extracted[2].astype(int)
    slot_series = extracted[3].astype(int)
    report.episodes = int(ep_series.nunique())

    pose_arr = df[list(POSE_COLUMNS)].to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.all(np.isfinite(pose_arr)):
        report.failures.append("NON_FINITE_POSES: Pose columns contain NaN or infinite values.")
    else:
        q_norms = np.linalg.norm(pose_arr[:, 3:], axis=-1)
        if np.any(q_norms < 1e-8):
            report.failures.append("ZERO_NORM_QUATERNION: Pose columns contain a degenerate quaternion.")
        else:
            report.checks_passed.append("pose_numerics_valid")

    # Provenance columns check
    for col in PROVENANCE_COLUMNS:
        blank = df[col].isna() | df[col].astype(str).str.strip().isin({"", "nan", "None", "NaN"})
        if blank.any():
            report.failures.append(
                f"EMPTY_PROVENANCE_COLUMN: `{col}` is empty on {int(blank.sum())} row(s)."
            )

    for col in UNIFORM_PROVENANCE_COLUMNS:
        vals = df[col].dropna().astype(str).str.strip().unique()
        if len(vals) != 1:
            report.failures.append(
                f"NON_UNIFORM_PROVENANCE: `{col}` must be identical on every row (found {len(vals)})."
            )
            continue
        val = str(vals[0])
        if col == "code_commit_url":
            report.code_commit_url = val
            c_fail, c_warn = _verify_commit_provenance(
                val,
                repo_root=repo_root,
                require_clean_git=require_clean_git,
                require_head_match=require_head_match,
            )
            report.failures.extend(c_fail)
            report.warnings.extend(c_warn)
            if not c_fail:
                report.checks_passed.append("commit_provenance_valid")
        else:
            synth_reason = _is_synthetic_sha256(val)
            if synth_reason:
                report.failures.append(f"SYNTHETIC_PROVENANCE_HASH: `{col}` has {synth_reason}.")
            elif col == "eval_script_sha256" and val.lower() not in RELEASED_EVAL_SCRIPT_SHA256:
                report.warnings.append(
                    f"UNRECOGNIZED_EVAL_SCRIPT_HASH: `eval_script_sha256` ({val[:16]}...) "
                    "differs from the released kit's `evaluate_policy.py` digest."
                )

    # Per-episode checkpoint_sha256 verification
    real_checkpoint_hashes: set[str] | None = None
    if checkpoint_dir is not None and checkpoint_dir.exists():
        real_checkpoint_hashes = set()
        for p in checkpoint_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".safetensors", ".pt", ".pth", ".ckpt", ".onnx", ".pkl"):
                real_checkpoint_hashes.add(sha256_file(p).lower())

    eval_script_val = str(df["eval_script_sha256"].iloc[0]).strip().lower()
    eval_code_val = str(df["eval_code_sha256"].iloc[0]).strip().lower()

    df_ep_ckpt = pd.DataFrame({"ep": ep_series, "ckpt": df["checkpoint_sha256"].astype(str).str.strip()})
    missing_ckpt_eps: list[int] = []
    for ep, grp in df_ep_ckpt.groupby("ep"):
        uniq = grp["ckpt"].unique()
        if len(uniq) != 1:
            report.failures.append(
                f"NON_UNIFORM_EPISODE_CHECKPOINT: Episode {ep} carries {len(uniq)} distinct "
                "`checkpoint_sha256` values."
            )
            continue
        ckpt_val = str(uniq[0]).lower()
        synth = _is_synthetic_sha256(ckpt_val, episode=int(ep))
        if synth:
            report.failures.append(
                f"SYNTHETIC_CHECKPOINT_HASH: Episode {ep} `checkpoint_sha256` has {synth}."
            )
        elif ckpt_val in (eval_script_val, eval_code_val):
            report.failures.append(
                f"REUSED_SCRIPT_HASH_AS_CHECKPOINT: Episode {ep} `checkpoint_sha256` "
                "copies the evaluator script/code hash instead of a policy checkpoint."
            )
        elif real_checkpoint_hashes is not None and ckpt_val not in real_checkpoint_hashes:
            missing_ckpt_eps.append(int(ep))

    if missing_ckpt_eps:
        report.failures.append(
            f"UNVERIFIED_CHECKPOINT_FILES: {len(missing_ckpt_eps)} episode(s) "
            f"({missing_ckpt_eps[:10]}) report `checkpoint_sha256` digests that do not match "
            f"any policy weight file in `{checkpoint_dir}`."
        )
    elif not any("CHECKPOINT" in f for f in report.failures):
        report.checks_passed.append("checkpoint_provenance_valid")

    # Track 2 Tier 2 / Tier 3 cross-tier ground-truth leakage screening
    if track in ("track_2_tier2", "track_2_tier3") and tier1_reference_dir is not None and tier1_reference_dir.exists():
        df_pose = df.copy()
        df_pose["ep"] = ep_series
        df_pose["f"] = frame_series
        df_pose["slot"] = slot_series
        for ep, grp in df_pose[df_pose["slot"] == 0].groupby("ep"):
            pq_candidates = list(tier1_reference_dir.rglob(f"episode_{int(ep):06d}.parquet"))
            if not pq_candidates:
                continue
            try:
                ref_df = pd.read_parquet(pq_candidates[0], columns=["observation.object.pose"])
            except Exception:  # noqa: BLE001
                continue
            sub_sorted = grp.sort_values("f")
            fs = sub_sorted["f"].to_numpy(dtype=int)
            if len(fs) == 0 or int(fs.max()) >= len(ref_df):
                continue
            q_sub = sub_sorted[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy(
                dtype=np.float64, copy=True
            )
            p_sub = sub_sorted[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64, copy=True)
            q_ref = np.zeros_like(q_sub)
            p_ref = np.zeros_like(p_sub)
            for idx_f, f_val in enumerate(fs):
                raw_pose = np.asarray(ref_df.iloc[int(f_val)]["observation.object.pose"], dtype=np.float64).reshape(-1, 7)
                p_ref[idx_f] = raw_pose[0, :3]
                # LeRobot Tier 1 parquet stores [x, y, z, qw, qx, qy, qz]
                q_ref[idx_f] = [raw_pose[0, 4], raw_pose[0, 5], raw_pose[0, 6], raw_pose[0, 3]]
            q_sub /= np.linalg.norm(q_sub, axis=-1, keepdims=True)
            q_ref /= np.linalg.norm(q_ref, axis=-1, keepdims=True)
            mean_dot = float(np.mean(np.abs(np.sum(q_sub * q_ref, axis=-1))))
            mean_pos_diff = float(np.mean(np.linalg.norm(p_sub - p_ref, axis=-1)))
            if mean_dot > 0.99995 or mean_pos_diff < 1e-4:
                report.failures.append(
                    f"CROSS_TIER_GROUND_TRUTH_LEAKAGE: `{track}` episode {ep} copies clean "
                    f"Tier 1 ground-truth trajectory (mean |quat_dot|={mean_dot:.7f}, "
                    f"mean pos_diff={mean_pos_diff:.2e} m), violating Rule 2.8.c."
                )
                break
        if not any("CROSS_TIER_GROUND_TRUTH_LEAKAGE" in f for f in report.failures):
            report.checks_passed.append("cross_tier_isolation_verified")

    # Track 3 embedded object_mesh_b64 & derived .cd-o.parquet bundle verification
    if is_track3:
        if "object_mesh_b64" not in df.columns:
            report.failures.append(
                "MISSING_OBJECT_MESH_B64: Track 3 combined Parquet must include `object_mesh_b64`."
            )
            return report

        non_empty = df[df["object_mesh_b64"].notna() & (df["object_mesh_b64"].astype(str).str.strip() != "")]
        if len(non_empty) == 0:
            report.failures.append(
                "EMPTY_OBJECT_MESH_B64: Track 3 submission contains no packed object meshes."
            )
            return report

        t3_meshes: list[tuple[str, np.ndarray, np.ndarray]] = []
        for _, row in non_empty.iterrows():
            rid = str(row["row_id"])
            try:
                v_arr, f_arr = _decode_track3_mesh_b64(str(row["object_mesh_b64"]).strip())
                t3_meshes.append((rid, v_arr, f_arr))
            except Exception as exc:  # noqa: BLE001
                report.failures.append(f"CORRUPT_OBJECT_MESH_B64: Could not decode mesh at {rid} ({exc}).")

        m_fail, m_warn = _audit_meshes(t3_meshes, reference_meshes=reference_meshes)
        report.failures.extend(m_fail)
        report.warnings.extend(m_warn)
        if not m_fail:
            report.checks_passed.append("track3_mesh_integrity_and_noise_screen")

        # If a derived .cd-o.parquet exists or was passed, verify commit & vertex parity
        cd_o_candidate = derived_cd_o_path or submission_path.with_name(submission_path.stem + ".cd-o.parquet")
        if cd_o_candidate.exists():
            report.derived_cd_o_sha256 = sha256_file(cd_o_candidate)
            try:
                cd_df = pd.read_parquet(cd_o_candidate)
                cd_commits = cd_df["code_commit_url"].dropna().astype(str).str.strip().unique()
                if len(cd_commits) != 1 or str(cd_commits[0]) != report.code_commit_url:
                    report.failures.append(
                        "TRACK3_CDO_BUNDLE_MISMATCH: `.cd-o.parquet` has a different "
                        "`code_commit_url` than the parent Track 3 pose Parquet."
                    )
                else:
                    report.checks_passed.append("track3_cdo_bundle_coupled")
            except Exception as exc:  # noqa: BLE001
                report.failures.append(f"CORRUPT_CDO_PARQUET: Could not read {cd_o_candidate} ({exc}).")

    return report


def _cmd_audit(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    ref_meshes: dict[str, np.ndarray] = {}
    if args.prohibited_mesh_dir and args.prohibited_mesh_dir.exists():
        try:
            import trimesh  # type: ignore[import-untyped]

            for p in sorted(args.prohibited_mesh_dir.rglob("*.glb")):
                try:
                    m = trimesh.load(p, force="mesh")
                    v = np.asarray(m.vertices, dtype=np.float64)
                    if len(v) > 0:
                        ref_meshes[f"{p.parent.name}/{p.name}"] = v
                except Exception:  # noqa: BLE001
                    continue
        except ImportError:
            pass

    report = audit_submission_artifact(
        args.submission,
        args.track,
        policy=policy,
        allowed_tracks_override=args.allowed_tracks,
        checkpoint_dir=args.checkpoint_dir,
        tier1_reference_dir=args.tier1_reference_dir,
        reference_meshes=ref_meshes or None,
        repo_root=args.repo_root,
        require_clean_git=args.require_clean_git,
        require_head_match=args.require_head_match,
    )

    if args.check_quota:
        receipt_dirs = [args.submission.parent, Path(".")]
        if args.receipt_dir:
            receipt_dirs.append(args.receipt_dir)
        report.quota = check_weekly_quota(
            args.track,
            submission_sha256=report.submission_sha256,
            receipt_dirs=receipt_dirs,
            ledger_path=args.ledger,
        )

    if args.receipt_out:
        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_out.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"V2D Compliance Gate: {report.submission_path} (track={report.track})")
        print(f"  SHA-256:  {report.submission_sha256}")
        print(f"  Rows:     {report.rows:,} across {report.episodes} episode(s)")
        if report.code_commit_url:
            print(f"  Commit:   {report.code_commit_url}")
        for passed in report.checks_passed:
            print(f"  [  OK  ] {passed}")
        for warn in report.warnings:
            print(f"  [ WARN ] {warn}")
        for fail in report.failures:
            print(f"  [ FAIL ] {fail}")
        if report.quota is not None:
            q = report.quota
            print(
                f"  [QUOTA ] {q.weekly_used_max}/{q.max_weekly_allowed} used in rolling 7d window "
                f"({q.weekly_remaining} remaining)"
            )
            for reason in q.blocking_reasons:
                print(f"  [ FAIL ] {reason}")

    return 0 if report.ok else 1


def _cmd_check_quota(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    _, scope_errors = check_track_scope(
        args.track,
        policy=policy,
        allowed_tracks_override=args.allowed_tracks,
    )
    if scope_errors:
        for err in scope_errors:
            print(f"[ FAIL ] {err}", file=sys.stderr)
        return 1

    receipt_dirs = [Path(".")]
    if args.receipt_dir:
        receipt_dirs.append(args.receipt_dir)
    status = check_weekly_quota(
        args.track,
        submission_sha256=args.file_sha256,
        receipt_dirs=receipt_dirs,
        ledger_path=args.ledger,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "ok": status.ok,
                    "track": status.track,
                    "now_utc": status.now_utc,
                    "in_unlimited_window": status.in_unlimited_window,
                    "after_freeze": status.after_freeze,
                    "max_weekly_allowed": status.max_weekly_allowed,
                    "weekly_used_max": status.weekly_used_max,
                    "weekly_remaining": status.weekly_remaining,
                    "weekly_used_by_competition": status.weekly_used_by_competition,
                    "next_slot_opens_utc": status.next_slot_opens_utc,
                    "blocking_reasons": status.blocking_reasons,
                },
                indent=2,
            )
        )
    else:
        print(
            f"Track {status.track}: {status.weekly_used_max}/{status.max_weekly_allowed} "
            f"weekly submissions used ({status.weekly_remaining} remaining)."
        )
        for reason in status.blocking_reasons:
            print(f"  [ FAIL ] {reason}")
    return 0 if status.ok else 2


def _cmd_init_policy(args: argparse.Namespace) -> int:
    tracks = args.tracks or list(CANONICAL_TRACK_COMPETITIONS.keys())
    unknown = [t for t in tracks if t not in CANONICAL_TRACK_COMPETITIONS]
    if unknown:
        print(f"Unknown track(s): {unknown}. Valid: {sorted(CANONICAL_TRACK_COMPETITIONS)}", file=sys.stderr)
        return 1
    payload = {
        "schema_version": 1,
        "allowed_tracks": tracks,
        "max_weekly_submissions_per_track": DEFAULT_MAX_WEEKLY_SUBMISSIONS,
        "require_atomic_track_bundle": True,
        "block_agent_live_uploads": True,
        "require_clean_git_worktree": True,
        "freeze_utc": DEFAULT_FREEZE_UTC.isoformat(),
        "unlimited_window_start_utc": DEFAULT_UNLIMITED_START_UTC.isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote V2D team compliance policy to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit_p = sub.add_parser("audit", help="Audit a packed submission file against all V2D rules")
    audit_p.add_argument("submission", type=Path, help="Packed CSV or Parquet submission file")
    audit_p.add_argument("--track", required=True, help="Target track key (e.g. track_1, track_2_tier1, track_3)")
    audit_p.add_argument("--policy", type=Path, default=None, help="Path to .v2d_policy.json")
    audit_p.add_argument("--allowed-tracks", nargs="+", default=None, help="Override allowed tracks")
    audit_p.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory of policy checkpoints to verify SHA-256")
    audit_p.add_argument("--tier1-reference-dir", type=Path, default=None, help="Track 2 Tier 1 reference dir for cross-tier leakage screening")
    audit_p.add_argument("--prohibited-mesh-dir", type=Path, default=None, help="Reference .glb mesh dir for near-duplicate screening")
    audit_p.add_argument("--repo-root", type=Path, default=None, help="Git repository root to verify code_commit_url")
    audit_p.add_argument("--require-clean-git", action="store_true", help="Fail if git working tree has uncommitted changes")
    audit_p.add_argument("--require-head-match", action="store_true", help="Fail if code_commit_url does not match git HEAD")
    audit_p.add_argument("--check-quota", action="store_true", help="Also check local rolling 7-day submission quota")
    audit_p.add_argument("--receipt-dir", type=Path, default=None, help="Directory containing *.kaggle-receipt.json files")
    audit_p.add_argument("--ledger", type=Path, default=Path(".v2d_submission_ledger.json"))
    audit_p.add_argument("--receipt-out", type=Path, default=None, help="Write signed compliance report JSON")
    audit_p.add_argument("--json", action="store_true", help="Print report as JSON")
    audit_p.set_defaults(func=_cmd_audit)

    quota_p = sub.add_parser("check-quota", help="Check rolling 7-day (5/week) submission quota for a track")
    quota_p.add_argument("--track", required=True)
    quota_p.add_argument("--file-sha256", default=None)
    quota_p.add_argument("--policy", type=Path, default=None)
    quota_p.add_argument("--allowed-tracks", nargs="+", default=None)
    quota_p.add_argument("--receipt-dir", type=Path, default=None)
    quota_p.add_argument("--ledger", type=Path, default=Path(".v2d_submission_ledger.json"))
    quota_p.add_argument("--json", action="store_true")
    quota_p.set_defaults(func=_cmd_check_quota)

    init_p = sub.add_parser("init-policy", help="Write a `.v2d_policy.json` file locking allowed tracks and quota")
    init_p.add_argument("--tracks", nargs="+", default=None, help="Tracks your team is entering")
    init_p.add_argument("-o", "--out", type=Path, default=Path(".v2d_policy.json"))
    init_p.set_defaults(func=_cmd_init_policy)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ComplianceError as exc:
        print(f"COMPLIANCE GATE FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
