from pathlib import Path
import sys

import pytest


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import runtime


def test_submission_hold_overrides_submit_authority(tmp_path, monkeypatch):
    hold = tmp_path / "SUBMISSION_HOLD"
    monkeypatch.setenv("MV_HOI_ALLOW_SUBMIT", "1")
    monkeypatch.setenv("MV_HOI_SUBMISSION_HOLD_FILE", str(hold))

    assert runtime.submission_allowed() is True
    runtime.require_submit_authority("submit a workflow")

    hold.touch()
    assert runtime.submission_allowed() is False
    with pytest.raises(RuntimeError, match="submission hold is active"):
        runtime.require_submit_authority("submit a workflow")


def test_dry_run_remains_available_during_submission_hold(tmp_path, monkeypatch):
    hold = tmp_path / "SUBMISSION_HOLD"
    hold.touch()
    monkeypatch.setenv("MV_HOI_ALLOW_SUBMIT", "1")
    monkeypatch.setenv("MV_HOI_SUBMISSION_HOLD_FILE", str(hold))

    runtime.require_submit_authority("dry-run a workflow", dry_run=True)
