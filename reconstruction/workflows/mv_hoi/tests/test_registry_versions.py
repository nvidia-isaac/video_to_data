import json
import subprocess
import sys
from pathlib import Path

import pytest


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import db, registry_versions


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["ngc"], returncode=returncode, stdout=stdout, stderr=stderr,
    )


def test_list_repository_versions_parses_and_sorts_strict_semvers(monkeypatch):
    payload = [
        {"tag": "latest"},
        {"tag": "1.10.0"},
        {"image": "nvstaging/isaac-amr/example:1.9.0"},
        {"tag": "1.10"},
        {"tag": "v2.0.0"},
    ]
    calls = []

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _completed(stdout=json.dumps(payload))

    monkeypatch.setattr(registry_versions.subprocess, "run", _run)

    assert registry_versions.list_repository_versions("example") == [
        "1.9.0",
        "1.10.0",
    ]
    assert calls == [
        (
            [
                "ngc",
                "registry",
                "image",
                "list",
                "--format_type",
                "json",
                "nvstaging/isaac-amr/example:*",
            ],
            {"capture_output": True, "text": True},
        )
    ]


def test_list_repository_versions_accepts_empty_repository(monkeypatch):
    monkeypatch.setattr(
        registry_versions.subprocess,
        "run",
        lambda *_args, **_kwargs: _completed(stdout="[]"),
    )

    assert registry_versions.list_repository_versions("example") == []


def test_list_repository_versions_reports_malformed_json(monkeypatch):
    monkeypatch.setattr(
        registry_versions.subprocess,
        "run",
        lambda *_args, **_kwargs: _completed(stdout="not-json"),
    )

    with pytest.raises(registry_versions.RegistryVersionError, match="malformed JSON"):
        registry_versions.list_repository_versions("example")


def test_list_repository_versions_reports_auth_failure(monkeypatch):
    monkeypatch.setattr(
        registry_versions.subprocess,
        "run",
        lambda *_args, **_kwargs: _completed(
            stderr="Authentication required", returncode=1,
        ),
    )

    with pytest.raises(registry_versions.RegistryVersionError) as exc_info:
        registry_versions.list_repository_versions("example")

    assert "Authentication required" in str(exc_info.value)
    assert "authentication configuration" in str(exc_info.value)


def test_resolve_submission_version_uses_latest_numeric_semver(monkeypatch):
    versions = {
        repository: ["1.9.0", "1.10.0"]
        for repository in registry_versions.managed_repositories()
    }
    monkeypatch.setattr(
        registry_versions,
        "list_repository_versions",
        lambda repository: versions[repository],
    )

    assert registry_versions.resolve_submission_version() == "1.10.0"


def test_resolve_submission_version_rejects_incomplete_release(monkeypatch):
    versions = {
        repository: ["1.10.0"]
        for repository in registry_versions.managed_repositories()
    }
    versions["mv_hoi_sam2"] = ["1.9.0"]
    monkeypatch.setattr(
        registry_versions,
        "list_repository_versions",
        lambda repository: versions[repository],
    )

    with pytest.raises(
        registry_versions.RegistryVersionError,
        match=r"1\.10\.0.*mv_hoi_sam2",
    ):
        registry_versions.resolve_submission_version()


def test_requested_submission_version_must_exist_on_every_image(monkeypatch):
    versions = {
        repository: ["2.0.0"]
        for repository in registry_versions.managed_repositories()
    }
    versions["mv_hoi_detectron2"] = []
    monkeypatch.setattr(
        registry_versions,
        "list_repository_versions",
        lambda repository: versions[repository],
    )

    with pytest.raises(
        registry_versions.RegistryVersionError,
        match=r"2\.0\.0.*mv_hoi_detectron2",
    ):
        registry_versions.resolve_submission_version("2.0.0")


def test_resolve_push_version_starts_at_initial_version(monkeypatch):
    monkeypatch.setattr(registry_versions, "latest_registry_version", lambda: None)

    assert registry_versions.resolve_push_version() == (None, "0.1.0")


def test_resolve_push_version_increments_remote_patch(monkeypatch):
    monkeypatch.setattr(
        registry_versions, "latest_registry_version", lambda: "1.10.9",
    )

    assert registry_versions.resolve_push_version() == ("1.10.9", "1.10.10")
    assert registry_versions.resolve_push_version("2.0.0") == ("1.10.9", "2.0.0")
    with pytest.raises(registry_versions.RegistryVersionError, match="must be greater"):
        registry_versions.resolve_push_version("1.10.9")


def test_resolve_push_version_rejects_invalid_semver_before_query(monkeypatch):
    monkeypatch.setattr(
        registry_versions,
        "latest_registry_version",
        lambda: pytest.fail("invalid semver should fail before querying NGC"),
    )

    with pytest.raises(registry_versions.RegistryVersionError, match="Invalid semver"):
        registry_versions.resolve_push_version("release-2")


def test_ensure_version_cached_is_idempotent_and_allows_remote_history(tmp_path):
    db_path = str(tmp_path / "processing.db")
    db.init_db(db_path)
    db.ensure_version_cached("2.0.0", "first", db_path=db_path)
    db.ensure_version_cached("1.9.0", "older remote release", db_path=db_path)
    db.ensure_version_cached("2.0.0", "replacement", db_path=db_path)

    conn = db.get_connection(db_path)
    rows = conn.execute(
        "SELECT version, message FROM pipeline_versions ORDER BY version"
    ).fetchall()
    conn.close()

    assert [(row["version"], row["message"]) for row in rows] == [
        ("1.9.0", "older remote release"),
        ("2.0.0", "first"),
    ]
