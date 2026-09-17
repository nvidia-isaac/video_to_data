"""Resumable subprocess runner with logs and durable stage manifests."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import socket
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .config import REPO_ROOT, E2EConfig

MANIFEST_VERSION = 1
INTERRUPTED_REASON = (
    "runner exited without recording a result; its stage lease is no longer held"
)
RUNTIME_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".ini",
    ".json",
    ".lock",
    ".py",
    ".sh",
    ".toml",
    ".urdf",
    ".usd",
    ".usda",
    ".xml",
    ".yaml",
    ".yml",
}
RUNTIME_SOURCE_EXCLUDED_PARTS = {
    ".codex",
    ".e2e",
    ".git",
    ".github",
    ".venv",
    ".vscode",
    "__pycache__",
    "artifacts",
    "docs",
    "logs",
    "out",
    "test",
    "tests",
    "venv",
}


class StageError(RuntimeError):
    """Raised when a stage cannot safely run or does not complete."""


class StageLeaseHeld(StageError):
    """Raised when another runner currently owns a stage lease."""


def _is_runtime_source(path: Path) -> bool:
    if any(part in RUNTIME_SOURCE_EXCLUDED_PARTS for part in path.parts):
        return False
    if path.name.startswith("test_") or path.name.endswith("_test.py"):
        return False
    return (
        path.name in {".gitmodules", "run_e2e.sh"}
        or path.name.startswith("Dockerfile")
        or path.suffix.lower() in RUNTIME_SOURCE_SUFFIXES
    )


def _git_output(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def repository_snapshot(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Fingerprint executable repository state without coupling stages to docs/tests."""
    try:
        revision = _git_output(repo_root, "rev-parse", "HEAD").decode().strip()
        listed = _git_output(repo_root, "ls-files", "-z")
        untracked = _git_output(
            repo_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
        submodules = _git_output(repo_root, "submodule", "status", "--recursive")
        candidates = {
            Path(raw.decode("utf-8", errors="surrogateescape"))
            for raw in (*listed.split(b"\0"), *untracked.split(b"\0"))
            if raw
        }
        source_paths = sorted(path for path in candidates if _is_runtime_source(path))
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision = "unavailable"
        submodules = b""
        source_paths = sorted(
            path.relative_to(repo_root)
            for path in repo_root.rglob("*")
            if path.is_file() and _is_runtime_source(path.relative_to(repo_root))
        )

    digest = hashlib.sha256()
    digest.update(b"submodules\0")
    digest.update(submodules)
    for relative in source_paths:
        digest.update(relative.as_posix().encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        path = repo_root / relative
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        elif path.is_file():
            digest.update(b"file\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"missing\0")
    return {
        "revision": revision,
        "runtime_tree_sha256": digest.hexdigest(),
        "runtime_file_count": len(source_paths),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-." else "-" for char in value)


@dataclass(frozen=True)
class Command:
    """One subprocess invocation within a stage."""

    argv: tuple[str, ...]
    cwd: Path
    label: str = "command"
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("command argv cannot be empty")

    def rendered(self) -> str:
        prefix = ""
        if self.env:
            prefix = (
                "env "
                + " ".join(
                    f"{key}={shlex.quote(value)}"
                    for key, value in sorted(self.env.items())
                )
                + " "
            )
        return prefix + shlex.join(self.argv)

    def serializable(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "env": dict(sorted(self.env.items())),
            "label": self.label,
        }


@dataclass(frozen=True)
class Stage:
    """A resumable unit with declared inputs and required outputs."""

    name: str
    commands: tuple[Command, ...] = ()
    inputs: tuple[Path, ...] = ()
    outputs: tuple[Path, ...] = ()
    description: str = ""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    """Build a stable-enough fingerprint for resume safety."""
    resolved = path.resolve()
    if not resolved.exists():
        raise StageError(f"required input does not exist: {resolved}")
    stat = resolved.stat()
    if resolved.is_file():
        return {
            "kind": "file",
            "path": str(resolved),
            "size": stat.st_size,
            "sha256": _hash_file(resolved),
        }
    if resolved.is_dir():
        digest = hashlib.sha256()
        count = 0
        for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
            child_stat = child.stat()
            relative = child.relative_to(resolved).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(str(child_stat.st_size).encode("ascii"))
            digest.update(str(child_stat.st_mtime_ns).encode("ascii"))
            count += 1
        return {
            "kind": "directory",
            "path": str(resolved),
            "file_count": count,
            "tree_sha256": digest.hexdigest(),
        }
    return {
        "kind": "other",
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _atomic_write(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _stage_lease_path(run_root: Path, stage_name: str) -> Path:
    return run_root / ".locks" / f"{_slug(stage_name)}.lock"


@contextmanager
def _stage_lease(run_root: Path, stage_name: str) -> Iterator[Path]:
    """Hold a process-scoped lease proving that a stage runner is alive."""
    path = _stage_lease_path(run_root, stage_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StageLeaseHeld(
                f"stage {stage_name!r} is already owned by another runner"
            ) from exc
        try:
            yield path
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _new_manifest(config_path: Path) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_VERSION,
        "config": str(config_path),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "stages": {},
    }


def _read_manifest(manifest_path: Path, config_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return _new_manifest(config_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageError(f"cannot read run manifest: {exc}") from exc
    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise StageError(
            f"unsupported run manifest schema: {manifest.get('schema_version')!r}"
        )
    if manifest.get("config") != str(config_path):
        raise StageError("run manifest belongs to a different configuration")
    if not isinstance(manifest.get("stages"), dict):
        raise StageError("run manifest has invalid stages data")
    return manifest


def _write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _atomic_write(manifest_path, manifest)


def _mark_interrupted(stage: dict[str, Any]) -> None:
    interrupted_at = _utc_now()
    attempts = stage.get("attempts", [])
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            if isinstance(attempt, dict) and attempt.get("status") == "running":
                attempt["status"] = "interrupted"
                attempt["finished_at"] = interrupted_at
                attempt["interrupted_at"] = interrupted_at
                attempt["error"] = INTERRUPTED_REASON
                break
    stage["status"] = "interrupted"
    stage["interrupted_at"] = interrupted_at
    stage["last_error"] = INTERRUPTED_REASON


def reconcile_interrupted_stages(config: E2EConfig) -> dict[str, Any]:
    """Persist ``interrupted`` for running stages whose runner lease is gone."""
    manifest_path = config.run_root / "run_manifest.json"
    manifest = _read_manifest(manifest_path, config.path)
    running = [
        name
        for name, stage in manifest["stages"].items()
        if isinstance(stage, dict) and stage.get("status") == "running"
    ]
    for stage_name in running:
        try:
            with _stage_lease(config.run_root, stage_name):
                # Reload after acquiring the lease so a just-finished runner is not
                # mistaken for an interrupted one.
                manifest = _read_manifest(manifest_path, config.path)
                stage = manifest["stages"].get(stage_name)
                if isinstance(stage, dict) and stage.get("status") == "running":
                    _mark_interrupted(stage)
                    _write_manifest(manifest_path, manifest)
        except StageLeaseHeld:
            # The lease is the liveness signal; a held lease means genuinely running.
            continue
    return _read_manifest(manifest_path, config.path)


class Runner:
    """Execute stages and preserve enough provenance to resume safely."""

    def __init__(
        self,
        config: E2EConfig,
        *,
        dry_run: bool = False,
        resume: bool = True,
        force: bool = False,
        repository: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.resume = resume
        self.force = force
        self.manifest_path = config.run_root / "run_manifest.json"
        self.logs_dir = config.run_root / "logs"
        self.repository = dict(repository or repository_snapshot())

    def _load_manifest(self) -> dict[str, Any]:
        return _read_manifest(self.manifest_path, self.config.path)

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        _write_manifest(self.manifest_path, manifest)

    def _input_snapshot(self, paths: Iterable[Path]) -> dict[str, Any]:
        return {str(path.resolve()): fingerprint(path) for path in paths}

    @staticmethod
    def _outputs_exist(paths: Iterable[Path]) -> bool:
        return all(path.exists() for path in paths)

    def _signature(self, stage: Stage, inputs: Mapping[str, Any]) -> str:
        payload = {
            "name": stage.name,
            "commands": [command.serializable() for command in stage.commands],
            "inputs": inputs,
            "outputs": [str(path.resolve()) for path in stage.outputs],
            "repository_runtime_tree_sha256": self.repository["runtime_tree_sha256"],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def preview(self, stages: Sequence[Stage]) -> None:
        for stage in stages:
            print(f"[DRY-RUN] {stage.name}: {stage.description or 'stage'}")
            if not stage.commands:
                print("  (artifact checks only)")
            for command in stage.commands:
                print(f"  cd {shlex.quote(str(command.cwd))}")
                print(f"  {command.rendered()}")
            if stage.outputs:
                print("  outputs:")
                for output in stage.outputs:
                    print(f"    {output}")

    def run(self, stage: Stage) -> str:
        """Execute one stage, returning ``completed`` or ``skipped``."""
        if self.dry_run:
            self.preview((stage,))
            return "dry-run"

        self.config.run_root.mkdir(parents=True, exist_ok=True)
        with _stage_lease(self.config.run_root, stage.name) as lease_path:
            return self._run_with_lease(stage, lease_path)

    def _run_with_lease(self, stage: Stage, lease_path: Path) -> str:
        """Execute a stage while its process-liveness lease is held."""
        manifest = self._load_manifest()
        stages_data: dict[str, Any] = manifest["stages"]
        previous = stages_data.get(stage.name)
        if isinstance(previous, dict) and previous.get("status") == "running":
            _mark_interrupted(previous)
            self._write_manifest(manifest)

        inputs = self._input_snapshot(stage.inputs)
        signature = self._signature(stage, inputs)
        if previous and previous.get("status") == "completed" and not self.force:
            if previous.get("signature") != signature:
                raise StageError(
                    f"stage {stage.name!r} inputs or command changed since completion; "
                    "use a new run root or pass --force"
                )
            if self.resume and self._outputs_exist(stage.outputs):
                print(f"[SKIP] {stage.name}: completed with matching inputs")
                return "skipped"

        attempts = (
            list(previous.get("attempts", [])) if isinstance(previous, dict) else []
        )
        attempt_number = len(attempts) + 1
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{_slug(stage.name)}.attempt-{attempt_number}.log"
        attempt: dict[str, Any] = {
            "number": attempt_number,
            "started_at": _utc_now(),
            "status": "running",
            "log": str(log_path),
            "commands": [command.serializable() for command in stage.commands],
            "repository": self.repository,
            "runner_pid": os.getpid(),
            "runner_host": socket.gethostname(),
            "lease": str(lease_path),
        }
        attempts.append(attempt)
        stages_data[stage.name] = {
            "status": "running",
            "description": stage.description,
            "signature": signature,
            "inputs": inputs,
            "outputs": [str(path.resolve()) for path in stage.outputs],
            "attempts": attempts,
            "repository": self.repository,
        }
        self._write_manifest(manifest)

        print(f"[RUN] {stage.name}")
        try:
            with log_path.open("w", encoding="utf-8") as log:
                for index, command in enumerate(stage.commands, start=1):
                    if not command.cwd.is_dir():
                        raise StageError(
                            f"working directory does not exist for {command.label}: {command.cwd}"
                        )
                    header = f"[{index}/{len(stage.commands)}] {command.label}: {command.rendered()}"
                    print(f"  {header}")
                    log.write(header + "\n")
                    log.flush()
                    environment = os.environ.copy()
                    environment.update(command.env)
                    process = subprocess.Popen(
                        command.argv,
                        cwd=command.cwd,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="")
                        log.write(line)
                    process.stdout.close()
                    return_code = process.wait()
                    if return_code != 0:
                        attempt["command_index"] = index
                        attempt["exit_code"] = return_code
                        raise StageError(
                            f"stage {stage.name!r} failed in {command.label} "
                            f"with exit code {return_code}; see {log_path}"
                        )
            missing = [str(path) for path in stage.outputs if not path.exists()]
            if missing:
                raise StageError(
                    f"stage {stage.name!r} did not produce required outputs: {missing}"
                )
        except BaseException as exc:
            attempt["status"] = "failed"
            attempt["finished_at"] = _utc_now()
            attempt["error"] = str(exc)
            stages_data[stage.name]["status"] = "failed"
            stages_data[stage.name]["last_error"] = str(exc)
            self._write_manifest(manifest)
            if isinstance(exc, StageError):
                raise
            raise StageError(f"stage {stage.name!r} failed: {exc}") from exc

        attempt["status"] = "completed"
        attempt["finished_at"] = _utc_now()
        attempt["exit_code"] = 0
        stages_data[stage.name]["status"] = "completed"
        stages_data[stage.name]["completed_at"] = attempt["finished_at"]
        stages_data[stage.name].pop("last_error", None)
        self._write_manifest(manifest)
        print(f"[DONE] {stage.name}")
        return "completed"

    def run_many(self, stages: Sequence[Stage]) -> list[str]:
        if self.dry_run:
            self.preview(stages)
            return ["dry-run"] * len(stages)
        return [self.run(stage) for stage in stages]


def _checkpoint_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def discover_checkpoint(root: Path) -> Path:
    """Return the newest numbered, serveable-looking checkpoint directory."""
    if not root.is_dir():
        raise StageError(f"checkpoint root does not exist: {root}")
    candidates = [path for path in root.rglob("checkpoint-*") if path.is_dir()]
    complete = [
        path
        for path in candidates
        if (path / "config.json").is_file()
        and (
            (path / "model.safetensors.index.json").is_file()
            or any(path.glob("model*.safetensors"))
        )
    ]
    pool = complete or candidates
    if not pool:
        if (root / "config.json").is_file() and any(root.glob("model*.safetensors")):
            return root.resolve()
        raise StageError(f"no checkpoint directory found under {root}")
    return max(
        pool,
        key=lambda path: (_checkpoint_number(path), path.stat().st_mtime_ns, str(path)),
    ).resolve()
