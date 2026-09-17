"""Portable host state and mutation-authority policy."""

from __future__ import annotations

import os
from pathlib import Path


MV_HOI_DIR = Path(__file__).resolve().parent.parent


def state_dir() -> Path:
    configured = os.environ.get("MV_HOI_STATE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return (root / "v2d-mv-hoi").resolve()


def state_path(kind: str) -> Path:
    if kind not in {"db", "generated", "manifests", "logs", "locks"}:
        raise ValueError(f"Unsupported MV-HOI state directory: {kind}")
    return state_dir() / kind


def env_file() -> Path:
    return Path(
        os.environ.get("MV_HOI_ENV_FILE", state_dir() / "host.env")
    ).expanduser().resolve()


def config_path() -> Path:
    return Path(
        os.environ.get("MV_HOI_CONFIG_PATH", MV_HOI_DIR / "config.yaml")
    ).expanduser().resolve()


def generated_dir() -> Path:
    return state_path("generated")


def submission_hold_path() -> Path:
    """Return the host-local sentinel that blocks all external mutations."""
    return Path(
        os.environ.get(
            "MV_HOI_SUBMISSION_HOLD_FILE",
            state_path("locks") / "SUBMISSION_HOLD",
        )
    ).expanduser().resolve()


def submission_hold_active() -> bool:
    return submission_hold_path().exists()


def submission_allowed() -> bool:
    return (
        os.environ.get("MV_HOI_ALLOW_SUBMIT") == "1"
        and not submission_hold_active()
    )


def require_submit_authority(action: str, *, dry_run: bool = False) -> None:
    """Reject external mutations unless this host was explicitly enabled."""
    if dry_run:
        return
    if submission_hold_active():
        raise RuntimeError(
            f"Refusing to {action}: submission hold is active at "
            f"{submission_hold_path()}"
        )
    if not submission_allowed():
        raise RuntimeError(
            f"Refusing to {action}: set MV_HOI_ALLOW_SUBMIT=1 only on the "
            "single authorized submit host"
        )
