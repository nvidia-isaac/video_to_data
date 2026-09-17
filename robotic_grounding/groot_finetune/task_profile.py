# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task profiles for language conditioning and closed-loop evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
LIFT_HOLD_EVALUATOR = "lift_hold"


@dataclass(frozen=True)
class TargetObject:
    """Select the evaluated scene object without relying on array position."""

    selector: str
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate the selector and optional object name."""
        if self.selector not in {"primary", "name"}:
            raise ValueError("target selector must be 'primary' or 'name'")
        if self.selector == "name" and not self.name:
            raise ValueError("target selector 'name' requires a non-empty name")
        if self.selector == "primary" and self.name is not None:
            raise ValueError("target selector 'primary' must not define a name")

    def resolve(self, object_names: tuple[str, ...]) -> int:
        """Resolve the selected object against an exact ordered name tuple."""
        if not object_names:
            raise ValueError("task profile cannot select from an empty object list")
        if len(set(object_names)) != len(object_names):
            raise ValueError("scene object names must be unique")
        if self.selector == "primary":
            return 0
        assert self.name is not None
        if self.name not in object_names:
            raise ValueError(
                f"target object {self.name!r} is not present; available={list(object_names)}"
            )
        return object_names.index(self.name)


@dataclass(frozen=True)
class LiftHoldEvaluator:
    """Configuration for the released lift-and-hold evaluator."""

    lift_threshold_m: float
    hold_threshold_m: float
    min_hold_steps: int
    evaluator_id: str = LIFT_HOLD_EVALUATOR

    def __post_init__(self) -> None:
        """Validate lift-and-hold thresholds."""
        if self.evaluator_id != LIFT_HOLD_EVALUATOR:
            raise ValueError(f"unsupported evaluator id: {self.evaluator_id!r}")
        if self.lift_threshold_m <= 0.0:
            raise ValueError("lift_threshold_m must be positive")
        if not 0.0 < self.hold_threshold_m <= self.lift_threshold_m:
            raise ValueError(
                "hold_threshold_m must be positive and no greater than lift_threshold_m"
            )
        if self.min_hold_steps <= 0:
            raise ValueError("min_hold_steps must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible evaluator representation."""
        return {
            "id": self.evaluator_id,
            "lift_threshold_m": self.lift_threshold_m,
            "hold_threshold_m": self.hold_threshold_m,
            "min_hold_steps": self.min_hold_steps,
        }


@dataclass(frozen=True)
class TaskProfile:
    """Task semantics kept separate from embodiment and environment configuration."""

    task_id: str
    object_prompt: str
    instruction: str
    target_object: TargetObject
    evaluator: LiftHoldEvaluator
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate required task semantics."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version={self.schema_version!r}; expected {SCHEMA_VERSION}"
            )
        for name, value in (
            ("task_id", self.task_id),
            ("object_prompt", self.object_prompt),
            ("instruction", self.instruction),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible profile representation."""
        value = asdict(self)
        value["evaluator"] = self.evaluator.as_dict()
        return value

    @property
    def sha256(self) -> str:
        """Return a stable semantic hash of the canonical representation."""
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def task_profile_from_dict(value: Mapping[str, Any]) -> TaskProfile:
    """Validate and construct a task profile from serialized data."""
    try:
        target = value["target_object"]
        evaluator = value["evaluator"]
        if not isinstance(target, Mapping) or not isinstance(evaluator, Mapping):
            raise TypeError("target_object and evaluator must be objects")
        return TaskProfile(
            schema_version=int(value["schema_version"]),
            task_id=str(value["task_id"]),
            object_prompt=str(value["object_prompt"]),
            instruction=str(value["instruction"]),
            target_object=TargetObject(
                selector=str(target["selector"]),
                name=str(target["name"]) if target.get("name") is not None else None,
            ),
            evaluator=LiftHoldEvaluator(
                evaluator_id=str(evaluator["id"]),
                lift_threshold_m=float(evaluator["lift_threshold_m"]),
                hold_threshold_m=float(evaluator["hold_threshold_m"]),
                min_hold_steps=int(evaluator["min_hold_steps"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid task profile: {exc}") from exc


def load_task_profile(value: str | Path | Mapping[str, Any]) -> TaskProfile:
    """Load a task profile from a JSON file or decoded object."""
    if isinstance(value, Mapping):
        return task_profile_from_dict(value)
    path = Path(value)
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read task profile {path}: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"task profile must be a JSON object: {path}")
    return task_profile_from_dict(decoded)


def write_task_profile(path: Path, profile: TaskProfile) -> None:
    """Write a canonical task-profile snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
