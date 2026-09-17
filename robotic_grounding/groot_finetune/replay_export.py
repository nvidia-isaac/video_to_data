# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate joint-rollout artifacts before semantic replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import SCHEMA_VERSION, EmbodimentContract

ROLLOUT_FORMATS = {"joint_rollout", "joint_rollout_selection"}


@dataclass(frozen=True)
class ReplayEpisode:
    """One validated source trajectory ready for kinematic replay."""

    path: Path
    joint_pos: np.ndarray
    action_target: np.ndarray
    object_pose: np.ndarray
    source_success: bool


@dataclass(frozen=True)
class ReplayExport:
    """Validated artifact, selected episodes, and source-eligibility counts."""

    manifest: dict
    episodes: tuple[ReplayEpisode, ...]
    total_episode_count: int
    source_successful_episode_count: int

    @property
    def source_unsuccessful_episode_count(self) -> int:
        """Return the number of source episodes that did not reach timeout."""
        return self.total_episode_count - self.source_successful_episode_count


def _validate_manifest(
    manifest: dict,
    contract: EmbodimentContract,
    expected_object_names: tuple[str, ...],
) -> None:
    if manifest.get("format") not in ROLLOUT_FORMATS:
        raise ValueError(
            f"unsupported rollout format={manifest.get('format')!r}; expected one of {sorted(ROLLOUT_FORMATS)}"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported rollout schema_version={manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    if manifest.get("embodiment_contract") != contract.contract_id:
        raise ValueError(
            f"rollout embodiment contract mismatch: {manifest.get('embodiment_contract')!r} != {contract.contract_id!r}"
        )
    if manifest.get("embodiment_contract_sha256") != contract.sha256:
        raise ValueError("rollout embodiment contract hash does not match")
    timeout_name = manifest.get("timeout_termination")
    if (
        not isinstance(timeout_name, str)
        or timeout_name not in contract.source_terminations
    ):
        raise ValueError(
            f"rollout timeout termination is missing or outside the contract: {timeout_name!r}"
        )
    if float(manifest.get("fps", -1.0)) != float(contract.fps):
        raise ValueError(
            f"rollout fps must be {contract.fps}, got {manifest.get('fps')!r}"
        )
    names = tuple(str(name) for name in manifest.get("joint_names", ()))
    if names != contract.joint_names:
        raise ValueError("rollout joint names or order do not match the contract")
    objects = tuple(str(name) for name in manifest.get("object_names", ()))
    if objects != expected_object_names:
        raise ValueError(
            f"rollout object names or order do not match the scene: {objects} != {expected_object_names}"
        )


def load_replay_export(
    export_dir: str | Path,
    contract: EmbodimentContract,
    expected_object_names: tuple[str, ...],
    *,
    source_success_only: bool = True,
) -> ReplayExport:
    """Load a strictly contract-matched joint-rollout artifact."""
    root = Path(export_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing rollout manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid rollout manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("rollout manifest must be a JSON object")
    _validate_manifest(manifest, contract, expected_object_names)

    episode_paths = sorted(root.glob("episode_*.npz"))
    episodes: list[ReplayEpisode] = []
    source_successful_episode_count = 0
    for path in episode_paths:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "joint_pos",
                "action_target",
                "object_pose",
                "source_success",
                "termination_reasons",
            }
            missing_keys = sorted(required - set(data.files))
            if missing_keys:
                raise ValueError(f"{path.name}: missing arrays {missing_keys}")
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            action_target = np.asarray(data["action_target"], dtype=np.float32)
            object_pose = np.asarray(data["object_pose"], dtype=np.float32)
            success_array = np.asarray(data["source_success"])
            if success_array.size != 1:
                raise ValueError(
                    f"{path.name}: source_success must be scalar, got {success_array.shape}"
                )
            source_success = bool(success_array.item())
            termination_reasons = tuple(
                str(reason) for reason in np.asarray(data["termination_reasons"])
            )
            if source_success != (
                str(manifest["timeout_termination"]) in termination_reasons
            ):
                raise ValueError(
                    f"{path.name}: source_success does not match timeout termination "
                    f"{manifest['timeout_termination']!r} in {termination_reasons}"
                )
        source_successful_episode_count += int(source_success)
        if source_success_only and not source_success:
            continue
        if joint_pos.ndim != 2 or joint_pos.shape[1] != contract.action_dim:
            raise ValueError(
                f"{path.name}: joint_pos must be (T, {contract.action_dim}), got {joint_pos.shape}"
            )
        if action_target.shape != joint_pos.shape:
            raise ValueError(
                f"{path.name}: action_target {action_target.shape} != joint_pos {joint_pos.shape}"
            )
        expected_object_shape = (
            joint_pos.shape[0],
            len(expected_object_names),
            7,
        )
        if object_pose.shape != expected_object_shape:
            raise ValueError(
                f"{path.name}: object_pose must be {expected_object_shape}, got {object_pose.shape}"
            )
        for key, value in (
            ("joint_pos", joint_pos),
            ("action_target", action_target),
            ("object_pose", object_pose),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{path.name}: {key} contains non-finite values")
        episodes.append(
            ReplayEpisode(
                path=path,
                joint_pos=joint_pos,
                action_target=action_target,
                object_pose=object_pose,
                source_success=source_success,
            )
        )
    if not episodes:
        qualifier = " timeout-eligible" if source_success_only else ""
        raise ValueError(
            f"no{qualifier} episode_*.npz files under {root} "
            f"(input={len(episode_paths)}, "
            f"source_successful={source_successful_episode_count})"
        )
    return ReplayExport(
        manifest=manifest,
        episodes=tuple(episodes),
        total_episode_count=len(episode_paths),
        source_successful_episode_count=source_successful_episode_count,
    )
