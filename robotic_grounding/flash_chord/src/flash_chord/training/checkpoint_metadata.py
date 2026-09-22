# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolved configuration and policy-schema metadata stored with checkpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

from flash_chord.envs.observation import ObservationDiagnostics
from flash_chord.runtime.actions import ActionDiagnostics

if TYPE_CHECKING:
    from flash_chord.data.reference import Reference

CONFIG_METADATA_KEY = "resolved_config_json"
POLICY_SCHEMA_METADATA_KEY = "policy_schema_json"
CRITIC_SCHEMA_METADATA_KEY = "critic_schema_json"
OBJECT_ARTICULATION_CONTRACT_VERSION = 1
_ACTION_SEMANTICS = frozenset(
    {
        ("raw", "identity"),
        ("normalized", "linear"),
        ("normalized", "rational"),
    }
)
_MISSING = object()


def _object_articulation_config(reference: Reference) -> dict[str, object]:
    """Serialize stable asset-derived articulation physics and drive semantics."""
    return {
        "version": OBJECT_ARTICULATION_CONTRACT_VERSION,
        "entries": [
            {
                "asset_name": asset.name,
                "simulation_joint_name": articulation.simulation_joint_name,
                "reference_index": articulation.reference_index,
                "physics": {
                    "armature": articulation.physics.armature,
                    "friction": articulation.physics.friction,
                },
                "drive": {
                    "kp": articulation.drive.kp,
                    "kd": articulation.drive.kd,
                    "effort_limit": articulation.drive.effort_limit,
                },
            }
            for asset in reference.object_assets()
            for articulation in asset.articulations
        ],
    }


def _reference_articulation_contract(
    config: Mapping[str, object],
    *,
    source: str,
) -> Mapping[str, object]:
    if "reference" not in config:
        raise ValueError(f"{source} config is missing current reference metadata")
    reference = config["reference"]
    if not isinstance(reference, Mapping):
        raise TypeError(f"{source} reference config must be a mapping")
    if "object_articulations" not in reference:
        raise ValueError(f"{source} reference config is missing the object articulation contract")
    contract = reference["object_articulations"]
    if not isinstance(contract, Mapping):
        raise TypeError(f"{source} object articulation contract must be a mapping")
    version = contract.get("version")
    if version != OBJECT_ARTICULATION_CONTRACT_VERSION:
        raise ValueError(f"{source} object articulation contract has unsupported version {version!r}")
    entries = contract.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError(f"{source} object articulation contract entries must be a sequence")
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise ValueError(f"{source} object articulation contract entries must be mappings")
    return contract


def _validate_resolved_reference_config(
    saved_config: Mapping[str, object],
    current_config: Mapping[str, object],
) -> None:
    saved_contract = _reference_articulation_contract(saved_config, source="checkpoint")
    current_contract = _reference_articulation_contract(current_config, source="current")
    if saved_contract != current_contract:
        raise ValueError(
            "checkpoint object articulation config does not match current environment: "
            f"saved={saved_contract}, current={current_contract}"
        )


def resolved_reference_config(reference: Reference, parquet: str) -> dict[str, object]:
    """Build checkpoint provenance for the loaded trajectory and object mechanism."""
    return {
        "parquet": str(parquet),
        "num_frames": reference.num_frames,
        "playback_fps": reference.fps,
        "object_articulations": _object_articulation_config(reference),
    }


def validate_reference_config(saved_config: Mapping[str, object], reference: Reference) -> None:
    """Reject a checkpoint whose current asset-derived articulation contract does not match."""
    current_config = {"reference": {"object_articulations": _object_articulation_config(reference)}}
    _validate_resolved_reference_config(saved_config, current_config)


def validate_resume_reference_config(
    saved_metadata: Mapping[str, str],
    current_metadata: Mapping[str, object] | None,
) -> None:
    """Validate the loaded mechanism before resuming a current-schema checkpoint."""
    saved_config = checkpoint_config(saved_metadata)
    current_strings = {key: value for key, value in (current_metadata or {}).items() if isinstance(value, str)}
    current_config = checkpoint_config(current_strings)
    _validate_resolved_reference_config(saved_config, current_config)


def _validate_action_semantics(input_mode: object, input_mapping: object, component: str) -> None:
    if (input_mode, input_mapping) not in _ACTION_SEMANTICS:
        raise ValueError(
            f"{component} requires raw/identity, normalized/linear, or normalized/rational "
            f"action semantics, got {input_mode!r}/{input_mapping!r}"
        )


def _component_schema(
    component: str,
    dimension: int,
    names: tuple[str, ...],
    ranges: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    if len(names) != len(ranges):
        raise ValueError(f"{component} schema has {len(names)} names and {len(ranges)} ranges")
    expected_start = 0
    blocks = []
    for name, (start, end) in zip(names, ranges, strict=True):
        if start != expected_start or end <= start:
            raise ValueError(
                f"{component} schema block {name!r} has range ({start}, {end}); "
                f"expected a positive contiguous block starting at {expected_start}"
            )
        blocks.append({"name": name, "start": start, "end": end})
        expected_start = end
    if expected_start != dimension:
        raise ValueError(f"{component} schema covers {expected_start} values; expected dimension {dimension}")
    return {"dimension": dimension, "blocks": blocks}


def policy_schema(action, observation) -> dict[str, object]:
    """Build the exact flattened policy I/O schema from runtime strategies."""
    if not isinstance(action, ActionDiagnostics):
        raise TypeError("policy checkpointing requires an action implementing ActionDiagnostics")
    if not isinstance(observation, ObservationDiagnostics):
        raise TypeError("policy checkpointing requires an observation implementing ObservationDiagnostics")
    input_mode = action.input_mode
    input_mapping = action.input_mapping
    _validate_action_semantics(input_mode, input_mapping, "policy action")
    action_schema = _component_schema("action", action.action_dim, action.block_names, action.block_ranges)
    action_schema["input_mode"] = input_mode
    action_schema["input_mapping"] = input_mapping
    return {
        "action": action_schema,
        "observation": _component_schema(
            "observation",
            observation.observation_dim,
            observation.block_names,
            observation.block_ranges,
        ),
    }


def critic_schema(action, observation, context_names: Sequence[str]) -> dict[str, object]:
    """Build the flattened asymmetric-critic input schema in exact concatenation order."""
    actor = policy_schema(action, observation)
    names = tuple(context_names)
    if not names:
        raise ValueError("critic context names must not be empty")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError(f"critic context names must be unique non-empty strings, got {names}")
    context = _component_schema(
        "critic context",
        len(names),
        names,
        tuple((index, index + 1) for index in range(len(names))),
    )
    observation = actor["observation"]
    action = actor["action"]
    return {
        "input_order": ["observation", "context", "action"],
        "input_dimension": observation["dimension"] + context["dimension"] + action["dimension"],
        "observation": observation,
        "context": context,
        "action": action,
    }


def build_checkpoint_metadata(
    resolved_config: Mapping[str, object],
    schema: Mapping[str, object],
    critic: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Serialize stable, self-contained experiment metadata for a checkpoint header."""
    metadata = {
        CONFIG_METADATA_KEY: json.dumps(resolved_config, sort_keys=True, separators=(",", ":")),
        POLICY_SCHEMA_METADATA_KEY: json.dumps(schema, sort_keys=True, separators=(",", ":")),
    }
    if critic is not None:
        metadata[CRITIC_SCHEMA_METADATA_KEY] = json.dumps(critic, sort_keys=True, separators=(",", ":"))
    return metadata


def _metadata_mapping(metadata: Mapping[str, str], key: str) -> dict[str, object]:
    serialized = metadata.get(key)
    if serialized is None:
        raise ValueError(f"checkpoint is missing required current-schema metadata {key!r}")
    try:
        value = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ValueError(f"checkpoint {key} is invalid JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint {key} must contain a mapping")
    return value


def checkpoint_config(metadata: Mapping[str, str]) -> dict[str, object]:
    """Deserialize the required current resolved configuration."""
    return _metadata_mapping(metadata, CONFIG_METADATA_KEY)


def checkpoint_policy_schema(metadata: Mapping[str, str]) -> dict[str, object]:
    """Deserialize the required current policy schema."""
    return _metadata_mapping(metadata, POLICY_SCHEMA_METADATA_KEY)


def checkpoint_critic_schema(metadata: Mapping[str, str]) -> dict[str, object]:
    """Deserialize the required current asymmetric-critic schema."""
    return _metadata_mapping(metadata, CRITIC_SCHEMA_METADATA_KEY)


def validate_policy_schema(saved: Mapping[str, object], current: Mapping[str, object]) -> None:
    """Reject policy I/O incompatibility before attempting learner restoration."""
    normalized_saved = _normalize_action_schema(saved, "policy")
    normalized_current = _normalize_action_schema(current, "policy")
    if normalized_saved != normalized_current:
        saved_summary = {
            name: normalized_saved.get(name, {}).get("dimension")
            for name in ("observation", "action")
            if isinstance(normalized_saved.get(name), dict)
        }
        current_summary = {
            name: normalized_current.get(name, {}).get("dimension")
            for name in ("observation", "action")
            if isinstance(normalized_current.get(name), dict)
        }
        saved_mode = normalized_saved.get("action", {}).get("input_mode")
        current_mode = normalized_current.get("action", {}).get("input_mode")
        saved_mapping = normalized_saved.get("action", {}).get("input_mapping")
        current_mapping = normalized_current.get("action", {}).get("input_mapping")
        raise ValueError(
            f"checkpoint policy schema does not match evaluation environment: "
            f"saved={saved_summary}/{saved_mode}/{saved_mapping}, "
            f"current={current_summary}/{current_mode}/{current_mapping}"
        )


def validate_critic_schema(saved: Mapping[str, object], current: Mapping[str, object]) -> None:
    """Reject asymmetric critic incompatibility, including semantic block ordering."""
    normalized_saved = _normalize_action_schema(saved, "critic")
    normalized_current = _normalize_action_schema(current, "critic")
    if normalized_saved != normalized_current:
        saved_action = normalized_saved.get("action", {})
        current_action = normalized_current.get("action", {})
        raise ValueError(
            "checkpoint critic schema does not match the training environment: "
            f"saved_order={normalized_saved.get('input_order')}, "
            f"current_order={normalized_current.get('input_order')}, "
            f"saved_dimension={normalized_saved.get('input_dimension')}, "
            f"current_dimension={normalized_current.get('input_dimension')}, "
            f"saved_action_mode={saved_action.get('input_mode')}, "
            f"current_action_mode={current_action.get('input_mode')}, "
            f"saved_action_mapping={saved_action.get('input_mapping')}, "
            f"current_action_mapping={current_action.get('input_mapping')}"
        )


def _normalize_action_schema(schema: Mapping[str, object], component: str) -> dict[str, object]:
    """Validate explicit action semantics and ignore an optional historical version label."""
    action = schema.get("action")
    if not isinstance(action, Mapping):
        raise TypeError(f"{component} schema action must be a mapping")
    input_mode = action.get("input_mode")
    input_mapping = action.get("input_mapping")
    _validate_action_semantics(
        input_mode,
        input_mapping,
        f"{component} schema",
    )
    normalized = {key: value for key, value in schema.items() if key != "version"}
    return {
        **normalized,
        "action": {
            **action,
            "input_mode": input_mode,
            "input_mapping": input_mapping,
        },
    }


def _delete_config_path(config: DictConfig, key: str) -> None:
    """Delete one existing dot path while preserving Hydra deletion semantics."""
    parent_key, separator, leaf = key.rpartition(".")
    parent = OmegaConf.select(config, parent_key, default=_MISSING) if separator else config
    if parent is _MISSING:
        return
    if OmegaConf.is_list(parent):
        try:
            index = int(leaf)
        except ValueError as error:
            raise ValueError(
                f"list-valued config path {parent_key!r} requires an integer index, got {leaf!r}"
            ) from error
        if -len(parent) <= index < len(parent):
            del parent[index]
        return
    if OmegaConf.is_dict(parent):
        if leaf in parent:
            del parent[leaf]
        return
    raise ValueError(f"cannot delete {key!r} below non-container config value {parent!r}")


def _plain_config_value(value):
    return OmegaConf.to_container(value, resolve=True, enum_to_str=True) if OmegaConf.is_config(value) else value


def _replace_config_path(config: DictConfig, source: str, destination: str) -> None:
    """Copy one complete ownership subtree, or preserve its deletion at the destination."""
    value = OmegaConf.select(config, source, default=_MISSING)
    if value is _MISSING:
        _delete_config_path(config, destination)
    else:
        OmegaConf.update(
            config,
            destination,
            _plain_config_value(value),
            merge=False,
            force_add=True,
        )


def checkpoint_evaluation_config(
    current: DictConfig,
    saved: Mapping[str, object],
    explicit_override_keys: Sequence[str] = (),
    *,
    use_evaluation_training_world_count: bool = True,
) -> DictConfig:
    """Create eval config from saved state plus eval-only and explicit current overrides.

    Full-learner evaluators need the training world count to match the evaluation environment. Actor-only
    evaluators can preserve the saved count so training-only budget validation remains tied to the saved run.
    """
    current_data = OmegaConf.to_container(current, resolve=True, enum_to_str=True)
    if not isinstance(current_data, dict):
        raise TypeError("current evaluation config must be a mapping")
    effective = OmegaConf.create(saved)
    OmegaConf.set_struct(effective, False)

    for key in ("evaluation", "viewer", "markers", "logging"):
        if key in current_data:
            OmegaConf.update(effective, key, current_data[key], merge=False)

    strategy_roots = {"action", "observation", "objective", "termination", "reset"}
    root_strategy_overrides: set[str] = set()
    env_owned_override_roots: set[str] = set()
    root_curriculum_override = False
    training_curriculum_override = False
    for raw_key in explicit_override_keys:
        key = raw_key.lstrip("+~")
        if not key or key.split(".", 1)[0] in {"evaluation", "viewer", "markers", "logging"}:
            continue
        parts = key.split(".")
        if parts[0] in strategy_roots | {"sim"}:
            root_strategy_overrides.add(parts[0])
        if len(parts) >= 2 and parts[0] == "env" and parts[1] in strategy_roots | {"sim"}:
            env_owned_override_roots.add(parts[1])
        if parts[0] == "curriculum":
            root_curriculum_override = True
        if len(parts) >= 2 and parts[0] == "training" and parts[1] == "curriculum":
            training_curriculum_override = True

        if raw_key.startswith("~"):
            _delete_config_path(effective, key)
            continue
        value = OmegaConf.select(current, key, default=_MISSING)
        if value is _MISSING:
            raise ValueError(f"explicit evaluation override {raw_key!r} is absent from the composed config")
        OmegaConf.update(
            effective,
            key,
            _plain_config_value(value),
            merge=False,
            force_add=True,
        )

    # Evaluation is intentionally one-world and unassisted after the reference reset hold.
    world_count = current_data["scene"]["world_count"]
    OmegaConf.update(effective, "scene.world_count", world_count, merge=False)
    if use_evaluation_training_world_count:
        OmegaConf.update(effective, "training.world_count", world_count, merge=False)
    OmegaConf.update(effective, "env.voc_scale", current_data["env"]["voc_scale"], merge=False)
    OmegaConf.update(effective, "env.capture_terminal_state", False, merge=False)

    # Saved env.* and training.curriculum are runtime-authoritative because those are the objects training consumed.
    # Explicit root overrides still propagate to their runtime copies; explicit runtime overrides take precedence.
    for root in strategy_roots | {"sim"}:
        if root in env_owned_override_roots:
            _replace_config_path(effective, f"env.{root}", root)
        elif root in root_strategy_overrides:
            _replace_config_path(effective, root, f"env.{root}")
        elif OmegaConf.select(effective, f"env.{root}", default=_MISSING) is not _MISSING:
            _replace_config_path(effective, f"env.{root}", root)
        else:
            _replace_config_path(effective, root, f"env.{root}")

    if training_curriculum_override:
        _replace_config_path(effective, "training.curriculum", "curriculum")
    elif root_curriculum_override:
        _replace_config_path(effective, "curriculum", "training.curriculum")
    elif OmegaConf.select(effective, "training.curriculum", default=_MISSING) is not _MISSING:
        _replace_config_path(effective, "training.curriculum", "curriculum")
    else:
        _replace_config_path(effective, "curriculum", "training.curriculum")
    return effective
