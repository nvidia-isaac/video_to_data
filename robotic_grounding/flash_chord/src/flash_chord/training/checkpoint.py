# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safe, complete learner-state checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import jax
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

_FORMAT = "learner_state"
_VERSION = "1"
_STATE_PREFIX = "state."
_USER_PREFIX = "user."


def save_checkpoint(
    path: str | Path,
    state,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Atomically save every dynamic learner-state leaf without pickle."""
    leaves, _ = jax.tree.flatten(state)
    tensors: dict[str, np.ndarray] = {}
    manifest = []
    for index, leaf in enumerate(leaves):
        value = np.asarray(leaf)
        shape = value.shape
        if value.ndim == 0:
            value = value.reshape(1)
        key = f"{_STATE_PREFIX}{index:04d}"
        tensors[key] = np.ascontiguousarray(value)
        manifest.append({"key": key, "shape": shape, "dtype": np.dtype(value.dtype).str})

    checkpoint_metadata = {
        "format": _FORMAT,
        "version": _VERSION,
        "manifest": json.dumps(manifest, separators=(",", ":")),
    }
    if metadata is not None:
        checkpoint_metadata.update({f"{_USER_PREFIX}{key}": str(value) for key, value in metadata.items()})

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        save_file(tensors, temporary, metadata=checkpoint_metadata)
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_checkpoint(path: str | Path):
    """Read and validate the format envelope shared by resume and inference restore."""
    with safe_open(str(path), framework="numpy") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("format") != _FORMAT or metadata.get("version") != _VERSION:
            raise ValueError(
                f"unsupported checkpoint format/version: {metadata.get('format')!r}/{metadata.get('version')!r}"
            )
        try:
            manifest = json.loads(metadata["manifest"])
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError("checkpoint manifest is missing or invalid") from error
        arrays = {entry["key"]: checkpoint.get_tensor(entry["key"]) for entry in manifest}
    user_metadata = {
        key.removeprefix(_USER_PREFIX): value for key, value in metadata.items() if key.startswith(_USER_PREFIX)
    }
    return manifest, arrays, user_metadata


def read_checkpoint_metadata(path: str | Path) -> dict[str, str]:
    """Read validated user metadata without constructing a learner template."""
    with safe_open(str(path), framework="numpy") as checkpoint:
        metadata = checkpoint.metadata() or {}
    if metadata.get("format") != _FORMAT or metadata.get("version") != _VERSION:
        raise ValueError(
            f"unsupported checkpoint format/version: {metadata.get('format')!r}/{metadata.get('version')!r}"
        )
    return {key.removeprefix(_USER_PREFIX): value for key, value in metadata.items() if key.startswith(_USER_PREFIX)}


def _restore_checkpoint(path: str | Path, template, template_leaf_paths: frozenset[str] = frozenset()):
    """Restore a checkpoint, optionally retaining explicitly named leaves from the template."""
    manifest, arrays, user_metadata = _read_checkpoint(path)

    path_leaves, tree = jax.tree_util.tree_flatten_with_path(template)
    if len(manifest) != len(path_leaves):
        raise ValueError(f"checkpoint has {len(manifest)} leaves; template has {len(path_leaves)}")

    restored = []
    for entry, (path_keys, template_leaf) in zip(manifest, path_leaves, strict=True):
        leaf_path = jax.tree_util.keystr(path_keys)
        value = arrays[entry["key"]].reshape(tuple(entry["shape"]))
        template_value = np.asarray(template_leaf)
        expected_shape = template_value.shape
        expected_dtype = template_value.dtype
        if leaf_path in template_leaf_paths:
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"checkpoint leaf {entry['key']} ({leaf_path}) has dtype {value.dtype}; expected {expected_dtype}"
                )
            restored.append(template_leaf)
            continue
        if value.shape != expected_shape or value.dtype != expected_dtype:
            raise ValueError(
                f"checkpoint leaf {entry['key']} ({leaf_path}) has {value.shape}/{value.dtype}; "
                f"expected {expected_shape}/{expected_dtype}"
            )
        restored.append(jax.device_put(value, getattr(template_leaf, "device", None)))

    return jax.tree.unflatten(tree, restored), user_metadata


def load_checkpoint(path: str | Path, template):
    """Restore every dynamic leaf into a compatible freshly initialized template."""
    return _restore_checkpoint(path, template)


def load_checkpoint_for_inference(path: str | Path, template):
    """Restore learner state while retaining the evaluation environment's current observation.

    The observation leaf has shape ``(world_count, observation_dim)`` and is rollout state rather
    than policy state. Keeping the fresh template value permits evaluation with a different number
    of worlds while all policy, optimizer, normalization, RNG, and scalar leaves remain strict.
    """
    return _restore_checkpoint(path, template, template_leaf_paths=frozenset({".observation"}))
