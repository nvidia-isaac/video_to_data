# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Algorithm-neutral tests for the Safetensors checkpoint envelope."""

import stat

import numpy as np
import pytest


def test_checkpoint_round_trips_arbitrary_pytree_and_metadata(tmp_path):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("safetensors")

    from flash_chord.training.checkpoint import load_checkpoint, read_checkpoint_metadata, save_checkpoint

    state = {
        "count": jnp.asarray(7, dtype=jnp.int32),
        "weights": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
    }
    path = tmp_path / "state.safetensors"
    save_checkpoint(path, state, metadata={"algorithm": "test", "iteration": 7})

    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert read_checkpoint_metadata(path) == {"algorithm": "test", "iteration": "7"}
    restored, metadata = load_checkpoint(path, jax.tree.map(jnp.zeros_like, state))
    assert metadata == {"algorithm": "test", "iteration": "7"}
    for expected, actual in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize(
    "template",
    [
        {"value": np.zeros((3,), dtype=np.float32)},
        {"value": np.zeros((2,), dtype=np.int32)},
    ],
)
def test_checkpoint_rejects_incompatible_shape_or_dtype(tmp_path, template):
    pytest.importorskip("jax")
    pytest.importorskip("safetensors")

    from flash_chord.training.checkpoint import load_checkpoint, save_checkpoint

    path = tmp_path / "state.safetensors"
    save_checkpoint(path, {"value": np.ones((2,), dtype=np.float32)})

    with pytest.raises(ValueError, match="checkpoint leaf"):
        load_checkpoint(path, template)


def test_checkpoint_metadata_rejects_unknown_format(tmp_path):
    safetensors = pytest.importorskip("safetensors.numpy")

    from flash_chord.training.checkpoint import read_checkpoint_metadata

    path = tmp_path / "unknown.safetensors"
    safetensors.save_file(
        {"state.0000": np.zeros((1,), dtype=np.float32)},
        path,
        metadata={"format": "other", "version": "1"},
    )

    with pytest.raises(ValueError, match="unsupported checkpoint format/version"):
        read_checkpoint_metadata(path)
