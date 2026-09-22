# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared zero-copy Warp/JAX array views."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_warp_jax_views_share_pointer_shape_dtype_and_mutations():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import warp as wp

    from flash_chord.runtime.jax import from_jax, to_jax

    with wp.ScopedDevice("cuda:0"):
        warp_array = wp.array(np.arange(6, dtype=np.float32), dtype=wp.float32)
        jax_view = to_jax(warp_array, shape=(2, 3))
        assert jax_view.shape == (2, 3)
        assert jax_view.dtype == jnp.float32
        assert jax_view.device.platform == "gpu"
        assert jax_view.unsafe_buffer_pointer() == warp_array.ptr

        warp_array.assign(np.arange(10, 16, dtype=np.float32))
        wp.synchronize()
        np.testing.assert_allclose(np.asarray(jax_view), [[10, 11, 12], [13, 14, 15]])

        jax_array = jax.device_put(
            jnp.arange(4, dtype=jnp.int32),
            device=wp.device_to_jax("cuda:0"),
        )
        jax_array.block_until_ready()
        warp_view = from_jax(jax_array)
        assert warp_view.shape == (4,)
        assert warp_view.dtype == wp.int32
        assert warp_view.device == wp.get_device("cuda:0")
        assert warp_view.ptr == jax_array.unsafe_buffer_pointer()

        warp_view.assign(np.array([7, 8, 9, 10], dtype=np.int32))
        wp.synchronize()
        np.testing.assert_array_equal(np.asarray(jax_array), [7, 8, 9, 10])


def test_to_jax_rejects_shape_with_different_element_count():
    pytest.importorskip("jax")
    import warp as wp

    from flash_chord.runtime.jax import to_jax

    with wp.ScopedDevice("cuda:0"):
        array = wp.zeros(6, dtype=wp.float32)
        with pytest.raises(ValueError, match="requested shape"):
            to_jax(array, shape=(2, 4))
