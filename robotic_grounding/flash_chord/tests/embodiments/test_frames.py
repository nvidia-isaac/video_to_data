# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for semantic body-frame resolution across fixed-joint collapse."""

from types import SimpleNamespace

import pytest

from flash_chord.embodiments.base import BodyFrame
from flash_chord.embodiments.frames import (
    DeviceBodyFrameMap,
    capture_body_ids,
    resolve_body_frames,
    resolve_retained_body_ids,
)


def test_capture_body_ids_uses_exact_terminal_labels_and_requested_order():
    builder = SimpleNamespace(body_label=("robot/root", "robot/palm", "robot/index_fingertip"))

    assert capture_body_ids(builder, ("index_fingertip", "palm")) == {
        "index_fingertip": 2,
        "palm": 1,
    }


@pytest.mark.parametrize(
    ("labels", "names", "message"),
    [
        (("robot/root",), ("tip",), "missing=['tip']"),
        (("left/palm", "right/palm"), ("palm",), "ambiguous={'palm': [0, 1]}"),
        (("robot/root",), ("root", "root"), "must be unique"),
    ],
)
def test_capture_body_ids_rejects_unresolved_or_duplicate_names(labels, names, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        capture_body_ids(SimpleNamespace(body_label=labels), names)


def test_resolve_body_frames_handles_retained_and_merged_bodies():
    collapse_result = {
        "body_remap": {0: 4},
        "body_merged_parent": {1: 0, 2: 0},
        "body_merged_transform": {
            1: SimpleNamespace(p=(0.1, 0.0, 0.0), q=(0.0, 0.0, 0.0, 1.0)),
            2: SimpleNamespace(p=(0.1, 0.2, 0.0), q=(0.0, 0.0, 0.707, 0.707)),
        },
    }

    root, palm, tip = resolve_body_frames({"root": 0, "palm": 1, "tip": 2}, collapse_result)

    assert root.body_id == palm.body_id == tip.body_id == 4
    assert root.body_to_frame_pos == (0.0, 0.0, 0.0)
    assert palm.body_to_frame_pos == (0.1, 0.0, 0.0)
    assert tip.body_to_frame_pos == (0.1, 0.2, 0.0)
    assert tip.body_to_frame_quat_xyzw == pytest.approx((0.0, 0.0, 2**-0.5, 2**-0.5))


def test_resolve_retained_body_ids_preserves_order_and_merged_aliases():
    collapse_result = {
        "body_remap": {0: 4, 3: 7},
        "body_merged_parent": {1: 0, 2: 0},
    }

    assert resolve_retained_body_ids((3, 1, 0, 2), collapse_result) == (7, 4, 4, 4)


@pytest.mark.parametrize(
    ("body_id", "collapse_result", "message"),
    [
        (-1, {"body_remap": {}, "body_merged_parent": {}}, "must be nonnegative"),
        (1, {"body_remap": {}, "body_merged_parent": {}}, "was not retained or merged"),
        (1, {"body_remap": {}, "body_merged_parent": {1: -1}}, "collapsed into the static world"),
    ],
)
def test_resolve_retained_body_ids_rejects_unusable_topology(body_id, collapse_result, message):
    with pytest.raises(ValueError, match=message):
        resolve_retained_body_ids((body_id,), collapse_result)


@pytest.mark.gpu
def test_device_body_frame_map_preserves_order_shared_bodies_and_normalizes_quaternions():
    import numpy as np
    import warp as wp

    frames = (
        BodyFrame("tip", 4, body_to_frame_pos=(0.1, 0.2, 0.3), body_to_frame_quat_xyzw=(0.0, 0.0, 0.707, 0.707)),
        BodyFrame("palm", 4),
    )
    with wp.ScopedDevice("cuda:0"):
        frame_map = DeviceBodyFrameMap.build(frames)

    assert frame_map.count == 2
    assert frame_map.body_ids.numpy().tolist() == [4, 4]
    np.testing.assert_allclose(frame_map.body_to_frame_pos.numpy(), [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        frame_map.body_to_frame_quat.numpy(),
        [[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], [0.0, 0.0, 0.0, 1.0]],
        atol=1.0e-6,
    )


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ((), "at least one"),
        ((BodyFrame("palm", 0), BodyFrame("palm", 1)), "unique and nonempty"),
    ],
)
def test_device_body_frame_map_rejects_invalid_semantic_mappings(frames, message):
    with pytest.raises(ValueError, match=message):
        DeviceBodyFrameMap.build(frames)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"name": "", "body_id": 0}, "nonempty string"),
        ({"name": "palm", "body_id": -1}, "nonnegative"),
        ({"name": "palm", "body_id": 0, "body_to_frame_pos": (0.0, float("nan"), 0.0)}, "finite vec3"),
        (
            {"name": "palm", "body_id": 0, "body_to_frame_quat_xyzw": (0.0, 0.0, 0.0, 0.0)},
            "nonzero norm",
        ),
    ],
)
def test_body_frame_rejects_invalid_semantics_at_construction(values, message):
    with pytest.raises(ValueError, match=message):
        BodyFrame(**values)


@pytest.mark.parametrize(
    ("collapse_result", "message"),
    [
        (
            {"body_remap": {}, "body_merged_parent": {}, "body_merged_transform": {}},
            "was not retained or merged",
        ),
        (
            {
                "body_remap": {},
                "body_merged_parent": {1: -1},
                "body_merged_transform": {1: SimpleNamespace(p=(0, 0, 0), q=(0, 0, 0, 1))},
            },
            "collapsed into the static world",
        ),
    ],
)
def test_resolve_body_frames_rejects_unusable_collapse_results(collapse_result, message):
    with pytest.raises(ValueError, match=message):
        resolve_body_frames({"tip": 1}, collapse_result)
