# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

estimate_mhr = pytest.importorskip("v2d.sam3d_body.lib.estimate_mhr_params")


class FakeMaskSource:
    def __init__(self, stems, masks):
        self.stems = stems
        self._masks = masks

    def __getitem__(self, idx):
        return self._masks[idx]


def test_bbox_from_mask_returns_tight_xyxy_box():
    mask = np.zeros((5, 7, 1), dtype=np.uint8)
    mask[1:4, 2:6, 0] = 255

    np.testing.assert_array_equal(
        estimate_mhr._bbox_from_mask(mask),
        np.array([2, 1, 6, 4], dtype=np.float32),
    )


def test_bbox_from_mask_returns_none_for_empty_mask():
    mask = np.zeros((5, 7, 1), dtype=np.uint8)

    assert estimate_mhr._bbox_from_mask(mask) is None


def test_select_frame_prompt_prefers_non_empty_mask():
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 2:6] = 255
    mask_source = FakeMaskSource(["000000"], [mask])
    bbox_track = np.array([[0, 0, 3, 3]], dtype=np.float32)

    bbox, selected_mask, source = estimate_mhr._select_frame_prompt(
        image=image,
        stem="000000",
        frame_idx=0,
        bbox_track=bbox_track,
        mask_source=mask_source,
        mask_stem_to_idx={"000000": 0},
    )

    np.testing.assert_array_equal(bbox, np.array([2, 1, 6, 4], dtype=np.float32))
    assert selected_mask.shape == (5, 7, 1)
    assert source == estimate_mhr.PROMPT_SOURCE_MASK


def test_select_frame_prompt_empty_mask_falls_back_to_bbox():
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    mask_source = FakeMaskSource(["000000"], [np.zeros((5, 7), dtype=np.uint8)])
    bbox_track = np.array([[1, 1, 5, 4]], dtype=np.float32)

    bbox, selected_mask, source = estimate_mhr._select_frame_prompt(
        image=image,
        stem="000000",
        frame_idx=0,
        bbox_track=bbox_track,
        mask_source=mask_source,
        mask_stem_to_idx={"000000": 0},
    )

    np.testing.assert_array_equal(bbox, bbox_track[0])
    assert selected_mask is None
    assert source == estimate_mhr.PROMPT_SOURCE_BBOX_FALLBACK


def test_select_frame_prompt_degenerate_bbox_falls_back_to_full_image():
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    bbox_track = np.array([[1, 1, 1, 4]], dtype=np.float32)

    bbox, selected_mask, source = estimate_mhr._select_frame_prompt(
        image=image,
        stem="000000",
        frame_idx=0,
        bbox_track=bbox_track,
        mask_source=None,
        mask_stem_to_idx={},
    )

    np.testing.assert_array_equal(bbox, np.array([0, 0, 7, 5], dtype=np.float32))
    assert selected_mask is None
    assert source == estimate_mhr.PROMPT_SOURCE_FULL_IMAGE_FALLBACK
