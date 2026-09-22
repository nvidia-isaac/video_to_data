# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SharpaWave dual-hand embodiment build + layout."""

import pytest

pytestmark = pytest.mark.gpu

_DIGITS = ("thumb", "index", "middle", "ring", "pinky")
_OBJECT_CONTACT_LINKS = {
    "hand_C_MC",
    "thumb_MC",
    "thumb_PP",
    "thumb_DP",
    "index_PP",
    "index_MP",
    "index_DP",
    "middle_PP",
    "middle_MP",
    "middle_DP",
    "ring_PP",
    "ring_MP",
    "ring_DP",
    "pinky_MC",
    "pinky_PP",
    "pinky_MP",
    "pinky_DP",
}


def _build(config=None):
    import warp as wp

    import newton
    from flash_chord.embodiments.sharpa_hands import SharpaHands

    with wp.ScopedDevice("cuda:0"):
        builder = newton.ModelBuilder()
        layout = SharpaHands(config).build(builder)
    return builder, layout


def _build_layout():
    return _build()[1]


def test_dual_hand_layout_matches_ground_truth():
    layout = _build_layout()
    assert layout.num_joint_q == 58
    assert layout.num_joint_dof == 56
    assert layout.sides == ("left", "right")

    left = layout.hand("left")
    assert left.wrist_pos_dof_ids == (0, 1, 2)
    assert left.wrist_orient_dof_ids == (3, 4, 5)
    assert left.wrist_orient_q_id == 3
    assert left.finger_dof_ids == tuple(range(6, 28))  # 22 finger DOFs
    assert left.finger_joint_names[0] == "left_thumb_CMC_FE"
    assert left.finger_joint_names[-1] == "left_pinky_DIP"
    assert left.wrist_body_id == 3  # left_sharpa_wave/left_hand_C_MC
    assert left.fingertip_body_ids == (8, 12, 16, 20, 25)  # the *_DP tips
    assert len(left.link_geometry) == 23
    assert left.contact_link_count == 17
    assert len(left.collision_shape_ids) == 54
    assert len(left.contact_shape_ids) == 44
    assert set(left.contact_shape_ids) < set(left.collision_shape_ids)
    left_links = {link.link_name: link for link in left.link_geometry}
    assert len(left_links["hand_C_MC"].shape_ids) == 2
    assert len(left_links["index_DP"].shape_ids) == 4  # DP plus its fixed elastomer

    right = layout.hand("right")
    assert right.wrist_pos_dof_ids == (28, 29, 30)
    assert right.wrist_orient_dof_ids == (31, 32, 33)
    assert right.wrist_orient_q_id == 32
    assert right.finger_dof_ids == tuple(range(34, 56))
    assert right.finger_joint_names[0] == "right_thumb_CMC_FE"
    assert right.finger_joint_names[-1] == "right_pinky_DIP"
    assert right.wrist_body_id == 29
    assert right.fingertip_body_ids == (34, 38, 42, 46, 51)
    assert len(right.link_geometry) == 23
    assert right.contact_link_count == 17
    assert len(right.collision_shape_ids) == 54
    assert len(right.contact_shape_ids) == 44
    assert set(right.contact_shape_ids) < set(right.collision_shape_ids)


def test_object_contact_links_match_the_explicit_sharpa_semantic_set():
    layout = _build_layout()
    for side in ("left", "right"):
        contact_names = {link.link_name for link in layout.hand(side).contact_links}
        assert contact_names == _OBJECT_CONTACT_LINKS


def test_object_contact_links_reject_unknown_configuration_names():
    from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig

    with pytest.raises(ValueError, match="object-contact links.*missing"):
        _build(SharpaHandsConfig(object_contact_links=["hand_C_MC", "missing"]))


def test_palm_dp_and_true_fingertip_frames_are_explicit():
    builder, layout = _build()
    for side in ("left", "right"):
        hand = layout.hand(side)
        assert hand.palm_frame.name == f"{side}_hand_C_MC"
        assert hand.palm_frame.body_id == hand.wrist_body_id
        assert hand.palm_frame.body_to_frame_pos == (0.0, 0.0, 0.0)
        assert tuple(frame.name for frame in hand.dp_frames) == tuple(f"{side}_{digit}_DP" for digit in _DIGITS)
        assert tuple(frame.name for frame in hand.fingertip_frames) == tuple(
            f"{side}_{digit}_fingertip" for digit in _DIGITS
        )
        assert hand.fingertip_body_ids == tuple(frame.body_id for frame in hand.dp_frames)
        for dp, tip in zip(hand.dp_frames, hand.fingertip_frames, strict=True):
            assert builder.body_label[dp.body_id].endswith(f"/{dp.name}")
            assert dp.body_to_frame_pos == (0.0, 0.0, 0.0)
            assert tip.body_id == dp.body_id
            assert sum(value * value for value in tip.body_to_frame_pos) > 0.0004


def test_curated_mechanical_exclusions_do_not_change_layout():
    from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig

    default_builder, default_layout = _build()
    adjacent_builder, adjacent_layout = _build(SharpaHandsConfig(exclude_contacts=[]))

    default_pairs = set(default_builder.shape_collision_filter_pairs)
    adjacent_pairs = set(adjacent_builder.shape_collision_filter_pairs)
    assert default_layout == adjacent_layout
    assert adjacent_pairs < default_pairs


@pytest.mark.parametrize(
    ("excluded_contacts", "message"),
    [
        ([("hand_C_MC", "missing")], "missing links"),
        ([("hand_C_MC", "hand_C_MC")], "cannot pair a link with itself"),
        (
            [("hand_C_MC", "thumb_MC"), ("thumb_MC", "hand_C_MC")],
            "duplicate Sharpa excluded contact pair",
        ),
    ],
)
def test_curated_mechanical_exclusions_reject_invalid_configuration(excluded_contacts, message):
    from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig

    with pytest.raises(ValueError, match=message):
        _build(SharpaHandsConfig(exclude_contacts=excluded_contacts))
