import pytest

from v2d.sam3d_body.lib import export_soma as export_soma_module
from v2d.sam3d_body.lib.export_soma import (
    build_leaf_weight,
    create_soma_layer,
    parse_pose_prior_weights,
)


def test_heel_contact_pose_prior_profile_matches_soma_x() -> None:
    weights = parse_pose_prior_weights("heel_contact")

    assert weights == {
        "Hips": 0.35,
        "Spine1": 0.35,
        "Spine2": 0.35,
        "Chest": 0.35,
        "LeftLeg": 0.35,
        "LeftShin": 6.0,
        "LeftFoot": 8.0,
        "LeftToeBase": 10.0,
        "LeftToeEnd": 10.0,
        "RightLeg": 0.35,
        "RightShin": 6.0,
        "RightFoot": 8.0,
        "RightToeBase": 10.0,
        "RightToeEnd": 10.0,
    }


def test_pose_prior_profile_accepts_custom_joint_weights() -> None:
    assert parse_pose_prior_weights("LeftFoot=8, RightFoot=9") == {
        "LeftFoot": 8.0,
        "RightFoot": 9.0,
    }
    assert parse_pose_prior_weights("uniform") is None

    with pytest.raises(ValueError, match="JointName=value"):
        parse_pose_prior_weights("unknown_profile")


def test_autograd_leaf_weights_are_region_specific() -> None:
    assert build_leaf_weight(1.0, hand_weight=5.0, foot_weight=20.0) == {
        "head": 1.0,
        "hands": 5.0,
        "feet": 20.0,
    }
    assert build_leaf_weight(1.0) == 1.0


def test_soma_layer_uses_public_asset_compatible_rig(monkeypatch) -> None:
    captured = {}

    def fake_soma_layer(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(export_soma_module, "SOMALayer", fake_soma_layer)

    layer = create_soma_layer("cuda")

    assert layer is not None
    assert captured["identity_model_type"] == "mhr"
    assert captured["enable_procedural_transforms"] is False
