from types import SimpleNamespace

import numpy as np
import pytest
import torch

from learning.datasets.mhr_input_materialization import MHR_SPATIAL_SCALE_KEY, MHR_XYZ_ANCHOR_KEY, build_mhr_spatial_normalization_contract, build_mhr_xyz_anchor_contract, prepare_mhr_spatial_batch, restore_mhr_spatial_normalization_contract, restore_mhr_xyz_anchor_contract, select_mhr_xyz_anchor
from learning.inference import object_pose_from_relative
from learning.training.mhr_losses import compose_mhr_output


def _config():
    return SimpleNamespace(mhr_xyz_anchor_type="root_joint_1", mhr_spatial_normalization_type="human_height_2m", mhr_spatial_target_height=2.0, rot_normalizer=1.0)


def _checkpoint(cfg):
    return {"cfg": vars(cfg).copy(), "mhr_xyz_anchor_contract": build_mhr_xyz_anchor_contract(cfg), "mhr_spatial_normalization_contract": build_mhr_spatial_normalization_contract(cfg)}


def test_checkpoint_restores_root_anchor_and_human_height_contracts():
    checkpoint_cfg = _config()
    runtime_cfg = _config()
    runtime_cfg.mhr_xyz_anchor_type = "body_world_translation"
    runtime_cfg.mhr_spatial_target_height = 1.0
    checkpoint = _checkpoint(checkpoint_cfg)
    assert restore_mhr_xyz_anchor_contract(checkpoint, runtime_cfg) == checkpoint["mhr_xyz_anchor_contract"]
    assert restore_mhr_spatial_normalization_contract(checkpoint, runtime_cfg) == checkpoint["mhr_spatial_normalization_contract"]
    assert runtime_cfg.mhr_xyz_anchor_type == "root_joint_1"
    assert runtime_cfg.mhr_spatial_target_height == 2.0


def test_legacy_checkpoint_is_rejected_instead_of_using_old_normalization():
    with pytest.raises(ValueError, match="predates the human-height spatial normalization contract"):
        restore_mhr_spatial_normalization_contract({"cfg": vars(_config())}, _config())


def test_root_joint_is_selected_as_xyz_anchor():
    body_translation = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    root_joint = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    np.testing.assert_array_equal(select_mhr_xyz_anchor(body_translation, root_joint, _config()), root_joint)


def test_spatial_batch_normalizes_inputs_and_restores_metric_predictions():
    cfg = _config()
    scale = torch.tensor([[2.0, 4.0]])
    pose = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    pose[:, :, :3, 3] = torch.tensor([[[0.25, 0.5, 1.0], [1.0, 0.5, 0.25]]])
    batch = {
        "input_xyz": torch.ones(1, 2, 3, 1, 1),
        "render_xyz": torch.ones(1, 2, 3, 1, 1) * 2.0,
        "pose_perturbed": pose,
        "mhr_trans_init": torch.zeros(1, 2, 3),
        MHR_SPATIAL_SCALE_KEY: scale,
        MHR_XYZ_ANCHOR_KEY: torch.zeros(1, 2, 3),
    }
    prepare_mhr_spatial_batch(batch, cfg, None)
    assert torch.equal(batch["input_xyz"][:, :, 0, 0, 0], scale)
    assert torch.equal(batch["render_xyz"][:, :, 0, 0, 0], scale * 2.0)
    assert torch.equal(batch["poseA_norm"][:, :, :3, 3], pose[:, :, :3, 3] * scale[:, :, None])
    output = {"delta_mhr_trans": torch.tensor([[[0.2, -0.4, 0.6], [0.8, 1.2, -1.6]]])}
    expected_metric = output["delta_mhr_trans"] / scale[:, :, None]
    assert torch.equal(compose_mhr_output(output, batch)["mhr_trans"], expected_metric)
    object_pose = object_pose_from_relative(batch, cfg, pose, torch.zeros(2, 3), output["delta_mhr_trans"].reshape(-1, 3))
    assert torch.allclose(object_pose[:, :, :3, 3], pose[:, :, :3, 3] + expected_metric)
