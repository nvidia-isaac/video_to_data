from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from pytorch3d.transforms.so3 import so3_exp_map

import Utils
from learning.datasets.mhr_input_materialization import MHR_SPATIAL_NORMALIZATION_HUMAN_HEIGHT, denormalize_mhr_translation, resolve_mhr_spatial_normalization_type
from learning.training.mhr_supervision import apply_mhr_supervision_contract


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def forward_batch_mhr(batch: Mapping[str, Any], model: Any, supervision_contract: Mapping[str, Any]) -> dict[str, Any]:
    if "mhr_model_output" in batch:
        output = batch["mhr_model_output"]
    else:
        required = ("input_rgbs", "render_rgbs", "input_xyz", "render_xyz")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"MHR inference batch is missing rendered input fields: {missing}")
        imgs_b, imgs_a = batch["input_rgbs"], batch["render_rgbs"]
        xyz_b, xyz_a = batch["input_xyz"], batch["render_xyz"]
        object_pose = batch.get("poseA_norm", batch.get("pose_perturbed"))
        output = model(torch.cat([imgs_a, xyz_a], 2), torch.cat([imgs_b, xyz_b], 2), object_pose, batch)

    if "mhr_trans_init" in batch:
        batch_size, frame_count = batch["mhr_trans_init"].shape[:2]
        for key, value in list(output.items()):
            if key.startswith("delta_mhr_") and len(value.shape) == 2 and value.shape[0] == batch_size * frame_count:
                output[key] = value.reshape(batch_size, frame_count, -1)
    if model is not None and not model.training:
        if supervision_contract is None:
            raise RuntimeError("MHR inference requires a resolved training supervision contract")
        output = apply_mhr_supervision_contract(output, supervision_contract)
    return output


def object_pose_from_relative(batch: Mapping[str, Any], cfg: Any, pose_a: torch.Tensor, rotation: torch.Tensor, translation_delta: torch.Tensor) -> torch.Tensor:
    batch_size, frame_count = pose_a.shape[:2]
    translation = denormalize_mhr_translation(translation_delta.reshape(batch_size, frame_count, 3), batch).reshape(-1, 3)
    rotation_matrix = so3_exp_map(rotation * _cfg_value(cfg, "rot_normalizer")).permute(0, 2, 1)
    return Utils.egocentric_delta_pose_to_pose(pose_a.reshape(-1, 4, 4), trans_delta=translation, rot_mat_delta=rotation_matrix).reshape(batch_size, frame_count, 4, 4)


class WildVideoDataProcessor:
    """Materialize the validation-time CoCoNet input tensors used by wild inference."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.render_h5_root = cfg["render_root"]

    @staticmethod
    def _one_channel_mask(mask_ho: torch.Tensor) -> torch.Tensor:
        mask_h = mask_ho[0] > 0.5
        mask_o = mask_ho[1] > 0.5
        combined = torch.zeros_like(mask_h).float()
        combined[mask_o] = 1.0
        combined[mask_h] = -1.0
        return combined

    def process_input(self, dmap_xyz_init, i, input_data, mesh_diameter, human_translation, pose_init, render_data, rgb_render):
        rgb = torch.from_numpy(input_data["rgbmB"][:, :, :3] / 255.0).permute(2, 0, 1).float()
        dmap_xyz = torch.from_numpy(input_data["xyzB"]).permute(2, 0, 1).float()
        if "behave-fp+input" in self.render_h5_root:
            dmap_xyz /= 1000.0
        defer_mhr_spatial_scale = _cfg_value(self.cfg, "body_model", "smpl") == "mhr" and resolve_mhr_spatial_normalization_type(self.cfg) == MHR_SPATIAL_NORMALIZATION_HUMAN_HEIGHT
        if self.cfg["normalize_xyz"] and not defer_mhr_spatial_scale:
            dmap_xyz *= 2 / mesh_diameter
            dmap_xyz_init *= 2 / mesh_diameter
        if self.cfg.add_ho_mask:
            mask_ho = torch.from_numpy(input_data["rgbmB"][:, :, 3:] / 255.0).permute(2, 0, 1).float()
            if self.cfg.mask_encode_type in ["stack", "stack-occ"]:
                dmap_xyz = torch.cat([dmap_xyz, mask_ho], axis=0)
            elif self.cfg.mask_encode_type == "hum-obj-fullobj":
                dmap_xyz = torch.cat([dmap_xyz, mask_ho, torch.from_numpy(render_data["mask_o"][:, :, 1][None]).float()], 0)
            elif self.cfg.mask_encode_type == "obj-fullobj":
                dmap_xyz = torch.cat([dmap_xyz, mask_ho[1:], torch.from_numpy(render_data["mask_o"][:, :, 1][None]).float()], 0)
            elif self.cfg.mask_encode_type == "one-channel":
                dmap_xyz = torch.cat([dmap_xyz, self._one_channel_mask(mask_ho)[None]], axis=0)
            else:
                raise ValueError("Unknown mask encode type: " + self.cfg.mask_encode_type)
        if self.cfg.mask_rgb_bkg:
            assert self.cfg.add_ho_mask, "mask add ho mask for this setup!"
            mask_foreground = ((mask_ho[0:1] > 0.5) | (mask_ho[1:2] > 0.5)).expand(3, -1, -1)
            rgb[~mask_foreground] = 0.0
            dmap_xyz[:3][~mask_foreground] = 0
        if self.cfg.add_ho_mask:
            mask_render_full = np.mean(rgb_render, -1) > 0.01
            assert ("mask_o" in render_data) | ("fp+smpl" not in self.render_h5_root), "incorrect data format!"
            if "mask_o" in render_data:
                mask_render_obj = render_data["mask_o"]
                if len(mask_render_obj.shape) == 3:
                    mask_render_obj = mask_render_obj[:, :, 0]
                mask_render_human = mask_render_full & (~mask_render_obj)
            else:
                mask_render_obj = mask_render_full
                mask_render_human = mask_ho[0].cpu().numpy()
            mask_ho_render = torch.from_numpy(np.stack([mask_render_human, mask_render_obj], 0)).float()
            if self.cfg.mask_encode_type == "stack":
                dmap_xyz_a = torch.cat([dmap_xyz_init, mask_ho_render], 0)
            elif self.cfg.mask_encode_type == "one-channel":
                dmap_xyz_a = torch.cat([dmap_xyz_init, self._one_channel_mask(mask_ho_render)[None]], axis=0)
            elif self.cfg.mask_encode_type == "hum-obj-fullobj":
                dmap_xyz_a = torch.cat([dmap_xyz_init, mask_ho_render, torch.from_numpy(render_data["mask_o"][:, :, 1][None]).float()], 0)
            elif self.cfg.mask_encode_type == "obj-fullobj":
                dmap_xyz_a = torch.cat([dmap_xyz_init, mask_ho_render[1:], torch.from_numpy(render_data["mask_o"][:, :, 1][None]).float()], 0)
            elif self.cfg.mask_encode_type == "stack-occ":
                mask_o_render = torch.from_numpy(mask_render_obj)
                mask_o_render[mask_ho[0] > 0.5] = 0
                dmap_xyz_a = torch.cat([dmap_xyz_init, torch.stack([mask_ho[0], mask_o_render], 0)], 0)
            else:
                raise ValueError("Unknown mask encode type: " + self.cfg.mask_encode_type)
        else:
            dmap_xyz_a = dmap_xyz_init.clone()
        if self.cfg.subtract_transl:
            assert self.cfg["normalize_xyz"]
            invalid = dmap_xyz_a[2:3] < 0.01
            spatial_scale = 1.0 if defer_mhr_spatial_scale else 2 / mesh_diameter
            trans_ref = torch.from_numpy(pose_init[:3, 3]).reshape((3, 1, 1)) * spatial_scale
            if human_translation is not None:
                if self.cfg.trans_ref_type == "frame":
                    trans_ref = torch.from_numpy(human_translation[i].copy()).reshape((3, 1, 1)) * spatial_scale
                elif self.cfg.trans_ref_type == "1st-frame":
                    trans_ref = torch.from_numpy(human_translation[0].copy()).reshape((3, 1, 1)) * spatial_scale
                else:
                    raise ValueError(f"Unknown translation reference type: {self.cfg.trans_ref_type}")
            dmap_xyz_a[:3] = dmap_xyz_a[:3] - trans_ref
            dmap_xyz_a[:3][invalid.repeat(3, 1, 1)] = 0.0
            invalid = dmap_xyz[2:3] < 0.01
            dmap_xyz[:3] = dmap_xyz[:3] - trans_ref
            dmap_xyz[:3][invalid.repeat(3, 1, 1)] = 0.0
            if self.cfg.crop_xyz_3d:
                bound_min, bound_max = np.array([-1, -1, -1.0]), np.array([1, 1, 1.0])
                inside = ((dmap_xyz[0] < bound_max[0]) & (dmap_xyz[0] > bound_min[0]) & (dmap_xyz[1] < bound_max[1]) & (dmap_xyz[1] > bound_min[1]) & (dmap_xyz[1] < bound_max[2]) & (dmap_xyz[1] > bound_min[2]))
                dmap_xyz[:3][~inside[None].repeat(3, 1, 1)] = 0.0
        return dmap_xyz, dmap_xyz_a, rgb
