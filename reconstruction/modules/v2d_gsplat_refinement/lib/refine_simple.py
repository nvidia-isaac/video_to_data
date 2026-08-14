# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal 2DGS hand/object/camera refinement.

This is intentionally much smaller than ``refine.py``. It optimizes only:
  - static Gaussian appearance params: scale, color, opacity;
  - per-frame object root pose;
  - optional global object scale;
  - per-frame hand wrist/root pose (MANO global_orient + cam_t only);
  - per-frame background/camera pose for the static depth-initialized scene.

It does not optimize Gaussian offsets or Gaussian orientation. MANO finger
pose, MANO shape, hand scale, and object scale are optional via their LR flags.
Image losses are full-frame
L1 photometric error plus optional L1 segmentation-mask supervision, with
optional relative depth-gradient supervision from MoGe depth, and with
optional temporal smoothness on the absolute and hand/object-relative dynamic
poses.

Run inside the gsplat refinement environment, for example:

    python -m v2d.gsplat_refinement.lib.refine_simple \
        --frames_dir data/outputs/clip/frames \
        --depth_dir data/outputs/clip/depth \
        --intrinsics_path data/outputs/clip/intrinsics_stable.json \
        --object_mesh_path data/outputs/clip/mesh_scaled.obj \
        --object_poses_dir data/outputs/clip/poses \
        --object_mask_dir data/outputs/clip/masks/1 \
        --right_hand_pose_dir data/outputs/clip/hamer_aligned_filled/2 \
        --right_hand_mask_dir data/outputs/clip/masks/2 \
        --mano_assets_root data/weights/hamer/_DATA/data \
        --refined_object_poses_dir data/outputs/clip/poses_refined_simple \
        --refined_right_hand_pose_dir data/outputs/clip/hamer_refined_simple/2 \
        --refined_camera_poses_dir data/outputs/clip/camera_refined_simple \
        --overlay_path data/outputs/clip/refined_simple.mp4
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from v2d.common.datatypes import Transform3d

from .background import (
    BackgroundGaussians,
    BackgroundPoseField,
    init_background_from_depth,
    init_background_multiframe,
)
from .gaussians import (
    GaussianFrame,
    axis_angle_to_quat,
    concat_frames,
    init_hand_face_gaussians,
    init_object_face_gaussians_from_mesh,
    quat_mul,
)
from .io import (
    HandPoseTrack,
    ObjectPoseTrack,
    load_hand_poses,
    load_intrinsics,
    load_object_mesh,
    load_object_poses,
    save_hand_poses,
    save_object_poses,
)
from .losses import (
    depth_gradient_loss,
    quat_smoothness,
    rotation_smoothness,
    temporal_smoothness,
)
from .pose_fields import HandPoseField, ObjectPoseField, _quat_to_axis_angle


@dataclass
class HandSlot:
    side: str
    track: HandPoseTrack
    pose_field: HandPoseField
    gaussians: torch.nn.Module
    mask_dir: str
    output_pose_dir: str | None
    frame_to_pos: dict[int, int]


class SimpleFrameCache:
    """CPU cache for the small set of supervision this script uses."""

    def __init__(
        self,
        frame_indices: list[int],
        frames_dir: str,
        depth_dir: str | None,
        object_mask_dir: str,
        hand_mask_dirs: list[str],
        height: int,
        width: int,
        valid_mask_threshold: float = 0.04,
        valid_mask_erode_iters: int = 2,
    ) -> None:
        self.frame_indices = list(frame_indices)
        self.height = int(height)
        self.width = int(width)
        n = len(frame_indices)
        self.rgb = torch.zeros((n, height, width, 3), dtype=torch.float32)
        self.depth = (
            torch.empty((n, height, width), dtype=torch.float32)
            if depth_dir is not None else None
        )
        self.obj_mask = torch.zeros((n, height, width), dtype=torch.float32)
        self.hand_masks = [
            torch.zeros((n, height, width), dtype=torch.float32)
            for _ in hand_mask_dirs
        ]

        for t, fidx in enumerate(tqdm(frame_indices, ncols=80, desc="caching", unit="frame")):
            self.rgb[t] = _load_rgb_resized(_find_frame_file(frames_dir, fidx), width, height)
            if self.depth is not None and depth_dir is not None:
                self.depth[t] = _load_depth_resized(os.path.join(depth_dir, f"{fidx:06d}.png"), width, height)
            self.obj_mask[t] = _load_mask_or_zero(object_mask_dir, fidx, width, height)
            for k, mask_dir in enumerate(hand_mask_dirs):
                self.hand_masks[k][t] = _load_mask_or_zero(mask_dir, fidx, width, height)

        if valid_mask_threshold <= 0:
            valid = torch.ones((height, width), dtype=torch.float32)
        else:
            max_brightness = self.rgb.amax(dim=0).amax(dim=-1)
            valid = (max_brightness > float(valid_mask_threshold)).float()
            if valid_mask_erode_iters > 0:
                m = valid.unsqueeze(0).unsqueeze(0)
                for _ in range(int(valid_mask_erode_iters)):
                    m = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
                valid = m.squeeze(0).squeeze(0)
        self.valid_mask = valid

        try:
            self.rgb = self.rgb.pin_memory()
            if self.depth is not None:
                self.depth = self.depth.pin_memory()
            self.obj_mask = self.obj_mask.pin_memory()
            self.hand_masks = [m.pin_memory() for m in self.hand_masks]
            self.valid_mask = self.valid_mask.pin_memory()
        except RuntimeError:
            pass

    def union_mask(self, t: int, device: torch.device | str) -> torch.Tensor:
        m = self.obj_mask[t].to(device, non_blocking=True).clone()
        for hm in self.hand_masks:
            m = torch.maximum(m, hm[t].to(device, non_blocking=True))
        return m


def _target_rgb(
    cache: SimpleFrameCache,
    t: int,
    device: torch.device | str,
    mask_background: bool,
) -> torch.Tensor:
    target = cache.rgb[t].to(device, non_blocking=True)
    if mask_background:
        target = target * cache.union_mask(t, device).unsqueeze(-1)
    return target


def _find_frame_file(frames_dir: str, fidx: int) -> str:
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(frames_dir, f"{fidx:06d}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Frame {fidx:06d} not found in {frames_dir}")


def _list_frame_indices(frames_dir: str) -> set[int]:
    out: set[int] = set()
    for path in glob.glob(os.path.join(frames_dir, "*.png")) + glob.glob(os.path.join(frames_dir, "*.jpg")):
        try:
            out.add(int(os.path.splitext(os.path.basename(path))[0]))
        except ValueError:
            continue
    return out


def _load_rgb_resized(path: str, width: int, height: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _load_depth_resized(path: str, width: int, height: int) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Depth frame not found: {path}")
    img = Image.open(path)
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    px = np.asarray(img).astype(np.float32)
    with np.errstate(divide="ignore"):
        depth = 1.0 / (px / 65535.0) - 1.0
    return torch.from_numpy(depth)


def _load_mask_or_zero(mask_dir: str, fidx: int, width: int, height: int) -> torch.Tensor:
    path = os.path.join(mask_dir, f"{fidx:06d}.png")
    if not os.path.exists(path):
        return torch.zeros((height, width), dtype=torch.float32)
    img = Image.open(path)
    if img.size != (width, height):
        img = img.resize((width, height), Image.NEAREST)
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.0:
        arr = arr / 255.0
    return torch.from_numpy((arr > 0.5).astype(np.float32))


def _restrict_object_track(track: ObjectPoseTrack, keep: set[int]) -> ObjectPoseTrack:
    idxs = [i for i, fidx in enumerate(track.frame_indices) if fidx in keep]
    if not idxs:
        raise RuntimeError("No object pose frames overlap the available RGB frames.")
    return ObjectPoseTrack(
        rotations=track.rotations[idxs],
        translations=track.translations[idxs],
        scales=track.scales[idxs],
        frame_indices=[track.frame_indices[i] for i in idxs],
    )


def _restrict_hand_track(track: HandPoseTrack, keep: set[int]) -> HandPoseTrack:
    idxs = [i for i, fidx in enumerate(track.frame_indices) if fidx in keep]
    if not idxs:
        raise RuntimeError("A hand pose track has no frames overlapping the training frames.")
    return HandPoseTrack(
        global_orient=track.global_orient[idxs],
        hand_pose=track.hand_pose[idxs],
        betas=track.betas[idxs],
        cam_t=track.cam_t[idxs],
        is_right=track.is_right,
        frame_indices=[track.frame_indices[i] for i in idxs],
        raw_records=[track.raw_records[i] for i in idxs],
        hand_scale=track.hand_scale,
    )


def _freeze_all(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def _enable(module: torch.nn.Module, names: list[str]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, torch.nn.Parameter):
            value.requires_grad_(True)
            params.append(value)
    return params


def _trainable_gaussian_params(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    _freeze_all(module)
    return _enable(module, ["_log_scale", "_opacity_logit", "_color"])


def _load_relative_w2c_poses(
    poses_dir: str,
    frame_indices: list[int],
    ref_t: int,
) -> dict[int, np.ndarray]:
    ref_fidx = frame_indices[ref_t]
    ref_path = os.path.join(poses_dir, f"{ref_fidx:06d}.json")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"background_pose_init_dir is set but reference pose {ref_path} is missing."
        )
    M_ref = Transform3d.load(ref_path).to_matrix()
    out: dict[int, np.ndarray] = {ref_t: np.eye(4)}
    for t, fidx in enumerate(frame_indices):
        if t == ref_t:
            continue
        path = os.path.join(poses_dir, f"{fidx:06d}.json")
        if not os.path.exists(path):
            continue
        tf = Transform3d.load(path)
        if any(abs(s - 1.0) > 1e-3 for s in tf.scale):
            continue
        out[t] = np.linalg.inv(tf.to_matrix()) @ M_ref
    return out


def _seed_background_pose_field(
    bg_pose_field: BackgroundPoseField,
    poses: dict[int, np.ndarray],
) -> tuple[int, int]:
    device = bg_pose_field.axis_angle.device
    aa = bg_pose_field.axis_angle.detach().clone()
    tr = bg_pose_field.translation.detach().clone()
    loaded = missing = 0
    for t in range(aa.shape[0]):
        if t not in poses:
            missing += 1
            continue
        M = poses[t]
        R = torch.from_numpy(M[:3, :3]).to(device=device, dtype=torch.float32)
        q = _rotmat_to_quat(R)
        aa[t] = _quat_to_axis_angle(q)
        tr[t] = torch.from_numpy(M[:3, 3]).to(device=device, dtype=torch.float32)
        loaded += 1
    with torch.no_grad():
        bg_pose_field.axis_angle.copy_(aa)
        bg_pose_field.translation.copy_(tr)
    return loaded, missing


def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    import roma

    q_xyzw = roma.rotmat_to_unitquat(R)
    return torch.cat([q_xyzw[..., 3:4], q_xyzw[..., :3]], dim=-1)


def _anchor_camera_reference(bg_pose_field: BackgroundPoseField) -> None:
    ref_t = int(bg_pose_field.ref_frame_t)
    with torch.no_grad():
        bg_pose_field.axis_angle[ref_t].zero_()
        bg_pose_field.translation[ref_t].zero_()


@dataclass
class ObjectSDFGrid:
    """Differentiable object-space SDF grid.

    ``values`` stores signed distance with positive outside the object and
    negative inside. Trimesh uses the opposite sign convention, so construction
    flips the sign once. Querying uses ``grid_sample`` so gradients flow to the
    queried points, hence to hand/object pose and scale parameters.
    """

    values: torch.Tensor      # (1, 1, D, H, W), D=z, H=y, W=x
    bounds_min: torch.Tensor  # (3,) object-frame xyz
    bounds_max: torch.Tensor  # (3,) object-frame xyz

    def query(self, points_obj: torch.Tensor) -> torch.Tensor:
        if points_obj.numel() == 0:
            return points_obj.new_zeros((0,))
        span = (self.bounds_max - self.bounds_min).clamp_min(1e-8)
        coords = 2.0 * (points_obj - self.bounds_min) / span - 1.0
        grid = coords.view(1, -1, 1, 1, 3)
        sampled = F.grid_sample(
            self.values.to(device=points_obj.device, dtype=points_obj.dtype),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.view(-1)


def _build_object_sdf_grid(
    vertices: torch.Tensor,
    faces: np.ndarray,
    resolution: int,
    margin: float,
    device: torch.device,
) -> ObjectSDFGrid:
    res = int(resolution)
    if res < 8:
        raise ValueError("object_sdf_resolution must be >= 8")
    verts_np = vertices.detach().cpu().numpy().astype(np.float64)
    faces_np = np.asarray(faces, dtype=np.int64)
    try:
        import trimesh
        from trimesh.proximity import signed_distance
    except ImportError as exc:
        raise RuntimeError(
            "Hand/object penetration loss requires trimesh and its proximity dependencies."
        ) from exc

    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    if not bool(getattr(mesh, "is_watertight", False)):
        print("Warning: object mesh is not watertight; hand/object SDF signs may be approximate.")

    bounds_min_np = verts_np.min(axis=0)
    bounds_max_np = verts_np.max(axis=0)
    extent = bounds_max_np - bounds_min_np
    pad = max(float(margin) * 4.0, float(extent.max()) * 0.05, 1e-3)
    bounds_min_np = bounds_min_np - pad
    bounds_max_np = bounds_max_np + pad

    xs = np.linspace(bounds_min_np[0], bounds_max_np[0], res, dtype=np.float32)
    ys = np.linspace(bounds_min_np[1], bounds_max_np[1], res, dtype=np.float32)
    zs = np.linspace(bounds_min_np[2], bounds_max_np[2], res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=-1)

    sdf = np.empty((points.shape[0],), dtype=np.float32)
    chunk = 200_000
    print(f"Building object SDF grid: resolution={res}, points={points.shape[0]}, padding={pad:.4g}")
    try:
        for start in tqdm(range(0, points.shape[0], chunk), ncols=80, desc="object_sdf", unit="chunk"):
            end = min(start + chunk, points.shape[0])
            # trimesh signed_distance is positive inside, negative outside.
            sdf[start:end] = -signed_distance(mesh, points[start:end]).astype(np.float32)
    except BaseException as exc:
        raise RuntimeError(
            "Failed to build object SDF grid for hand/object penetration. "
            "This usually means trimesh's optional rtree dependency is missing "
            "or the object mesh is invalid."
        ) from exc

    sdf_xyz = torch.from_numpy(sdf.reshape(res, res, res))
    values = sdf_xyz.permute(2, 1, 0).contiguous().view(1, 1, res, res, res).to(device)
    bounds_min = torch.tensor(bounds_min_np, dtype=torch.float32, device=device)
    bounds_max = torch.tensor(bounds_max_np, dtype=torch.float32, device=device)
    return ObjectSDFGrid(values=values, bounds_min=bounds_min, bounds_max=bounds_max)


def _hand_object_penetration_loss(
    t: int,
    fidx: int,
    obj_pose_field: ObjectPoseField,
    obj_gaussians: torch.nn.Module,
    hand_slots: list[HandSlot],
    object_sdf: ObjectSDFGrid,
    margin: float,
    max_hand_verts: int,
) -> torch.Tensor:
    R_obj, t_obj = obj_pose_field(t)
    if hasattr(obj_gaussians, "object_scale"):
        s_obj = obj_gaussians.object_scale().clamp_min(1e-6)
    else:
        s_obj = R_obj.new_ones(())
    losses: list[torch.Tensor] = []
    for slot in hand_slots:
        local_t = slot.frame_to_pos.get(fidx)
        if local_t is None:
            continue
        verts_cam, _ = slot.pose_field.posed_verts_and_rotmats_camera(local_t)
        if max_hand_verts > 0 and verts_cam.shape[0] > int(max_hand_verts):
            idx = torch.linspace(
                0,
                verts_cam.shape[0] - 1,
                int(max_hand_verts),
                device=verts_cam.device,
            ).round().to(torch.long)
            verts_cam = verts_cam[idx]
        verts_obj = ((verts_cam - t_obj) @ R_obj) / s_obj
        sdf_metric = object_sdf.query(verts_obj) * s_obj
        penetration = (float(margin) - sdf_metric).clamp_min(0.0)
        losses.append(penetration.square().mean())
    if not losses:
        return R_obj.sum() * 0.0
    return torch.stack(losses).mean()


class VGG16PerceptualLoss(torch.nn.Module):
    """VGG16 feature-space L1 loss for RGB images in [0, 1]."""

    def __init__(
        self,
        weights_path: str | None = None,
        resize: int = 224,
        layer_indices: tuple[int, ...] = (3, 8, 15, 22),
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights, vgg16
        except ImportError as exc:
            raise RuntimeError(
                "VGG perceptual loss requires torchvision in the gsplat refinement environment."
            ) from exc

        self.layer_indices = set(int(i) for i in layer_indices)
        self.resize = int(resize)
        if weights_path is not None:
            weights_path = os.path.abspath(os.path.expanduser(weights_path))
            model = vgg16(weights=None)
            if not os.path.exists(weights_path):
                weights = VGG16_Weights.IMAGENET1K_V1
                os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
                print(f"Downloading VGG16 ImageNet weights -> {weights_path}")
                try:
                    state = torch.hub.load_state_dict_from_url(
                        weights.url,
                        model_dir=os.path.dirname(weights_path) or ".",
                        file_name=os.path.basename(weights_path),
                        map_location="cpu",
                        progress=True,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to download torchvision VGG16 ImageNet weights to {weights_path}."
                    ) from exc
            else:
                state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
        else:
            try:
                model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load torchvision VGG16 ImageNet weights. "
                    "Provide --vgg_weights_path with a local vgg16 state_dict, "
                    "or ensure torchvision's weight cache/network access is available."
                ) from exc
        self.features = model.features[: max(self.layer_indices) + 1].eval()
        for param in self.features.parameters():
            param.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def _preprocess(
        self,
        image: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = image.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0)
        if valid_mask is not None:
            x = x * valid_mask.to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        if self.resize > 0 and (x.shape[-2] != self.resize or x.shape[-1] != self.resize):
            x = F.interpolate(x, size=(self.resize, self.resize), mode="bilinear", align_corners=False)
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                outputs.append(x)
        return outputs

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_features = self._features(self._preprocess(pred_rgb, valid_mask))
        with torch.no_grad():
            target_features = self._features(self._preprocess(target_rgb, valid_mask))
        loss = pred_rgb.new_zeros(())
        for pred_f, target_f in zip(pred_features, target_features):
            loss = loss + F.l1_loss(pred_f, target_f)
        return loss / float(max(len(pred_features), 1))


def _render_2dgs_features(
    frame: GaussianFrame,
    K: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    colors: torch.Tensor | None = None,
    clamp_colors: bool = True,
) -> torch.Tensor:
    try:
        from gsplat.rendering import rasterization_2dgs
    except ImportError as exc:
        raise RuntimeError(
            "refine_simple.py requires gsplat.rendering.rasterization_2dgs. "
            "Install/run in the gsplat refinement Docker image with gsplat>=1.4."
        ) from exc

    device = frame.means.device
    render_colors = frame.colors if colors is None else colors
    if clamp_colors:
        render_colors = render_colors.clamp(0.0, 1.0)
    n_channels = int(render_colors.shape[-1])
    viewmats = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    kwargs = dict(
        means=frame.means,
        quats=frame.quats,
        scales=frame.scales,
        opacities=frame.opacities,
        colors=render_colors,
        viewmats=viewmats,
        Ks=K.unsqueeze(0),
        width=int(width),
        height=int(height),
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        render_mode="RGB",
        backgrounds=torch.zeros(1, n_channels, device=device, dtype=torch.float32),
    )
    try:
        result = rasterization_2dgs(**kwargs)
    except TypeError:
        # Some gsplat builds expose a narrower 2DGS signature. Retry with the
        # universally required raster inputs before surfacing the error.
        kwargs.pop("render_mode", None)
        result = rasterization_2dgs(**kwargs)

    colors = result[0] if isinstance(result, tuple) else result
    if colors.ndim == 4:
        colors = colors[0]
    return colors


def _render_2dgs_rgb(
    frame: GaussianFrame,
    K: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> torch.Tensor:
    colors = _render_2dgs_features(frame, K, width, height, near_plane, far_plane)
    return colors[..., :3]


def _render_2dgs_depth(
    frame: GaussianFrame,
    K: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth_feature = frame.means[:, 2:3].clamp_min(float(near_plane))
    alpha_feature = torch.ones_like(depth_feature)
    rendered = _render_2dgs_features(
        frame,
        K,
        width,
        height,
        near_plane=near_plane,
        far_plane=far_plane,
        colors=torch.cat([depth_feature, alpha_feature], dim=-1),
        clamp_colors=False,
    )
    depth_accum = rendered[..., 0]
    alpha = rendered[..., 1].clamp(0.0, 1.0)
    depth = depth_accum / alpha.clamp_min(1e-4)
    return depth, alpha


def _current_frame(
    t: int,
    fidx: int,
    obj_gaussians: torch.nn.Module,
    obj_pose_field: ObjectPoseField,
    hand_slots: list[HandSlot],
    bg_gaussians: BackgroundGaussians | None,
    bg_pose_field: BackgroundPoseField | None,
) -> GaussianFrame:
    R_obj, t_obj = obj_pose_field(t)
    frames = [obj_gaussians(R_obj, t_obj)]
    for slot in hand_slots:
        local_t = slot.frame_to_pos.get(fidx)
        if local_t is None:
            continue
        verts, Rv = slot.pose_field.posed_verts_and_rotmats_camera(local_t)
        frames.append(slot.gaussians(verts, Rv))
    if bg_gaussians is not None and bg_pose_field is not None:
        R_bg, t_bg = bg_pose_field(t)
        frames.append(bg_gaussians(R_bg, t_bg))
    return concat_frames(frames)


def _label_block(
    n: int,
    n_classes: int,
    class_idx: int | None,
    device: torch.device,
) -> torch.Tensor:
    labels = torch.zeros((int(n), int(n_classes)), device=device, dtype=torch.float32)
    if class_idx is not None and n > 0:
        labels[:, int(class_idx)] = 1.0
    return labels


def _current_frame_with_labels(
    t: int,
    fidx: int,
    obj_gaussians: torch.nn.Module,
    obj_pose_field: ObjectPoseField,
    hand_slots: list[HandSlot],
    bg_gaussians: BackgroundGaussians | None,
    bg_pose_field: BackgroundPoseField | None,
    n_classes: int,
) -> tuple[GaussianFrame, torch.Tensor]:
    R_obj, t_obj = obj_pose_field(t)
    obj_frame = obj_gaussians(R_obj, t_obj)
    frames = [obj_frame]
    labels = [_label_block(obj_frame.means.shape[0], n_classes, 0, obj_frame.means.device)]

    for slot_i, slot in enumerate(hand_slots):
        local_t = slot.frame_to_pos.get(fidx)
        if local_t is None:
            continue
        verts, Rv = slot.pose_field.posed_verts_and_rotmats_camera(local_t)
        hand_frame = slot.gaussians(verts, Rv)
        frames.append(hand_frame)
        labels.append(_label_block(
            hand_frame.means.shape[0], n_classes, slot_i + 1, hand_frame.means.device
        ))

    if bg_gaussians is not None and bg_pose_field is not None:
        R_bg, t_bg = bg_pose_field(t)
        bg_frame = bg_gaussians(R_bg, t_bg)
        frames.append(bg_frame)
        labels.append(_label_block(bg_frame.means.shape[0], n_classes, None, bg_frame.means.device))

    return concat_frames(frames), torch.cat(labels, dim=0).contiguous()


def _segmentation_mask_loss(
    rendered_class: torch.Tensor,
    cache: SimpleFrameCache,
    t: int,
    valid_mask: torch.Tensor,
    valid_denom: torch.Tensor,
) -> torch.Tensor:
    n_classes = int(rendered_class.shape[-1])
    target = rendered_class.new_zeros(rendered_class.shape)
    target[..., 0] = cache.obj_mask[t].to(rendered_class.device, non_blocking=True)
    for slot_i, hand_mask in enumerate(cache.hand_masks):
        if slot_i + 1 >= n_classes:
            break
        target[..., slot_i + 1] = hand_mask[t].to(rendered_class.device, non_blocking=True)
    diff = (rendered_class - target).abs()
    return (diff * valid_mask.unsqueeze(-1)).sum() / (
        valid_denom * float(max(n_classes, 1))
    )


def _normalized_smooth(loss: torch.Tensor, n: int) -> torch.Tensor:
    return loss / float(max(n - 1, 1))


def _cosine_lr_scale(step: int, total_steps: int, min_factor: float) -> float:
    if total_steps <= 1:
        return float(min_factor)
    progress = min(max(int(step), 0), int(total_steps) - 1) / float(int(total_steps) - 1)
    return float(min_factor) + 0.5 * (1.0 - float(min_factor)) * (1.0 + math.cos(math.pi * progress))


def _first_nonfinite_tensor(
    tensors: list[tuple[str, torch.Tensor | None]],
) -> str | None:
    """Return the first named tensor containing NaN or Inf, if any."""
    for name, tensor in tensors:
        if tensor is not None and not bool(torch.isfinite(tensor).all()):
            return name
    return None


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _hand_root_quats_camera(slot: HandSlot, t_idx: torch.Tensor) -> torch.Tensor:
    q_hand = axis_angle_to_quat(slot.pose_field.global_orient[t_idx])
    if not slot.pose_field.is_right:
        q_hand = torch.cat([q_hand[..., :2], -q_hand[..., 2:]], dim=-1)
    return q_hand


def _hand_object_relative_smoothness(
    obj_pose_field: ObjectPoseField,
    slot: HandSlot,
    w_smooth_hand_object_relative_rot: float,
    w_smooth_hand_object_relative_trans: float,
) -> torch.Tensor:
    loss = obj_pose_field.axis_angle.new_zeros(())
    if (
        float(w_smooth_hand_object_relative_rot) <= 0.0
        and float(w_smooth_hand_object_relative_trans) <= 0.0
    ):
        return loss

    object_frame_to_pos = {fidx: i for i, fidx in enumerate(obj_pose_field.frame_indices)}
    paired: list[tuple[int, int]] = []
    for hand_t, fidx in enumerate(slot.pose_field.frame_indices):
        obj_t = object_frame_to_pos.get(fidx)
        if obj_t is not None:
            paired.append((hand_t, obj_t))
    if len(paired) < 2:
        return loss

    device = obj_pose_field.axis_angle.device
    hand_idx = torch.tensor([h for h, _ in paired], device=device, dtype=torch.long)
    obj_idx = torch.tensor([o for _, o in paired], device=device, dtype=torch.long)

    R_obj, t_obj = obj_pose_field.batched_forward(obj_idx)
    q_obj = axis_angle_to_quat(obj_pose_field.axis_angle[obj_idx])
    q_hand = _hand_root_quats_camera(slot, hand_idx)
    t_hand = slot.pose_field.cam_t[hand_idx]

    if float(w_smooth_hand_object_relative_rot) > 0.0:
        q_rel = quat_mul(_quat_conj(q_obj), q_hand)
        loss = loss + float(w_smooth_hand_object_relative_rot) * _normalized_smooth(
            quat_smoothness(q_rel),
            len(paired),
        )
    if float(w_smooth_hand_object_relative_trans) > 0.0:
        # Express hand root translation in the object frame, removing common
        # camera-frame motion before applying temporal smoothness.
        t_rel = torch.matmul((t_hand - t_obj).unsqueeze(1), R_obj).squeeze(1)
        loss = loss + float(w_smooth_hand_object_relative_trans) * _normalized_smooth(
            temporal_smoothness(t_rel),
            len(paired),
        )
    return loss


def _pose_smoothness(
    obj_pose_field: ObjectPoseField,
    hand_slots: list[HandSlot],
    bg_pose_field: BackgroundPoseField | None,
    w_smooth_object_rot: float,
    w_smooth_object_trans: float,
    w_smooth_hand_rot: float,
    w_smooth_hand_articulation: float,
    w_smooth_hand_trans: float,
    w_smooth_hand_object_relative_rot: float,
    w_smooth_hand_object_relative_trans: float,
    w_smooth_camera_rot: float,
    w_smooth_camera_trans: float,
) -> torch.Tensor:
    loss = obj_pose_field.axis_angle.new_zeros(())
    T = obj_pose_field.num_frames()
    loss = loss + float(w_smooth_object_rot) * _normalized_smooth(
        rotation_smoothness(obj_pose_field.axis_angle), T
    )
    loss = loss + float(w_smooth_object_trans) * _normalized_smooth(
        temporal_smoothness(obj_pose_field.translation), T
    )
    if bg_pose_field is not None:
        loss = loss + float(w_smooth_camera_rot) * _normalized_smooth(
            rotation_smoothness(bg_pose_field.axis_angle), T
        )
        loss = loss + float(w_smooth_camera_trans) * _normalized_smooth(
            temporal_smoothness(bg_pose_field.translation), T
        )
    for slot in hand_slots:
        n = slot.pose_field.num_frames()
        loss = loss + float(w_smooth_hand_rot) * _normalized_smooth(
            rotation_smoothness(slot.pose_field.global_orient), n
        )
        loss = loss + float(w_smooth_hand_articulation) * _normalized_smooth(
            rotation_smoothness(slot.pose_field.hand_pose.view(n, 15, 3)), n
        )
        loss = loss + float(w_smooth_hand_trans) * _normalized_smooth(
            temporal_smoothness(slot.pose_field.cam_t), n
        )
        loss = loss + _hand_object_relative_smoothness(
            obj_pose_field,
            slot,
            w_smooth_hand_object_relative_rot,
            w_smooth_hand_object_relative_trans,
        )
    return loss


def _save_camera_poses(
    bg_pose_field: BackgroundPoseField,
    frame_indices: list[int],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    q = axis_angle_to_quat(bg_pose_field.axis_angle.detach()).cpu().tolist()
    tr = bg_pose_field.translation.detach().cpu().tolist()
    for t, fidx in enumerate(frame_indices):
        Transform3d(rotation=q[t], translation=tr[t], scale=[1.0, 1.0, 1.0]).save(
            os.path.join(output_dir, f"{fidx:06d}.json")
        )


def _write_overlay_video(
    output_path: str,
    cache: SimpleFrameCache,
    K: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    obj_gaussians: torch.nn.Module,
    obj_pose_field: ObjectPoseField,
    hand_slots: list[HandSlot],
    bg_gaussians: BackgroundGaussians | None,
    bg_pose_field: BackgroundPoseField | None,
    fps: float = 30.0,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        with torch.no_grad():
            for t, fidx in enumerate(tqdm(cache.frame_indices, desc="render overlay", ncols=80)):
                frame = _current_frame(
                    t, fidx, obj_gaussians, obj_pose_field, hand_slots,
                    bg_gaussians, bg_pose_field,
                )
                rgb = _render_2dgs_rgb(frame, K, width, height)
                arr = (rgb.clamp(0.0, 1.0) * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(arr).save(os.path.join(tmp, f"{t:06d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-r", str(fps),
            "-i", os.path.join(tmp, "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            output_path,
        ], check=True)


def _class_render_to_rgb(rendered_class: torch.Tensor) -> torch.Tensor:
    """Colorize alpha-composited class channels for progress visualization."""
    if rendered_class.numel() == 0 or rendered_class.shape[-1] == 0:
        h, w = rendered_class.shape[:2]
        return rendered_class.new_zeros((h, w, 3))
    palette = rendered_class.new_tensor([
        [1.00, 0.24, 0.12],  # object
        [0.10, 0.76, 1.00],  # hand slot 0
        [1.00, 0.22, 0.86],  # hand slot 1
        [0.70, 1.00, 0.18],
        [1.00, 0.78, 0.12],
    ])
    n_classes = int(rendered_class.shape[-1])
    if n_classes > palette.shape[0]:
        repeats = (n_classes + palette.shape[0] - 1) // palette.shape[0]
        palette = palette.repeat(repeats, 1)
    return (rendered_class.clamp(0.0, 1.0) @ palette[:n_classes]).clamp(0.0, 1.0)


def _dump_progress_frame(
    progress_dir: str,
    step: int,
    t: int,
    cache: SimpleFrameCache,
    K: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    obj_gaussians: torch.nn.Module,
    obj_pose_field: ObjectPoseField,
    hand_slots: list[HandSlot],
    bg_gaussians: BackgroundGaussians | None,
    bg_pose_field: BackgroundPoseField | None,
    mask_background: bool = False,
) -> None:
    os.makedirs(progress_dir, exist_ok=True)
    with torch.no_grad():
        fidx = cache.frame_indices[t]
        target = _target_rgb(cache, t, device, mask_background)
        n_mask_classes = 1 + len(hand_slots)
        frame, labels = _current_frame_with_labels(
            t, fidx, obj_gaussians, obj_pose_field, hand_slots,
            bg_gaussians, bg_pose_field, n_mask_classes,
        )
        render_frame = replace(
            frame,
            colors=torch.cat([frame.colors, labels.to(frame.colors.dtype)], dim=-1),
        )
        rendered = _render_2dgs_features(render_frame, K, width, height)
        pred = rendered[..., :3].clamp(0.0, 1.0)
        rendered_mask = _class_render_to_rgb(rendered[..., 3:])
        err = (pred - target).abs().mul(4.0).clamp(0.0, 1.0)
        panel = torch.cat([target, pred, rendered_mask, err], dim=1)
        arr = (panel * 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(arr).save(os.path.join(progress_dir, f"{step:06d}.png"))


def _stitch_progress_video(progress_dir: str, fps: float = 12.0) -> None:
    if not os.path.isdir(progress_dir):
        return
    pngs = [p for p in os.listdir(progress_dir) if p.endswith(".png")]
    if not pngs:
        return
    out_path = os.path.join(os.path.dirname(progress_dir.rstrip("/")), "progress_simple.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-r", str(fps),
        "-pattern_type", "glob",
        "-i", os.path.join(progress_dir, "*.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        out_path,
    ], check=True)
    print(f"Wrote simple-refinement progress video -> {out_path}")


def refine_simple(
    frames_dir: str,
    depth_dir: str | None,
    intrinsics_path: str,
    object_mesh_path: str,
    object_poses_dir: str,
    object_mask_dir: str,
    refined_object_poses_dir: str,
    overlay_path: str,
    left_hand_pose_dir: str | None = None,
    left_hand_mask_dir: str | None = None,
    right_hand_pose_dir: str | None = None,
    right_hand_mask_dir: str | None = None,
    refined_left_hand_pose_dir: str | None = None,
    refined_right_hand_pose_dir: str | None = None,
    refined_camera_poses_dir: str | None = None,
    mano_assets_root: str | None = None,
    background_pose_init_dir: str | None = None,
    n_epochs: int = 20,
    batch_size: int = 4,
    train_resolution_scale: float = 0.5,
    lr_gaussians: float = 1e-2,
    lr_object_pose: float = 1e-3,
    lr_object_scale: float = 0.0,
    lr_hand_pose: float = 1e-3,
    lr_hand_articulation: float = 0.0,
    lr_hand_shape: float = 0.0,
    lr_hand_scale: float = 0.0,
    lr_camera_pose: float = 1e-4,
    lr_schedule: str = "constant",
    lr_cosine_min_factor: float = 0.0,
    w_smooth_object_rot: float = 0.01,
    w_smooth_object_trans: float = 0.01,
    w_smooth_hand_rot: float = 0.01,
    w_smooth_hand_articulation: float = 0.01,
    w_smooth_hand_trans: float = 0.01,
    w_smooth_hand_object_relative_rot: float = 0.0,
    w_smooth_hand_object_relative_trans: float = 0.0,
    w_smooth_camera_rot: float = 0.01,
    w_smooth_camera_trans: float = 0.01,
    w_mask: float = 1.0,
    w_relative_depth: float = 0.0,
    w_perceptual: float = 0.0,
    vgg_weights_path: str | None = None,
    perceptual_resize: int = 224,
    w_hand_object_penetration: float = 0.0,
    hand_object_penetration_margin: float = 0.003,
    object_sdf_resolution: int = 96,
    hand_object_penetration_max_verts: int = 0,
    mask_background: bool = False,
    bg_ref_frame: int | None = None,
    bg_init_stride: int = 10,
    bg_voxel_size: float = 0.005,
    bg_max_points: int = 50000,
    valid_mask_threshold: float = 0.04,
    valid_mask_erode_iters: int = 2,
    face_normal_thin_factor_obj: float = 0.25,
    face_normal_thin_factor_hand: float = 0.25,
    init_opacity_obj: float = 0.9,
    init_opacity_hand: float = 0.9,
    init_opacity_bg: float = 0.9,
    init_gaussian_scale_factor: float = 1.0,
    render_every: int = 25,
    progress_dir: str | None = None,
    debug_frame_idx: int | None = None,
    fps: float = 30.0,
    seed: int = 0,
    device: str = "cuda",
) -> None:
    if (left_hand_pose_dir or right_hand_pose_dir) and mano_assets_root is None:
        raise ValueError("mano_assets_root is required when refining hand tracks")
    if left_hand_pose_dir and left_hand_mask_dir is None:
        raise ValueError("left_hand_mask_dir is required with left_hand_pose_dir")
    if right_hand_pose_dir and right_hand_mask_dir is None:
        raise ValueError("right_hand_mask_dir is required with right_hand_pose_dir")
    if not mask_background and depth_dir is None:
        raise ValueError("depth_dir is required unless --mask-background is set")
    if float(w_relative_depth) > 0.0 and depth_dir is None:
        raise ValueError("depth_dir is required when --w_relative_depth is nonzero")
    lr_schedule = str(lr_schedule).lower()
    if lr_schedule not in {"constant", "cosine"}:
        raise ValueError("lr_schedule must be 'constant' or 'cosine'")
    lr_cosine_min_factor = float(lr_cosine_min_factor)
    if lr_cosine_min_factor < 0.0 or lr_cosine_min_factor > 1.0:
        raise ValueError("lr_cosine_min_factor must be in [0, 1]")

    torch.manual_seed(seed)
    device_t = torch.device(device)

    K, width, height = load_intrinsics(intrinsics_path, str(device_t))
    if train_resolution_scale != 1.0:
        s = float(train_resolution_scale)
        K = K.clone()
        K[0, 0] *= s
        K[1, 1] *= s
        K[0, 2] *= s
        K[1, 2] *= s
        width = max(1, int(round(width * s)))
        height = max(1, int(round(height * s)))
        print(f"Training/render resolution: {width}x{height} (scale={s:.3f})")

    available_frames = _list_frame_indices(frames_dir)
    object_track = _restrict_object_track(load_object_poses(object_poses_dir, str(device_t)), available_frames)
    frame_indices = list(object_track.frame_indices)
    keep = set(frame_indices)

    raw_hand_specs: list[tuple[str, str, str, str | None]] = []
    if left_hand_pose_dir is not None:
        raw_hand_specs.append(("left", left_hand_pose_dir, left_hand_mask_dir, refined_left_hand_pose_dir))
    if right_hand_pose_dir is not None:
        raw_hand_specs.append(("right", right_hand_pose_dir, right_hand_mask_dir, refined_right_hand_pose_dir))

    hand_tracks: list[tuple[str, HandPoseTrack, str, str | None]] = []
    for side, pose_dir, mask_dir, out_dir in raw_hand_specs:
        ht = _restrict_hand_track(load_hand_poses(pose_dir, str(device_t)), keep)
        hand_tracks.append((side, ht, mask_dir, out_dir))

    cache = SimpleFrameCache(
        frame_indices=frame_indices,
        frames_dir=frames_dir,
        depth_dir=depth_dir if (not mask_background or float(w_relative_depth) > 0.0) else None,
        object_mask_dir=object_mask_dir,
        hand_mask_dirs=[h[2] for h in hand_tracks],
        height=height,
        width=width,
        valid_mask_threshold=valid_mask_threshold,
        valid_mask_erode_iters=valid_mask_erode_iters,
    )
    valid_mask = cache.valid_mask.to(device_t, non_blocking=True)
    valid_denom = valid_mask.sum().clamp_min(1.0)

    obj_pose_field = ObjectPoseField(object_track).to(device_t)
    obj_verts, obj_colors, obj_faces = load_object_mesh(object_mesh_path, str(device_t))
    obj_gaussians = init_object_face_gaussians_from_mesh(
        obj_verts, obj_faces, obj_colors,
        normal_thin_factor=face_normal_thin_factor_obj,
        init_opacity=init_opacity_obj,
        init_scale_factor=init_gaussian_scale_factor,
    ).to(device_t)
    obj_g_params = _trainable_gaussian_params(obj_gaussians)
    if float(lr_object_scale) > 0.0 and hasattr(obj_gaussians, "_log_scale_global"):
        obj_gaussians._log_scale_global.requires_grad_(True)
    object_trainable = "appearance"
    if getattr(obj_gaussians, "_log_scale_global", None) is not None and obj_gaussians._log_scale_global.requires_grad:
        object_trainable += ", global scale"
    print(f"Object: {obj_gaussians.num_gaussians()} face-anchored Gaussians "
          f"(trainable: {object_trainable})")

    hand_slots: list[HandSlot] = []
    for slot_i, (side, ht, mask_dir, out_dir) in enumerate(hand_tracks):
        pf = HandPoseField(
            ht,
            mano_assets_root,
            device=device_t,
            learn_hand_scale=float(lr_hand_scale) > 0.0,
        ).to(device_t)
        _freeze_all(pf)
        if float(lr_hand_pose) > 0.0:
            pf.global_orient.requires_grad_(True)
            pf.cam_t.requires_grad_(True)
        if float(lr_hand_articulation) > 0.0:
            pf.hand_pose.requires_grad_(True)
        if float(lr_hand_shape) > 0.0:
            pf.betas.requires_grad_(True)
        if float(lr_hand_scale) > 0.0:
            pf.hand_scale.requires_grad_(True)
        with torch.no_grad():
            zero_pose = torch.zeros(1, 48, device=device_t)
            zero_betas = torch.zeros(1, 10, device=device_t)
            rest = pf.mano(zero_pose, zero_betas).verts[0].detach().clone()
            if not ht.is_right:
                rest = rest * rest.new_tensor([-1.0, 1.0, 1.0])
        faces_np = pf.mano.th_faces.detach().cpu().numpy()
        hg = init_hand_face_gaussians(
            rest_vertices=rest,
            faces=faces_np,
            is_right=ht.is_right,
            normal_thin_factor=face_normal_thin_factor_hand,
            hand_scale_init=float(getattr(ht, "hand_scale", 1.0) or 1.0),
            init_opacity=init_opacity_hand,
            init_scale_factor=init_gaussian_scale_factor,
            device=device_t,
        ).to(device_t)
        if cache.hand_masks[slot_i].sum() > 0:
            mask_sum = cache.hand_masks[slot_i].sum().clamp_min(1.0)
            mean_rgb = (cache.rgb * cache.hand_masks[slot_i].unsqueeze(-1)).sum(dim=(0, 1, 2)) / mask_sum
            with torch.no_grad():
                hg._color.copy_(mean_rgb.to(device_t).expand_as(hg._color))
        hg_params = _trainable_gaussian_params(hg)
        trainable_hand = []
        if pf.global_orient.requires_grad or pf.cam_t.requires_grad:
            trainable_hand.append("root")
        if pf.hand_pose.requires_grad:
            trainable_hand.append("fingers")
        if pf.betas.requires_grad:
            trainable_hand.append("shape")
        if pf.hand_scale.requires_grad:
            trainable_hand.append("scale")
        trainable_desc = ", ".join(trainable_hand) if trainable_hand else "none"
        print(f"Hand {side}: {hg.num_gaussians()} face-anchored Gaussians "
              f"(trainable: {trainable_desc})")
        hand_slots.append(HandSlot(
            side=side,
            track=ht,
            pose_field=pf,
            gaussians=hg,
            mask_dir=mask_dir,
            output_pose_dir=out_dir,
            frame_to_pos={fidx: i for i, fidx in enumerate(ht.frame_indices)},
        ))
        del hg_params  # collected below by module reference

    if bg_ref_frame is not None and int(bg_ref_frame) in frame_indices:
        ref_t = frame_indices.index(int(bg_ref_frame))
    else:
        ref_t = 0
    bg_gaussians: BackgroundGaussians | None = None
    bg_pose_field: BackgroundPoseField | None = None
    bg_g_params: list[torch.nn.Parameter] = []
    if mask_background:
        if background_pose_init_dir is not None:
            print("Ignoring background_pose_init_dir because --mask-background omits background Gaussians.")
        if refined_camera_poses_dir is not None:
            print("Ignoring refined_camera_poses_dir because --mask-background omits camera/background pose refinement.")
        print("Mask background: target background is black and no background Gaussians are rendered.")
    else:
        if cache.depth is None:
            raise RuntimeError("Depth cache is missing; depth_dir is required for background refinement.")
        slam_poses = None
        if background_pose_init_dir is not None:
            slam_poses = _load_relative_w2c_poses(background_pose_init_dir, frame_indices, ref_t)

        def union_for_bg(t: int) -> torch.Tensor:
            m = cache.union_mask(t, device_t)
            m = torch.maximum(m, 1.0 - valid_mask)
            return m

        if slam_poses is not None and bg_init_stride > 1:
            positions = sorted({ref_t} | set(range(0, len(frame_indices), int(bg_init_stride))))
            positions = [t for t in positions if t in slam_poses]
            T_w2c_list = [torch.from_numpy(slam_poses[t]).to(device_t, dtype=torch.float32) for t in positions]
            anchors, colors, init_scales = init_background_multiframe(
                rgbs=[cache.rgb[t].to(device_t) for t in positions],
                depths=[cache.depth[t].to(device_t) for t in positions],
                union_masks=[union_for_bg(t) for t in positions],
                T_w2c_list=T_w2c_list,
                K=K,
                voxel_size=bg_voxel_size,
                max_points=bg_max_points,
            )
            print(f"Background: initialized from {len(positions)} frames")
        else:
            anchors, colors, init_scales = init_background_from_depth(
                rgb=cache.rgb[ref_t].to(device_t),
                depth=cache.depth[ref_t].to(device_t),
                union_mask=union_for_bg(ref_t),
                K=K,
                max_points=bg_max_points,
            )
            print(f"Background: initialized from frame {frame_indices[ref_t]:06d}")
        bg_gaussians = BackgroundGaussians(
            anchors, colors, init_scales,
            init_opacity=init_opacity_bg,
            init_scale_factor=init_gaussian_scale_factor,
        ).to(device_t)
        bg_g_params = _trainable_gaussian_params(bg_gaussians)
        bg_pose_field = BackgroundPoseField(len(frame_indices), device=device_t, ref_frame_t=ref_t).to(device_t)
        if slam_poses is not None:
            loaded, missing = _seed_background_pose_field(bg_pose_field, slam_poses)
            print(f"Camera/background pose seed: loaded {loaded}, missing {missing}")
        _anchor_camera_reference(bg_pose_field)
        print(f"Background: {bg_gaussians.num_gaussians()} depth-initialized Gaussians")

    use_mask_loss = float(w_mask) > 0.0
    n_mask_classes = 1 + len(hand_slots)
    if use_mask_loss:
        print(f"Segmentation mask L1: weight={float(w_mask):.4g}, classes={n_mask_classes}")

    use_relative_depth_loss = float(w_relative_depth) > 0.0
    if use_relative_depth_loss:
        if cache.depth is None:
            raise RuntimeError("Depth cache is missing; depth_dir is required for relative depth loss.")
        region = "foreground masks" if mask_background else "valid image region"
        print(f"Relative MoGe depth-gradient loss: weight={float(w_relative_depth):.4g}, "
              f"region={region}")

    if (
        float(w_smooth_hand_object_relative_rot) > 0.0
        or float(w_smooth_hand_object_relative_trans) > 0.0
    ):
        if hand_slots:
            print(
                "Hand/object relative smoothness: "
                f"rot={float(w_smooth_hand_object_relative_rot):.4g}, "
                f"trans={float(w_smooth_hand_object_relative_trans):.4g}"
            )
        else:
            print("Ignoring hand/object relative smoothness because no hand tracks are present.")

    use_perceptual_loss = float(w_perceptual) > 0.0
    perceptual_loss_fn: VGG16PerceptualLoss | None = None
    if use_perceptual_loss:
        perceptual_loss_fn = VGG16PerceptualLoss(
            weights_path=vgg_weights_path,
            resize=perceptual_resize,
        ).to(device_t)
        print(f"VGG16 perceptual L1: weight={float(w_perceptual):.4g}, "
              f"resize={int(perceptual_resize)}")

    object_sdf: ObjectSDFGrid | None = None
    if float(w_hand_object_penetration) > 0.0:
        if hand_slots:
            object_sdf = _build_object_sdf_grid(
                obj_verts,
                obj_faces,
                object_sdf_resolution,
                hand_object_penetration_margin,
                device_t,
            )
            print(
                f"Hand/object penetration: weight={float(w_hand_object_penetration):.4g}, "
                f"margin={float(hand_object_penetration_margin):.4g}, "
                f"sdf_res={int(object_sdf_resolution)}"
            )
        else:
            print("Ignoring hand/object penetration loss because no hand tracks are present.")

    progress_t = ref_t
    if debug_frame_idx is not None:
        try:
            progress_t = frame_indices.index(int(debug_frame_idx))
        except ValueError:
            print(f"Warning: debug_frame_idx={debug_frame_idx} not in sequence; "
                  f"using frame {frame_indices[progress_t]:06d} for progress.")
    if render_every > 0 and progress_dir is None:
        progress_dir = os.path.join(
            os.path.dirname(os.path.abspath(overlay_path)) or ".",
            "refined_simple_progress",
        )
    if render_every > 0 and progress_dir is not None:
        print(f"Simple-refinement progress frames -> {progress_dir} "
              f"(frame {frame_indices[progress_t]:06d}, every {render_every} steps)")
        _dump_progress_frame(
            progress_dir, 0, progress_t, cache, K, width, height, device_t,
            obj_gaussians, obj_pose_field, hand_slots, bg_gaussians, bg_pose_field,
            mask_background=mask_background,
        )

    param_groups: list[dict] = []
    if obj_g_params:
        param_groups.append({"params": obj_g_params, "lr": lr_gaussians})
    for slot in hand_slots:
        params = [p for p in (slot.gaussians._log_scale, slot.gaussians._opacity_logit, slot.gaussians._color) if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr_gaussians})
    if bg_g_params:
        param_groups.append({"params": bg_g_params, "lr": lr_gaussians})
    if float(lr_object_scale) > 0.0 and hasattr(obj_gaussians, "_log_scale_global"):
        param_groups.append({"params": [obj_gaussians._log_scale_global], "lr": lr_object_scale})
    param_groups.extend([
        {"params": [obj_pose_field.axis_angle], "lr": lr_object_pose},
        {"params": [obj_pose_field.translation], "lr": lr_object_pose},
    ])
    if bg_pose_field is not None:
        param_groups.extend([
            {"params": [bg_pose_field.axis_angle], "lr": lr_camera_pose},
            {"params": [bg_pose_field.translation], "lr": lr_camera_pose},
        ])
    for slot in hand_slots:
        if float(lr_hand_pose) > 0.0:
            param_groups.append({"params": [slot.pose_field.global_orient], "lr": lr_hand_pose})
            param_groups.append({"params": [slot.pose_field.cam_t], "lr": lr_hand_pose})
        if float(lr_hand_articulation) > 0.0:
            param_groups.append({"params": [slot.pose_field.hand_pose], "lr": lr_hand_articulation})
        if float(lr_hand_shape) > 0.0:
            param_groups.append({"params": [slot.pose_field.betas], "lr": lr_hand_shape})
        if float(lr_hand_scale) > 0.0:
            param_groups.append({"params": [slot.pose_field.hand_scale], "lr": lr_hand_scale})

    optimizer = torch.optim.Adam(param_groups)
    # The rasterizer can occasionally emit an invalid backward value for a
    # plausible initial trajectory. Keep that one mini-batch from poisoning
    # every pose through the sequence-wide smoothness terms.
    trainable_params = [
        param for group in optimizer.param_groups for param in group["params"]
    ]
    max_grad_norm = 1.0
    print(f"Gradient clipping: global norm <= {max_grad_norm:g}")
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    T = len(frame_indices)
    n_batches = max(1, (T + int(batch_size) - 1) // int(batch_size))
    total_steps = int(n_epochs) * n_batches
    lr_scale = 1.0
    if lr_schedule == "cosine":
        print(f"LR schedule: cosine, min_factor={lr_cosine_min_factor:.4g}, steps={total_steps}")
    pbar = tqdm(total=total_steps, ncols=100, desc="refine_simple")
    step = 0
    for _epoch in range(int(n_epochs)):
        order = torch.randperm(T).tolist()
        for start in range(0, T, int(batch_size)):
            batch = order[start:start + int(batch_size)]
            if lr_schedule == "cosine":
                lr_scale = _cosine_lr_scale(step, total_steps, lr_cosine_min_factor)
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * lr_scale
            optimizer.zero_grad(set_to_none=True)
            photo = obj_pose_field.axis_angle.new_zeros(())
            mask_l1 = obj_pose_field.axis_angle.new_zeros(())
            relative_depth = obj_pose_field.axis_angle.new_zeros(())
            perceptual = obj_pose_field.axis_angle.new_zeros(())
            penetration = obj_pose_field.axis_angle.new_zeros(())
            for t in batch:
                fidx = frame_indices[t]
                target = _target_rgb(cache, t, device_t, mask_background)
                if use_mask_loss:
                    frame, labels = _current_frame_with_labels(
                        t, fidx, obj_gaussians, obj_pose_field, hand_slots,
                        bg_gaussians, bg_pose_field, n_mask_classes,
                    )
                    render_frame = replace(
                        frame,
                        colors=torch.cat([frame.colors, labels.to(frame.colors.dtype)], dim=-1),
                    )
                    rendered = _render_2dgs_features(render_frame, K, width, height)
                    pred = rendered[..., :3]
                    class_pred = rendered[..., 3:]
                    mask_l1 = mask_l1 + _segmentation_mask_loss(
                        class_pred, cache, t, valid_mask, valid_denom,
                    )
                else:
                    frame = _current_frame(
                        t, fidx, obj_gaussians, obj_pose_field, hand_slots,
                        bg_gaussians, bg_pose_field,
                    )
                    pred = _render_2dgs_rgb(frame, K, width, height)
                per_pixel = (pred - target).abs().sum(dim=-1)
                photo = photo + (per_pixel * valid_mask).sum() / valid_denom
                if perceptual_loss_fn is not None:
                    perceptual = perceptual + perceptual_loss_fn(pred, target, valid_mask)
                if use_relative_depth_loss:
                    rendered_depth, rendered_alpha = _render_2dgs_depth(frame, K, width, height)
                    target_depth = cache.depth[t].to(device_t, non_blocking=True)
                    depth_mask = (
                        cache.union_mask(t, device_t)
                        if mask_background else valid_mask
                    )
                    depth_mask = depth_mask * (rendered_alpha > 1e-3).to(depth_mask.dtype)
                    relative_depth = relative_depth + depth_gradient_loss(
                        rendered_depth,
                        target_depth,
                        depth_mask,
                    )
                if object_sdf is not None:
                    penetration = penetration + _hand_object_penetration_loss(
                        t,
                        fidx,
                        obj_pose_field,
                        obj_gaussians,
                        hand_slots,
                        object_sdf,
                        hand_object_penetration_margin,
                        hand_object_penetration_max_verts,
                    )
            photo = photo / float(max(len(batch), 1))
            mask_l1 = mask_l1 / float(max(len(batch), 1))
            relative_depth = relative_depth / float(max(len(batch), 1))
            perceptual = perceptual / float(max(len(batch), 1))
            penetration = penetration / float(max(len(batch), 1))
            smooth = _pose_smoothness(
                obj_pose_field, hand_slots, bg_pose_field,
                w_smooth_object_rot, w_smooth_object_trans,
                w_smooth_hand_rot, w_smooth_hand_articulation, w_smooth_hand_trans,
                w_smooth_hand_object_relative_rot, w_smooth_hand_object_relative_trans,
                w_smooth_camera_rot, w_smooth_camera_trans,
            ) / float(n_batches)
            loss = (
                photo
                + float(w_mask) * mask_l1
                + float(w_relative_depth) * relative_depth
                + float(w_perceptual) * perceptual
                + float(w_hand_object_penetration) * penetration
                + smooth
            )
            if not bool(torch.isfinite(loss)):
                print(
                    f"WARNING: skipping non-finite refinement loss at step={step}, "
                    f"frames={[frame_indices[t] for t in batch]}"
                )
                optimizer.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)
                continue

            loss.backward()
            bad_gradient = _first_nonfinite_tensor([
                (f"gradient for group={group_i} param={param_i} shape={tuple(param.shape)}", param.grad)
                for group_i, group in enumerate(optimizer.param_groups)
                for param_i, param in enumerate(group["params"])
            ])
            if bad_gradient is not None:
                print(
                    f"WARNING: skipping non-finite refinement gradient at step={step}, "
                    f"frames={[frame_indices[t] for t in batch]}: {bad_gradient}"
                )
                optimizer.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=max_grad_norm, error_if_nonfinite=False
            )
            if not bool(torch.isfinite(grad_norm)):
                print(
                    f"WARNING: skipping non-finite gradient norm at step={step}, "
                    f"frames={[frame_indices[t] for t in batch]}"
                )
                optimizer.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)
                continue

            optimizer.step()
            bad_parameter = _first_nonfinite_tensor([
                (f"group={group_i} param={param_i} shape={tuple(param.shape)}", param)
                for group_i, group in enumerate(optimizer.param_groups)
                for param_i, param in enumerate(group["params"])
            ])
            if bad_parameter is not None:
                raise FloatingPointError(
                    f"Non-finite refinement parameter after optimizer step {step}, "
                    f"frames={[frame_indices[t] for t in batch]}: {bad_parameter}"
                )
            if bg_pose_field is not None:
                _anchor_camera_reference(bg_pose_field)
            step += 1
            if render_every > 0 and progress_dir is not None and step % int(render_every) == 0:
                _dump_progress_frame(
                    progress_dir, step, progress_t, cache, K, width, height, device_t,
                    obj_gaussians, obj_pose_field, hand_slots, bg_gaussians, bg_pose_field,
                    mask_background=mask_background,
                )
            pbar.update(1)
            pbar.set_postfix(
                photo=f"{float(photo.detach()):.4f}",
                mask=f"{float(mask_l1.detach()):.4f}",
                rdepth=f"{float(relative_depth.detach()):.4f}",
                perc=f"{float(perceptual.detach()):.4f}",
                pen=f"{float(penetration.detach()):.4f}",
                lr=f"{lr_scale:.3g}",
                smooth=f"{float(smooth.detach()):.4f}",
            )
    pbar.close()
    if render_every > 0 and progress_dir is not None:
        _dump_progress_frame(
            progress_dir, step, progress_t, cache, K, width, height, device_t,
            obj_gaussians, obj_pose_field, hand_slots, bg_gaussians, bg_pose_field,
            mask_background=mask_background,
        )
        _stitch_progress_video(progress_dir)

    refined_obj_track = obj_pose_field.export_track()
    s_obj_learned = float(obj_gaussians.object_scale().detach())
    refined_obj_track.scales = refined_obj_track.scales * s_obj_learned
    if float(lr_object_scale) > 0.0:
        print(f"Learned object scale: {s_obj_learned:.4f}")
    save_object_poses(refined_obj_track, refined_object_poses_dir)
    print(f"Wrote refined object poses -> {refined_object_poses_dir}")
    for slot in hand_slots:
        if slot.output_pose_dir is None:
            continue
        save_hand_poses(slot.pose_field.export_track(slot.track.raw_records), slot.output_pose_dir)
        print(f"Wrote refined {slot.side} hand poses -> {slot.output_pose_dir}")
    if refined_camera_poses_dir is not None and bg_pose_field is not None:
        _save_camera_poses(bg_pose_field, frame_indices, refined_camera_poses_dir)
        print(f"Wrote refined camera/background poses -> {refined_camera_poses_dir}")

    _write_overlay_video(
        output_path=overlay_path,
        cache=cache,
        K=K,
        width=width,
        height=height,
        device=device_t,
        obj_gaussians=obj_gaussians,
        obj_pose_field=obj_pose_field,
        hand_slots=hand_slots,
        bg_gaussians=bg_gaussians,
        bg_pose_field=bg_pose_field,
        fps=fps,
    )
    print(f"Wrote overlay -> {overlay_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--depth_dir", default=None,
                   help="Depth frames for background initialization. Required unless --mask-background is set.")
    p.add_argument("--intrinsics_path", required=True)
    p.add_argument("--object_mesh_path", required=True)
    p.add_argument("--object_poses_dir", required=True)
    p.add_argument("--object_mask_dir", required=True)
    p.add_argument("--refined_object_poses_dir", required=True)
    p.add_argument("--overlay_path", required=True)
    p.add_argument("--left_hand_pose_dir", default=None)
    p.add_argument("--left_hand_mask_dir", default=None)
    p.add_argument("--right_hand_pose_dir", default=None)
    p.add_argument("--right_hand_mask_dir", default=None)
    p.add_argument("--refined_left_hand_pose_dir", default=None)
    p.add_argument("--refined_right_hand_pose_dir", default=None)
    p.add_argument("--refined_camera_poses_dir", default=None)
    p.add_argument("--mano_assets_root", default=None)
    p.add_argument("--background_pose_init_dir", default=None)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--train_resolution_scale", type=float, default=0.5)
    p.add_argument("--lr_gaussians", type=float, default=1e-2)
    p.add_argument("--lr_object_pose", type=float, default=1e-3)
    p.add_argument("--lr_object_scale", type=float, default=0.0,
                   help="LR for global object scale. 0 disables.")
    p.add_argument("--lr_hand_pose", type=float, default=1e-3,
                   help="LR for hand root global_orient and cam_t. 0 disables root pose refinement.")
    p.add_argument("--lr_hand_articulation", type=float, default=0.0,
                   help="LR for MANO finger articulation hand_pose. 0 disables.")
    p.add_argument("--lr_hand_shape", type=float, default=0.0,
                   help="LR for shared MANO betas/shape per hand track. 0 disables.")
    p.add_argument("--lr_hand_scale", type=float, default=0.0,
                   help="LR for shared per-track hand_scale. 0 disables.")
    p.add_argument("--lr_camera_pose", type=float, default=1e-4)
    p.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant",
                   help="Learning-rate schedule for all optimizer groups.")
    p.add_argument("--lr_cosine_min_factor", type=float, default=0.0,
                   help="Final LR multiplier for --lr_schedule cosine. 0 decays to zero.")
    p.add_argument("--w_smooth_object_rot", type=float, default=0.01)
    p.add_argument("--w_smooth_object_trans", type=float, default=0.01)
    p.add_argument("--w_smooth_hand_rot", type=float, default=0.01)
    p.add_argument("--w_smooth_hand_articulation", type=float, default=0.01,
                   help="Temporal rotation smoothness for MANO finger hand_pose.")
    p.add_argument("--w_smooth_hand_trans", type=float, default=0.01)
    p.add_argument("--w_smooth_hand_object_relative_rot", type=float, default=0.0,
                   help="Temporal smoothness for hand root rotation relative to object pose. 0 disables.")
    p.add_argument("--w_smooth_hand_object_relative_trans", type=float, default=0.0,
                   help="Temporal smoothness for hand root translation in the object frame. "
                        "0 disables.")
    p.add_argument("--w_smooth_camera_rot", type=float, default=0.01)
    p.add_argument("--w_smooth_camera_trans", type=float, default=0.01)
    p.add_argument("--w_mask", type=float, default=1.0,
                   help="Weight for L1 segmentation mask loss. 0 disables.")
    p.add_argument("--w_relative_depth", type=float, default=0.0,
                   help="Weight for log-depth-gradient loss against MoGe depth. 0 disables.")
    p.add_argument("--w_perceptual", type=float, default=0.0,
                   help="Weight for VGG16 perceptual feature L1 loss. 0 disables.")
    p.add_argument("--vgg_weights_path", default=None,
                   help="Optional local VGG16 state_dict path. If missing, torchvision ImageNet weights are downloaded there.")
    p.add_argument("--perceptual_resize", type=int, default=224,
                   help="Square resize for VGG perceptual loss. <=0 uses training resolution.")
    p.add_argument("--w_hand_object_penetration", type=float, default=0.0,
                   help="Weight for 3D hand/object SDF penetration loss. 0 disables.")
    p.add_argument("--hand_object_penetration_margin", type=float, default=0.003,
                   help="Signed-distance margin in meters/object units for hand/object separation.")
    p.add_argument("--object_sdf_resolution", type=int, default=96,
                   help="Resolution per axis for the object SDF grid used by hand/object penetration.")
    p.add_argument("--hand_object_penetration_max_verts", type=int, default=0,
                   help="Max MANO verts per hand for penetration loss. 0 uses all vertices.")
    p.add_argument("--mask-background", "--mask_background", dest="mask_background", action="store_true",
                   help="Black out target background and omit background Gaussians/camera refinement.")
    p.add_argument("--bg_ref_frame", type=int, default=None)
    p.add_argument("--bg_init_stride", type=int, default=10)
    p.add_argument("--bg_voxel_size", type=float, default=0.005)
    p.add_argument("--bg_max_points", type=int, default=50000)
    p.add_argument("--valid_mask_threshold", type=float, default=0.04)
    p.add_argument("--valid_mask_erode_iters", type=int, default=2)
    p.add_argument("--face_normal_thin_factor_obj", type=float, default=0.25)
    p.add_argument("--face_normal_thin_factor_hand", type=float, default=0.25)
    p.add_argument("--init_opacity_obj", type=float, default=0.9)
    p.add_argument("--init_opacity_hand", type=float, default=0.9)
    p.add_argument("--init_opacity_bg", type=float, default=0.9)
    p.add_argument("--init_gaussian_scale_factor", type=float, default=1.0,
                   help="Multiplier for all initial Gaussian scales/radii.")
    p.add_argument("--render_every", type=int, default=25,
                   help="Write a target/render/error progress panel every N optimizer steps. 0 disables.")
    p.add_argument("--progress_dir", default=None,
                   help="Directory for simple-refinement progress PNGs. Defaults next to overlay_path.")
    p.add_argument("--debug_frame_idx", type=int, default=None,
                   help="Frame index to use for intermediate progress visualizations.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    refine_simple(**vars(args))


if __name__ == "__main__":
    main()
