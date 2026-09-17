from __future__ import annotations

import time

import cv2
import numpy as np
import torch

import Utils
from lib_mhr.object_texture import render_original_object_parts
from prep.mhr_geometry_crop import build_geometry_guided_crop
from tools import img_utils


def _crop_resize(image, center, crop_size, render_size, mode):
    crop = img_utils.crop(image, center, crop_size)
    height, width = crop.shape[:2]
    if height != width:
        pad_height = max(width - height, 0)
        pad_width = max(height - width, 0)
        pad_spec = ((0, pad_height), (0, pad_width)) if crop.ndim == 2 else ((0, pad_height), (0, pad_width), (0, 0))
        crop = np.pad(crop, pad_spec)
    return img_utils.resize(crop, render_size, mode=mode)


def _crop_intrinsics(bottom_right, top_left, render_size, focal, principal_point):
    crop_size = np.mean(bottom_right - top_left)
    scale = render_size[0] / crop_size
    focal_roi = focal * scale
    principal_roi = (principal_point - top_left) * scale
    return np.array([[focal_roi[0], 0, principal_roi[0]], [0, focal_roi[1], principal_roi[1]], [0, 0, 1.0]])


def build_input_payload_from_modalities(rgb, depth_m, mask_h, mask_o, K_rgb, render_size, human_vertices_camera, human_faces, object_vertices_camera, object_faces, *, human_pose_valid, object_pose_valid, timings=None):
    render_size = tuple(int(value) for value in render_size)
    if len(render_size) != 2 or render_size[0] != render_size[1]:
        raise ValueError(f"Renderer crop size must be square, got {render_size}")
    crop_started = time.perf_counter()
    crop = build_geometry_guided_crop(mask_h, mask_o, human_vertices_camera, human_faces, object_vertices_camera, object_faces, K_rgb, human_pose_valid=human_pose_valid, object_pose_valid=object_pose_valid, pad=1.1)
    if timings is not None:
        timings["geometry_guided_crop"] = time.perf_counter() - crop_started
    top_left, bottom_right, crop_size = crop.top_left, crop.bottom_right, crop.crop_size
    center = (top_left + bottom_right) / 2

    resize_started = time.perf_counter()
    rgb_crop = _crop_resize(rgb, center, crop_size, render_size, cv2.INTER_LINEAR).astype(np.uint8)
    mask_h_crop = _crop_resize(mask_h, center, crop_size, render_size, cv2.INTER_NEAREST).astype(np.uint8)
    mask_o_crop = _crop_resize(mask_o, center, crop_size, render_size, cv2.INTER_NEAREST).astype(np.uint8)
    depth_scale = np.array([depth_m.shape[1] / rgb.shape[1], depth_m.shape[0] / rgb.shape[0]], dtype=np.float32)
    depth_center = center * depth_scale
    depth_crop_size = crop_size * float(depth_scale.mean())
    K_depth = K_rgb.copy()
    K_depth[0] *= depth_scale[0]
    K_depth[1] *= depth_scale[1]
    K_roi = _crop_intrinsics(bottom_right * depth_scale, top_left * depth_scale, render_size, np.array([K_depth[0, 0], K_depth[1, 1]], dtype=np.float32), np.array([K_depth[0, 2], K_depth[1, 2]], dtype=np.float32)).astype(np.float32)
    depth_crop = _crop_resize(depth_m, depth_center, depth_crop_size, render_size, cv2.INTER_NEAREST)
    dmap_xyz = Utils.depth2xyzmap(depth_crop, K_roi)
    dmap_xyz[depth_crop <= 0] = 0
    if timings is not None:
        timings["crop_resize_and_xyz"] = time.perf_counter() - resize_started
    rgbm = np.concatenate([rgb_crop, mask_h_crop[:, :, None], mask_o_crop[:, :, None]], axis=-1)
    bbox = np.concatenate([top_left, bottom_right]).astype(np.float32)
    return rgbm, dmap_xyz.astype(np.float32), K_roi, bbox, crop.diagnostics


def compose_human_object_render(human_rgb, human_depth, object_rgb, object_depth):
    human_valid = torch.isfinite(human_depth) & (human_depth > 0)
    object_valid = torch.isfinite(object_depth) & (object_depth > 0)
    object_front = object_valid & (~human_valid | (object_depth < human_depth))
    return torch.where(object_front[..., None], object_rgb, human_rgb), torch.where(object_front, object_depth, human_depth)


def render_textured_object_parts(mesh_tensors_parts, object_poses, K, height, width, glctx, output_size, bbox2d=None):
    return render_original_object_parts(mesh_tensors_parts, object_poses, K, height, width, glctx, output_size, bbox2d=bbox2d)
