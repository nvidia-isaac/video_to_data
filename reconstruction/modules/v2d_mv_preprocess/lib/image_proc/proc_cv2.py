# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import logging

import cv2
import numpy as np

from v2d.mv.rig import CameraParam
from v2d.mv.rig.edex import DistortionModel

from .base import ImageProcessorBase


logger = logging.getLogger(__name__)


class RectifyProcessor(ImageProcessorBase):
    def __init__(
        self,
        mapx: np.ndarray,
        mapy: np.ndarray,
        roi: tuple[int, int, int, int],
        K: np.ndarray,
        D: np.ndarray,
        R_rect: np.ndarray,
        P_rect: np.ndarray,
    ):
        self.mapx = mapx
        self.mapy = mapy
        self.roi = roi
        self.K = K
        self.D = D
        self.R_rect = R_rect
        self.P_rect = P_rect

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi
        img = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)
        img = img[y:y+h, x:x+w]
        return img

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.reshape(-1, 1, 2).astype(np.float64)
        rect_pts = cv2.undistortPoints(pts, self.K, self.D, R=self.R_rect, P=self.P_rect)
        rect_pts = rect_pts.reshape(-1, 2)
        rect_pts[:, 0] -= self.roi[0]
        rect_pts[:, 1] -= self.roi[1]
        return rect_pts


def image_proc_build_rectify(
    left_param: CameraParam,
    right_param: CameraParam,
) -> tuple[tuple[RectifyProcessor, RectifyProcessor], tuple[CameraParam, CameraParam]]:
    """Build stereo rectification processors and update camera parameters.

    Uses cv2.stereoRectify to compute rectification transforms, then builds
    undistortion+rectification remap tables. After rectification, distortion
    model is set to PINHOLE (no distortion).

    Returns:
        (left_processor, right_processor), (left_param, right_param)
    """
    K = np.stack([left_param.K, right_param.K]).copy()
    D = np.stack([left_param.D, right_param.D]).copy()
    P = np.stack([left_param.P, right_param.P]).copy()
    R = np.stack([left_param.R, right_param.R]).copy()
    size = left_param.resolution

    Rot = R[1].T @ R[0]
    Trans = P[1][:3, 3] / np.diag(P[1])

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K[0], D[0], K[1], D[1], size, Rot, Trans, alpha=0)
    R, P = [R1, R2], [P1, P2]

    roi = cv2.getValidDisparityROI(roi1, roi2, 0, 64, 1)
    logger.debug(f"Undistorted ROI (X, Y, W, H): {roi}")
    mapx, mapy = [], []
    for i in [0, 1]:
        x, y = cv2.initUndistortRectifyMap(K[i], D[i], R[i], P[i], size, 5)
        mapx.append(x)
        mapy.append(y)

    params = []
    for i, param in enumerate([left_param, right_param]):
        new_param = copy.deepcopy(param)
        new_param.resolution = np.array([roi[2], roi[3]], dtype=np.int32)

        new_param.D_model = DistortionModel.PINHOLE.value
        new_param.D = np.array([], dtype=np.float32)

        # Update intrinsics, accounting for ROI crop
        new_param.K = P[i][:3, :3].copy()
        new_param.K[0, 2] -= roi[0]  # Adjust cx by x_offset
        new_param.K[1, 2] -= roi[1]  # Adjust cy by y_offset

        new_param.P = P[i].copy()
        new_param.P[:3, :3] = new_param.K
        new_param.R = R[i]

        # Apply rectification rotation to camera orientation
        if new_param.T is not None:
            new_param.T[:3, :3] = new_param.T[:3, :3] @ R[i].T

        params.append(new_param)

    return (
        RectifyProcessor(mapx[0], mapy[0], roi, K[0], D[0], R[0], P[0]),
        RectifyProcessor(mapx[1], mapy[1], roi, K[1], D[1], R[1], P[1]),
    ), tuple(params)


class RescaleProcessor(ImageProcessorBase):
    def __init__(self, dsize: tuple[int, int], scale: float, interpolation: int = cv2.INTER_LINEAR):
        self.dsize = dsize
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, dsize=self.dsize, interpolation=self.interpolation)

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        return pts * self.scale


def image_proc_build_rescale(
    camera_param: CameraParam,
    scale: float,
) -> tuple[RescaleProcessor, CameraParam]:
    """Build a rescale processor and update camera parameters."""
    new_param = camera_param.scale(scale)
    return RescaleProcessor(tuple(new_param.resolution), scale), new_param


class CropProcessor(ImageProcessorBase):
    def __init__(self, roi: tuple[int, int, int, int]):
        self.roi = roi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x, y, w, h = self.roi
        return img[y:y+h, x:x+w]

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.copy()
        pts[:, 0] -= self.roi[0]
        pts[:, 1] -= self.roi[1]
        return pts


def image_proc_build_center_crop(
    camera_param: CameraParam,
    w_target: int,
    h_target: int,
) -> tuple[CropProcessor, CameraParam]:
    """Build a center crop processor and update camera parameters."""
    w, h = camera_param.resolution
    assert w_target > 0 and h_target > 0, "w_target and h_target must be positive"
    assert w_target <= w and h_target <= h, "w_target and h_target must be <= camera resolution"

    w_pad = (w - w_target) // 2
    h_pad = (h - h_target) // 2
    roi = (w_pad, h_pad, w_target, h_target)
    logger.debug(f"Cropping from {camera_param.resolution} to ({w_target}, {h_target})")

    new_param = copy.deepcopy(camera_param)
    new_param.resolution = np.array([w_target, h_target], dtype=np.int32)
    new_param.K[0, 2] -= w_pad
    new_param.K[1, 2] -= h_pad
    if new_param.P is not None:
        new_param.P[:3, :3] = new_param.K

    return CropProcessor(roi), new_param
