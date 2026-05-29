# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .edex import Camera, DistortionModel, EDEXMetadata


@dataclass
class CameraParam:
    resolution: np.ndarray  # (2,) [width, height]
    D_model: str
    D: np.ndarray  # distortion coefficients
    K: np.ndarray  # (3, 3) intrinsic matrix
    P: np.ndarray | None = None  # (3, 4) projection matrix (stereo)
    R: np.ndarray | None = None  # (3, 3) rectification matrix (stereo)
    T: np.ndarray | None = None  # (4, 4) transform from camera to world coordinates
    
    def scale(self, scale: float):
        assert scale > 0, "scale must be positive"
        new_param = deepcopy(self)
        old_resolution = new_param.resolution.copy()
        new_param.resolution = np.round(new_param.resolution * scale).astype(int)
        x_scale = new_param.resolution[0] / old_resolution[0]
        y_scale = new_param.resolution[1] / old_resolution[1]
        new_param.K[0, :] *= x_scale
        new_param.K[1, :] *= y_scale
        if new_param.P is not None:
            new_param.P[0, :] *= x_scale
            new_param.P[1, :] *= y_scale
        return new_param


def edex_camera_to_param(camera: Camera) -> CameraParam:
    """Convert an EDEX Camera to a CameraParam."""
    intrinsics = camera.intrinsics
    fx, fy = intrinsics.focal
    cx, cy = intrinsics.principal

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float32)

    T = None
    if camera.transform is not None:
        T = np.eye(4, dtype=np.float32)
        T[:3, :] = camera.transform

    return CameraParam(
        resolution=intrinsics.resolution.astype(int),
        D_model=intrinsics.distortion_model.value,
        D=intrinsics.distortion_params,
        K=K,
        P=intrinsics.projection,
        R=intrinsics.rectification,
        T=T,
    )


def load_camera_params(source_path: Path) -> list[CameraParam]:
    """Load camera parameters from a calibration file.

    Dispatches to the appropriate reader based on file suffix.
    """
    source_path = Path(source_path)
    name = source_path.name.lower()
    suffix = source_path.suffix.lower()

    if name == "edex" or suffix == ".edex":
        edex = EDEXMetadata.read(source_path)
        return [
            edex_camera_to_param(cam)
            for cam in edex.header.cameras
        ]

    raise ValueError(f"Unsupported camera params format: {name}")


def param_overwrite_in_edex(edex: EDEXMetadata, cam_id: int, param: CameraParam) -> None:
    """Write a CameraParam back into an EDEXMetadata object (inverse of edex_camera_to_param)."""
    camera = edex.header.cameras[cam_id]
    camera.intrinsics.resolution = param.resolution.astype(np.int32)
    camera.intrinsics.distortion_model = DistortionModel(param.D_model)
    camera.intrinsics.distortion_params = param.D
    camera.intrinsics.focal = np.array([param.K[0, 0], param.K[1, 1]], dtype=np.float32)
    camera.intrinsics.principal = np.array([param.K[0, 2], param.K[1, 2]], dtype=np.float32)
    camera.intrinsics.projection = param.P
    camera.intrinsics.rectification = param.R
    if param.T is not None:
        camera.transform = param.T[:3, :].astype(np.float32)
    else:
        camera.transform = None


def edex_merge_extrinsics(target: EDEXMetadata, source: EDEXMetadata) -> None:
    """Copy camera extrinsic transforms from source EDEX into target EDEX.

    Copies the transform (3x4 [R|t]) from each camera in source into the
    corresponding camera in target, matched by index. Intrinsics are left
    unchanged.
    """
    assert len(target.header.cameras) == len(source.header.cameras), (
        f"Camera count mismatch: target has {len(target.header.cameras)}, "
        f"source has {len(source.header.cameras)}"
    )
    for cam_id in range(len(target.header.cameras)):
        target.header.cameras[cam_id].transform = source.header.cameras[cam_id].transform
