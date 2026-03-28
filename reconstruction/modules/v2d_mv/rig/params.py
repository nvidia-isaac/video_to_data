from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .edex import Camera, EDEXMetadata


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
