"""OpenCV-based undistortion for the camera models AnyCalib emits.

We support the two most common output models:
- ``opencv``:         Brown-Conrady (radial+tangential) → ``cv2.undistort`` family
- ``opencv_fisheye``: Kannala-Brandt (4 params)         → ``cv2.fisheye`` family

For ``pinhole`` we short-circuit (no remap needed).
"""
from __future__ import annotations

import numpy as np
import cv2

from v2d.common.datatypes import CameraDistortion, CameraIntrinsics


def _opencv_dist_coeffs(distortion: CameraDistortion) -> np.ndarray:
    """OpenCV ``cv2.undistort`` accepts (4|5|8|12|14)-element coefficient
    vectors. Pad shorter param lists out to length 4 with zeros."""
    p = list(distortion.params)
    if len(p) < 4:
        p = p + [0.0] * (4 - len(p))
    return np.asarray(p, dtype=np.float64).reshape(-1, 1)


def build_undistort_maps(
    intrinsics: CameraIntrinsics,
    distortion: CameraDistortion,
    balance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    """Compute (map1, map2) for ``cv2.remap`` plus the new pinhole intrinsics
    matching the rectified output.

    Args:
        intrinsics: Original (distorted) camera intrinsics.
        distortion: Distortion description (model + params).
        balance:    Fisheye-only. 0 → crop tightly (no black borders).
                    1 → keep the full FoV (visible black corners).
    """
    K = intrinsics.to_matrix().astype(np.float64)
    size = (intrinsics.width, intrinsics.height)

    if distortion.model == "pinhole":
        raise ValueError("No undistortion needed for pinhole cameras")

    if distortion.model == "opencv_fisheye":
        D = np.asarray(distortion.params, dtype=np.float64).reshape(-1, 1)
        if D.shape[0] != 4:
            raise ValueError(f"opencv_fisheye expects 4 params, got {D.shape[0]}")
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, size, np.eye(3), balance=balance,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, size, cv2.CV_16SC2,
        )
    elif distortion.model == "opencv":
        D = _opencv_dist_coeffs(distortion)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, size, alpha=balance, newImgSize=size)
        map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), new_K, size, cv2.CV_16SC2)
    else:
        raise NotImplementedError(
            f"Undistortion not implemented for model '{distortion.model}'. "
            "Supported: 'opencv', 'opencv_fisheye'."
        )

    new_intrinsics = CameraIntrinsics(
        fx=float(new_K[0, 0]), fy=float(new_K[1, 1]),
        cx=float(new_K[0, 2]), cy=float(new_K[1, 2]),
        width=intrinsics.width, height=intrinsics.height,
    )
    return map1, map2, new_intrinsics


def undistort_image(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    distortion: CameraDistortion,
    balance: float = 0.0,
) -> tuple[np.ndarray, CameraIntrinsics]:
    """Undistort a single HWC image. Returns (rectified_image, new_intrinsics)."""
    map1, map2, new_intrinsics = build_undistort_maps(intrinsics, distortion, balance)
    rectified = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    return rectified, new_intrinsics
