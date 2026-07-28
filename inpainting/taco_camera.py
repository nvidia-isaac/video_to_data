"""Strict loading and projection helpers for official TACO egocentric cameras."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import ContractError


@dataclass(frozen=True)
class TacoCamera:
    intrinsic: np.ndarray
    world_to_camera: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.world_to_camera.shape[0])


def load_taco_camera(
    intrinsic_path: str | Path,
    extrinsic_path: str | Path,
    expected_frames: int,
    width: int,
    height: int,
) -> TacoCamera:
    """Load official TACO K and per-frame world-to-camera matrices."""

    intrinsic_path = Path(intrinsic_path)
    extrinsic_path = Path(extrinsic_path)
    if not intrinsic_path.is_file():
        raise FileNotFoundError(intrinsic_path)
    if not extrinsic_path.is_file():
        raise FileNotFoundError(extrinsic_path)
    intrinsic = np.asarray(np.loadtxt(intrinsic_path), dtype=np.float64)
    world_to_camera = np.asarray(np.load(extrinsic_path), dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ContractError(f"TACO intrinsic must be (3,3), got {intrinsic.shape}")
    expected_shape = (expected_frames, 4, 4)
    if world_to_camera.shape != expected_shape:
        raise ContractError(
            f"TACO world-to-camera transforms must be {expected_shape}, got {world_to_camera.shape}"
        )
    if not np.isfinite(intrinsic).all() or not np.isfinite(world_to_camera).all():
        raise ContractError("Camera calibration contains non-finite values")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ContractError("Camera focal lengths must be positive")
    if not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-6):
        raise ContractError(f"Unexpected intrinsic bottom row: {intrinsic[2]}")
    if not (0 <= intrinsic[0, 2] <= width and 0 <= intrinsic[1, 2] <= height):
        raise ContractError(
            f"Principal point {(intrinsic[0, 2], intrinsic[1, 2])} lies outside {width}x{height}"
        )
    if not np.allclose(world_to_camera[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ContractError("World-to-camera matrices have invalid homogeneous bottom rows")
    rotations = world_to_camera[:, :3, :3]
    identity = np.eye(3)[None]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), identity, atol=2e-3):
        raise ContractError("World-to-camera rotation is not orthonormal")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, atol=2e-3):
        raise ContractError("World-to-camera rotation determinant is not +1")
    return TacoCamera(intrinsic=intrinsic, world_to_camera=world_to_camera)


def world_to_camera_points(points_world: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points_world.shape[-1] != 3:
        raise ContractError(f"Points must end in xyz, got {points_world.shape}")
    if transform.shape != (4, 4):
        raise ContractError(f"Transform must be (4,4), got {transform.shape}")
    return points_world @ transform[:3, :3].T + transform[:3, 3]


def project_camera_points(
    points_camera: np.ndarray, intrinsic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel xy and validity (`z > 0`) without clipping to image bounds."""

    points_camera = np.asarray(points_camera, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if points_camera.shape[-1] != 3 or intrinsic.shape != (3, 3):
        raise ContractError("Expected points (...,3) and intrinsic (3,3)")
    valid = np.isfinite(points_camera).all(axis=-1) & (points_camera[..., 2] > 1e-6)
    homogeneous = points_camera @ intrinsic.T
    pixels = np.full(points_camera.shape[:-1] + (2,), np.nan, dtype=np.float64)
    pixels[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    return pixels, valid


def project_world_points(
    points_world: np.ndarray, intrinsic: np.ndarray, world_to_camera: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_camera = world_to_camera_points(points_world, world_to_camera)
    pixels, valid = project_camera_points(points_camera, intrinsic)
    return pixels, points_camera[..., 2], valid
