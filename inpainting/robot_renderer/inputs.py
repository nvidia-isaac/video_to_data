"""Load and strictly validate renderer trajectory and TACO calibration inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from inpainting.contracts import VideoGeometry, validate_robot_trajectory_arrays

from .transforms import (
    invert_rigid_transform,
    pose_matrix,
    validate_transform_batch,
)


SUPPORTED_ROBOT = "dexmate_vega"
SUPPORTED_GRIPPER = "sharpa_wave"


class RenderInputError(ValueError):
    """Raised when inputs cannot be rendered without guessing semantics."""


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise RenderInputError(f"{name} must be one scalar string, got {array.shape}")
    return str(array.reshape(-1)[0])


def _load_numeric_array(path: Path, *, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            preferred = [key for key in ("world_to_camera", "intrinsic", "K") if key in archive]
            if len(preferred) == 1:
                array = np.asarray(archive[preferred[0]])
            elif len(archive.files) == 1:
                array = np.asarray(archive[archive.files[0]])
            else:
                raise RenderInputError(
                    f"{label} archive must contain one array or a conventional key; "
                    f"found {archive.files}"
                )
    else:
        try:
            array = np.loadtxt(path)
        except Exception as exc:
            raise RenderInputError(f"Could not parse {label} from {path}: {exc}") from exc
    return np.asarray(array)


def load_intrinsic(path: str | Path, *, width: int, height: int) -> np.ndarray:
    """Load a TACO intrinsic matrix (3x3 or ``fx fy cx cy`` text)."""

    source = Path(path)
    array = np.asarray(_load_numeric_array(source, label="intrinsic"), dtype=np.float64)
    if array.shape == (4,):
        fx, fy, cx, cy = (float(value) for value in array)
        array = np.array(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)))
    if array.shape != (3, 3):
        raise RenderInputError(
            f"intrinsic must have shape (3,3) or be fx/fy/cx/cy, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise RenderInputError("intrinsic contains non-finite values")
    if array[0, 0] <= 0.0 or array[1, 1] <= 0.0:
        raise RenderInputError("intrinsic focal lengths fx and fy must be positive")
    expected_last = np.array((0.0, 0.0, 1.0))
    if not np.allclose(array[2], expected_last, atol=1e-8, rtol=0.0):
        raise RenderInputError(
            f"intrinsic last row must be [0,0,1], got {array[2].tolist()}"
        )
    if not np.isclose(array[0, 1], 0.0, atol=1e-8) or not np.isclose(
        array[1, 0], 0.0, atol=1e-8
    ):
        raise RenderInputError("skewed intrinsics are unsupported; K[0,1] and K[1,0] must be zero")
    cx, cy = float(array[0, 2]), float(array[1, 2])
    if not (0.0 <= cx < float(width) and 0.0 <= cy < float(height)):
        raise RenderInputError(
            f"intrinsic principal point ({cx:.3f},{cy:.3f}) lies outside "
            f"source geometry {width}x{height}; rescaling calibration silently is forbidden"
        )
    return array


def load_world_to_camera(path: str | Path, *, frame_count: int) -> np.ndarray:
    """Load exactly one rigid world-to-camera matrix per source frame."""

    source = Path(path)
    matrices = _load_numeric_array(source, label="world-to-camera calibration")
    matrices = validate_transform_batch(matrices, name="world_to_camera")
    if matrices.shape[0] != frame_count:
        raise RenderInputError(
            f"world_to_camera has {matrices.shape[0]} frames but trajectory/source has "
            f"{frame_count}"
        )
    return matrices


def _strict_trajectory_validation(arrays: Mapping[str, np.ndarray], frame_count: int) -> None:
    robot = _scalar_text(arrays["robot"], "robot")
    gripper = _scalar_text(arrays["gripper"], "gripper")
    if robot != SUPPORTED_ROBOT:
        raise RenderInputError(f"renderer supports robot {SUPPORTED_ROBOT!r}, got {robot!r}")
    if gripper != SUPPORTED_GRIPPER:
        raise RenderInputError(f"renderer supports gripper {SUPPORTED_GRIPPER!r}, got {gripper!r}")

    for side in ("left", "right"):
        valid_raw = np.asarray(arrays[f"{side}_valid"])
        if valid_raw.dtype != np.bool_:
            if not np.issubdtype(valid_raw.dtype, np.integer) or not np.isin(
                valid_raw, (0, 1)
            ).all():
                raise RenderInputError(f"{side}_valid must be boolean or binary integer")
        valid = valid_raw.astype(bool)
        invalid_count = int(frame_count - valid.sum())
        if invalid_count:
            raise RenderInputError(
                f"{side}_valid marks {invalid_count}/{frame_count} frames invalid. "
                "The v1 renderer deliberately has no pose hallucination/hold policy; "
                "interpolate or reject invalid tracks upstream."
            )

        for suffix in ("wrist_position", "wrist_wxyz", "finger_joints"):
            values = np.asarray(arrays[f"{side}_{suffix}"])
            if not np.issubdtype(values.dtype, np.number):
                raise RenderInputError(f"{side}_{suffix} must be numeric")
            if not np.isfinite(values).all():
                raise RenderInputError(f"{side}_{suffix} contains non-finite values")
        quaternion = np.asarray(arrays[f"{side}_wrist_wxyz"], dtype=np.float64)
        norms = np.linalg.norm(quaternion, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3, rtol=0.0):
            bad = int(np.argmax(np.abs(norms - 1.0)))
            raise RenderInputError(
                f"{side}_wrist_wxyz[{bad}] has norm {norms[bad]:.8g}, expected 1"
            )

        names = np.asarray(arrays[f"{side}_finger_joint_names"])
        text_names = [str(name) for name in names.tolist()]
        if any(not name for name in text_names):
            raise RenderInputError(f"{side}_finger_joint_names contains an empty name")
        if len(set(text_names)) != len(text_names):
            raise RenderInputError(f"{side}_finger_joint_names contains duplicates")


def _pose_array(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    result = np.empty((position.shape[0], 4, 4), dtype=np.float64)
    for frame in range(position.shape[0]):
        result[frame] = pose_matrix(position[frame], quaternion[frame])
    return result


@dataclass(frozen=True)
class RenderInputs:
    """Fully validated inputs expressed as world-frame wrist poses."""

    trajectory_path: Path
    intrinsic_path: Path
    world_to_camera_path: Path
    geometry: VideoGeometry
    coordinate_frame: str
    intrinsic: np.ndarray
    world_to_camera: np.ndarray
    trajectory: dict[str, np.ndarray]
    left_world_wrist: np.ndarray
    right_world_wrist: np.ndarray

    @property
    def frame_count(self) -> int:
        return self.geometry.frame_count

    def projection_report(self) -> dict:
        """Summarize wrist projection without claiming visibility of full meshes."""

        report: dict[str, object] = {}
        total_in_front = 0
        total_inside = 0
        for side, poses in (
            ("left", self.left_world_wrist),
            ("right", self.right_world_wrist),
        ):
            camera_poses = self.world_to_camera @ poses
            xyz = camera_poses[:, :3, 3]
            depth = xyz[:, 2]
            in_front = depth > 1e-6
            total_in_front += int(in_front.sum())
            uv = np.full((self.frame_count, 2), np.nan, dtype=np.float64)
            uv[in_front, 0] = (
                self.intrinsic[0, 0] * xyz[in_front, 0] / depth[in_front]
                + self.intrinsic[0, 2]
            )
            uv[in_front, 1] = (
                self.intrinsic[1, 1] * xyz[in_front, 1] / depth[in_front]
                + self.intrinsic[1, 2]
            )
            inside = (
                in_front
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] < self.geometry.width)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] < self.geometry.height)
            )
            total_inside += int(inside.sum())
            front_depth = depth[in_front]
            front_uv = uv[in_front]
            report[side] = {
                "sample_count": self.frame_count,
                "positive_depth_count": int(in_front.sum()),
                "inside_image_count": int(inside.sum()),
                "depth_m_range": (
                    [float(front_depth.min()), float(front_depth.max())]
                    if front_depth.size
                    else None
                ),
                "pixel_bounds": (
                    {
                        "u": [float(front_uv[:, 0].min()), float(front_uv[:, 0].max())],
                        "v": [float(front_uv[:, 1].min()), float(front_uv[:, 1].max())],
                    }
                    if front_uv.size
                    else None
                ),
            }
            if not in_front.any():
                raise RenderInputError(
                    f"all valid {side} wrists project behind the camera; world-to-camera "
                    "convention is likely inverted or frame alignment is wrong"
                )
        if total_in_front == 0:
            raise RenderInputError(
                "all valid wrists project behind the camera; world-to-camera convention is "
                "likely inverted"
            )
        if total_inside == 0:
            raise RenderInputError(
                "no wrist center projects inside the source image on any frame; calibration "
                "scale, frame alignment, or world-to-camera convention is likely wrong"
            )
        return report


def load_render_inputs(
    *,
    trajectory_path: str | Path,
    intrinsic_path: str | Path,
    world_to_camera_path: str | Path,
    width: int,
    height: int,
    fps: float,
) -> RenderInputs:
    """Load the common trajectory and calibration with no inferred transforms."""

    if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width <= 0:
        raise RenderInputError(f"width must be a positive integer, got {width!r}")
    if isinstance(height, bool) or not isinstance(height, (int, np.integer)) or height <= 0:
        raise RenderInputError(f"height must be a positive integer, got {height!r}")
    if width % 2 or height % 2:
        raise RenderInputError(
            f"source geometry {width}x{height} is not even; robot_rgb.mp4 uses yuv420p"
        )
    if not np.isfinite(fps) or fps <= 0.0:
        raise RenderInputError(f"fps must be positive and finite, got {fps!r}")

    trajectory_source = Path(trajectory_path).resolve()
    if not trajectory_source.is_file():
        raise FileNotFoundError(trajectory_source)
    with np.load(trajectory_source, allow_pickle=False) as archive:
        trajectory = {key: np.asarray(archive[key]) for key in archive.files}
    try:
        frame_count = validate_robot_trajectory_arrays(trajectory)
    except (ValueError, KeyError) as exc:
        raise RenderInputError(f"invalid robot trajectory: {exc}") from exc
    if frame_count <= 0:
        raise RenderInputError("robot trajectory contains zero frames")
    if not np.issubdtype(trajectory["frame_indices"].dtype, np.integer):
        raise RenderInputError("robot trajectory frame_indices must have an integer dtype")
    _strict_trajectory_validation(trajectory, frame_count)
    coordinate_frame = _scalar_text(trajectory["coordinate_frame"], "coordinate_frame")

    geometry = VideoGeometry(
        frame_count=frame_count,
        width=int(width),
        height=int(height),
        fps=float(fps),
    )
    intrinsic_source = Path(intrinsic_path).resolve()
    w2c_source = Path(world_to_camera_path).resolve()
    intrinsic = load_intrinsic(intrinsic_source, width=width, height=height)
    world_to_camera = load_world_to_camera(w2c_source, frame_count=frame_count)

    local_poses: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        local_poses[side] = _pose_array(
            np.asarray(trajectory[f"{side}_wrist_position"], dtype=np.float64),
            np.asarray(trajectory[f"{side}_wrist_wxyz"], dtype=np.float64),
        )

    if coordinate_frame == "world":
        left_world = local_poses["left"]
        right_world = local_poses["right"]
    elif coordinate_frame == "camera":
        # A complete robot cannot be fixed independently in every moving camera
        # frame. Convert each wrist back into the shared TACO world frame first.
        camera_to_world = np.empty_like(world_to_camera)
        for frame in range(frame_count):
            camera_to_world[frame] = invert_rigid_transform(world_to_camera[frame])
        left_world = camera_to_world @ local_poses["left"]
        right_world = camera_to_world @ local_poses["right"]
    else:  # common contract currently limits this; keep an explicit guard here.
        raise RenderInputError(f"unsupported coordinate frame {coordinate_frame!r}")

    result = RenderInputs(
        trajectory_path=trajectory_source,
        intrinsic_path=intrinsic_source,
        world_to_camera_path=w2c_source,
        geometry=geometry,
        coordinate_frame=coordinate_frame,
        intrinsic=intrinsic,
        world_to_camera=world_to_camera,
        trajectory=trajectory,
        left_world_wrist=left_world,
        right_world_wrist=right_world,
    )
    result.projection_report()
    return result
