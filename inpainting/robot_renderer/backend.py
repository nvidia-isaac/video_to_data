"""Lazy IK + pyrender backend for offline Vega/Sharpa image generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
import uuid

import numpy as np

from .assets import RobotAssets, validate_named_joint_trajectory
from .external_ik import (
    FLANGES,
    HAND_MOUNT,
    IK_CONSTRUCTOR_KWARGS,
    ExternalIKModules,
    load_external_ik,
)
from .inputs import RenderInputs
from .provenance import build_provenance, sha256_file
from .transforms import (
    CV_TO_OPENGL,
    invert_rigid_transform,
    matrix_to_quaternion_wxyz,
    pose_matrix,
    validate_rigid_transform,
)


RENDER_METADATA_SCHEMA = "v2d.inpainting.robot-render/v1"


class RenderBackendError(RuntimeError):
    """Raised when IK, OpenGL rendering, or media encoding fails."""


def validate_render_visibility(
    pixel_counts: list[int] | np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, int | float]:
    """Reject blank/nearly blank renders before committing artifacts."""

    counts = np.asarray(pixel_counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or np.any(counts < 0):
        raise RenderBackendError("per-frame robot pixel counts must be a non-empty nonnegative array")
    image_pixels = int(width) * int(height)
    if image_pixels <= 0:
        raise RenderBackendError("render width and height must be positive")
    minimum_pixels = max(16, int(np.ceil(image_pixels * 1e-5)))
    required_frames = max(1, int(np.ceil(counts.size * 0.10)))
    visible_frames = int(np.count_nonzero(counts >= minimum_pixels))
    if visible_frames < required_frames:
        raise RenderBackendError(
            "robot render is blank or nearly blank: only "
            f"{visible_frames}/{counts.size} frames contain at least {minimum_pixels} robot "
            f"pixels (required {required_frames}); recheck calibration and camera convention"
        )
    return {
        "robot_pixel_count": int(counts.sum()),
        "mean_robot_pixels_per_frame": float(counts.mean()),
        "min_robot_pixels_per_frame": int(counts.min()),
        "max_robot_pixels_per_frame": int(counts.max()),
        "visibility_pixel_threshold": minimum_pixels,
        "visible_frame_count": visible_frames,
        "required_visible_frame_count": required_frames,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Pinocchio-compatible fixed-axis RPY matrix (Rz @ Ry @ Rx)."""

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def flange_to_hand_transforms() -> dict[str, np.ndarray]:
    """Return calibrated physical l8-flange -> Sharpa-root matrices."""

    transforms: dict[str, np.ndarray] = {}
    for flange in FLANGES:
        specification = HAND_MOUNT[flange]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _rpy_matrix(*specification["rpy"])
        transform[:3, 3] = np.asarray(specification["xyz"], dtype=np.float64)
        transforms[flange] = transform
    return transforms


def world_wrist_rows(poses: np.ndarray) -> np.ndarray:
    """Convert a world pose batch to ``xyz+wxyz`` rows expected by mount search."""

    result = np.empty((poses.shape[0], 7), dtype=np.float64)
    result[:, :3] = poses[:, :3, 3]
    for frame, transform in enumerate(poses):
        result[frame, 3:] = matrix_to_quaternion_wxyz(transform[:3, :3])
    return result


@dataclass(frozen=True)
class KinematicsResult:
    arm_center_world: np.ndarray
    arm_joint_names: tuple[str, ...]
    arm_joint_values: np.ndarray
    max_position_residual_m: float
    p95_position_residual_m: float
    max_joint_step_rad: float
    mount_method: str
    external_sources: dict[str, Any]

    def as_dict(self) -> dict:
        return {
            "arm_center_world": self.arm_center_world.tolist(),
            "arm_joint_names": list(self.arm_joint_names),
            "max_position_residual_m": self.max_position_residual_m,
            "p95_position_residual_m": self.p95_position_residual_m,
            "max_joint_step_rad": self.max_joint_step_rad,
            "mount_method": self.mount_method,
            "external_sources": self.external_sources,
        }


def _mount_from_external_search(
    inputs: RenderInputs,
    assets: RobotAssets,
    external: ExternalIKModules,
) -> np.ndarray:
    left_rows = world_wrist_rows(inputs.left_world_wrist)
    right_rows = world_wrist_rows(inputs.right_world_wrist)
    placement = external.arm_mount_opt.place_hub_from_wrists(
        left_rows,
        right_rows,
        str(assets.arms.path),
        FLANGES,
        hand_mount=HAND_MOUNT,
        ik_kwargs=IK_CONSTRUCTOR_KWARGS,
        object_xyz=None,
        verbose=True,
    )
    if placement is None:
        raise RenderBackendError(
            "external arm_mount_opt.place_hub_from_wrists reported that a fixed Vega hub "
            "cannot reach this trajectory"
        )
    if not isinstance(placement, tuple) or len(placement) < 2:
        raise RenderBackendError(
            "unexpected place_hub_from_wrists result; expected (position, wxyz, ...)"
        )
    return pose_matrix(np.asarray(placement[0]), np.asarray(placement[1]))


def _build_flange_targets(
    inputs: RenderInputs,
    arm_center_world: np.ndarray,
) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    arm_from_world = invert_rigid_transform(arm_center_world)
    flange_to_hand = flange_to_hand_transforms()
    hand_to_flange = {
        flange: invert_rigid_transform(transform)
        for flange, transform in flange_to_hand.items()
    }
    targets: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    for frame in range(inputs.frame_count):
        frame_targets = {}
        for flange, wrist_poses in (
            (FLANGES[0], inputs.left_world_wrist),
            (FLANGES[1], inputs.right_world_wrist),
        ):
            arm_from_hand = arm_from_world @ wrist_poses[frame]
            arm_from_flange = arm_from_hand @ hand_to_flange[flange]
            frame_targets[flange] = (
                arm_from_flange[:3, 3].copy(),
                matrix_to_quaternion_wxyz(arm_from_flange[:3, :3]),
            )
        targets.append(frame_targets)
    return targets


def solve_kinematics(
    inputs: RenderInputs,
    assets: RobotAssets,
    *,
    scene_utils_root: str | Path,
    arm_center_world: np.ndarray | None = None,
    max_position_residual_m: float = 0.01,
    max_joint_step_rad: float = 0.4,
) -> KinematicsResult:
    """Place the rigid hub and solve one smooth dual-arm IK trajectory."""

    if (
        not np.isfinite(max_position_residual_m)
        or not np.isfinite(max_joint_step_rad)
        or max_position_residual_m <= 0.0
        or max_joint_step_rad <= 0.0
    ):
        raise ValueError("IK residual and joint-step thresholds must be positive")
    external = load_external_ik(scene_utils_root)
    if arm_center_world is None:
        mount = _mount_from_external_search(inputs, assets, external)
        mount_method = "external_arm_mount_opt.place_hub_from_wrists"
    else:
        mount = validate_rigid_transform(arm_center_world, name="arm_center_world")
        mount_method = "explicit_world_transform"

    targets = _build_flange_targets(inputs, mount)
    ik = external.arm_ik.ArmIK(
        str(assets.arms.path),
        flanges=FLANGES,
        **IK_CONSTRUCTOR_KWARGS,
    )
    joint_values = np.asarray(ik.solve_trajectory(targets), dtype=np.float64)
    joint_names = tuple(str(name) for name in ik.joint_names)
    if joint_values.shape != (inputs.frame_count, len(joint_names)):
        raise RenderBackendError(
            "external ArmIK returned an unexpected trajectory shape: "
            f"{joint_values.shape}, expected {(inputs.frame_count, len(joint_names))}"
        )
    if not np.isfinite(joint_values).all():
        raise RenderBackendError("external ArmIK returned non-finite joint values")
    expected_arm_names = set(assets.arms.actuated_joint_names)
    if set(joint_names) != expected_arm_names or len(joint_names) != len(expected_arm_names):
        raise RenderBackendError(
            "Pinocchio arm joint names do not exactly match the rendered URDF: "
            f"pinocchio={joint_names}, urdf={assets.arms.actuated_joint_names}"
        )
    validate_named_joint_trajectory(
        joint_values,
        np.asarray(joint_names),
        assets.arms,
        label="arm_joint_values",
    )

    residuals: list[float] = []
    for frame in range(inputs.frame_count):
        ik.configuration.q = joint_values[frame].copy()
        for flange in FLANGES:
            actual_position, _ = ik.flange_pose(flange)
            residuals.append(
                float(np.linalg.norm(np.asarray(actual_position) - targets[frame][flange][0]))
            )
    residual_array = np.asarray(residuals)
    residual_max = float(residual_array.max())
    residual_p95 = float(np.percentile(residual_array, 95.0))
    if inputs.frame_count > 1:
        joint_step = float(np.max(np.abs(np.diff(joint_values, axis=0))))
    else:
        joint_step = 0.0
    if residual_max > max_position_residual_m:
        raise RenderBackendError(
            f"Vega IK max attachment residual {residual_max:.4f} m exceeds "
            f"threshold {max_position_residual_m:.4f} m"
        )
    if joint_step > max_joint_step_rad:
        raise RenderBackendError(
            f"Vega IK max frame-to-frame joint step {joint_step:.4f} rad exceeds "
            f"threshold {max_joint_step_rad:.4f} rad"
        )
    return KinematicsResult(
        arm_center_world=mount,
        arm_joint_names=joint_names,
        arm_joint_values=joint_values.astype(np.float32),
        max_position_residual_m=residual_max,
        p95_position_residual_m=residual_p95,
        max_joint_step_rad=joint_step,
        mount_method=mount_method,
        external_sources=external.sources.as_dict(),
    )


class _UrdfVisual:
    """A yourdfpy model whose visual nodes are mirrored into one pyrender scene."""

    def __init__(self, urdf_path: Path, render_scene, pyrender_module, yourdfpy_module):
        self.model = yourdfpy_module.URDF.load(
            str(urdf_path),
            build_scene_graph=True,
            build_collision_scene_graph=False,
            load_meshes=True,
            load_collision_meshes=False,
            force_mesh=False,
        )
        self._render_scene = render_scene
        self._nodes: list[tuple[str, Any]] = []
        for graph_node_name in sorted(self.model.scene.graph.nodes_geometry):
            _, geometry_name = self.model.scene.graph[graph_node_name]
            mesh = self.model.scene.geometry[geometry_name]
            render_mesh = pyrender_module.Mesh.from_trimesh(mesh, smooth=False)
            render_node = render_scene.add(render_mesh, pose=np.eye(4))
            self._nodes.append((graph_node_name, render_node))

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.model.actuated_joint_names)

    def update(self, configuration: dict[str, float], root_pose_opengl: np.ndarray) -> None:
        self.model.update_cfg(configuration)
        for graph_node_name, render_node in self._nodes:
            local_pose, _ = self.model.scene.graph[graph_node_name]
            self._render_scene.set_pose(render_node, pose=root_pose_opengl @ local_pose)


class _FFmpegWriter:
    def __init__(self, path: Path, *, width: int, height: int, fps: float):
        executable = shutil.which("ffmpeg")
        if executable is None:
            try:
                import imageio_ffmpeg

                bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
            except (ImportError, RuntimeError, OSError) as exc:
                raise RenderBackendError(
                    "ffmpeg is required to encode robot_rgb.mp4; neither PATH nor "
                    "imageio-ffmpeg provides it"
                ) from exc
            if not bundled.is_file():
                raise RenderBackendError(
                    f"imageio-ffmpeg reported a missing executable at {bundled}"
                )
            executable = str(bundled)
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.12g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise RenderBackendError("ffmpeg frame must be uint8 HxWx3 RGB")
        if self._process.stdin is None:
            raise RenderBackendError("ffmpeg stdin is unavailable")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            error = self._read_error()
            raise RenderBackendError(f"ffmpeg stopped while encoding: {error}") from exc

    def _read_error(self) -> str:
        if self._process.stderr is None:
            return "no stderr"
        return self._process.stderr.read().decode("utf-8", errors="replace").strip()

    def close(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        error = self._read_error()
        return_code = self._process.wait()
        if return_code:
            raise RenderBackendError(f"ffmpeg failed with exit {return_code}: {error}")

    def abort(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()


def _verify_video(path: Path, inputs: RenderInputs) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if executable is None:
        try:
            import cv2
        except ImportError as exc:
            raise RenderBackendError(
                "ffprobe or OpenCV is required to verify robot_rgb.mp4"
            ) from exc
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RenderBackendError(f"OpenCV could not open encoded video {path}")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        while capture.grab():
            frame_count += 1
        capture.release()
        actual = (width, height, frame_count)
        expected = (inputs.geometry.width, inputs.geometry.height, inputs.frame_count)
        if actual != expected:
            raise RenderBackendError(
                f"encoded video geometry/count {actual} != expected {expected}"
            )
        if not np.isfinite(fps) or not np.isclose(
            fps, inputs.geometry.fps, atol=max(1e-3, inputs.geometry.fps * 1e-4), rtol=0.0
        ):
            raise RenderBackendError(
                f"encoded video fps {fps:.8g} != expected {inputs.geometry.fps:.8g}"
            )
        return {
            "verification_backend": "opencv",
            "width": width,
            "height": height,
            "decoded_frame_count": frame_count,
            "fps": fps,
        }
    command = [
        executable,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_read_frames,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RenderBackendError(f"ffprobe failed: {completed.stderr.strip()}")
    try:
        stream = json.loads(completed.stdout)["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RenderBackendError(f"ffprobe returned malformed metadata: {completed.stdout}") from exc
    actual = (
        int(stream["width"]),
        int(stream["height"]),
        int(stream["nb_read_frames"]),
    )
    expected = (inputs.geometry.width, inputs.geometry.height, inputs.frame_count)
    if actual != expected:
        raise RenderBackendError(f"encoded video geometry/count {actual} != expected {expected}")
    try:
        encoded_fps = float(Fraction(stream["avg_frame_rate"]))
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise RenderBackendError(f"ffprobe returned invalid frame rate {stream}") from exc
    if not np.isclose(
        encoded_fps,
        inputs.geometry.fps,
        atol=max(1e-3, inputs.geometry.fps * 1e-4),
        rtol=0.0,
    ):
        raise RenderBackendError(
            f"encoded video fps {encoded_fps:.8g} != expected {inputs.geometry.fps:.8g}"
        )
    stream["verification_backend"] = "ffprobe"
    return stream


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.stem}.partial{path.suffix}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _restore_host_ownership(paths: list[Path]) -> None:
    """Chown container outputs back to the host caller when the wrapper requests it."""

    uid_text = os.environ.get("V2D_RENDER_HOST_UID")
    gid_text = os.environ.get("V2D_RENDER_HOST_GID")
    if uid_text is None and gid_text is None:
        return
    if uid_text is None or gid_text is None:
        raise RenderBackendError("host ownership requires both V2D_RENDER_HOST_UID and GID")
    try:
        uid, gid = int(uid_text), int(gid_text)
    except ValueError as exc:
        raise RenderBackendError("host UID/GID must be non-negative integers") from exc
    if uid < 0 or gid < 0:
        raise RenderBackendError("host UID/GID must be non-negative integers")
    for path in paths:
        if path.exists():
            os.chown(path, uid, gid)


def _metadata_base(
    inputs: RenderInputs,
    assets: RobotAssets,
    *,
    output_dir: Path,
    background_rgb: tuple[int, int, int],
    max_position_residual_m: float,
    max_joint_step_rad: float,
) -> dict[str, Any]:
    return {
        "schema_version": RENDER_METADATA_SCHEMA,
        "run_id": str(uuid.uuid4()),
        "started_at": _utc_now(),
        "state": "running",
        "container_image": os.environ.get("V2D_RENDER_CONTAINER_IMAGE"),
        "container_image_id": os.environ.get("V2D_RENDER_CONTAINER_IMAGE_ID"),
        "host_output_dir": os.environ.get("V2D_RENDER_HOST_OUTPUT_DIR"),
        "geometry": inputs.geometry.as_dict(),
        "trajectory": str(inputs.trajectory_path),
        "trajectory_coordinate_frame": inputs.coordinate_frame,
        "intrinsic": str(inputs.intrinsic_path),
        "world_to_camera": str(inputs.world_to_camera_path),
        "projection_validation": inputs.projection_report(),
        "provenance": build_provenance(
            trajectory=inputs.trajectory_path,
            intrinsic=inputs.intrinsic_path,
            world_to_camera=inputs.world_to_camera_path,
            capture_mode="render_time",
        ),
        "assets": assets.as_dict(),
        "background_rgb": list(background_rgb),
        "kinematics_policy": {
            "max_position_residual_m": float(max_position_residual_m),
            "max_joint_step_rad": float(max_joint_step_rad),
        },
        "coordinate_conventions": {
            "input_camera": "OpenCV +x right, +y down, +z forward",
            "calibration": "T_camera_world (world-to-camera)",
            "quaternion": "wxyz",
            "depth": "positive metric camera z; invalid inf",
        },
        "artifacts": {
            "rgb": str((output_dir / "robot_rgb.mp4").resolve()),
            "mask": str((output_dir / "robot_mask.npy").resolve()),
            "depth": str((output_dir / "robot_depth.npy").resolve()),
        },
    }


def render_robot(
    inputs: RenderInputs,
    assets: RobotAssets,
    *,
    scene_utils_root: str | Path,
    output_dir: str | Path,
    arm_center_world: np.ndarray | None = None,
    background_rgb: tuple[int, int, int] = (0, 0, 0),
    max_position_residual_m: float = 0.01,
    max_joint_step_rad: float = 0.4,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render RGB/mask/depth artifacts; metadata is committed last."""

    if len(background_rgb) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 0 <= int(value) <= 255
        for value in background_rgb
    ):
        raise ValueError("background_rgb must contain three integer values in [0,255]")
    if (
        not np.isfinite(max_position_residual_m)
        or not np.isfinite(max_joint_step_rad)
        or max_position_residual_m <= 0.0
        or max_joint_step_rad <= 0.0
    ):
        raise ValueError("IK residual and joint-step thresholds must be positive")
    background_rgb = tuple(int(value) for value in background_rgb)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "rgb": destination / "robot_rgb.mp4",
        "mask": destination / "robot_mask.npy",
        "depth": destination / "robot_depth.npy",
        "metadata": destination / "render_metadata.json",
    }
    existing = [str(path) for path in final_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "renderer outputs already exist; pass --overwrite to replace exactly these files: "
            + ", ".join(existing)
        )
    temporary_paths = {
        "rgb": destination / "robot_rgb.partial.mp4",
        "mask": destination / "robot_mask.partial.npy",
        "depth": destination / "robot_depth.partial.npy",
    }
    stale = [str(path) for path in temporary_paths.values() if path.exists()]
    if stale and not overwrite:
        raise FileExistsError(
            "stale partial renderer outputs exist; inspect or pass --overwrite: " + ", ".join(stale)
        )
    if overwrite:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    metadata = _metadata_base(
        inputs,
        assets,
        output_dir=destination,
        background_rgb=background_rgb,
        max_position_residual_m=max_position_residual_m,
        max_joint_step_rad=max_joint_step_rad,
    )
    writer: _FFmpegWriter | None = None
    renderer = None
    mask_memmap = None
    depth_memmap = None
    try:
        kinematics = solve_kinematics(
            inputs,
            assets,
            scene_utils_root=scene_utils_root,
            arm_center_world=arm_center_world,
            max_position_residual_m=max_position_residual_m,
            max_joint_step_rad=max_joint_step_rad,
        )
        metadata["kinematics"] = kinematics.as_dict()

        # Set before importing pyrender/OpenGL. The container wrapper also sets it
        # at process start, while this fallback keeps direct invocations explicit.
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        try:
            import pyrender
            import yourdfpy
        except ImportError as exc:
            raise RenderBackendError(
                "pyrender and yourdfpy are required; use container_runner.py with "
                "robotic-grounding:photo-render-v6"
            ) from exc

        bg_float = np.array((*background_rgb, 0), dtype=np.float64) / 255.0
        render_scene = pyrender.Scene(
            bg_color=bg_float,
            ambient_light=np.array((0.45, 0.45, 0.45), dtype=np.float64),
        )
        camera = pyrender.IntrinsicsCamera(
            fx=float(inputs.intrinsic[0, 0]),
            fy=float(inputs.intrinsic[1, 1]),
            cx=float(inputs.intrinsic[0, 2]),
            cy=float(inputs.intrinsic[1, 2]),
            znear=0.01,
            zfar=10.0,
        )
        render_scene.add(camera, pose=np.eye(4))
        key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
        render_scene.add(key_light, pose=np.eye(4))

        arms_visual = _UrdfVisual(assets.arms.path, render_scene, pyrender, yourdfpy)
        left_visual = _UrdfVisual(assets.left_hand.path, render_scene, pyrender, yourdfpy)
        right_visual = _UrdfVisual(assets.right_hand.path, render_scene, pyrender, yourdfpy)
        expected_names = {
            "arms": tuple(assets.arms.actuated_joint_names),
            "left": tuple(assets.left_hand.actuated_joint_names),
            "right": tuple(assets.right_hand.actuated_joint_names),
        }
        actual_names = {
            "arms": arms_visual.joint_names,
            "left": left_visual.joint_names,
            "right": right_visual.joint_names,
        }
        for part in expected_names:
            if set(actual_names[part]) != set(expected_names[part]):
                raise RenderBackendError(
                    f"yourdfpy {part} joint names {actual_names[part]} differ from XML "
                    f"inspection {expected_names[part]}"
                )

        renderer = pyrender.OffscreenRenderer(
            viewport_width=inputs.geometry.width,
            viewport_height=inputs.geometry.height,
            point_size=1.0,
        )
        shape = (inputs.frame_count, inputs.geometry.height, inputs.geometry.width)
        mask_memmap = np.lib.format.open_memmap(
            temporary_paths["mask"], mode="w+", dtype=np.bool_, shape=shape
        )
        depth_memmap = np.lib.format.open_memmap(
            temporary_paths["depth"], mode="w+", dtype=np.float32, shape=shape
        )
        writer = _FFmpegWriter(
            temporary_paths["rgb"],
            width=inputs.geometry.width,
            height=inputs.geometry.height,
            fps=inputs.geometry.fps,
        )

        left_names = [str(name) for name in inputs.trajectory["left_finger_joint_names"]]
        right_names = [str(name) for name in inputs.trajectory["right_finger_joint_names"]]
        left_values = np.asarray(inputs.trajectory["left_finger_joints"], dtype=np.float64)
        right_values = np.asarray(inputs.trajectory["right_finger_joints"], dtype=np.float64)
        flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
        background_array = np.asarray(background_rgb, dtype=np.uint8)
        frame_pixel_counts: list[int] = []
        for frame in range(inputs.frame_count):
            world_to_gl = CV_TO_OPENGL @ inputs.world_to_camera[frame]
            arms_visual.update(
                dict(
                    zip(
                        kinematics.arm_joint_names,
                        (float(value) for value in kinematics.arm_joint_values[frame]),
                    )
                ),
                world_to_gl @ kinematics.arm_center_world,
            )
            left_visual.update(
                dict(zip(left_names, (float(value) for value in left_values[frame]))),
                world_to_gl @ inputs.left_world_wrist[frame],
            )
            right_visual.update(
                dict(zip(right_names, (float(value) for value in right_values[frame]))),
                world_to_gl @ inputs.right_world_wrist[frame],
            )
            color_rgba, raw_depth = renderer.render(render_scene, flags=flags)
            frame_mask = np.isfinite(raw_depth) & (raw_depth > 0.0)
            frame_pixel_counts.append(int(frame_mask.sum()))
            frame_depth = np.asarray(raw_depth, dtype=np.float32)
            frame_depth[~frame_mask] = np.inf
            frame_rgb = np.broadcast_to(
                background_array,
                (inputs.geometry.height, inputs.geometry.width, 3),
            ).copy()
            frame_rgb[frame_mask] = color_rgba[frame_mask, :3]
            mask_memmap[frame] = frame_mask
            depth_memmap[frame] = frame_depth
            writer.write(frame_rgb)

        visibility_statistics = validate_render_visibility(
            frame_pixel_counts,
            width=inputs.geometry.width,
            height=inputs.geometry.height,
        )
        writer.close()
        writer = None
        mask_memmap.flush()
        depth_memmap.flush()
        del mask_memmap
        del depth_memmap
        mask_memmap = None
        depth_memmap = None
        renderer.delete()
        renderer = None
        video_stream = _verify_video(temporary_paths["rgb"], inputs)

        # The sidecar is the multi-file commit marker. Invalidate any prior
        # complete generation before the first sequential artifact rename so a
        # concurrent consumer can never treat mixed generations as complete.
        metadata["state"] = "committing"
        metadata["committing_at"] = _utc_now()
        _write_json_atomic(final_paths["metadata"], metadata)
        temporary_paths["rgb"].replace(final_paths["rgb"])
        temporary_paths["mask"].replace(final_paths["mask"])
        temporary_paths["depth"].replace(final_paths["depth"])

        metadata["state"] = "complete"
        metadata["completed_at"] = _utc_now()
        metadata["render_statistics"] = {
            **visibility_statistics,
            "video_verification": video_stream,
        }
        metadata["artifact_bytes"] = {
            name: final_paths[name].stat().st_size for name in ("rgb", "mask", "depth")
        }
        metadata["artifact_sha256"] = {
            name: sha256_file(final_paths[name]) for name in ("rgb", "mask", "depth")
        }
        _write_json_atomic(final_paths["metadata"], metadata)
        _restore_host_ownership(list(final_paths.values()))
        return metadata
    except Exception as exc:
        if writer is not None:
            writer.abort()
        if mask_memmap is not None:
            mask_memmap.flush()
        if depth_memmap is not None:
            depth_memmap.flush()
        if renderer is not None:
            renderer.delete()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        metadata["state"] = "failed"
        metadata["failed_at"] = _utc_now()
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        # A failed sidecar is useful, but never overwrite a prior complete run
        # unless the caller explicitly authorized replacement.
        if overwrite or not final_paths["metadata"].exists():
            _write_json_atomic(final_paths["metadata"], metadata)
            _restore_host_ownership([final_paths["metadata"]])
        raise
