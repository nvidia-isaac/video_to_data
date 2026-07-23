"""Lazy yourdfpy+pyrender backend for complete parallel-jaw robot URDFs."""

from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping
import uuid

import numpy as np

from .bundle import RobotBundle
from .gripper import GripperTrajectory, map_aperture_trajectory
from .inputs import ParallelJawInputs
from .kinematics import KinematicsResult, solve_kinematics
from .provenance import build_provenance, sha256_file
from .transforms import CV_TO_OPENGL, validate_transform


# Deliberately identical to the established renderer contract so the unchanged
# depth compositor can consume either Vega/Sharpa or a parallel-jaw embodiment.
RENDER_METADATA_SCHEMA = "v2d.inpainting.robot-render/v1"


class RenderError(RuntimeError):
    """Raised when IK, rasterization, encoding, or publication fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _restore_host_ownership(paths: list[Path]) -> None:
    uid_text = os.environ.get("V2D_RENDER_HOST_UID")
    gid_text = os.environ.get("V2D_RENDER_HOST_GID")
    if uid_text is None and gid_text is None:
        return
    if uid_text is None or gid_text is None:
        raise RenderError("host ownership requires both V2D_RENDER_HOST_UID and GID")
    try:
        uid, gid = int(uid_text), int(gid_text)
    except ValueError as exc:
        raise RenderError("host UID/GID must be integers") from exc
    if uid < 0 or gid < 0:
        raise RenderError("host UID/GID must be non-negative")
    for path in paths:
        if path.exists():
            os.chown(path, uid, gid)


def validate_render_visibility(
    pixel_counts: np.ndarray | list[int],
    *,
    width: int,
    height: int,
) -> dict[str, int | float]:
    counts = np.asarray(pixel_counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or np.any(counts < 0):
        raise RenderError("robot pixel counts must be a non-empty nonnegative vector")
    image_pixels = int(width) * int(height)
    if image_pixels <= 0:
        raise RenderError("render geometry must be positive")
    minimum_pixels = max(16, int(np.ceil(image_pixels * 1e-5)))
    required_frames = max(1, int(np.ceil(counts.size * 0.10)))
    visible_frames = int(np.count_nonzero(counts >= minimum_pixels))
    if visible_frames < required_frames:
        raise RenderError(
            f"robot render is blank/nearly blank: {visible_frames}/{counts.size} "
            f"frames have at least {minimum_pixels} pixels, required {required_frames}"
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


def build_frame_configuration(
    bundle: RobotBundle,
    kinematics: KinematicsResult,
    left_gripper: GripperTrajectory,
    right_gripper: GripperTrajectory,
    frame: int,
) -> dict[str, float]:
    configuration = {
        name: float(value) for name, value in bundle.fixed_root_joint_values.items()
    }
    configuration.update(
        {
            name: float(value)
            for name, value in zip(
                kinematics.arm_joint_names,
                kinematics.arm_joint_values[frame],
            )
        }
    )
    configuration.update(
        {
            name: float(value)
            for name, value in zip(
                left_gripper.names,
                left_gripper.values[frame],
            )
        }
    )
    configuration.update(
        {
            name: float(value)
            for name, value in zip(
                right_gripper.names,
                right_gripper.values[frame],
            )
        }
    )
    expected = set(bundle.render_inspection.independent_joint_names)
    if set(configuration) != expected:
        raise RenderError(
            "frame configuration does not exactly cover independent render joints; "
            f"missing={sorted(expected - set(configuration))}, "
            f"extra={sorted(set(configuration) - expected)}"
        )
    return configuration


class _UrdfVisual:
    """Mirror one yourdfpy scene into pyrender; yourdfpy expands URDF mimics."""

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
    def independent_joint_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.model.actuated_joint_names)

    def update(
        self, configuration: Mapping[str, float], root_pose_opengl: np.ndarray
    ) -> None:
        # update_cfg computes mimic followers from their source joint. Supplying
        # only independent joints prevents accidental double application.
        self.model.update_cfg(dict(configuration))
        for graph_node_name, render_node in self._nodes:
            local_pose, _ = self.model.scene.graph[graph_node_name]
            self._render_scene.set_pose(
                render_node,
                pose=root_pose_opengl @ local_pose,
            )


class _FFmpegWriter:
    def __init__(self, path: Path, *, width: int, height: int, fps: float):
        executable = shutil.which("ffmpeg")
        if executable is None:
            try:
                import imageio_ffmpeg

                executable = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, OSError, RuntimeError) as exc:
                raise RenderError("ffmpeg or imageio-ffmpeg is required") from exc
        command = [
            str(executable),
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
            raise RenderError("ffmpeg frame must be uint8 HxWx3 RGB")
        if self._process.stdin is None:
            raise RenderError("ffmpeg stdin is unavailable")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RenderError(f"ffmpeg stopped: {self._read_error()}") from exc

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
            raise RenderError(f"ffmpeg failed with exit {return_code}: {error}")

    def abort(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()


def _verify_video(path: Path, inputs: ParallelJawInputs) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if executable is None:
        try:
            import cv2
        except ImportError as exc:
            raise RenderError(
                "ffprobe or OpenCV is required to verify the encoded robot video"
            ) from exc
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RenderError(f"OpenCV could not open encoded video {path}")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        while capture.grab():
            frame_count += 1
        capture.release()
        actual = (width, height, frame_count)
        expected = (
            inputs.geometry.width,
            inputs.geometry.height,
            inputs.frame_count,
        )
        if actual != expected:
            raise RenderError(
                f"encoded video geometry/count {actual} != expected {expected}"
            )
        if not np.isfinite(encoded_fps) or not np.isclose(
            encoded_fps,
            inputs.geometry.fps,
            atol=max(1e-3, inputs.geometry.fps * 1e-4),
            rtol=0.0,
        ):
            raise RenderError(
                f"encoded fps {encoded_fps:.8g} != expected {inputs.geometry.fps:.8g}"
            )
        return {
            "verification_backend": "opencv",
            "width": width,
            "height": height,
            "decoded_frame_count": frame_count,
            "fps": encoded_fps,
        }
    completed = subprocess.run(
        [
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
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RenderError(f"ffprobe failed: {completed.stderr.strip()}")
    try:
        stream = json.loads(completed.stdout)["streams"][0]
        actual = (
            int(stream["width"]),
            int(stream["height"]),
            int(stream["nb_read_frames"]),
        )
        encoded_fps = float(Fraction(stream["avg_frame_rate"]))
    except (
        KeyError,
        IndexError,
        ValueError,
        ZeroDivisionError,
        json.JSONDecodeError,
    ) as exc:
        raise RenderError(
            f"ffprobe returned malformed metadata: {completed.stdout}"
        ) from exc
    expected = (
        inputs.geometry.width,
        inputs.geometry.height,
        inputs.frame_count,
    )
    if actual != expected:
        raise RenderError(
            f"encoded video geometry/count {actual} != expected {expected}"
        )
    if not np.isclose(
        encoded_fps,
        inputs.geometry.fps,
        atol=max(1e-3, inputs.geometry.fps * 1e-4),
        rtol=0.0,
    ):
        raise RenderError(
            f"encoded fps {encoded_fps:.8g} != expected {inputs.geometry.fps:.8g}"
        )
    stream["verification_backend"] = "ffprobe"
    return stream


def _metadata_base(
    inputs: ParallelJawInputs,
    bundle: RobotBundle,
    *,
    output_dir: Path,
    background_rgb: tuple[int, int, int],
    T_world_hub: np.ndarray,
    orientation_cost: float,
    max_position_residual_m: float,
    max_orientation_residual_deg: float,
    max_joint_step_rad: float,
) -> dict[str, Any]:
    return {
        "schema_version": RENDER_METADATA_SCHEMA,
        "renderer_kind": "generic_parallel_jaw",
        "run_id": str(uuid.uuid4()),
        "started_at": _utc_now(),
        "container_image": os.environ.get("V2D_RENDER_CONTAINER_IMAGE"),
        "container_image_id": os.environ.get("V2D_RENDER_CONTAINER_IMAGE_ID"),
        "host_output_dir": os.environ.get("V2D_RENDER_HOST_OUTPUT_DIR"),
        "geometry": inputs.geometry.as_dict(),
        "rendered_source_frame_indices": [
            int(value) for value in np.asarray(inputs.target["frame_indices"]).tolist()
        ],
        "preview": inputs.preview_source_frame_index is not None,
        "tracker": inputs.tracker,
        "target": str(inputs.target_path),
        "trajectory_coordinate_frame": "world",
        "intrinsic": str(inputs.intrinsic_path),
        "world_to_camera": str(inputs.world_to_camera_path),
        "projection_validation": inputs.projection_report(),
        "robot_bundle": bundle.as_dict(),
        "background_rgb": list(background_rgb),
        "kinematics_policy": {
            "max_position_residual_m": float(max_position_residual_m),
            "max_orientation_residual_deg": float(max_orientation_residual_deg),
            "max_joint_step_rad": float(max_joint_step_rad),
            "elbow_out_gain": 0.0,
            "orientation_cost": float(orientation_cost),
            "root_placement": "explicit_shared_T_world_hub",
        },
        "T_world_hub": T_world_hub.tolist(),
        "coordinate_conventions": {
            "input_target": "T_world_semantic; quaternion wxyz",
            "tcp_rotation": "R_world_tcp = R_world_semantic @ semantic_target_to_tcp_rotation",
            "bundle_root_transform": (
                "T_robot_root_hub maps hub-frame points into robot-root frame"
            ),
            "input_camera": "OpenCV +x right, +y down, +z forward",
            "calibration": "T_camera_world (world-to-camera)",
            "depth": "positive metric camera z; invalid inf",
        },
        "artifacts": {
            "rgb": str((output_dir / "robot_rgb.mp4").resolve()),
            "mask": str((output_dir / "robot_mask.npy").resolve()),
            "depth": str((output_dir / "robot_depth.npy").resolve()),
        },
    }


def render_parallel_jaw_robot(
    inputs: ParallelJawInputs,
    bundle: RobotBundle,
    *,
    scene_utils_root: str | Path,
    output_dir: str | Path,
    T_world_hub: np.ndarray,
    world_hub_source_path: str | Path | None = None,
    background_rgb: tuple[int, int, int] = (0, 0, 0),
    orientation_cost: float = 0.010,
    max_position_residual_m: float = 0.01,
    max_orientation_residual_deg: float = 20.0,
    max_joint_step_rad: float = 0.4,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render full-resolution RGB/mask/depth and publish metadata last."""

    T_world_hub = validate_transform(T_world_hub, name="T_world_hub")
    if len(background_rgb) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 0 <= int(value) <= 255
        for value in background_rgb
    ):
        raise ValueError("background_rgb must contain three integers in [0,255]")
    background_rgb = tuple(int(value) for value in background_rgb)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    finals = {
        "rgb": destination / "robot_rgb.mp4",
        "mask": destination / "robot_mask.npy",
        "depth": destination / "robot_depth.npy",
        "metadata": destination / "render_metadata.json",
    }
    existing = [str(path) for path in finals.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "renderer outputs already exist; pass --overwrite to replace: "
            + ", ".join(existing)
        )

    token = uuid.uuid4().hex
    temporary = {
        "rgb": destination / f".robot_rgb.{token}.partial.mp4",
        "mask": destination / f".robot_mask.{token}.partial.npy",
        "depth": destination / f".robot_depth.{token}.partial.npy",
    }
    failure_path = destination / "render_failure.json"
    metadata = _metadata_base(
        inputs,
        bundle,
        output_dir=destination,
        background_rgb=background_rgb,
        T_world_hub=T_world_hub,
        orientation_cost=orientation_cost,
        max_position_residual_m=max_position_residual_m,
        max_orientation_residual_deg=max_orientation_residual_deg,
        max_joint_step_rad=max_joint_step_rad,
    )
    writer: _FFmpegWriter | None = None
    renderer = None
    mask_memmap = None
    depth_memmap = None
    publication_started = False
    try:
        left_gripper = map_aperture_trajectory(
            np.asarray(inputs.target["left_aperture_m"], dtype=np.float64),
            side="left",
            spec=bundle.gripper_mapping,
            render_inspection=bundle.render_inspection,
        )
        right_gripper = map_aperture_trajectory(
            np.asarray(inputs.target["right_aperture_m"], dtype=np.float64),
            side="right",
            spec=bundle.gripper_mapping,
            render_inspection=bundle.render_inspection,
        )
        metadata["gripper_mapping"] = {
            "left": dict(left_gripper.report),
            "right": dict(right_gripper.report),
        }
        kinematics = solve_kinematics(
            inputs,
            bundle,
            scene_utils_root=scene_utils_root,
            T_world_hub=T_world_hub,
            orientation_cost=orientation_cost,
            max_position_residual_m=max_position_residual_m,
            max_orientation_residual_deg=max_orientation_residual_deg,
            max_joint_step_rad=max_joint_step_rad,
        )
        metadata["kinematics"] = kinematics.as_dict()
        metadata["provenance"] = build_provenance(
            target=inputs.target_path,
            intrinsic=inputs.intrinsic_path,
            world_to_camera=inputs.world_to_camera_path,
            world_hub=world_hub_source_path,
            bundle=bundle,
            arm_ik_source=kinematics.external_ik["source"],
            capture_mode="render_time",
        )

        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        try:
            import pyrender
            import yourdfpy
        except ImportError as exc:
            raise RenderError(
                "pyrender and yourdfpy are required; use the pinned photo-render container"
            ) from exc

        background_float = np.asarray((*background_rgb, 0), dtype=np.float64) / 255.0
        scene = pyrender.Scene(
            bg_color=background_float,
            ambient_light=np.asarray((0.45, 0.45, 0.45), dtype=np.float64),
        )
        camera = pyrender.IntrinsicsCamera(
            fx=float(inputs.intrinsic[0, 0]),
            fy=float(inputs.intrinsic[1, 1]),
            cx=float(inputs.intrinsic[0, 2]),
            cy=float(inputs.intrinsic[1, 2]),
            znear=0.01,
            zfar=10.0,
        )
        scene.add(camera, pose=np.eye(4))
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=2.5),
            pose=np.eye(4),
        )
        visual = _UrdfVisual(bundle.render_urdf, scene, pyrender, yourdfpy)
        expected_independent = set(bundle.render_inspection.independent_joint_names)
        actual_independent = set(visual.independent_joint_names)
        if actual_independent != expected_independent:
            raise RenderError(
                "yourdfpy actuated joints differ from URDF inspection; mimic handling "
                f"or parser semantics changed: yourdfpy={sorted(actual_independent)}, "
                f"expected={sorted(expected_independent)}"
            )

        renderer = pyrender.OffscreenRenderer(
            viewport_width=inputs.geometry.width,
            viewport_height=inputs.geometry.height,
            point_size=1.0,
        )
        shape = (
            inputs.frame_count,
            inputs.geometry.height,
            inputs.geometry.width,
        )
        mask_memmap = np.lib.format.open_memmap(
            temporary["mask"], mode="w+", dtype=np.bool_, shape=shape
        )
        depth_memmap = np.lib.format.open_memmap(
            temporary["depth"], mode="w+", dtype=np.float32, shape=shape
        )
        writer = _FFmpegWriter(
            temporary["rgb"],
            width=inputs.geometry.width,
            height=inputs.geometry.height,
            fps=inputs.geometry.fps,
        )
        flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
        background_array = np.asarray(background_rgb, dtype=np.uint8)
        pixel_counts: list[int] = []
        for frame in range(inputs.frame_count):
            configuration = build_frame_configuration(
                bundle,
                kinematics,
                left_gripper,
                right_gripper,
                frame,
            )
            world_to_opengl = CV_TO_OPENGL @ inputs.world_to_camera[frame]
            visual.update(
                configuration,
                world_to_opengl @ kinematics.T_world_robot_root,
            )
            color_rgba, raw_depth = renderer.render(scene, flags=flags)
            frame_mask = np.isfinite(raw_depth) & (raw_depth > 0.0)
            pixel_counts.append(int(frame_mask.sum()))
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

        visibility = validate_render_visibility(
            pixel_counts,
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
        video_verification = _verify_video(temporary["rgb"], inputs)

        # Invalidate an older commit marker immediately before the first
        # artifact rename. The new complete marker is atomically written only
        # after all three new artifacts are in their final locations.
        publication_started = True
        finals["metadata"].unlink(missing_ok=True)
        temporary["rgb"].replace(finals["rgb"])
        temporary["mask"].replace(finals["mask"])
        temporary["depth"].replace(finals["depth"])

        metadata["state"] = "complete"
        metadata["completed_at"] = _utc_now()
        metadata["render_statistics"] = {
            **visibility,
            "video_verification": video_verification,
        }
        metadata["artifact_bytes"] = {
            key: finals[key].stat().st_size for key in ("rgb", "mask", "depth")
        }
        metadata["artifact_sha256"] = {
            key: sha256_file(finals[key]) for key in ("rgb", "mask", "depth")
        }
        _write_json_atomic(finals["metadata"], metadata)
        failure_path.unlink(missing_ok=True)
        _restore_host_ownership(list(finals.values()))
        return metadata
    except Exception as exc:
        if writer is not None:
            writer.abort()
        if mask_memmap is not None:
            mask_memmap.flush()
            del mask_memmap
        if depth_memmap is not None:
            depth_memmap.flush()
            del depth_memmap
        if renderer is not None:
            renderer.delete()
        for path in temporary.values():
            path.unlink(missing_ok=True)
        failure = {
            "schema_version": "v2d.inpainting.parallel-jaw-render-failure/v1",
            "state": "failed",
            "failed_at": _utc_now(),
            "publication_started": publication_started,
            "robot_id": bundle.robot_id,
            "tracker": inputs.tracker,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        _write_json_atomic(failure_path, failure)
        _restore_host_ownership([failure_path])
        raise
