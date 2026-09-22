# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Newton-viewer app base for flash_chord debug/visualization scripts.

Wraps the ``newton.examples`` viewer harness (GL / MP4 / Viser / null): :meth:`ViewerApp.launch`
builds the parser, opens the viewer, constructs the app, and runs the loop via
``newton.examples.run`` (which dispatches ``step()`` / ``render()`` / ``gui()``).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import newton.examples

_NATIVE_VIEWER_BACKENDS = ("gl", "null", "viser")


@dataclass(frozen=True)
class ViewerConfig:
    """Viewer transport settings shared by Hydra-configured replay and evaluation."""

    backend: str = "gl"
    output_path: str = "output.mp4"
    num_frames: int = 100
    headless: bool = False
    quiet: bool = False
    realtime: bool = True
    width: int = 1920
    height: int = 1080
    video_fps: float = 20.0
    camera_position: tuple[float, float, float] | None = None
    camera_target: tuple[float, float, float] | None = None
    background_color: tuple[float, float, float] | None = None
    ambient_sky: tuple[float, float, float] | None = None
    ambient_ground: tuple[float, float, float] | None = None
    diffuse_scale: float | None = None
    specular_scale: float | None = None
    exposure: float | None = None
    spotlight_enabled: bool | None = None
    draw_shadows: bool | None = None
    support_color: tuple[float, float, float] | None = None
    ground_color: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError(f"viewer num_frames must be positive, got {self.num_frames}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"viewer dimensions must be positive, got {self.width}x{self.height}")
        if self.backend == "mp4" and (self.width % 2 or self.height % 2):
            raise ValueError(f"viewer dimensions must be even for MP4 encoding, got {self.width}x{self.height}")
        if not math.isfinite(self.video_fps) or self.video_fps <= 0.0:
            raise ValueError(f"viewer video_fps must be finite and positive, got {self.video_fps}")
        if self.backend == "mp4" and Path(self.output_path).suffix.lower() != ".mp4":
            raise ValueError(f"MP4 viewer output_path must end in .mp4, got {self.output_path!r}")
        vector_fields = (
            "camera_position",
            "camera_target",
            "background_color",
            "ambient_sky",
            "ambient_ground",
            "support_color",
            "ground_color",
        )
        for name in vector_fields:
            value = getattr(self, name)
            if value is None:
                continue
            values = tuple(float(component) for component in value)
            if len(values) != 3 or not all(math.isfinite(component) for component in values):
                raise ValueError(f"viewer {name} must contain three finite values, got {value}")
            if name not in {"camera_position", "camera_target"} and not all(
                0.0 <= component <= 1.0 for component in values
            ):
                raise ValueError(f"viewer {name} components must lie in [0, 1], got {value}")
            object.__setattr__(self, name, values)
        if (self.camera_position is None) != (self.camera_target is None):
            raise ValueError("viewer camera_position and camera_target must be configured together")
        if self.exposure is not None and (not math.isfinite(self.exposure) or self.exposure < 0.0):
            raise ValueError(f"viewer exposure must be finite and non-negative, got {self.exposure}")
        for name in ("diffuse_scale", "specular_scale"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"viewer {name} must be finite and non-negative, got {value}")


def _open_video_writer(output_path: Path, width: int, height: int, fps: float):
    """Open and prime Newton's imageio-ffmpeg streaming encoder."""
    import imageio_ffmpeg

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        size=(width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        quality=5,
        ffmpeg_log_level="warning",
    )
    signal_setting = os.environ.get("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT")
    os.environ["IMAGEIO_FFMPEG_NO_PREVENT_SIGINT"] = "1"
    try:
        writer.send(None)
    finally:
        if signal_setting is None:
            os.environ.pop("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", None)
        else:
            os.environ["IMAGEIO_FFMPEG_NO_PREVENT_SIGINT"] = signal_setting
    return writer


class MP4Viewer:
    """Bounded MP4 recorder around Newton's ordinary headless GL viewer."""

    def __init__(self, viewer, output_path: str | Path, num_frames: int, fps: float):
        self._viewer = viewer
        self.output_path = Path(output_path).expanduser().resolve()
        self.num_frames = num_frames
        self.fps = fps
        self.frame_count = 0
        self._frame_buffer = None
        self._writer = None
        self._closed = False

    def __getattr__(self, name: str):
        return getattr(self._viewer, name)

    def is_running(self) -> bool:
        return self.frame_count < self.num_frames and self._viewer.is_running()

    def end_frame(self) -> None:
        self._viewer.end_frame()
        if not self._viewer.is_running():
            return
        self._frame_buffer = self._viewer.get_frame(target_image=self._frame_buffer)
        if self._writer is None:
            height, width, channels = self._frame_buffer.shape
            if channels != 3:
                raise ValueError(f"viewer frame must be RGB, got shape {self._frame_buffer.shape}")
            self._writer = _open_video_writer(self.output_path, width, height, self.fps)
        self._writer.send(self._frame_buffer.numpy())
        self.frame_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        writer_error = None
        try:
            if self._writer is not None:
                self._writer.close()
        except BaseException as error:  # noqa: BLE001  # ensure GL closes even during cancellation
            writer_error = error
        finally:
            self._viewer.close()
        if writer_error is not None:
            raise RuntimeError(f"failed to finalize MP4 video {self.output_path}") from writer_error
        print(f"Video: {self.output_path} ({self.frame_count} frames at {self.fps:g} fps)")


def create_viewer(config: ViewerConfig, device: str | None = None):
    """Create one Newton viewer from a typed config without parsing process arguments."""
    import newton.viewer
    import warp as wp

    wp.config.quiet = config.quiet
    if device is not None:
        wp.set_device(device)
    if config.backend == "gl":
        return newton.viewer.ViewerGL(width=config.width, height=config.height, headless=config.headless)
    if config.backend == "mp4":
        viewer = newton.viewer.ViewerGL(width=config.width, height=config.height, headless=True)
        return MP4Viewer(viewer, config.output_path, config.num_frames, config.video_fps)
    if config.backend in {"none", "null"}:
        return newton.viewer.ViewerNull(num_frames=config.num_frames)
    if config.backend == "viser":
        return newton.viewer.ViewerViser()
    raise ValueError(f"unsupported viewer backend {config.backend!r}")


def configure_camera(viewer, config: ViewerConfig) -> None:
    """Apply a position/look-at camera after the app has installed its model."""
    if config.camera_position is None or config.camera_target is None:
        return
    camera = getattr(viewer, "camera", None)
    if camera is None or not hasattr(viewer, "set_camera"):
        raise ValueError(f"viewer backend {config.backend!r} does not support camera configuration")

    direction = tuple(target - position for position, target in zip(config.camera_position, config.camera_target))
    distance = math.sqrt(sum(component * component for component in direction))
    if distance <= 1.0e-8:
        raise ValueError("viewer camera_position and camera_target must be distinct")
    unit = tuple(component / distance for component in direction)
    if camera.up_axis == 0:
        pitch = math.degrees(math.asin(unit[0]))
        yaw = math.degrees(math.atan2(unit[2], unit[1]))
    elif camera.up_axis == 1:
        pitch = math.degrees(math.asin(unit[1]))
        yaw = math.degrees(math.atan2(unit[2], unit[0]))
    else:
        pitch = math.degrees(math.asin(unit[2]))
        yaw = math.degrees(math.atan2(unit[1], unit[0]))
    viewer.set_camera(config.camera_position, pitch, yaw)
    camera.set_pivot(config.camera_target)


def configure_render_style(viewer, config: ViewerConfig, scene=None) -> None:
    """Apply optional GL lighting/background and semantic scene colors after model setup."""
    renderer_values = (
        config.background_color,
        config.ambient_sky,
        config.ambient_ground,
        config.diffuse_scale,
        config.specular_scale,
        config.exposure,
        config.spotlight_enabled,
        config.draw_shadows,
    )
    if any(value is not None for value in renderer_values):
        renderer = getattr(viewer, "renderer", None)
        if renderer is None:
            raise ValueError(f"viewer backend {config.backend!r} does not support GL render styling")
        if config.background_color is not None:
            renderer.background_color = config.background_color
            renderer.sky_upper = config.background_color
            renderer.sky_lower = config.background_color
            renderer.draw_sky = False
        if config.ambient_sky is not None:
            renderer.ambient_sky = config.ambient_sky
        if config.ambient_ground is not None:
            renderer.ambient_ground = config.ambient_ground
        if config.diffuse_scale is not None:
            renderer.diffuse_scale = config.diffuse_scale
        if config.specular_scale is not None:
            renderer.specular_scale = config.specular_scale
        if config.exposure is not None:
            renderer.exposure = config.exposure
        if config.spotlight_enabled is not None:
            renderer.spotlight_enabled = config.spotlight_enabled
        if config.draw_shadows is not None:
            renderer.draw_shadows = config.draw_shadows

    if config.support_color is None and config.ground_color is None:
        return
    if scene is None:
        raise ValueError("support/ground render colors require an app exposing its semantic scene")
    if scene.model.shape_color is None:
        raise ValueError("support/ground render colors require model-owned shape colors")

    import warp as wp

    if config.support_color is not None and scene.support is not None:
        shapes_per_world = scene.collision_layout.shape_count
        for world in range(scene.world_count):
            start = world * shapes_per_world + scene.support.shapes.start
            stop = world * shapes_per_world + scene.support.shapes.stop
            scene.model.shape_color[start:stop].fill_(wp.vec3(*config.support_color))
    if config.ground_color is not None and scene.collision_policy.ground:
        ground_shape = scene.world_count * scene.collision_layout.shape_count
        scene.model.shape_color[ground_shape : ground_shape + 1].fill_(wp.vec3(*config.ground_color))


def _restrict_native_viewer_parser(parser) -> None:
    """Limit Newton's generic debug parser to the interactive/headless release transports."""
    import argparse

    for action in parser._actions:
        if action.dest == "viewer":
            action.choices = _NATIVE_VIEWER_BACKENDS
            action.help = "Viewer to use (gl, null, or viser)."
        elif action.dest in {"rerun_address", "output_path"}:
            action.help = argparse.SUPPRESS


class ViewerApp:
    """Base for a flash_chord viewer app.

    Subclasses set ``self.model`` and ``self.state`` in ``__init__`` (after
    ``super().__init__``) and call ``self.viewer.set_model(self.model)``; they may
    override :meth:`step`, :meth:`render`, ``gui(ui)``, or :meth:`add_arguments`.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0
        self.state = None
        configured_viewer = getattr(getattr(args, "config", None), "viewer", None)
        backend = getattr(configured_viewer, "backend", None)
        self._realtime = bool(getattr(configured_viewer, "realtime", False)) and backend in {
            "gl",
            "viser",
        }
        self._realtime_wall_origin = None
        self._realtime_sim_origin = 0.0
        self._realtime_last_sim_time = 0.0

    def step(self) -> None:
        """Advance one frame (no-op for a static view)."""

    def pace_realtime(self) -> None:
        """Throttle interactive rendering to simulation time when configured."""
        if not self._realtime:
            return
        now = time.perf_counter()
        if self._realtime_wall_origin is None or self.sim_time < self._realtime_last_sim_time:
            self._realtime_wall_origin = now
            self._realtime_sim_origin = self.sim_time
        else:
            target = self._realtime_wall_origin + self.sim_time - self._realtime_sim_origin
            remaining = target - now
            if remaining > 0.0:
                time.sleep(remaining)
        self._realtime_last_sim_time = self.sim_time

    def render(self) -> None:
        self.pace_realtime()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    @staticmethod
    def add_arguments(parser) -> None:
        """Override to add app-specific CLI arguments."""

    @classmethod
    def launch(cls) -> None:
        parser = newton.examples.create_parser()
        _restrict_native_viewer_parser(parser)
        cls.add_arguments(parser)
        viewer, args = newton.examples.init(parser)
        app = cls(viewer, args)
        newton.examples.run(app, args)

    @classmethod
    def launch_configured(cls, config: Any, viewer_config: ViewerConfig, device: str | None = None) -> None:
        """Launch from an already-composed application config instead of argparse."""
        viewer = create_viewer(viewer_config, device=device)
        args = SimpleNamespace(test=False, config=config)
        app = cls(viewer, args)
        configure_render_style(viewer, viewer_config, getattr(app, "scene", None))
        configure_camera(viewer, viewer_config)
        try:
            newton.examples.run(app, args)
        except KeyboardInterrupt:
            viewer.close()
            print("Viewer stopped.")
