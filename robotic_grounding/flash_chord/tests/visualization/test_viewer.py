# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for configured viewer timing."""

from types import SimpleNamespace

import pytest

from flash_chord.visualization import viewer as viewer_module
from flash_chord.visualization.viewer import (
    MP4Viewer,
    ViewerApp,
    ViewerConfig,
    _restrict_native_viewer_parser,
    configure_camera,
    configure_render_style,
)


def _args(backend: str, realtime: bool = True):
    viewer = SimpleNamespace(backend=backend, realtime=realtime)
    return SimpleNamespace(config=SimpleNamespace(viewer=viewer))


def test_interactive_viewer_paces_against_simulation_time(monkeypatch):
    times = iter((10.0, 10.01))
    sleeps = []
    monkeypatch.setattr(viewer_module.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(viewer_module.time, "sleep", sleeps.append)
    app = ViewerApp(None, _args("viser"))

    app.pace_realtime()
    app.sim_time = 0.05
    app.pace_realtime()

    assert sleeps == [pytest.approx(0.04)]


def test_headless_viewer_does_not_pace(monkeypatch):
    monkeypatch.setattr(viewer_module.time, "perf_counter", lambda: pytest.fail("clock should not be read"))
    app = ViewerApp(None, _args("none"))

    app.sim_time = 1.0
    app.pace_realtime()


class _Frame:
    shape = (2, 4, 3)

    def numpy(self):
        return b"rgb-frame"


class _GLViewer:
    def __init__(self):
        self.frame = _Frame()
        self.frame_targets = []
        self.end_count = 0
        self.close_count = 0

    def is_running(self):
        return True

    def end_frame(self):
        self.end_count += 1

    def get_frame(self, target_image=None):
        self.frame_targets.append(target_image)
        return self.frame

    def close(self):
        self.close_count += 1


class _Writer:
    def __init__(self):
        self.frames = []
        self.close_count = 0

    def send(self, frame):
        self.frames.append(frame)

    def close(self):
        self.close_count += 1


def test_mp4_viewer_records_exact_frame_count_with_reused_buffer(monkeypatch, tmp_path):
    writer = _Writer()
    opened = []

    def open_writer(path, width, height, fps):
        opened.append((path, width, height, fps))
        writer.send(None)
        return writer

    monkeypatch.setattr(viewer_module, "_open_video_writer", open_writer)
    gl = _GLViewer()
    viewer = MP4Viewer(gl, tmp_path / "policy.mp4", num_frames=2, fps=20.0)

    while viewer.is_running():
        viewer.end_frame()
    viewer.close()
    viewer.close()

    assert opened == [(tmp_path / "policy.mp4", 4, 2, 20.0)]
    assert gl.frame_targets == [None, gl.frame]
    assert writer.frames == [None, b"rgb-frame", b"rgb-frame"]
    assert (viewer.frame_count, gl.end_count, writer.close_count, gl.close_count) == (2, 2, 1, 1)


def test_configured_camera_uses_look_at_after_model_setup():
    camera = SimpleNamespace(up_axis=2, pivots=[])
    camera.set_pivot = camera.pivots.append
    viewer = SimpleNamespace(camera=camera, poses=[])
    viewer.set_camera = lambda position, pitch, yaw: viewer.poses.append((position, pitch, yaw))
    config = ViewerConfig(camera_position=(1.0, 0.0, 1.0), camera_target=(0.0, 0.0, 0.0))

    configure_camera(viewer, config)

    position, pitch, yaw = viewer.poses[0]
    assert position == (1.0, 0.0, 1.0)
    assert pitch == pytest.approx(-45.0)
    assert abs(yaw) == pytest.approx(180.0)
    assert camera.pivots == [(0.0, 0.0, 0.0)]


def test_viewer_config_rejects_incomplete_camera_and_odd_video_dimensions():
    with pytest.raises(ValueError, match="configured together"):
        ViewerConfig(camera_position=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="must be even"):
        ViewerConfig(backend="mp4", output_path="policy.mp4", width=1279, height=720)
    assert ViewerConfig(backend="gl", width=1279, height=719).width == 1279
    with pytest.raises(ValueError, match="must end in .mp4"):
        ViewerConfig(backend="mp4", output_path="policy.usd")


def test_native_debug_parser_excludes_usd_and_rerun_transports():
    parser = viewer_module.newton.examples.create_parser()
    _restrict_native_viewer_parser(parser)
    viewer_action = next(action for action in parser._actions if action.dest == "viewer")

    assert tuple(viewer_action.choices) == ("gl", "null", "viser")
    with pytest.raises(SystemExit):
        parser.parse_args(["--viewer", "usd"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--viewer", "rerun"])


def test_configured_viewer_closes_cleanly_on_keyboard_interrupt(monkeypatch, capsys):
    viewer = SimpleNamespace(close_count=0)
    viewer.close = lambda: setattr(viewer, "close_count", viewer.close_count + 1)
    monkeypatch.setattr(viewer_module, "create_viewer", lambda *_args, **_kwargs: viewer)
    monkeypatch.setattr(viewer_module, "configure_render_style", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(viewer_module, "configure_camera", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        viewer_module.newton.examples,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    ViewerApp.launch_configured(SimpleNamespace(), ViewerConfig(backend="viser"))

    assert viewer.close_count == 1
    assert capsys.readouterr().out == "Viewer stopped.\n"


class _ShapeColorSlice:
    def __init__(self, fills, shape_slice):
        self.fills = fills
        self.shape_slice = shape_slice

    def fill_(self, color):
        self.fills.append((self.shape_slice, tuple(color)))


class _ShapeColors:
    def __init__(self):
        self.fills = []

    def __getitem__(self, shape_slice):
        return _ShapeColorSlice(self.fills, shape_slice)


def test_render_style_uses_semantic_support_and_ground_shape_ranges():
    renderer = SimpleNamespace()
    viewer = SimpleNamespace(renderer=renderer)
    shape_colors = _ShapeColors()
    scene = SimpleNamespace(
        world_count=2,
        support=SimpleNamespace(shapes=SimpleNamespace(start=7, stop=9)),
        collision_layout=SimpleNamespace(shape_count=10),
        collision_policy=SimpleNamespace(ground=True),
        model=SimpleNamespace(shape_color=shape_colors),
    )
    config = ViewerConfig(
        background_color=(1.0, 1.0, 1.0),
        ambient_sky=(1.0, 1.0, 1.0),
        ambient_ground=(0.65, 0.65, 0.65),
        diffuse_scale=1.15,
        specular_scale=0.8,
        exposure=1.8,
        spotlight_enabled=False,
        draw_shadows=True,
        support_color=(0.78, 0.78, 0.78),
        ground_color=(0.92, 0.92, 0.92),
    )

    configure_render_style(viewer, config, scene)

    assert renderer.draw_sky is False
    assert renderer.sky_upper == renderer.sky_lower == (1.0, 1.0, 1.0)
    assert (renderer.ambient_sky, renderer.ambient_ground) == ((1.0, 1.0, 1.0), (0.65, 0.65, 0.65))
    assert (renderer.diffuse_scale, renderer.specular_scale, renderer.exposure) == (1.15, 0.8, 1.8)
    assert (renderer.spotlight_enabled, renderer.draw_shadows) == (False, True)
    assert shape_colors.fills == [
        (slice(7, 9), pytest.approx((0.78, 0.78, 0.78))),
        (slice(17, 19), pytest.approx((0.78, 0.78, 0.78))),
        (slice(20, 21), pytest.approx((0.92, 0.92, 0.92))),
    ]
