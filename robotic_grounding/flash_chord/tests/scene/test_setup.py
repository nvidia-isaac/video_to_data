# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for one-time task/reference/scene composition."""

from pathlib import Path
from types import SimpleNamespace

import pytest


def _reference(path: Path):
    return SimpleNamespace(metadata=SimpleNamespace(source_path=str(path)))


def test_setup_scene_forwards_explicit_task_and_scene_inputs(monkeypatch):
    from flash_chord.scene import setup

    requested = Path("input/reference")
    resolved = Path("/resolved/reference/data.parquet")
    support = Path("/resolved/support.usda")
    reference = _reference(resolved)
    embodiment = object()
    collision = object()
    scene = object()
    calls = {}

    def load_reference(parquet, *, control_fps, motion_speed, source_frame_playback):
        calls["load"] = (parquet, control_fps, motion_speed, source_frame_playback)
        return reference

    def build_scene(bound_embodiment, bound_reference, **kwargs):
        calls["build"] = (bound_embodiment, bound_reference, kwargs)
        return scene

    monkeypatch.setattr(setup, "load_reference", load_reference)
    monkeypatch.setattr(setup, "support_usda_for_reference", lambda bound_reference: support)
    monkeypatch.setattr(setup, "build_scene", build_scene)

    result = setup.setup_scene(
        parquet=requested,
        control_fps=20.0,
        motion_speed=0.5,
        embodiment=embodiment,
        collision=collision,
        world_count=4096,
        include_support=True,
        decompose_objects=False,
        contact_friction=0.5,
        object_free_joint_damping=0.01,
    )

    assert calls["load"] == (requested, 20.0, 0.5, False)
    assert calls["build"] == (
        embodiment,
        reference,
        {
            "world_count": 4096,
            "support_usda": support,
            "collision": collision,
            "decompose_objects": False,
            "contact_friction": 0.5,
            "object_free_joint_damping": 0.01,
        },
    )
    assert result.parquet_path == resolved
    assert result.reference is reference
    assert result.support_usda == support
    assert result.scene is scene


def test_setup_scene_requires_configured_support_before_building(monkeypatch):
    from flash_chord.scene import setup

    reference = _reference(Path("/resolved/reference/data.parquet"))
    monkeypatch.setattr(setup, "load_reference", lambda *args, **kwargs: reference)
    monkeypatch.setattr(setup, "support_usda_for_reference", lambda bound_reference: None)
    monkeypatch.setattr(
        setup,
        "build_scene",
        lambda *args, **kwargs: pytest.fail("scene construction must not run without configured support"),
    )

    with pytest.raises(FileNotFoundError, match="no support asset could be resolved"):
        setup.setup_scene(
            parquet="reference",
            control_fps=20.0,
            motion_speed=1.0,
            embodiment=object(),
            collision=object(),
            world_count=1,
            include_support=True,
            decompose_objects=True,
        )


def test_setup_scene_skips_support_resolution_when_disabled(monkeypatch):
    from flash_chord.scene import setup

    reference = _reference(Path("/resolved/reference/data.parquet"))
    scene = object()
    build_kwargs = {}
    monkeypatch.setattr(setup, "load_reference", lambda *args, **kwargs: reference)
    monkeypatch.setattr(
        setup,
        "support_usda_for_reference",
        lambda bound_reference: pytest.fail("disabled support must not be resolved"),
    )

    def build_scene(*args, **kwargs):
        build_kwargs.update(kwargs)
        return scene

    monkeypatch.setattr(setup, "build_scene", build_scene)
    result = setup.setup_scene(
        parquet="reference",
        control_fps=None,
        motion_speed=1.0,
        embodiment=object(),
        collision=object(),
        world_count=1,
        include_support=False,
        decompose_objects=True,
    )

    assert result.support_usda is None
    assert result.scene is scene
    assert build_kwargs["support_usda"] is None


def test_setup_scene_applies_reference_window_before_support_and_scene_binding(monkeypatch):
    from flash_chord.scene import setup

    source = _reference(Path("/resolved/reference/data.parquet"))
    windowed = _reference(Path("/resolved/reference/data.parquet"))
    scene = object()
    calls = []
    monkeypatch.setattr(setup, "load_reference", lambda *args, **kwargs: source)

    def frame_window(reference, start_frame, end_frame):
        calls.append((reference, start_frame, end_frame))
        return windowed

    monkeypatch.setattr(setup, "frame_window", frame_window)
    monkeypatch.setattr(setup, "support_usda_for_reference", lambda reference: None)
    monkeypatch.setattr(
        setup,
        "build_scene",
        lambda embodiment, reference, **kwargs: scene if reference is windowed else pytest.fail("untrimmed reference"),
    )

    result = setup.setup_scene(
        parquet="reference",
        control_fps=50.0,
        motion_speed=1.0,
        embodiment=object(),
        collision=object(),
        world_count=1,
        include_support=False,
        decompose_objects=True,
        motion_start_frame=300,
        motion_end_frame=-1,
    )

    assert calls == [(source, 300, -1)]
    assert result.reference is windowed
    assert result.scene is scene
