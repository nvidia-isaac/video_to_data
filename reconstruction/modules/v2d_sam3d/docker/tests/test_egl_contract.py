# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


_DOCKER_DIR = Path(__file__).parents[1]


def test_sam3d_image_declares_headless_egl_runtime_contract():
    dockerfile = (_DOCKER_DIR / "Dockerfile").read_text()

    for dependency in (
        "libegl1",
        "libgles2",
        "libxrender1",
        "libxext6",
        "libsm6",
        "libx11-6",
        'pyrender==0.1.45',
        'pyglet==2.1.15',
        "networkx",
    ):
        assert dependency in dockerfile

    assert "ENV PYOPENGL_PLATFORM=egl" in dockerfile
    assert "import networkx, pyrender" in dockerfile


def test_renderer_selects_egl_before_importing_pyrender():
    renderer = (_DOCKER_DIR.parent / "lib" / "render_textured_video.py").read_text()

    env_selection = 'os.environ.setdefault("PYOPENGL_PLATFORM", "egl")'
    assert env_selection in renderer
    assert renderer.index(env_selection) < renderer.index("import pyrender")
