# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.mesh.docker.run_mesh_render_depth import run_mesh_render_depth
from v2d.mesh.docker.run_mesh_render_image import run_mesh_render_image
from v2d.mesh.docker.run_mesh_render_mask import run_mesh_render_mask

from .conftest import is_png


# ---------------------------------------------------------------------------
# render_depth
# ---------------------------------------------------------------------------

def test_render_depth(output_dir, mesh, intrinsics):
    out = str(output_dir / "depth.png")
    run_mesh_render_depth(mesh, intrinsics, out)
    assert is_png(out)


def test_render_depth_with_transform(output_dir, mesh, intrinsics, transform):
    out = str(output_dir / "depth.png")
    run_mesh_render_depth(mesh, intrinsics, out, transform_path=transform)
    assert is_png(out)


def test_render_depth_transform_changes_output(output_dir, mesh, intrinsics, transform):
    plain = str(output_dir / "plain.png")
    transformed = str(output_dir / "transformed.png")
    run_mesh_render_depth(mesh, intrinsics, plain)
    run_mesh_render_depth(mesh, intrinsics, transformed, transform_path=transform)
    assert open(plain, "rb").read() != open(transformed, "rb").read()


def test_render_depth_broadcast_transforms(output_dir, mesh, intrinsics, transforms_glob):
    run_mesh_render_depth(mesh, intrinsics, str(output_dir / "*.png"), transform_path=transforms_glob)
    assert is_png(str(output_dir / "0.png"))
    assert is_png(str(output_dir / "1.png"))


# ---------------------------------------------------------------------------
# render_image
# ---------------------------------------------------------------------------

def test_render_image(output_dir, mesh, intrinsics):
    out = str(output_dir / "image.png")
    run_mesh_render_image(mesh, intrinsics, out)
    assert is_png(out)


def test_render_image_with_transform(output_dir, mesh, intrinsics, transform):
    out = str(output_dir / "image.png")
    run_mesh_render_image(mesh, intrinsics, out, transform_path=transform)
    assert is_png(out)


def test_render_image_broadcast_transforms(output_dir, mesh, intrinsics, transforms_glob):
    run_mesh_render_image(mesh, intrinsics, str(output_dir / "*.png"), transform_path=transforms_glob)
    assert is_png(str(output_dir / "0.png"))
    assert is_png(str(output_dir / "1.png"))


def test_render_image_with_background(output_dir, mesh, intrinsics, transform, background_image):
    out = str(output_dir / "image_with_background.png")
    run_mesh_render_image(mesh, intrinsics, out, transform_path=transform, background_path=background_image)
    assert is_png(out)


def test_render_image_background_differs_from_no_background(output_dir, mesh, intrinsics, transform, background_image):
    plain = str(output_dir / "plain.png")
    with_bg = str(output_dir / "with_background.png")
    run_mesh_render_image(mesh, intrinsics, plain, transform_path=transform)
    run_mesh_render_image(mesh, intrinsics, with_bg, transform_path=transform, background_path=background_image)
    assert open(plain, "rb").read() != open(with_bg, "rb").read()


# ---------------------------------------------------------------------------
# render_mask
# ---------------------------------------------------------------------------

def test_render_mask(output_dir, mesh, intrinsics):
    out = str(output_dir / "mask.png")
    run_mesh_render_mask(mesh, intrinsics, out)
    assert is_png(out)


def test_render_mask_with_transform(output_dir, mesh, intrinsics, transform):
    out = str(output_dir / "mask.png")
    run_mesh_render_mask(mesh, intrinsics, out, transform_path=transform)
    assert is_png(out)


def test_render_mask_broadcast_transforms(output_dir, mesh, intrinsics, transforms_glob):
    run_mesh_render_mask(mesh, intrinsics, str(output_dir / "*.png"), transform_path=transforms_glob)
    assert is_png(str(output_dir / "0.png"))
    assert is_png(str(output_dir / "1.png"))
