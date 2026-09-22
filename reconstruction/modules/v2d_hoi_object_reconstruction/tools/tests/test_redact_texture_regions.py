# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh
from PIL import Image
from reconstruction.modules.v2d_hoi_object_reconstruction.tools import (
    redact_texture_regions as redactor,
)
from reconstruction.modules.v2d_hoi_object_reconstruction.tools.redact_texture_regions import (
    GLB_BIN_CHUNK,
    GLB_JSON_CHUNK,
    _build_glb,
    _parse_glb,
    _prepare_report_path,
    _validate_usd_texture_dependency,
    mesh_contract_inventory,
    redact_array,
    redact_glb,
    redact_package,
)


def _config(width: int, height: int) -> dict:
    return {
        "schema_version": 1,
        "expected_size": [width, height],
        "regions": [
            {
                "name": "test_mark",
                "shape": "ellipse",
                "center": [width // 2, height // 2],
                "radii": [8, 6],
                "operation": "inpaint",
                "radius_px": 3,
            }
        ],
    }


def _texture(width: int = 64, height: int = 48) -> np.ndarray:
    x = np.arange(width, dtype=np.uint8)[None, :]
    y = np.arange(height, dtype=np.uint8)[:, None]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = x
    image[:, :, 1] = y
    image[:, :, 2] = 80
    image[20:29, 27:38] = [255, 255, 255]
    return image


def _write_textured_glb(path, source_texture: np.ndarray) -> None:
    mesh = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        image=Image.fromarray(source_texture),
    )
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    scene.add_geometry(mesh, geom_name="object", node_name="placed", transform=transform)
    path.write_bytes(scene.export(file_type="glb"))


def test_redact_array_changes_only_masked_pixels() -> None:
    source = _texture()
    redacted, mask, _ = redact_array(source, _config(source.shape[1], source.shape[0]))

    outside = mask == 0
    assert np.array_equal(redacted[outside], source[outside])
    assert np.any(redacted[~outside] != source[~outside])


def test_redact_array_preserves_rgba_alpha_channel() -> None:
    rgb = _texture()
    alpha = np.arange(rgb.shape[0] * rgb.shape[1], dtype=np.uint8).reshape(rgb.shape[:2])
    source = np.dstack((rgb, alpha))
    config = _config(source.shape[1], source.shape[0])
    config["regions"][0] = {
        "name": "test_mark",
        "shape": "rectangle",
        "top_left": [20, 15],
        "bottom_right": [42, 33],
        "operation": "blur",
        "radius_px": 5,
    }

    redacted, _, _ = redact_array(source, config)

    assert np.array_equal(redacted[:, :, 3], source[:, :, 3])
    assert np.any(redacted[:, :, :3] != source[:, :, :3])


def test_redact_array_rejects_a_no_op() -> None:
    source = np.zeros((48, 64, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="did not change any pixels"):
        redact_array(source, _config(64, 48))


def test_redact_glb_preserves_geometry_uvs_and_transforms(tmp_path) -> None:
    source_texture = _texture()
    source = tmp_path / "source.glb"
    _write_textured_glb(source, source_texture)

    document, chunks = _parse_glb(source)
    document["extensionsUsed"] = [
        "KHR_materials_ior",
        "KHR_materials_specular",
        "KHR_materials_transmission",
    ]
    document["materials"][0]["extensions"] = {
        "KHR_materials_ior": {"ior": 1.45},
        "KHR_materials_specular": {"specularFactor": 0.75},
        "KHR_materials_transmission": {"transmissionFactor": 0.2},
    }
    document["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    document["textures"][0]["sampler"] = 0
    source.write_bytes(
        _build_glb(
            [
                (
                    chunk_type,
                    json.dumps(document, separators=(",", ":")).encode("utf-8")
                    if chunk_type == GLB_JSON_CHUNK
                    else payload,
                )
                for chunk_type, payload in chunks
            ]
        )
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(64, 48)), encoding="utf-8")
    source_document, source_chunks = _parse_glb(source)
    before = mesh_contract_inventory(trimesh.load(source, force="scene", process=False))
    output = tmp_path / "redacted.glb"
    report = redact_glb(source, output, config_path)
    output_document, output_chunks = _parse_glb(output)
    after = mesh_contract_inventory(trimesh.load(output, force="scene", process=False))

    assert before == after
    assert output_document["materials"] == source_document["materials"]
    assert output_document["samplers"] == source_document["samplers"]
    assert output_document["extensionsUsed"] == source_document["extensionsUsed"]
    source_bin = next(data for kind, data in source_chunks if kind == GLB_BIN_CHUNK)
    output_bin = next(data for kind, data in output_chunks if kind == GLB_BIN_CHUNK)
    image = source_document["images"][0]
    image_view = source_document["bufferViews"][image["bufferView"]]
    image_start = image_view.get("byteOffset", 0)
    image_end = image_start + image_view["byteLength"]
    assert output_bin[:image_start] == source_bin[:image_start]
    assert output_bin[image_end : len(source_bin)] == source_bin[image_end:]
    assert not any(output_bin[image_start:image_end])
    assert report["geometry_uv_and_transforms_preserved"] is True
    assert report["non_texture_glb_content_preserved"] is True
    assert report["glb_texture_replacement"]["source_texture_payload_removed"] is True
    assert report["outside_mask_byte_identical"] is True


def test_redact_package_changes_only_glb_and_external_texture(tmp_path, monkeypatch) -> None:
    source_texture = _texture()
    source = tmp_path / "source"
    (source / "textures").mkdir(parents=True)
    _write_textured_glb(source / "output.glb", source_texture)
    (source / "output.usd").write_text("root layer", encoding="utf-8")
    (source / "visual_asset.usd").write_text("visual layer", encoding="utf-8")
    Image.fromarray(source_texture).save(source / "textures" / "object.png")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(64, 48)), encoding="utf-8")
    monkeypatch.setattr(
        redactor,
        "_validate_usd_texture_dependency",
        lambda package, texture: {
            "root_references_visual_asset": True,
            "selected_texture_resolved": True,
        },
    )

    output = tmp_path / "output"
    report = redact_package(source, output, config_path)

    assert report["geometry_uv_and_transforms_preserved"] is True
    assert report["external_texture_matches_glb_embedded_texture"] is True
    assert report["other_package_files_byte_identical"] is True
    assert report["source_usd_texture_dependency"]["selected_texture_resolved"] is True
    assert report["output_usd_texture_dependency"]["selected_texture_resolved"] is True
    assert report["glb"]["output"] == str(output / "output.glb")
    assert report["glb"]["texture_output"] == str(output / "textures" / "object.png")
    assert (output / "output.usd").read_bytes() == (source / "output.usd").read_bytes()
    assert (output / "visual_asset.usd").read_bytes() == (source / "visual_asset.usd").read_bytes()
    assert (output / "output.glb").read_bytes() != (source / "output.glb").read_bytes()
    assert (output / "textures" / "object.png").read_bytes() != (
        source / "textures" / "object.png"
    ).read_bytes()


def test_validate_usd_texture_dependency(tmp_path) -> None:
    pytest.importorskip("pxr")
    package = tmp_path / "package"
    texture = package / "textures" / "object.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"png")
    (package / "visual_asset.usd").write_text(
        """#usda 1.0
(
    defaultPrim = "Visual"
)
def Xform "Visual"
{
    custom asset texture = @./textures/object.png@
}
""",
        encoding="utf-8",
    )
    (package / "output.usd").write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset" (
    prepend references = @./visual_asset.usd@
)
{
}
""",
        encoding="utf-8",
    )

    report = _validate_usd_texture_dependency(package, texture)

    assert report["root_references_visual_asset"] is True
    assert report["selected_texture_resolved"] is True
    assert str(texture.resolve()) in report["resolved_assets"]


def test_validate_usd_texture_dependency_rejects_wrong_texture(tmp_path) -> None:
    pytest.importorskip("pxr")
    package = tmp_path / "package"
    texture = package / "textures" / "object.png"
    other_texture = package / "textures" / "other.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"png")
    other_texture.write_bytes(b"other")
    (package / "visual_asset.usd").write_text(
        """#usda 1.0
(
    defaultPrim = "Visual"
)
def Xform "Visual"
{
    custom asset texture = @./textures/other.png@
}
""",
        encoding="utf-8",
    )
    (package / "output.usd").write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset" (
    prepend references = @./visual_asset.usd@
)
{
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not resolve the selected texture"):
        _validate_usd_texture_dependency(package, texture)


def test_redact_glb_refuses_existing_output_without_force(tmp_path) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "output.glb"
    config = tmp_path / "config.json"
    _write_textured_glb(source, _texture())
    output.write_bytes(b"keep")
    config.write_text(json.dumps(_config(64, 48)), encoding="utf-8")

    with pytest.raises(FileExistsError, match="--force"):
        redact_glb(source, output, config)

    assert output.read_bytes() == b"keep"
    report = redact_glb(source, output, config, overwrite=True)
    assert report["status"] == "selected_glb_texture_regions_redacted"


def test_report_path_must_not_collide_with_assets(tmp_path) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "output.glb"
    config = tmp_path / "config.json"

    with pytest.raises(ValueError, match="collides"):
        _prepare_report_path(
            str(config),
            forbidden_paths={source, output, config},
        )

    report = tmp_path / "report.json"
    report.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--force"):
        _prepare_report_path(
            str(report),
            forbidden_paths={source, output, config},
        )
