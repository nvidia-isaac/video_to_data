#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Redact only configured regions of a texture image, GLB, or asset package.

Pixels outside the configured masks are required to remain byte-identical.  For
GLBs, mesh geometry, scene transforms, and UV coordinates are also verified
before the output is accepted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image

REPORT_SCHEMA_VERSION = 1
SUPPORTED_IMAGE_MODES = {"L", "RGB", "RGBA"}
GLB_MAGIC = b"glTF"
GLB_VERSION = 2
GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942


def _array_hash(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(np.asarray(canonical.shape, dtype=np.int64).tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_glb_bytes(payload: bytes, source: str) -> tuple[dict[str, Any], list[tuple[int, bytes]]]:
    if len(payload) < 12:
        raise ValueError(f"GLB is too short: {source}")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != GLB_MAGIC or version != GLB_VERSION:
        raise ValueError(f"expected a GLB 2.0 file: {source}")
    if declared_length != len(payload):
        raise ValueError(f"GLB length header is {declared_length}, actual size is {len(payload)}")

    chunks: list[tuple[int, bytes]] = []
    offset = 12
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError(f"truncated GLB chunk header: {source}")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_end > len(payload):
            raise ValueError(f"truncated GLB chunk payload: {source}")
        chunks.append((chunk_type, payload[offset:chunk_end]))
        offset = chunk_end

    json_chunks = [data for kind, data in chunks if kind == GLB_JSON_CHUNK]
    bin_chunks = [data for kind, data in chunks if kind == GLB_BIN_CHUNK]
    if len(json_chunks) != 1 or len(bin_chunks) != 1:
        raise ValueError("GLB must contain exactly one JSON chunk and one binary chunk")
    document = json.loads(json_chunks[0].rstrip(b" \t\r\n\x00").decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("GLB JSON chunk must contain an object")
    return document, chunks


def _parse_glb(path: Path) -> tuple[dict[str, Any], list[tuple[int, bytes]]]:
    return _parse_glb_bytes(path.read_bytes(), str(path))


def _base_color_image_index(document: dict[str, Any]) -> int:
    materials = document.get("materials", [])
    textures = document.get("textures", [])
    images = document.get("images", [])
    used_materials = {
        primitive["material"]
        for mesh in document.get("meshes", [])
        for primitive in mesh.get("primitives", [])
        if isinstance(primitive, dict) and isinstance(primitive.get("material"), int)
    }
    image_indices: set[int] = set()
    for material_index in used_materials:
        try:
            material = materials[material_index]
            texture_index = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
            image_index = textures[texture_index]["source"]
            images[image_index]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                "GLB material does not have a valid embedded base-color texture"
            ) from exc
        if not isinstance(image_index, int):
            raise ValueError("GLB texture source must be an image index")
        image_indices.add(image_index)
    if len(image_indices) != 1:
        raise ValueError(
            "expected exactly one embedded base-color image used by the mesh, "
            f"found {len(image_indices)}"
        )
    return next(iter(image_indices))


def _padded(payload: bytes, padding: bytes) -> bytes:
    return payload + padding * ((-len(payload)) % 4)


def _build_glb(chunks: list[tuple[int, bytes]]) -> bytes:
    encoded_chunks = []
    for chunk_type, chunk in chunks:
        padding = b" " if chunk_type == GLB_JSON_CHUNK else b"\x00"
        padded = _padded(chunk, padding)
        encoded_chunks.append(struct.pack("<II", len(padded), chunk_type) + padded)
    body = b"".join(encoded_chunks)
    return struct.pack("<4sII", GLB_MAGIC, GLB_VERSION, 12 + len(body)) + body


def _png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _glb_with_replaced_base_color(
    source_path: Path, redacted_image: Image.Image
) -> tuple[bytes, dict[str, Any]]:
    """Replace only the embedded base-color image in a GLB.

    The source image payload is zeroed so the unredacted pixels are not retained
    as unused data.  The replacement PNG is appended in a new buffer view; all
    material, sampler, extension, geometry, animation, and scene metadata stays
    untouched.
    """

    source_document, source_chunks = _parse_glb(source_path)
    document = copy.deepcopy(source_document)
    image_index = _base_color_image_index(document)
    image = document["images"][image_index]
    if not isinstance(image, dict) or not isinstance(image.get("bufferView"), int):
        raise ValueError("base-color image must be embedded through a bufferView")
    if "uri" in image:
        raise ValueError("base-color image must not use an external URI")

    buffer_views = document.get("bufferViews")
    buffers = document.get("buffers")
    if not isinstance(buffer_views, list) or not isinstance(buffers, list):
        raise ValueError("GLB is missing buffers or bufferViews")
    if len(buffers) != 1:
        raise ValueError(f"expected exactly one GLB buffer, found {len(buffers)}")
    if not isinstance(buffers[0], dict) or not isinstance(buffers[0].get("byteLength"), int):
        raise ValueError("GLB buffer must have an integer byteLength")

    original_view_index = image["bufferView"]
    try:
        original_view = buffer_views[original_view_index]
        original_view_buffer = original_view["buffer"]
        original_view_offset = original_view.get("byteOffset", 0)
        original_view_length = original_view["byteLength"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("base-color image has an invalid bufferView") from exc
    if (
        not isinstance(original_view, dict)
        or original_view_buffer != 0
        or not isinstance(original_view_offset, int)
        or not isinstance(original_view_length, int)
        or original_view_offset < 0
        or original_view_length <= 0
    ):
        raise ValueError("base-color image bufferView must be a non-empty byte range")

    shared_images = [
        index
        for index, candidate in enumerate(document["images"])
        if index != image_index
        and isinstance(candidate, dict)
        and candidate.get("bufferView") == original_view_index
    ]
    if shared_images:
        raise ValueError(f"base-color image bufferView is shared by other images: {shared_images}")

    bin_index = next(
        index for index, (kind, _) in enumerate(source_chunks) if kind == GLB_BIN_CHUNK
    )
    source_bin = source_chunks[bin_index][1]
    original_buffer_length = buffers[0]["byteLength"]
    if original_buffer_length > len(source_bin):
        raise ValueError("declared GLB buffer length exceeds its binary chunk")
    original_view_end = original_view_offset + original_view_length
    if original_view_end > original_buffer_length:
        raise ValueError("base-color image bufferView extends beyond the GLB buffer")
    for index, candidate in enumerate(buffer_views):
        if index == original_view_index:
            continue
        if not isinstance(candidate, dict) or candidate.get("buffer") != 0:
            raise ValueError(f"bufferView {index} does not use the GLB binary buffer")
        offset = candidate.get("byteOffset", 0)
        length = candidate.get("byteLength")
        if (
            not isinstance(offset, int)
            or not isinstance(length, int)
            or offset < 0
            or length < 0
            or offset + length > original_buffer_length
        ):
            raise ValueError(f"bufferView {index} has an invalid byte range")
        end = offset + length
        if max(original_view_offset, offset) < min(original_view_end, end):
            raise ValueError(
                "base-color image bufferView overlaps another bufferView; "
                "refusing to alter shared binary data"
            )

    original_image_payload = source_bin[original_view_offset:original_view_end]
    redacted_source_bin = bytearray(source_bin)
    redacted_source_bin[original_view_offset:original_view_end] = b"\x00" * len(
        original_image_payload
    )
    replacement = _png_bytes(redacted_image)
    replacement_offset = len(source_bin)
    replacement_view_index = len(buffer_views)
    buffer_views.append(
        {
            "buffer": 0,
            "byteOffset": replacement_offset,
            "byteLength": len(replacement),
        }
    )
    original_image = copy.deepcopy(image)
    image["bufferView"] = replacement_view_index
    image["mimeType"] = "image/png"
    buffers[0]["byteLength"] = replacement_offset + len(replacement)

    json_payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    output_chunks = list(source_chunks)
    json_index = next(
        index for index, (kind, _) in enumerate(source_chunks) if kind == GLB_JSON_CHUNK
    )
    output_chunks[json_index] = (GLB_JSON_CHUNK, json_payload)
    output_chunks[bin_index] = (
        GLB_BIN_CHUNK,
        bytes(redacted_source_bin) + replacement,
    )
    payload = _build_glb(output_chunks)

    output_document, parsed_chunks = _parse_glb_bytes(payload, "generated GLB")
    output_bin = next(data for kind, data in parsed_chunks if kind == GLB_BIN_CHUNK)
    if output_bin[: len(source_bin)] != bytes(redacted_source_bin):
        raise RuntimeError("GLB replacement changed binary data outside the source image")
    if output_bin[replacement_offset : replacement_offset + len(replacement)] != replacement:
        raise RuntimeError("GLB replacement image payload was not preserved")
    if any(output_bin[original_view_offset:original_view_end]):
        raise RuntimeError("source image payload was not removed from the GLB")
    if original_image_payload in output_bin:
        raise RuntimeError("unredacted source image payload remains in the output GLB")
    normalized_output = copy.deepcopy(output_document)
    appended = normalized_output["bufferViews"].pop()
    if appended != buffer_views[-1]:
        raise RuntimeError("GLB replacement bufferView was not preserved")
    normalized_output["images"][image_index] = original_image
    normalized_output["buffers"][0]["byteLength"] = original_buffer_length
    if normalized_output != source_document:
        raise RuntimeError("GLB replacement changed non-texture JSON content")

    return payload, {
        "image_index": image_index,
        "source_buffer_view": original_view_index,
        "replacement_buffer_view": replacement_view_index,
        "source_mime_type": original_image.get("mimeType"),
        "output_mime_type": "image/png",
        "source_texture_payload_removed": True,
        "non_texture_binary_ranges_byte_identical": True,
        "non_texture_glb_json_preserved": True,
    }


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    regions = config.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("config must contain a non-empty regions list")
    return config_path, config


def _validate_size(config: dict[str, Any], width: int, height: int) -> None:
    expected = config.get("expected_size")
    if expected is None:
        return
    if expected != [width, height]:
        raise ValueError(f"texture size is {width}x{height}, expected {expected[0]}x{expected[1]}")


def _region_mask(region: dict[str, Any], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    shape = region.get("shape")
    if shape == "ellipse":
        center = region.get("center")
        radii = region.get("radii")
        if not (
            isinstance(center, list)
            and len(center) == 2
            and isinstance(radii, list)
            and len(radii) == 2
        ):
            raise ValueError("ellipse requires center [x,y] and radii [rx,ry]")
        cv2.ellipse(
            mask,
            tuple(int(value) for value in center),
            tuple(int(value) for value in radii),
            float(region.get("angle_degrees", 0.0)),
            0,
            360,
            255,
            thickness=-1,
        )
    elif shape == "polygon":
        points = region.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("polygon requires at least three [x,y] points")
        polygon = np.asarray(points, dtype=np.int32)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError("polygon points must be [x,y] pairs")
        cv2.fillPoly(mask, [polygon], 255)
    elif shape == "rectangle":
        top_left = region.get("top_left")
        bottom_right = region.get("bottom_right")
        if not (
            isinstance(top_left, list)
            and len(top_left) == 2
            and isinstance(bottom_right, list)
            and len(bottom_right) == 2
        ):
            raise ValueError("rectangle requires top_left and bottom_right [x,y]")
        cv2.rectangle(
            mask,
            tuple(int(value) for value in top_left),
            tuple(int(value) for value in bottom_right),
            255,
            thickness=-1,
        )
    else:
        raise ValueError(f"unsupported mask shape: {shape!r}")

    dilation = int(region.get("dilate_px", 0))
    if dilation < 0:
        raise ValueError("dilate_px must be non-negative")
    if dilation:
        size = 2 * dilation + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel)
    if not np.any(mask):
        raise ValueError(f"region {region.get('name', '<unnamed>')!r} has an empty mask")
    return mask


def _apply_operation(image: np.ndarray, mask: np.ndarray, region: dict[str, Any]) -> np.ndarray:
    operation = region.get("operation", "inpaint")
    if operation == "inpaint":
        radius = float(region.get("radius_px", 7.0))
        if radius <= 0:
            raise ValueError("inpaint radius_px must be positive")
        method_name = region.get("method", "telea")
        methods = {"telea": cv2.INPAINT_TELEA, "navier_stokes": cv2.INPAINT_NS}
        if method_name not in methods:
            raise ValueError("inpaint method must be telea or navier_stokes")
        if image.ndim == 2:
            edited = cv2.inpaint(image, mask, radius, methods[method_name])
        else:
            color = cv2.inpaint(image[:, :, :3], mask, radius, methods[method_name])
            edited = image.copy()
            edited[:, :, :3] = color
    elif operation == "blur":
        radius = int(region.get("radius_px", 15))
        if radius <= 0:
            raise ValueError("blur radius_px must be positive")
        kernel_size = 2 * radius + 1
        edited = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif operation == "clone":
        offset = region.get("source_offset")
        if not isinstance(offset, list) or len(offset) != 2:
            raise ValueError("clone requires source_offset [dx,dy]")
        dx, dy = (int(value) for value in offset)
        target_y, target_x = np.nonzero(mask)
        source_x = target_x + dx
        source_y = target_y + dy
        height, width = image.shape[:2]
        if (
            np.any(source_x < 0)
            or np.any(source_x >= width)
            or np.any(source_y < 0)
            or np.any(source_y >= height)
        ):
            raise ValueError("clone source_offset samples outside the texture")

        feather = float(region.get("feather_px", 0.0))
        if feather < 0:
            raise ValueError("clone feather_px must be non-negative")
        edited = image.copy()
        cloned = image[source_y, source_x]
        if feather:
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            alpha = np.clip(distance[target_y, target_x] / feather, 0.0, 1.0)
            if image.ndim == 3:
                alpha = alpha[:, None]
            blended = (
                image[target_y, target_x].astype(np.float32) * (1.0 - alpha)
                + cloned.astype(np.float32) * alpha
            )
            cloned = np.rint(blended).astype(np.uint8)
        edited[target_y, target_x] = cloned
    else:
        raise ValueError(f"unsupported redaction operation: {operation!r}")

    result = image.copy()
    selected = mask.astype(bool)
    result[selected] = edited[selected]
    return result


def redact_array(
    source: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Apply configured redactions and return output, union mask, and details."""

    if source.dtype != np.uint8 or source.ndim not in (2, 3):
        raise ValueError("source must be a uint8 grayscale, RGB, or RGBA array")
    if source.ndim == 3 and source.shape[2] not in (3, 4):
        raise ValueError("source must be a uint8 grayscale, RGB, or RGBA array")
    height, width = source.shape[:2]
    _validate_size(config, width, height)

    result = source.copy()
    union_mask = np.zeros((height, width), dtype=np.uint8)
    details: list[dict[str, Any]] = []
    for index, region in enumerate(config["regions"]):
        if not isinstance(region, dict):
            raise ValueError(f"region {index} must be an object")
        mask = _region_mask(region, width, height)
        result = _apply_operation(result, mask, region)
        if source.ndim == 3 and source.shape[2] == 4:
            result[:, :, 3] = source[:, :, 3]
        union_mask = cv2.bitwise_or(union_mask, mask)
        details.append(
            {
                "name": region.get("name", f"region_{index}"),
                "shape": region.get("shape"),
                "operation": region.get("operation", "inpaint"),
                "mask_pixels": int(np.count_nonzero(mask)),
            }
        )

    outside = union_mask == 0
    if not np.array_equal(result[outside], source[outside]):
        raise RuntimeError("redaction changed pixels outside the configured masks")
    if np.array_equal(result, source):
        raise RuntimeError("configured redaction did not change any pixels")
    return result, union_mask, details


def _pil_array(image: Image.Image) -> tuple[np.ndarray, str]:
    image.load()
    if image.mode not in SUPPORTED_IMAGE_MODES:
        raise ValueError(
            f"unsupported image mode {image.mode!r}; expected one of {sorted(SUPPORTED_IMAGE_MODES)}"
        )
    return np.asarray(image).copy(), image.mode


def _redact_pil(
    image: Image.Image,
    config: dict[str, Any],
) -> tuple[Image.Image, np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray]:
    source, mode = _pil_array(image)
    redacted, mask, details = redact_array(source, config)
    output = Image.fromarray(redacted)
    if output.mode != mode:
        raise RuntimeError(f"redacted image mode changed from {mode} to {output.mode}")
    return output, mask, details, source, redacted


def _change_summary(source: np.ndarray, redacted: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    changed = source != redacted
    if changed.ndim == 3:
        changed = np.any(changed, axis=2)
    outside = mask == 0
    return {
        "mask_pixels": int(np.count_nonzero(mask)),
        "changed_pixels": int(np.count_nonzero(changed)),
        "outside_mask_pixel_count": int(np.count_nonzero(outside)),
        "outside_mask_byte_identical": bool(np.array_equal(source[outside], redacted[outside])),
        "source_pixels_sha256": _array_hash(source),
        "output_pixels_sha256": _array_hash(redacted),
    }


def _require_available(destination: Path, *, overwrite: bool, label: str) -> None:
    if destination.is_dir():
        raise IsADirectoryError(f"{label} output is a directory: {destination}")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing {label}: {destination}; pass --force"
        )


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.write-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        staged.write_bytes(payload)
        os.replace(staged, destination)


def _write_png(
    image: Image.Image, destination: Path, expected: np.ndarray, *, overwrite: bool
) -> None:
    if destination.suffix.lower() != ".png":
        raise ValueError("external redacted textures must use a .png extension")
    _require_available(destination, overwrite=overwrite, label="texture")
    payload = _png_bytes(image)
    with Image.open(io.BytesIO(payload)) as reloaded_image:
        reloaded, _ = _pil_array(reloaded_image)
    if not np.array_equal(reloaded, expected):
        raise RuntimeError("PNG encoding did not preserve the redacted pixels")
    _atomic_write_bytes(destination, payload)


def redact_image(
    input_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source_path == destination:
        raise ValueError("refusing to overwrite the source image")
    config_file, config = _load_config(config_path)
    if destination == config_file:
        raise ValueError("output image must not overwrite the redaction config")
    _require_available(destination, overwrite=overwrite, label="image")
    with Image.open(source_path) as image:
        redacted_image, mask, details, source, redacted = _redact_pil(image, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.redaction-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        redacted_image.save(staged)
        with Image.open(staged) as reloaded_image:
            reloaded, _ = _pil_array(reloaded_image)
        if not np.array_equal(reloaded, redacted):
            raise RuntimeError("output image encoding did not preserve the redacted pixels")
        os.replace(staged, destination)

    height, width = source.shape[:2]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "selected_texture_regions_redacted",
        "input": str(source_path),
        "output": str(destination),
        "config": str(config_file),
        "input_sha256": _file_hash(source_path),
        "output_sha256": _file_hash(destination),
        "size": [width, height],
        "regions": details,
        **_change_summary(source, redacted, mask),
    }


def _load_scene(path: Path) -> trimesh.Scene:
    loaded = trimesh.load(path, force="scene", process=False)
    if not isinstance(loaded, trimesh.Scene):
        raise TypeError(f"expected a mesh scene, got {type(loaded).__name__}")
    meshes = [mesh for mesh in loaded.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
    if not meshes or not any(len(mesh.faces) for mesh in meshes):
        raise ValueError(f"no triangle meshes found in {path}")
    return loaded


def mesh_contract_inventory(scene: trimesh.Scene) -> list[dict[str, Any]]:
    """Hash geometry, placement, and UVs that redaction must not change."""

    inventory: list[dict[str, Any]] = []
    for node_name in sorted(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name]
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        uv = getattr(mesh.visual, "uv", None)
        inventory.append(
            {
                "node": str(node_name),
                "geometry": str(geometry_name),
                "vertex_count": int(len(mesh.vertices)),
                "face_count": int(len(mesh.faces)),
                "vertices_sha256": _array_hash(np.asarray(mesh.vertices, dtype=np.float64)),
                "faces_sha256": _array_hash(np.asarray(mesh.faces, dtype=np.int64)),
                "transform_sha256": _array_hash(np.asarray(transform, dtype=np.float64)),
                "uv_sha256": (
                    _array_hash(np.asarray(uv, dtype=np.float64)) if uv is not None else None
                ),
            }
        )
    return inventory


def _texture_entries(scene: trimesh.Scene) -> list[tuple[str, Any, Image.Image]]:
    entries: list[tuple[str, Any, Image.Image]] = []
    for geometry_name, mesh in scene.geometry.items():
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        material = getattr(mesh.visual, "material", None)
        if material is None:
            continue
        image = getattr(material, "baseColorTexture", None)
        if image is None:
            image = getattr(material, "image", None)
        if isinstance(image, Image.Image):
            entries.append((str(geometry_name), material, image))
    return entries


def redact_glb(
    input_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    *,
    texture_output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source_path.suffix.lower() != ".glb" or destination.suffix.lower() != ".glb":
        raise ValueError("input and output must both be .glb files")
    if source_path == destination:
        raise ValueError("refusing to overwrite the source GLB")
    config_file, config = _load_config(config_path)
    if destination == config_file:
        raise ValueError("output GLB must not overwrite the redaction config")
    _require_available(destination, overwrite=overwrite, label="GLB")
    texture_path: Path | None = None
    if texture_output_path is not None:
        texture_path = Path(texture_output_path).expanduser().resolve()
        if texture_path in {source_path, destination, config_file}:
            raise ValueError("texture output collides with an input or GLB output")
        _require_available(texture_path, overwrite=overwrite, label="texture")

    scene = _load_scene(source_path)
    contract_before = mesh_contract_inventory(scene)
    entries = _texture_entries(scene)
    if len(entries) != 1:
        raise ValueError(f"expected exactly one embedded texture, found {len(entries)}")

    geometry_name, _, embedded = entries[0]
    redacted_image, mask, details, source, redacted = _redact_pil(embedded, config)
    payload, glb_replacement = _glb_with_replaced_base_color(source_path, redacted_image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.redaction-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        staged.write_bytes(payload)
        reloaded_scene = _load_scene(staged)
        if mesh_contract_inventory(reloaded_scene) != contract_before:
            raise RuntimeError("GLB replacement changed geometry, placement, or UV coordinates")
        reloaded_entries = _texture_entries(reloaded_scene)
        if len(reloaded_entries) != 1:
            raise RuntimeError("output GLB does not contain exactly one embedded texture")
        reloaded, _ = _pil_array(reloaded_entries[0][2])
        if not np.array_equal(reloaded, redacted):
            raise RuntimeError("embedded output texture does not match the redacted pixels")
        os.replace(staged, destination)

    texture_output: str | None = None
    if texture_path is not None:
        _write_png(redacted_image, texture_path, redacted, overwrite=overwrite)
        texture_output = str(texture_path)

    height, width = source.shape[:2]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "selected_glb_texture_regions_redacted",
        "input": str(source_path),
        "output": str(destination),
        "texture_output": texture_output,
        "config": str(config_file),
        "input_sha256": _file_hash(source_path),
        "output_sha256": _file_hash(destination),
        "geometry_uv_and_transforms_preserved": True,
        "non_texture_glb_content_preserved": True,
        "glb_texture_replacement": glb_replacement,
        "geometry": contract_before,
        "texture_geometry": geometry_name,
        "size": [width, height],
        "regions": details,
        **_change_summary(source, redacted, mask),
    }


def _package_texture(source: Path, texture_name: str | None) -> Path:
    texture_root = (source / "textures").resolve()
    if texture_name:
        candidate = (texture_root / texture_name).resolve()
        if texture_root not in candidate.parents:
            raise ValueError("texture name must remain inside the textures directory")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.suffix.lower() != ".png":
            raise ValueError("package texture must be a PNG")
        return candidate

    candidates = sorted(texture_root.glob("*.png"))
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one PNG in the package textures directory; "
            f"found {len(candidates)}. Pass --texture-name to select one."
        )
    return candidates[0]


def _resolved_asset_path(value: Any, package: Path) -> Path:
    text = str(value)
    candidate = Path(text)
    return candidate.resolve() if candidate.is_absolute() else (package / candidate).resolve()


def _validate_usd_texture_dependency(package: Path, texture: Path) -> dict[str, Any]:
    """Verify that the package root composes the selected external texture."""

    try:
        from pxr import Sdf, UsdUtils
    except ImportError as exc:
        raise RuntimeError(
            "package redaction requires OpenUSD Python bindings; install "
            "usd-core==26.5 or use the documented uv command"
        ) from exc

    package = package.resolve()
    texture = texture.resolve()
    if package not in texture.parents or not texture.is_file():
        raise ValueError("selected USD texture must be an existing package file")
    root_path = package / "output.usd"
    visual_path = package / "visual_asset.usd"
    root_layer = Sdf.Layer.FindOrOpen(str(root_path))
    visual_layer = Sdf.Layer.FindOrOpen(str(visual_path))
    if root_layer is None:
        raise ValueError(f"could not open USD package root: {root_path}")
    if visual_layer is None:
        raise ValueError(f"could not open USD visual layer: {visual_path}")

    references = sorted(str(value) for value in root_layer.GetExternalReferences())
    resolved_references = sorted(
        {_resolved_asset_path(value, package) for value in references}, key=str
    )
    if visual_path.resolve() not in resolved_references:
        raise ValueError("output.usd does not reference visual_asset.usd")

    try:
        _, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(root_path)))
    except Exception as exc:
        raise RuntimeError(f"could not compute USD package dependencies: {exc}") from exc
    resolved_assets = sorted({_resolved_asset_path(value, package) for value in assets}, key=str)
    if texture not in resolved_assets:
        raise ValueError(f"USD package does not resolve the selected texture: {texture}")

    return {
        "root_references_visual_asset": True,
        "selected_texture_resolved": True,
        "root_external_references": references,
        "resolved_root_external_references": [str(path) for path in resolved_references],
        "resolved_assets": [str(path) for path in resolved_assets],
        "unresolved_assets": sorted(str(value) for value in unresolved),
    }


def _package_file_hashes(root: Path, excluded: set[Path]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path not in excluded
    }


def redact_package(
    input_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    texture_name: str | None = None,
) -> dict[str, Any]:
    """Copy a GLB/USD package and redact its embedded and external texture."""

    source = Path(input_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination == source or source in destination.parents:
        raise ValueError("output package must be outside the source package")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output package: {destination}")

    source_glb = source / "output.glb"
    required_usd = [source / "output.usd", source / "visual_asset.usd"]
    missing = [path for path in [source_glb, *required_usd] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"package is missing required files: {missing}")
    source_texture = _package_texture(source, texture_name)
    relative_texture = source_texture.relative_to(source)
    source_usd_dependency = _validate_usd_texture_dependency(source, source_texture)
    with Image.open(source_texture) as external_image:
        source_external, _ = _pil_array(external_image)
    source_embedded_entries = _texture_entries(_load_scene(source_glb))
    if len(source_embedded_entries) != 1:
        raise ValueError("source package GLB must contain exactly one embedded texture")
    source_embedded, _ = _pil_array(source_embedded_entries[0][2])
    if not np.array_equal(source_external, source_embedded):
        raise ValueError("source package external texture does not match its embedded GLB texture")
    excluded = {source_glb, source_texture}
    unchanged_before = _package_file_hashes(source, excluded)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.redaction-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        shutil.copytree(source, staged)
        glb_report = redact_glb(
            source_glb,
            staged / "output.glb",
            config_path,
            texture_output_path=staged / relative_texture,
            overwrite=True,
        )

        unchanged_after = _package_file_hashes(
            staged,
            {staged / "output.glb", staged / relative_texture},
        )
        if unchanged_after != unchanged_before:
            raise RuntimeError("package copy changed a file outside the redacted textures")

        with Image.open(staged / relative_texture) as external_image:
            external, _ = _pil_array(external_image)
        embedded_entries = _texture_entries(_load_scene(staged / "output.glb"))
        if len(embedded_entries) != 1:
            raise RuntimeError("redacted package GLB does not contain exactly one texture")
        embedded, _ = _pil_array(embedded_entries[0][2])
        if not np.array_equal(external, embedded):
            raise RuntimeError("external package texture does not match the GLB texture")

        _validate_usd_texture_dependency(staged, staged / relative_texture)

        staged.rename(destination)
        try:
            output_usd_dependency = _validate_usd_texture_dependency(
                destination, destination / relative_texture
            )
        except Exception:
            destination.rename(staged)
            raise

    glb_report["output"] = str(destination / "output.glb")
    glb_report["texture_output"] = str(destination / relative_texture)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "selected_package_texture_regions_redacted",
        "input": str(source),
        "output": str(destination),
        "texture": relative_texture.as_posix(),
        "geometry_uv_and_transforms_preserved": True,
        "external_texture_matches_glb_embedded_texture": True,
        "other_package_files_byte_identical": True,
        "source_usd_texture_dependency": source_usd_dependency,
        "output_usd_texture_dependency": output_usd_dependency,
        "unchanged_file_sha256": unchanged_before,
        "glb": glb_report,
    }


def _prepare_report_path(
    report_path: str | None,
    *,
    forbidden_paths: set[Path],
    forbidden_roots: tuple[Path, ...] = (),
    overwrite: bool = False,
) -> Path | None:
    if report_path is None:
        return None
    destination = Path(report_path).expanduser().resolve()
    if destination.suffix.lower() != ".json":
        raise ValueError("report output must use a .json extension")
    resolved_forbidden = {path.expanduser().resolve() for path in forbidden_paths}
    if destination in resolved_forbidden or any(
        destination == root or root in destination.parents for root in forbidden_roots
    ):
        raise ValueError("report output collides with a protected input or asset")
    _require_available(destination, overwrite=overwrite, label="report")
    return destination


def _write_report(report: dict[str, Any], destination: Path | None) -> None:
    if destination is not None:
        _atomic_write_bytes(destination, (json.dumps(report, indent=2) + "\n").encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="redact a standalone texture image")
    image_parser.add_argument("input")
    image_parser.add_argument("output")
    image_parser.add_argument("--config", required=True)
    image_parser.add_argument("--report")
    image_parser.add_argument(
        "--force", action="store_true", help="replace existing output and report files"
    )

    glb_parser = subparsers.add_parser("glb", help="redact the embedded texture in a GLB")
    glb_parser.add_argument("input")
    glb_parser.add_argument("output")
    glb_parser.add_argument("--config", required=True)
    glb_parser.add_argument("--texture-output")
    glb_parser.add_argument("--report")
    glb_parser.add_argument(
        "--force", action="store_true", help="replace existing output and report files"
    )

    package_parser = subparsers.add_parser(
        "package", help="copy and redact a standard GLB/USD asset package"
    )
    package_parser.add_argument("input_dir")
    package_parser.add_argument("output_dir")
    package_parser.add_argument("--config", required=True)
    package_parser.add_argument("--texture-name")
    package_parser.add_argument("--report")

    args = parser.parse_args()
    if args.command == "image":
        report_path = _prepare_report_path(
            args.report,
            forbidden_paths={
                Path(args.input),
                Path(args.output),
                Path(args.config),
            },
            overwrite=args.force,
        )
        report = redact_image(args.input, args.output, args.config, overwrite=args.force)
    elif args.command == "glb":
        protected = {
            Path(args.input),
            Path(args.output),
            Path(args.config),
        }
        if args.texture_output:
            protected.add(Path(args.texture_output))
        report_path = _prepare_report_path(
            args.report, forbidden_paths=protected, overwrite=args.force
        )
        report = redact_glb(
            args.input,
            args.output,
            args.config,
            texture_output_path=args.texture_output,
            overwrite=args.force,
        )
    else:
        source = Path(args.input_dir).expanduser().resolve()
        destination = Path(args.output_dir).expanduser().resolve()
        report_path = _prepare_report_path(
            args.report,
            forbidden_paths={
                Path(args.config),
                destination,
                destination / "output.glb",
                destination / "output.usd",
                destination / "visual_asset.usd",
            },
            forbidden_roots=(source,),
        )
        if report_path is not None and destination in report_path.parents:
            relative_report = report_path.relative_to(destination)
            if (source / relative_report).exists():
                raise ValueError(
                    "report output would overwrite a file copied from the source package"
                )
        report = redact_package(
            args.input_dir,
            args.output_dir,
            args.config,
            texture_name=args.texture_name,
        )
    _write_report(report, report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
