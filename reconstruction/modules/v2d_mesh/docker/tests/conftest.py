# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import json
import math
import struct

import pytest

ASSETS = Path(__file__).parent.parent.parent / "assets"


def pytest_addoption(parser):
    parser.addoption(
        "--output-dir",
        default=None,
        help="Save test output artifacts here instead of a temp dir (useful for local inspection)",
    )


@pytest.fixture
def output_dir(request, tmp_path):
    """
    Output directory for a single test.

    With --output-dir /some/path: writes to /some/path/<test_name>/
    Without:                      writes to pytest's tmp_path/output/
    """
    custom = request.config.getoption("--output-dir")
    if custom:
        d = Path(custom) / request.node.name
        d.mkdir(parents=True, exist_ok=True)
        return d
    return tmp_path / "output"


def _write_test_mesh(path: Path) -> None:
    """Write a deterministic GLB sphere without committing a binary fixture."""
    rings, segments = 16, 32
    vertices = [(0.0, 0.0, 2.5), (0.0, 0.0, 1.5)]
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append((
                0.5 * math.sin(phi) * math.cos(theta),
                0.5 * math.sin(phi) * math.sin(theta),
                2.0 + 0.5 * math.cos(phi),
            ))

    indices = []
    first_ring, last_ring = 2, 2 + (rings - 2) * segments
    for segment in range(segments):
        following = (segment + 1) % segments
        indices.extend((0, first_ring + following, first_ring + segment))
        indices.extend((1, last_ring + segment, last_ring + following))
    for ring in range(rings - 2):
        current = first_ring + ring * segments
        following = current + segments
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            indices.extend((current + segment, current + next_segment, following + next_segment))
            indices.extend((current + segment, following + next_segment, following + segment))

    positions = b"".join(struct.pack("<3f", *vertex) for vertex in vertices)
    index_data = struct.pack(f"<{len(indices)}H", *indices)
    binary = positions + index_data
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(index_data), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_data = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_data += b" " * ((-len(json_data)) % 4)
    binary += b"\0" * ((-len(binary)) % 4)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_data) + 8 + len(binary))
        + struct.pack("<I4s", len(json_data), b"JSON") + json_data
        + struct.pack("<I4s", len(binary), b"BIN\0") + binary
    )


@pytest.fixture
def mesh(tmp_path):
    path = tmp_path / "mesh.glb"
    _write_test_mesh(path)
    return str(path)


@pytest.fixture
def intrinsics():
    return str(ASSETS / "intrinsics.json")


@pytest.fixture
def transform():
    return str(ASSETS / "transform.json")


@pytest.fixture
def transforms_glob():
    return str(ASSETS / "transforms/*.json")


@pytest.fixture
def background_image():
    return str(ASSETS / "test_image.jpg")


def is_glb(path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"glTF"


def is_png(path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"\x89PNG"
