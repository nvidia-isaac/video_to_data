# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys

import pytest
from PIL import Image
from reconstruction.modules.v2d_hoi_object_reconstruction.tools import (
    locate_logo_regions as locator,
)
from reconstruction.modules.v2d_hoi_object_reconstruction.tools.locate_logo_regions import (
    _deduplicate,
    _extract_json,
    _validate_output_paths,
    _validated_regions,
    _write_review_overlay,
    locate_image,
)


class _FakeModel:
    def __init__(self, response: str | list[str]) -> None:
        self.responses = [response] if isinstance(response, str) else list(response)
        self.calls = []

    def generate_from_frames(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_extract_json_accepts_fenced_response() -> None:
    assert _extract_json('result\n```json\n{"recognizable": false}\n```') == {"recognizable": False}


def test_validated_regions_clamps_and_converts_coordinates() -> None:
    parsed = {
        "regions": [
            {"label": "Example", "bbox": [-20, 250, 1100, 750]},
        ]
    }

    assert _validated_regions(parsed, width=101, height=201) == [
        {
            "label": "Example",
            "bbox_normalized_1000": [0.0, 250.0, 1000.0, 750.0],
            "bbox_pixels": [0, 50, 100, 150],
        }
    ]


def test_validated_regions_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="has no area"):
        _validated_regions(
            {"regions": [{"label": "invalid", "bbox": [40, 40, 20, 60]}]},
            width=101,
            height=201,
        )


def test_locate_image_uses_public_frame_interface_and_writes_candidate_config(
    tmp_path,
) -> None:
    image_path = tmp_path / "texture.png"
    Image.new("RGB", (101, 201), "gray").save(image_path)
    model = _FakeModel(
        '{"recognizable": true, "regions": '
        '[{"label": "Acme", "bbox": [100, 200, 400, 600]}], "notes": "one"}'
    )

    result = locate_image(
        image_path,
        ["Acme"],
        model,
        model_name="test/model",
        backend="local",
        max_new_tokens=42,
    )

    assert result["scan_status"] == "complete"
    assert result["failed_passes"] == []
    assert result["recognizable_proposal"] is True
    assert result["regions"][0]["bbox_pixels"] == [10, 40, 40, 120]
    assert result["candidate_redaction_config"]["expected_size"] == [101, 201]
    assert result["candidate_redaction_config"]["regions"][0]["operation"] == "blur"
    assert len(model.calls) == 1
    assert model.calls[0]["temperature"] == 0.0
    assert model.calls[0]["max_new_tokens"] == 42
    assert model.calls[0]["frames"][0].size == (101, 201)


def test_false_recognizable_flag_does_not_create_redaction_config(tmp_path) -> None:
    image_path = tmp_path / "texture.png"
    Image.new("RGB", (20, 20), "gray").save(image_path)
    model = _FakeModel('{"recognizable": false, "regions": [], "notes": "none"}')

    result = locate_image(
        image_path,
        ["Acme"],
        model,
        model_name="test/model",
        backend="local",
    )

    assert result["recognizable_proposal"] is False
    assert result["regions"] == []
    assert result["scan_status"] == "complete"
    assert result["candidate_redaction_config"] is None


def test_invalid_recognizable_type_is_recorded_as_parse_error(tmp_path) -> None:
    image_path = tmp_path / "texture.png"
    Image.new("RGB", (20, 20), "gray").save(image_path)
    model = _FakeModel('{"recognizable": "false", "regions": []}')

    result = locate_image(
        image_path,
        ["Acme"],
        model,
        model_name="test/model",
        backend="local",
    )

    assert result["regions"] == []
    assert result["scan_status"] == "incomplete"
    assert result["failed_passes"] == ["whole_image"]
    assert result["candidate_redaction_config"] is None
    assert result["passes"][0]["parse_error"] == ("model response recognizable must be a boolean")


def test_recognizable_flag_and_regions_must_agree(tmp_path) -> None:
    image_path = tmp_path / "texture.png"
    Image.new("RGB", (20, 20), "gray").save(image_path)
    model = _FakeModel(
        '{"recognizable": false, "regions": [{"label": "uncertain", "bbox": [100, 100, 500, 500]}]}'
    )

    result = locate_image(
        image_path,
        ["Acme"],
        model,
        model_name="test/model",
        backend="local",
    )

    assert result["scan_status"] == "incomplete"
    assert result["candidate_redaction_config"] is None
    assert result["passes"][0]["parse_error"] == (
        "model response recognizable flag and regions disagree"
    )


def test_tiled_scan_maps_regions_to_atlas_coordinates(tmp_path) -> None:
    image_path = tmp_path / "texture.png"
    Image.new("RGB", (101, 51), "gray").save(image_path)
    model = _FakeModel(
        [
            '{"recognizable": false, "regions": []}',
            '{"recognizable": true, "regions": [{"label": "Acme", "bbox": [500, 200, 900, 800]}]}',
            '{"recognizable": false, "regions": []}',
        ]
    )

    result = locate_image(
        image_path,
        ["Acme"],
        model,
        model_name="test/model",
        backend="local",
        tile_grid=(2, 1),
        tile_overlap=0.0,
    )

    assert len(model.calls) == 3
    assert result["scan"] == {
        "whole_image": True,
        "tile_grid": [2, 1],
        "tile_overlap": 0.0,
    }
    assert result["regions"] == [
        {
            "label": "Acme",
            "bbox_normalized_1000": [250.0, 200.0, 440.0, 800.0],
            "bbox_pixels": [25, 10, 44, 40],
            "proposed_by": ["tile_r1_c1"],
        }
    ]


def test_deduplicate_unions_overlapping_proposals() -> None:
    regions = [
        {
            "label": "Acme",
            "bbox_normalized_1000": [100.0, 100.0, 400.0, 400.0],
            "bbox_pixels": [10, 10, 40, 40],
            "proposed_by": ["whole_image"],
        },
        {
            "label": "Acme",
            "bbox_normalized_1000": [150.0, 150.0, 450.0, 450.0],
            "bbox_pixels": [15, 15, 45, 45],
            "proposed_by": ["tile_r1_c1"],
        },
    ]

    assert _deduplicate(regions) == [
        {
            "label": "Acme",
            "bbox_normalized_1000": [100.0, 100.0, 450.0, 450.0],
            "bbox_pixels": [10, 10, 45, 45],
            "proposed_by": ["tile_r1_c1", "whole_image"],
        }
    ]


def test_review_overlay_draws_box_without_changing_source(tmp_path) -> None:
    source = tmp_path / "texture.png"
    output = tmp_path / "overlay.png"
    Image.new("RGB", (100, 80), "gray").save(source)
    source_bytes = source.read_bytes()

    _write_review_overlay(
        source,
        [{"label": "Acme", "bbox_pixels": [20, 30, 60, 50]}],
        output,
    )

    with Image.open(output) as overlay:
        assert overlay.getpixel((20, 30)) == (255, 215, 0)
    assert source.read_bytes() == source_bytes


def test_output_paths_reject_collisions_and_existing_files(tmp_path) -> None:
    source = tmp_path / "texture.png"
    source.write_bytes(b"source")
    output = tmp_path / "proposal.json"

    with pytest.raises(ValueError, match="must not overwrite"):
        _validate_output_paths(source, [source], overwrite=False)
    with pytest.raises(ValueError, match="must be different"):
        _validate_output_paths(source, [output, output], overwrite=False)

    output.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--force"):
        _validate_output_paths(source, [output], overwrite=False)
    assert _validate_output_paths(source, [output], overwrite=True) == [output.resolve()]


def test_cli_exits_nonzero_and_does_not_write_config_for_incomplete_scan(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "texture.png"
    output = tmp_path / "proposal.json"
    config = tmp_path / "config.json"
    overlay = tmp_path / "overlay.png"
    Image.new("RGB", (20, 20), "gray").save(source)
    config.write_text("stale", encoding="utf-8")
    overlay.write_text("stale", encoding="utf-8")
    model = _FakeModel("not json")
    monkeypatch.setattr(locator, "_load_model", lambda args: model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "locate_logo_regions.py",
            str(source),
            "--expected-mark",
            "Acme",
            "--output",
            str(output),
            "--config-output",
            str(config),
            "--overlay-output",
            str(overlay),
            "--whole-image-only",
            "--force",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        locator.main()

    assert exc_info.value.code == 2
    assert json.loads(output.read_text(encoding="utf-8"))["scan_status"] == ("incomplete")
    assert not config.exists()
    assert not overlay.exists()
