import json
import sys
from pathlib import Path

import pytest

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

import upload_hitl


def _write_metadata(path: Path) -> None:
    path.write_text(
        "calib_seq_name: 2026-01-01_calibration\n"
        "object:\n"
        "  id: wooden_sword\n"
        "action_desc: pass object\n"
    )


def test_upload_hitl_uses_tiled_overlay(tmp_path):
    overlay_dir = tmp_path / "render_hoi_overlay"
    overlay_dir.mkdir()
    tiled = overlay_dir / "tiled_hoi_overlay.mp4"
    tiled.write_bytes(b"tiled video")
    (overlay_dir / "back_stereo_camera_left_hoi_overlay.mp4").write_bytes(
        b"single view"
    )
    metadata = tmp_path / "hoi_metadata.yaml"
    _write_metadata(metadata)
    output_dir = tmp_path / "upload_hitl"

    result = upload_hitl.upload_hitl(
        overlay_dir=str(overlay_dir),
        hoi_metadata_path=str(metadata),
        output_dir=str(output_dir),
        hitl_s3_base="s3://hitl-bucket/prefix",
        hitl_batch_name="batch_20260507",
        video_name="workflow_name",
    )

    video_dst = output_dir / "dataset" / "workflow_name.mp4"
    json_dst = output_dir / "jsons" / "workflow_name.json"
    assert video_dst.read_bytes() == b"tiled video"
    assert result["video_path"] == str(video_dst)
    assert result["json_path"] == str(json_dst)
    assert json.loads(json_dst.read_text()) == {
        "video_url": (
            "https://hitl-bucket.s3.us-west-2.amazonaws.com/"
            "prefix/batch_20260507/dataset/workflow_name.mp4"
        ),
        "object_id": "wooden_sword",
        "action_desc": "pass object",
    }


def test_upload_hitl_fails_without_tiled_overlay(tmp_path):
    overlay_dir = tmp_path / "render_hoi_overlay"
    overlay_dir.mkdir()
    (overlay_dir / "back_stereo_camera_left_hoi_overlay.mp4").write_bytes(b"back")
    (overlay_dir / "front_stereo_camera_left_hoi_overlay.mp4").write_bytes(b"front")
    metadata = tmp_path / "hoi_metadata.yaml"
    _write_metadata(metadata)

    with pytest.raises(FileNotFoundError) as exc_info:
        upload_hitl.upload_hitl(
            overlay_dir=str(overlay_dir),
            hoi_metadata_path=str(metadata),
            output_dir=str(tmp_path / "upload_hitl"),
            hitl_s3_base="s3://hitl-bucket/prefix",
            hitl_batch_name="batch_20260507",
            video_name="workflow_name",
        )

    message = str(exc_info.value)
    assert "HITL upload blocked" in message
    assert "tiled_hoi_overlay.mp4" in message
    assert "back_stereo_camera_left_hoi_overlay.mp4" in message
    assert "front_stereo_camera_left_hoi_overlay.mp4" in message
    assert "single-view fallback" in message
