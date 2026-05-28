import json
import sys
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import backfill_failure_segments as backfill


class _Body:
    def __init__(self, text: str):
        self.text = text

    def read(self):
        return self.text.encode("utf-8")


class _RequestPage:
    def __init__(self, objects: dict[str, str]):
        self.objects = objects

    def paginate(self, **kwargs):
        prefix = kwargs["Prefix"]
        contents = [
            {"Key": key}
            for key in sorted(self.objects)
            if key.startswith(prefix)
        ]
        return [{"Contents": contents}]


class _FakeS3:
    def __init__(self, objects: dict[str, str]):
        self.objects = dict(objects)
        self.puts: list[tuple[str, str]] = []

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return _RequestPage(self.objects)

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        if key not in self.objects:
            raise FileNotFoundError(key)
        return {"Body": _Body(self.objects[key])}

    def head_object(self, **kwargs):
        key = kwargs["Key"]
        if key not in self.objects:
            raise FileNotFoundError(key)
        return {}

    def put_object(self, **kwargs):
        body = kwargs["Body"]
        if isinstance(body, bytes):
            text = body.decode("utf-8")
        else:
            text = str(body)
        self.objects[kwargs["Key"]] = text
        self.puts.append((kwargs["Key"], text))


def _edex(frame_count: int) -> str:
    return json.dumps([{"frame_start": 0, "frame_end": frame_count}])


def _segments(items):
    return json.dumps(items) + "\n"


def test_backfill_segment_doubles_half_open_frames_and_preserves_fields():
    segment = {
        "id": "abc",
        "start_frame": 10,
        "end_frame": 20,
        "failure_category": "Mesh Penetration",
        "reason": "bad frame span",
    }

    updated, clipped = backfill.backfill_segment(segment, frame_count=100)

    assert updated == {
        "id": "abc",
        "start_frame": 20,
        "end_frame": 40,
        "failure_category": "Mesh Penetration",
        "reason": "bad frame span",
    }
    assert clipped is False


def test_backfill_segment_clips_one_frame_end_overflow():
    updated, clipped = backfill.backfill_segment(
        {"start_frame": 136, "end_frame": 251},
        frame_count=501,
    )

    assert updated["start_frame"] == 272
    assert updated["end_frame"] == 501
    assert clipped is True


def test_backfill_segment_rejects_larger_overflow():
    with pytest.raises(ValueError, match="invalid doubled segment"):
        backfill.backfill_segment(
            {"start_frame": 136, "end_frame": 252},
            frame_count=501,
        )


def test_process_empty_file_skips_without_writes():
    client = _FakeS3(
        {
            "data_export/seq/failure_segments.json": "[]\n",
            "data_export/seq/edex": _edex(100),
        }
    )

    manifest = backfill.run_backfill(
        client,
        "recordings",
        "data_export",
        dataset="dataset",
        export_url="swift://host/AUTH/recordings/data_export",
        apply=False,
    )

    assert manifest["summary"]["empty_files"] == 1
    assert manifest["summary"]["would_update_files"] == 0
    assert client.puts == []


def test_dry_run_reports_updates_without_writes():
    client = _FakeS3(
        {
            "data_export/seq/failure_segments.json": _segments(
                [{"id": "seg", "start_frame": 2, "end_frame": 5}]
            ),
            "data_export/seq/edex": _edex(20),
        }
    )

    manifest = backfill.run_backfill(
        client,
        "recordings",
        "data_export",
        dataset="dataset",
        export_url="swift://host/AUTH/recordings/data_export",
        apply=False,
    )

    assert manifest["summary"]["candidate_files"] == 1
    assert manifest["summary"]["would_update_files"] == 1
    assert manifest["files"][0]["new_segments"][0]["start_frame"] == 4
    assert manifest["files"][0]["new_segments"][0]["end_frame"] == 10
    assert client.puts == []


def test_apply_writes_backup_updated_file_and_manifest():
    client = _FakeS3(
        {
            "data_export/seq/failure_segments.json": _segments(
                [{"id": "seg", "start_frame": 2, "end_frame": 5}]
            ),
            "data_export/seq/edex": _edex(20),
        }
    )

    manifest = backfill.run_backfill(
        client,
        "recordings",
        "data_export",
        dataset="dataset",
        export_url="swift://host/AUTH/recordings/data_export",
        apply=True,
        timestamp="20260528_120000_123456",
        write_manifest=False,
    )

    assert manifest["summary"]["updated_files"] == 1
    assert json.loads(
        client.objects["data_export/seq/failure_segments.pre_subsampling_backfill.json"]
    ) == [{"id": "seg", "start_frame": 2, "end_frame": 5}]
    assert json.loads(client.objects["data_export/seq/failure_segments.json"]) == [
        {"id": "seg", "start_frame": 4, "end_frame": 10}
    ]


def test_existing_backup_prevents_double_application():
    client = _FakeS3(
        {
            "data_export/seq/failure_segments.json": _segments(
                [{"id": "seg", "start_frame": 4, "end_frame": 10}]
            ),
            "data_export/seq/failure_segments.pre_subsampling_backfill.json": _segments(
                [{"id": "seg", "start_frame": 2, "end_frame": 5}]
            ),
            "data_export/seq/edex": _edex(20),
        }
    )

    manifest = backfill.run_backfill(
        client,
        "recordings",
        "data_export",
        dataset="dataset",
        export_url="swift://host/AUTH/recordings/data_export",
        apply=True,
        write_manifest=False,
    )

    assert manifest["summary"]["skipped_already_backfilled"] == 1
    assert json.loads(client.objects["data_export/seq/failure_segments.json"]) == [
        {"id": "seg", "start_frame": 4, "end_frame": 10}
    ]
    assert client.puts == []


def test_rejected_file_is_not_written():
    client = _FakeS3(
        {
            "data_export/seq/failure_segments.json": _segments(
                [{"id": "seg", "start_frame": 9, "end_frame": 12}]
            ),
            "data_export/seq/edex": _edex(20),
        }
    )

    manifest = backfill.run_backfill(
        client,
        "recordings",
        "data_export",
        dataset="dataset",
        export_url="swift://host/AUTH/recordings/data_export",
        apply=True,
        write_manifest=False,
    )

    assert manifest["summary"]["rejected_files"] == 1
    assert "invalid doubled segment" in manifest["files"][0]["reason"]
    assert client.puts == []
