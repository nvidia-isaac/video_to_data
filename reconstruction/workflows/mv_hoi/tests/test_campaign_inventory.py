from io import BytesIO
import hashlib
from pathlib import Path
import sys

from botocore.exceptions import ClientError


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration.campaign_inventory import (
    build_inventories,
    _has_legacy_pass_evidence,
    _is_selected_data_output_path,
    _list,
    _list_selected_data_outputs,
    select_canary,
    _write_sequence_records,
)
from orchestration.config_utils import (
    EXPORT_CONFIG_PIPELINE,
    RECON_PIPELINE,
    REVALIDATION_PIPELINE,
)


class _Paginator:
    def __init__(self, payloads):
        self.payloads = payloads

    def paginate(self, **kwargs):
        prefix = kwargs.get("Prefix", "")
        return [{"Contents": [
            {"Key": key, "Size": len(value), "ETag": f'"etag-{index}"'}
            for index, (key, value) in enumerate(self.payloads.items())
            if key.startswith(prefix)
        ]}]


class _Client:
    def __init__(self, payloads):
        self.payloads = payloads

    def get_paginator(self, _name):
        return _Paginator(self.payloads)

    def get_object(self, *, Bucket, Key):
        del Bucket
        return {"Body": BytesIO(self.payloads[Key])}

    def head_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.payloads:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey"},
                 "ResponseMetadata": {"HTTPStatusCode": 404}},
                "HeadObject",
            )
        keys = list(self.payloads)
        return {
            "ContentLength": len(self.payloads[Key]),
            "ETag": f'"etag-{keys.index(Key)}"',
        }


def test_legacy_pass_evidence_accepts_qc_pass_or_completed_export():
    assert _has_legacy_pass_evidence({
        "reconstruction_status": "SUCCEEDED",
        "latest_qc_decision": "PASS",
        "export_status": None,
    })


def test_cleanup_manifest_is_part_of_future_campaign_inventory():
    assert _is_selected_data_output_path("intermediate_cleanup.json")
    assert _has_legacy_pass_evidence({
        "reconstruction_status": "SUCCEEDED",
        "reconstruction_run_id": 7,
        "latest_qc_decision": None,
        "export_status": "SUCCEEDED",
        "export_reconstruction_run_id": 7,
    })


def test_old_export_does_not_make_newer_qc_pending_reconstruction_legacy_pass():
    assert not _has_legacy_pass_evidence({
        "reconstruction_status": "SUCCEEDED",
        "reconstruction_run_id": 8,
        "latest_qc_decision": None,
        "export_status": "SUCCEEDED",
        "export_reconstruction_run_id": 7,
    })


def test_qc_pending_failed_and_unfinished_export_are_not_legacy_pass():
    for identity in (
        {
            "reconstruction_status": "SUCCEEDED",
            "latest_qc_decision": None,
            "export_status": None,
        },
        {
            "reconstruction_status": "SUCCEEDED",
            "latest_qc_decision": "FAIL",
            "export_status": None,
        },
        {
            "reconstruction_status": "SUCCEEDED",
            "reconstruction_run_id": 7,
            "latest_qc_decision": None,
            "export_status": "RUNNING",
            "export_reconstruction_run_id": 7,
        },
        {
            "reconstruction_status": "FAILED",
            "reconstruction_run_id": 7,
            "latest_qc_decision": "PASS",
            "export_status": "SUCCEEDED",
            "export_reconstruction_run_id": 7,
        },
    ):
        assert not _has_legacy_pass_evidence(identity)


def test_inventory_hashes_pose_mesh_metadata_and_bbox_but_not_large_frame_payloads():
    payloads = {
        "root/sequence/foundation_pose/poses.npy": b"poses",
        "root/sequence/mv_preprocess/object_mesh/output_aligned.glb": b"mesh",
        "root/sequence/mv_preprocess/hoi_metadata.yaml": b"metadata",
        "root/sequence/mv_preprocess/object_bbox_source.txt": (
            b"manual_labeled_bboxes\n"
        ),
        "root/sequence/mv_preprocess/labeled_bboxes/left.json": b"bbox",
        "root/sequence/mv_preprocess/images/left.h5": b"large-rgb",
    }

    records = {
        item["key"]: item for item in _list(
            _Client(payloads), "bucket", "root", content_hashes=True,
        )
    }

    for key in set(payloads) - {"root/sequence/mv_preprocess/images/left.h5"}:
        assert records[key]["sha256"] == hashlib.sha256(payloads[key]).hexdigest()
    assert "sha256" not in records["root/sequence/mv_preprocess/images/left.h5"]
    assert (
        records["root/sequence/mv_preprocess/object_bbox_source.txt"]["content"]
        == "manual_labeled_bboxes"
    )


def test_selected_data_output_paths_are_bounded_to_consumed_artifacts():
    assert _is_selected_data_output_path(
        "mv_preprocess/images/front_stereo_camera_right.h5"
    )
    assert _is_selected_data_output_path(
        "mv_preprocess/images/front_stereo_camera_right/000001.png"
    )
    assert _is_selected_data_output_path(
        "foundation_stereo/front_stereo_camera_left/depth.h5"
    )
    assert _is_selected_data_output_path(
        "foundation_stereo/front_stereo_camera_left/depth/000001.png"
    )
    assert _is_selected_data_output_path(
        "sam2_object_masks/front_stereo_camera_left/0.h5"
    )
    assert _is_selected_data_output_path(
        "sam2_object_masks/front_stereo_camera_left/0/000001.png"
    )
    assert _is_selected_data_output_path(
        "sam2_human_masks/front_stereo_camera_left/0/000001.png"
    )
    assert _is_selected_data_output_path(
        "mv_preprocess/labeled_bboxes/front_stereo_camera_left_bbox.png"
    )
    assert _is_selected_data_output_path(
        "mv_preprocess/object_bbox_source.txt"
    )
    assert not _is_selected_data_output_path(
        "foundation_stereo/front_stereo_camera_left/debug/frame_000001.png"
    )
    assert not _is_selected_data_output_path(
        "render_hoi_overlay/front_stereo_camera_left/frame_000001.png"
    )


def test_list_discards_unselected_objects_before_manifest_materialization():
    payloads = {
        "root/sequence/foundation_pose/poses.npy": b"poses",
        "root/sequence/debug/frame.png": b"debug",
    }
    records = _list(
        _Client(payloads), "bucket", "root",
        include_key=lambda key: "/debug/" not in key,
    )
    assert [item["key"] for item in records] == [
        "root/sequence/foundation_pose/poses.npy"
    ]


def test_sequence_record_manifest_contains_only_its_frozen_entry(tmp_path):
    inventory = {
        "schema": "schema", "dataset": "dataset", "kind": "legacy_revalidation",
        "sequence_count": 2,
        "sequences": [{"sequence": "one", "value": 1}, {"sequence": "two", "value": 2}],
    }
    directory = tmp_path / "records"
    _write_sequence_records(directory, inventory)
    one = __import__("json").loads((directory / "one.json").read_text())
    assert one["sequence_count"] == 1
    assert one["sequences"] == [{"sequence": "one", "value": 1}]


def test_selected_output_listing_uses_exact_pass_inputs_and_only_backlog_labels():
    payloads = {
        "root/pass/foundation_pose/poses.npy": b"pass-pose",
        "root/pass/mv_preprocess/labeled_bboxes/left.json": b"pass-label",
        "root/pass/mv_preprocess/object_bbox_source.txt": b"manual_labeled_bboxes",
        "root/pass/mv_preprocess/images/back_stereo_camera_left.h5": b"metadata",
        "root/pass/mv_preprocess/images/back_stereo_camera_left.ffv1.abc.mkv": b"frames",
        "root/pass/debug/frame.png": b"debug",
        "root/backlog/foundation_pose/poses.npy": b"old-pose",
        "root/backlog/mv_preprocess/labeled_bboxes/left.json": b"backlog-label",
        "root/backlog/mv_preprocess/object_bbox_source.txt": b"manual_labeled_bboxes",
    }
    records = _list_selected_data_outputs(
        _Client(payloads), "bucket", "root", ["pass", "backlog"],
        full_sequences=["pass"], workers=2,
    )
    assert {item["key"] for item in records} == {
        "root/pass/foundation_pose/poses.npy",
        "root/pass/mv_preprocess/labeled_bboxes/left.json",
        "root/pass/mv_preprocess/object_bbox_source.txt",
        "root/pass/mv_preprocess/images/back_stereo_camera_left.h5",
        "root/pass/mv_preprocess/images/back_stereo_camera_left.ffv1.abc.mkv",
        "root/backlog/mv_preprocess/labeled_bboxes/left.json",
        "root/backlog/mv_preprocess/object_bbox_source.txt",
    }


def test_selected_output_listing_freezes_legacy_frame_directories():
    payloads = {
        "root/pass/mv_preprocess/images/front_stereo_camera_left/000001.png": b"rgb",
        "root/pass/foundation_stereo/front_stereo_camera_left/depth/000001.png": b"depth",
        "root/pass/sam2_object_masks/front_stereo_camera_left/0/000001.png": b"object",
        "root/pass/sam2_human_masks/front_stereo_camera_left/0/000001.png": b"human",
    }

    records = _list_selected_data_outputs(
        _Client(payloads), "bucket", "root", ["pass"],
        full_sequences=["pass"], workers=1,
    )

    assert {item["key"] for item in records} == set(payloads)


def test_inventory_reads_legacy_revalidation_input_not_new_reconstruction_output(
    monkeypatch,
):
    observed = {}

    monkeypatch.setattr(
        "orchestration.campaign_inventory._db_sequences",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "orchestration.campaign_inventory._list",
        lambda *_args, **_kwargs: [],
    )

    def list_outputs(_client, _bucket, root, _names, **_kwargs):
        observed["root"] = root
        return []

    monkeypatch.setattr(
        "orchestration.campaign_inventory._list_selected_data_outputs",
        list_outputs,
    )
    monkeypatch.setattr(
        "orchestration.campaign_inventory._load_accuracy_statuses",
        lambda *_args, **_kwargs: {},
    )
    dataset_cfg = {
        "swift_base": "swift://host/account/bucket/root",
        "pipelines": {
            RECON_PIPELINE: {
                "input_path": "data",
                "output_path": "data_output_2",
            },
            REVALIDATION_PIPELINE: {
                "input_path": "data_output",
            },
            EXPORT_CONFIG_PIPELINE: {
                "legacy_output_path": "data_export",
            },
        },
    }

    master, _, _ = build_inventories(
        dataset="dataset",
        dataset_cfg=dataset_cfg,
        db_path="unused",
        cutover_at="2026-07-28T00:00:00Z",
        client=object(),
        bucket="bucket",
        base_prefix="root",
    )

    assert observed["root"] == "root/data_output"
    assert (
        master["source_prefixes"]["data_output"]
        == "swift://host/account/bucket/root/data_output"
    )


def test_canary_prioritizes_object_diversity_deterministically():
    records = [
        {
            "sequence": f"sequence-{object_index}-{variant}",
            "object_id": f"object-{object_index}",
            "source_size": object_index * 100 + variant,
            "has_symmetry": bool(variant % 2),
            "legacy_metric_status": "PASS" if variant % 2 else "unknown",
        }
        for object_index in range(25)
        for variant in range(3)
    ]
    selected = select_canary(records, count=20)
    selected_objects = {
        record["object_id"] for record in records if record["sequence"] in selected
    }
    assert len(selected) == 20
    assert len(selected_objects) == 20
    assert selected == select_canary(list(reversed(records)), count=20)
