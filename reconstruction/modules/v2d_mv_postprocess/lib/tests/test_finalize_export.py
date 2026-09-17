import json
import sys
from pathlib import Path

import pytest


LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

import finalize_export as finalize
import interaction_trim


def _write_metric_segments(source: Path, value: list | None = None) -> Path:
    path = source / "check_accuracy" / "failure_segments.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([] if value is None else value) + "\n")
    return path


def _trim_manifest(frame_count: int = 2, start: int = 0) -> dict:
    manifest = {
        "schema": interaction_trim.TRIM_SCHEMA,
        "source_frame_count": frame_count,
        "export_source_start_frame": start,
        "export_source_end_frame": frame_count,
        "export_frame_count": frame_count - start,
        "contact_frame_source": start,
        "first_contact_frame_source": start,
        "last_contact_frame_source": frame_count - 1,
        "trimmed_prefix_frames": start,
        "trimmed_suffix_frames": 0,
        "trimmed_suffix_seconds": 0.0,
    }
    manifest["decision_sha256"] = interaction_trim._canonical_sha256(manifest)
    return manifest


def _stub_trim(
    monkeypatch,
    *,
    frame_count: int = 2,
    start: int = 0,
    observed_prepare: dict | None = None,
) -> None:
    manifest = _trim_manifest(frame_count, start)
    monkeypatch.setattr(
        finalize, "detect_interaction_trim", lambda *_args, **_kwargs: manifest,
    )
    def fake_prepare(source, _output, _manifest, **kwargs):
        if observed_prepare is not None:
            observed_prepare.update(kwargs)
        return Path(source)

    monkeypatch.setattr(finalize, "prepare_trimmed_source", fake_prepare)


def test_finalize_export_validates_and_commits_all_payloads(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _write_metric_segments(source)
    segments = tmp_path / "segments.json"
    segments.write_text("[]\n")

    def fake_export_sequence(**kwargs):
        output = Path(kwargs["output_dir"])
        (output / "images").mkdir(parents=True)
        (output / "images" / "camera.h5").write_bytes(b"metadata")

    sha256_calls = []
    real_sha256 = finalize._sha256

    def counting_sha256(path):
        sha256_calls.append(path)
        return real_sha256(path)

    monkeypatch.setattr(finalize, "export_sequence", fake_export_sequence)
    _stub_trim(monkeypatch)
    monkeypatch.setattr(finalize, "_sha256", counting_sha256)
    monkeypatch.setattr(
        finalize, "validate_export_payload",
        lambda _root, **_kwargs: {
            "schema": "v2d.mv_hoi.export_validation.v2", "frame_count": 2,
        },
    )
    output = tmp_path / "output"
    commit = finalize.finalize_export(
        source_dir=source, failure_segments_path=segments, output_dir=output,
        sequence_name="sequence", pipeline_version="1.6.2",
        authorization_type="QC", reconstruction_run_id=42,
    )

    assert commit["complete"] is True
    assert commit["export_validation"]["frame_count"] == 2
    committed_paths = {item["path"] for item in commit["files"]}
    assert committed_paths >= {
        "failure_segments.json", "images/camera.h5", "interaction_trim.json",
        "metrics/reconstruction/42/failure_segments/source_failure_segments.json",
        "metrics/reconstruction/42/failure_segments/export_failure_segments.json",
        "metrics/reconstruction/42/interaction_trim/interaction_trim.json",
        "metrics/reconstruction/42/manifest.json",
    }
    metrics_manifest = json.loads(
        (output / "metrics/reconstruction/42/manifest.json").read_text()
    )
    assert metrics_manifest["request_id"] == 42
    assert all(item["sha256"] for item in metrics_manifest["files"])
    assert len(sha256_calls) == commit["file_count"]
    assert finalize.verify_export_commit(output)["reconstruction_run_id"] == 42
    (output / "failure_segments.json").write_text("[x]")
    with pytest.raises(ValueError, match="SHA-256"):
        finalize.verify_export_commit(output)


def test_new_commit_reuses_ffv1_content_address_but_verification_rehashes(
    tmp_path,
):
    digest = "a" * 64
    sidecar = tmp_path / f"camera.ffv1.{digest}.mkv"
    sidecar.write_bytes(b"not-the-named-digest")

    trusted = finalize._payloads(
        tmp_path, trust_new_content_addresses=True,
    )
    independently_hashed = finalize._payloads(tmp_path)

    assert trusted[0]["sha256"] == digest
    assert independently_hashed[0]["sha256"] == finalize._sha256(sidecar)
    assert independently_hashed[0]["sha256"] != digest


def test_finalize_export_refuses_uncommitted_existing_destination(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "partial").write_text("partial")
    with pytest.raises(ValueError, match="rebuild-incomplete"):
        finalize.finalize_export(
            source_dir=tmp_path, failure_segments_path=tmp_path / "missing",
            output_dir=output, sequence_name="sequence", pipeline_version="1.6.2",
            authorization_type="QC", reconstruction_run_id=1,
        )


def test_finalize_export_preserves_mounted_output_root(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _write_metric_segments(source)
    segments = tmp_path / "segments.json"
    segments.write_text("[]\n")
    output = tmp_path / "output"
    output.mkdir()
    original_rmdir = Path.rmdir

    def reject_output_rmdir(path):
        if path == output:
            raise OSError(16, "Device or resource busy", str(path))
        return original_rmdir(path)

    def fake_export_sequence(**kwargs):
        Path(kwargs["output_dir"], "poses.npy").write_bytes(b"pose")

    monkeypatch.setattr(Path, "rmdir", reject_output_rmdir)
    monkeypatch.setattr(finalize, "export_sequence", fake_export_sequence)
    _stub_trim(monkeypatch)
    monkeypatch.setattr(
        finalize, "validate_export_payload",
        lambda _root, **_kwargs: {"schema": "v2d.mv_hoi.export_validation.v2"},
    )

    commit = finalize.finalize_export(
        source_dir=source, failure_segments_path=segments, output_dir=output,
        sequence_name="sequence", pipeline_version="1.6.2",
        authorization_type="QC", reconstruction_run_id=1,
    )
    assert commit["complete"] is True
    assert output.is_dir()


def test_finalize_export_merges_request_isolated_lineage(tmp_path, monkeypatch):
    reconstruction = tmp_path / "reconstruction"
    (reconstruction / "foundation_pose").mkdir(parents=True)
    _write_metric_segments(reconstruction)
    preprocess = tmp_path / "preprocess"
    (preprocess / "mv_preprocess").mkdir(parents=True)
    (preprocess / "face_detector").mkdir()
    segments = tmp_path / "segments.json"
    segments.write_text("[]\n")
    observed = {}

    def fake_export_sequence(**kwargs):
        source = Path(kwargs["source_dir"])
        observed["foundation_pose"] = (source / "foundation_pose").is_dir()
        observed["mv_preprocess"] = (source / "mv_preprocess").is_dir()
        observed["face_detector"] = (source / "face_detector").is_dir()
        Path(kwargs["output_dir"], "poses.npy").write_bytes(b"pose")

    monkeypatch.setattr(finalize, "export_sequence", fake_export_sequence)
    _stub_trim(monkeypatch)
    monkeypatch.setattr(
        finalize, "validate_export_payload",
        lambda _root, **_kwargs: {"schema": "v2d.mv_hoi.export_validation.v2"},
    )
    commit = finalize.finalize_export(
        source_dir=reconstruction,
        preprocess_source_dir=preprocess,
        failure_segments_path=segments,
        output_dir=tmp_path / "output",
        sequence_name="sequence",
        pipeline_version="1.6.19",
        authorization_type="QC",
        reconstruction_run_id=19,
    )

    assert observed == {
        "foundation_pose": True,
        "mv_preprocess": True,
        "face_detector": True,
    }
    assert commit["complete"] is True


def test_finalize_export_clips_failure_segments_to_trimmed_timeline(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    _write_metric_segments(source)
    segments = tmp_path / "segments.json"
    segments.write_text(json.dumps([
        {"id": "prefix", "start_frame": 0, "end_frame": 10},
        {"id": "crossing", "start_frame": 15, "end_frame": 30},
        {"id": "retained", "start_frame": 60, "end_frame": 80},
    ]))

    observed_export = {}

    def fake_export_sequence(**kwargs):
        observed_export.update(kwargs)
        Path(kwargs["output_dir"], "poses.npy").write_bytes(b"pose")

    monkeypatch.setattr(finalize, "export_sequence", fake_export_sequence)
    observed_prepare = {}
    _stub_trim(
        monkeypatch,
        frame_count=100,
        start=20,
        observed_prepare=observed_prepare,
    )
    monkeypatch.setattr(
        finalize,
        "validate_export_payload",
        lambda _root, **_kwargs: {
            "schema": "v2d.mv_hoi.export_validation.v2",
            "frame_count": 80,
        },
    )

    output = tmp_path / "output"
    commit = finalize.finalize_export(
        source_dir=source,
        failure_segments_path=segments,
        output_dir=output,
        sequence_name="sequence",
        pipeline_version="1.6.22",
        authorization_type="QC",
        reconstruction_run_id=22,
    )

    assert json.loads((output / "failure_segments.json").read_text()) == [
        {
            "id": "crossing",
            "source_start_frame": 15,
            "source_end_frame": 30,
            "start_frame": 0,
            "end_frame": 10,
        },
        {
            "id": "retained",
            "source_start_frame": 60,
            "source_end_frame": 80,
            "start_frame": 40,
            "end_frame": 60,
        },
    ]
    assert commit["trimmed_qc"]["failure_coverage_frames"] == 30
    assert commit["trimmed_qc"]["failure_coverage"] == pytest.approx(0.375)
    assert observed_prepare["defer_frame_archives"] is True
    assert observed_prepare["max_video_workers"] >= 1
    assert observed_prepare["video_codec_threads"] >= 1
    assert observed_export["source_start_frame"] == 20
    assert observed_export["source_end_frame"] == 100
    assert observed_export["max_camera_workers"] >= 1


def test_finalize_export_applies_annotation_limit_only_after_trim(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    _write_metric_segments(source)
    segments = tmp_path / "segments.json"
    segments.write_text(json.dumps([
        {"id": str(index), "start_frame": index, "end_frame": index + 1}
        for index in range(7)
    ]))

    monkeypatch.setattr(
        finalize,
        "export_sequence",
        lambda **kwargs: Path(
            kwargs["output_dir"], "poses.npy",
        ).write_bytes(b"pose"),
    )
    _stub_trim(monkeypatch, frame_count=100, start=50)
    monkeypatch.setattr(
        finalize,
        "validate_export_payload",
        lambda _root, **_kwargs: {
            "schema": "v2d.mv_hoi.export_validation.v2",
            "frame_count": 50,
        },
    )

    commit = finalize.finalize_export(
        source_dir=source,
        failure_segments_path=segments,
        output_dir=tmp_path / "output",
        sequence_name="sequence",
        pipeline_version="1.6.22",
        authorization_type="QC",
        reconstruction_run_id=22,
        max_failure_annotations=6,
    )

    assert commit["trimmed_qc"]["failure_annotation_count"] == 0
    assert json.loads(
        (tmp_path / "output" / "failure_segments.json").read_text()
    ) == []


def test_finalize_export_rejects_excess_annotations_in_retained_interval(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    _write_metric_segments(source)
    segments = tmp_path / "segments.json"
    segments.write_text(json.dumps([
        {"id": str(index), "start_frame": 60 + index, "end_frame": 61 + index}
        for index in range(7)
    ]))

    monkeypatch.setattr(
        finalize,
        "export_sequence",
        lambda **kwargs: Path(
            kwargs["output_dir"], "poses.npy",
        ).write_bytes(b"pose"),
    )
    _stub_trim(monkeypatch, frame_count=100, start=50)

    output = tmp_path / "output"
    with pytest.raises(ValueError, match="human_qc_failure_count"):
        finalize.finalize_export(
            source_dir=source,
            failure_segments_path=segments,
            output_dir=output,
            sequence_name="sequence",
            pipeline_version="1.6.22",
            authorization_type="QC",
            reconstruction_run_id=22,
            max_failure_annotations=6,
        )
    assert not (output / "commit.json").exists()
    assert {path.name for path in output.iterdir()} == {"export_rejection.json"}
    rejection = json.loads((output / "export_rejection.json").read_text())
    assert rejection["schema"] == "v2d.mv_hoi.export_rejection.v1"
    assert rejection["gates"][0]["observed"] == 7
