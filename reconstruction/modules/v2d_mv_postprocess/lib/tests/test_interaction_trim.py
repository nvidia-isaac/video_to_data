import json
from fractions import Fraction
import sys
import threading
from pathlib import Path

import av
import h5py
import numpy as np
import pytest
import torch
import yaml


LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

import interaction_trim as trim


def _manifest(source_count: int, start: int, end: int | None = None) -> dict:
    end = source_count if end is None else end
    value = {
        "schema": trim.TRIM_SCHEMA,
        "reason": "stable_contact",
        "source_frame_count": source_count,
        "export_source_start_frame": start,
        "export_source_end_frame": end,
        "export_frame_count": end - start,
        "contact_frame_source": start + 1,
        "first_contact_frame_source": start + 1,
        "last_contact_frame_source": end - 1,
        "trimmed_prefix_frames": start,
        "trimmed_prefix_seconds": start / 30.0,
        "trimmed_suffix_frames": source_count - end,
        "trimmed_suffix_seconds": (source_count - end) / 30.0,
    }
    value["decision_sha256"] = trim._canonical_sha256(value)
    return value


def test_stable_contact_uses_first_under_threshold_frame_in_earliest_window():
    distances = [0.5, 0.05, 0.4, 0.04, 0.03, 0.02, 0.01, 0.5]

    contact = trim.select_stable_contact_frame(
        distances,
        distance_threshold_m=0.10,
        window_frames=7,
        required_under_threshold_frames=5,
    )

    assert contact == 1


def test_isolated_distance_spikes_do_not_trigger_contact():
    distances = [0.5, 0.05, 0.5, 0.04, 0.5, 0.5, 0.5, 0.03, 0.5]

    assert trim.select_stable_contact_frame(distances) is None


def test_invalid_distances_do_not_count_toward_stable_contact():
    distances = [np.nan, 0.01, np.inf, 0.01, np.nan, 0.01, 0.01]

    assert trim.select_stable_contact_frame(distances) is None


def test_stable_contact_interval_uses_latest_stable_window_not_tail_noise():
    distances = np.full(30, 0.5)
    distances[5:10] = 0.10
    distances[20:25] = 0.10
    distances[29] = 0.01

    assert trim.select_stable_contact_frames(distances) == (5, 24)


def test_stable_contact_interval_returns_no_partial_decision():
    distances = [np.nan, 0.01, 0.5, 0.01, np.inf, 0.01, 0.5]

    assert trim.select_stable_contact_frames(distances) == (None, None)


def test_contact_preserves_exactly_three_seconds_at_30_fps():
    assert trim.padded_export_start_frame(
        240,
        source_frame_count=600,
        fps=30,
        pre_contact_padding_seconds=3.0,
    ) == 150
    assert trim.padded_export_start_frame(
        60,
        source_frame_count=600,
        fps=30,
        pre_contact_padding_seconds=3.0,
    ) == 0
    assert trim.padded_export_end_frame(
        240,
        source_frame_count=600,
        fps=30,
        post_contact_padding_seconds=3.0,
    ) == 331
    assert trim.padded_export_end_frame(
        570,
        source_frame_count=600,
        fps=30,
        post_contact_padding_seconds=3.0,
    ) == 600


def test_trim_manifest_v1_read_compatibility_and_v2_suffix_validation():
    legacy = {
        "schema": trim.TRIM_SCHEMA_V1,
        "source_frame_count": 100,
        "export_source_start_frame": 20,
        "export_source_end_frame": 100,
        "export_frame_count": 80,
        "contact_frame_source": 21,
    }
    legacy["decision_sha256"] = trim._canonical_sha256(legacy)
    assert trim.validate_trim_manifest(legacy) == (20, 100, 80)

    symmetric = _manifest(100, 20, 80)
    assert trim.validate_trim_manifest(symmetric) == (20, 80, 60)
    symmetric["trimmed_suffix_frames"] = 19
    symmetric["decision_sha256"] = trim._canonical_sha256(
        {key: value for key, value in symmetric.items() if key != "decision_sha256"}
    )
    with pytest.raises(ValueError, match="suffix count"):
        trim.validate_trim_manifest(symmetric)


def test_distance_computation_transforms_human_vertices_into_object_frame(
    monkeypatch,
):
    class FakeSurface:
        def __init__(self, _vertices, _faces):
            pass

        def minimum_distance(self, points):
            return float(np.linalg.norm(points, axis=1).min())

    monkeypatch.setattr(trim, "_ObjectSurfaceDistance", FakeSurface)
    humans = np.asarray([
        [[1.05, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    ])
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[:, 0, 3] = 1.0

    distances = trim.compute_human_object_distances(
        humans,
        poses,
        np.zeros((3, 3)),
        np.asarray([[0, 1, 2]]),
    )

    assert distances == pytest.approx([0.05, 1.0])


def test_trim_failure_segments_clips_shifts_and_drops_prefix():
    segments = [
        {"id": "before", "start_frame": 0, "end_frame": 10},
        {"id": "crossing", "start_frame": 15, "end_frame": 25},
        {"id": "after", "start_frame": 30, "end_frame": 50},
    ]

    result = trim.trim_failure_segments(
        segments, start_frame=20, end_frame=40,
    )

    assert result == [
        {
            "id": "crossing",
            "source_start_frame": 15,
            "source_end_frame": 25,
            "start_frame": 0,
            "end_frame": 5,
        },
        {
            "id": "after",
            "source_start_frame": 30,
            "source_end_frame": 50,
            "start_frame": 10,
            "end_frame": 20,
        },
    ]
    assert trim.merged_interval_coverage(result) == 15


def test_trim_failure_segments_obeys_all_half_open_boundary_cases():
    segments = [
        {"id": "entirely_before", "start_frame": 0, "end_frame": 19},
        {"id": "touches_start_outside", "start_frame": 5, "end_frame": 20},
        {"id": "crosses_start", "start_frame": 15, "end_frame": 25},
        {"id": "inside", "start_frame": 25, "end_frame": 35},
        {"id": "spans_both", "start_frame": 10, "end_frame": 50},
        {"id": "crosses_end", "start_frame": 35, "end_frame": 45},
        {"id": "touches_end_outside", "start_frame": 40, "end_frame": 45},
        {"id": "entirely_after", "start_frame": 41, "end_frame": 50},
    ]

    result = trim.trim_failure_segments(segments, start_frame=20, end_frame=40)

    assert [(item["id"], item["start_frame"], item["end_frame"]) for item in result] == [
        ("crosses_start", 0, 5),
        ("inside", 5, 15),
        ("spans_both", 0, 20),
        ("crosses_end", 15, 20),
    ]
    assert max(item["end_frame"] for item in result) == 20


def test_torch_and_soma_temporal_artifacts_are_sliced_schema_aware(tmp_path):
    params_source = tmp_path / "mhr_params.pt"
    params_output = tmp_path / "trimmed_params.pt"
    params = {
        key: torch.arange(6 * 2).reshape(6, 2)
        for key in trim.MHR_PARAMETER_TEMPORAL_KEYS
    }
    params["static_note"] = "preserved"
    torch.save(params, params_source)

    trim._slice_torch_mapping(
        params_source,
        params_output,
        start=2,
        end=5,
        source_count=6,
        temporal_keys=trim.MHR_PARAMETER_TEMPORAL_KEYS,
    )

    trimmed_params = torch.load(
        params_output, weights_only=False, map_location="cpu",
    )
    assert trimmed_params["static_note"] == "preserved"
    for key in trim.MHR_PARAMETER_TEMPORAL_KEYS:
        assert torch.equal(trimmed_params[key], params[key][2:5])

    soma_source = tmp_path / "soma.npz"
    soma_output = tmp_path / "trimmed_soma.npz"
    np.savez(
        soma_source,
        poses=np.arange(6 * 3).reshape(6, 3),
        transl=np.arange(6 * 3).reshape(6, 3),
        identity_coeffs=np.arange(6 * 2).reshape(6, 2),
        scale_params=np.arange(6 * 4).reshape(6, 4),
        bone_length_flexibles=np.arange(6 * 6).reshape(6, 6),
        joint_names=np.asarray(["a", "b"]),
    )

    trim._slice_soma(
        soma_source,
        soma_output,
        start=2,
        end=5,
        source_count=6,
    )

    with np.load(soma_output, allow_pickle=False) as soma:
        assert soma["poses"].shape[0] == 3
        assert np.array_equal(soma["poses"], np.arange(18).reshape(6, 3)[2:5])
        assert np.array_equal(
            soma["bone_length_flexibles"],
            np.arange(36).reshape(6, 6)[2:5],
        )
        assert np.array_equal(soma["joint_names"], ["a", "b"])


def test_metadata_and_edex_are_rebased_and_record_provenance(tmp_path):
    manifest = _manifest(100, 25, 80)
    metadata_source = tmp_path / "metadata.yaml"
    metadata_output = tmp_path / "trimmed_metadata.yaml"
    metadata_source.write_text("frame_count: 100\nobject:\n  id: book\n")

    trim._trim_metadata(
        metadata_source, metadata_output, manifest=manifest,
    )

    metadata = yaml.safe_load(metadata_output.read_text())
    assert metadata["frame_count"] == 55
    assert metadata["interaction_trim"]["export_source_start_frame"] == 25
    assert metadata["interaction_trim"]["export_source_end_frame"] == 80
    assert metadata["interaction_trim"]["trimmed_suffix_frames"] == 20
    assert (
        metadata["interaction_trim"]["decision_sha256"]
        == manifest["decision_sha256"]
    )

    edex_source = tmp_path / "edex"
    edex_output = tmp_path / "trimmed_edex"
    edex_source.write_text(json.dumps([{
        "frame_start": 0, "frame_end": 100, "cameras": [],
    }]))
    trim._trim_edex(edex_source, edex_output, output_count=55)
    header = json.loads(edex_output.read_text())[0]
    assert header["frame_start"] == 0
    assert header["frame_end"] == 55


def test_unknown_temporal_soma_array_is_rejected(tmp_path):
    source = tmp_path / "soma.npz"
    np.savez(
        source,
        poses=np.zeros((6, 2)),
        transl=np.zeros((6, 3)),
        identity_coeffs=np.zeros((6, 4)),
        scale_params=np.zeros((6, 5)),
        unexpected=np.zeros((6, 1)),
    )

    with pytest.raises(ValueError, match="ambiguously temporal"):
        trim._slice_soma(
            source,
            tmp_path / "out.npz",
            start=1,
            end=5,
            source_count=6,
        )


def test_video_trim_is_frame_accurate_and_rebased(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "trimmed.mp4"
    with av.open(str(source), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=Fraction(30, 1))
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            frame = av.VideoFrame.from_ndarray(
                np.full((24, 32, 3), index * 20, dtype=np.uint8),
                format="rgb24",
            )
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    result = trim.trim_video(
        source,
        output,
        start_frame=2,
        end_frame=7,
        expected_source_frames=8,
    )

    assert result["source_frames"] == 8
    assert result["frames"] == 5
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 5
    assert [frame.pts for frame in frames] == [0, 512, 1024, 1536, 2048]


def test_source_timeline_uses_archive_count_without_decoding_video(
    tmp_path, monkeypatch,
):
    images = tmp_path / "mv_preprocess" / "images"
    videos = tmp_path / "mv_preprocess" / "videos"
    images.mkdir(parents=True)
    videos.mkdir(parents=True)
    with h5py.File(images / "camera.h5", "w") as h5_file:
        h5_file.create_dataset(
            "frames", data=np.zeros((6, 4, 5, 3), dtype=np.uint8),
        )
        h5_file.attrs["fps"] = 30
    (videos / "camera.mp4").touch()
    monkeypatch.setattr(
        trim, "_video_stream_info", lambda _path: (6, Fraction(30, 1)),
    )

    def reject_decode(_path):
        raise AssertionError("video frames should not be decoded")

    monkeypatch.setattr(trim, "_video_info", reject_decode)

    assert trim.resolve_source_timeline(tmp_path) == (6, Fraction(30, 1))


def test_video_tree_trimming_uses_bounded_parallelism(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    for index in range(4):
        (source / f"{index}.mp4").touch()
    (source / "note.txt").write_text("preserve")
    entered = 0
    peak = 0
    lock = threading.Lock()
    workers_ready = threading.Event()
    observed_threads = []

    def fake_trim(_source, output, **kwargs):
        nonlocal entered, peak
        observed_threads.append(kwargs["codec_threads"])
        with lock:
            entered += 1
            peak = max(peak, entered)
            if entered == 2:
                workers_ready.set()
        assert workers_ready.wait(timeout=2)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        with lock:
            entered -= 1
        return {}

    monkeypatch.setattr(trim, "trim_video", fake_trim)
    trim._slice_video_tree(
        source,
        destination,
        start=1,
        end=3,
        source_count=4,
        max_workers=2,
        codec_threads=3,
    )

    assert peak == 2
    assert observed_threads == [3, 3, 3, 3]
    assert (destination / "note.txt").is_symlink()


def test_deferred_trim_leaves_large_frame_archives_for_export(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        "mv_preprocess",
        "face_detector",
        "foundation_stereo",
        "sam2_object_masks",
        "sam2_human_masks",
        "foundation_pose",
        "sam3d_body",
        "export_soma",
        "render_hoi_overlay",
    ):
        (source / name).mkdir()
    calls = {}
    monkeypatch.setattr(
        trim, "_copy_directory_view", lambda _source, _output: None,
    )

    def record_replacements(_source, _output, name, replacements):
        calls[name] = set(replacements)

    monkeypatch.setattr(trim, "_replace_task_directory", record_replacements)
    output = trim.prepare_trimmed_source(
        source,
        tmp_path / "trimmed",
        _manifest(10, 2),
        defer_frame_archives=True,
        max_video_workers=4,
        video_codec_threads=2,
    )

    assert output == tmp_path / "trimmed"
    assert calls["mv_preprocess"] == {"hoi_metadata.yaml", "edex", "videos"}
    assert calls["face_detector"] == {"videos"}
    assert "foundation_stereo" not in calls
    assert "sam2_object_masks" not in calls
    assert "sam2_human_masks" not in calls


def test_suffix_only_trim_still_slices_every_temporal_artifact(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        "mv_preprocess",
        "face_detector",
        "foundation_stereo",
        "sam2_object_masks",
        "sam2_human_masks",
        "foundation_pose",
        "sam3d_body",
        "export_soma",
        "render_hoi_overlay",
    ):
        (source / name).mkdir()
    calls = {}
    archive_calls = []
    monkeypatch.setattr(
        trim, "_copy_directory_view", lambda _source, _output: None,
    )

    def record_replacements(_source, _output, name, replacements):
        calls[name] = set(replacements)

    monkeypatch.setattr(trim, "_replace_task_directory", record_replacements)
    monkeypatch.setattr(
        trim,
        "_slice_archive_tree",
        lambda source, destination, **kwargs: archive_calls.append(
            (source.name, destination.name, kwargs["start"], kwargs["end"])
        ),
    )

    output = trim.prepare_trimmed_source(
        source,
        tmp_path / "trimmed",
        _manifest(10, 0, 8),
        defer_frame_archives=False,
    )

    assert output == tmp_path / "trimmed"
    assert calls["mv_preprocess"] == {
        "hoi_metadata.yaml", "edex", "images", "videos",
    }
    assert calls["face_detector"] == {"images", "videos"}
    assert calls["foundation_pose"] == {"poses.npy", "pose_valid_mask.npy"}
    assert calls["sam3d_body"] == {"mhr_params_mv.pt", "mhr_mesh_mv.pt"}
    assert calls["export_soma"] == {"soma_params.npz"}
    assert calls["render_hoi_overlay"] == {"tiled_hoi_overlay.mp4"}
    assert {(name, start, end) for name, _, start, end in archive_calls} == {
        ("foundation_stereo", 0, 8),
        ("sam2_object_masks", 0, 8),
        ("sam2_human_masks", 0, 8),
    }
