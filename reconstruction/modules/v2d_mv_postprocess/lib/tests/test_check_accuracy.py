import json
import statistics
import sys
from pathlib import Path

import pytest


LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

import check_accuracy as ca


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _silhouette(values_by_camera: dict[str, list[float | None]]) -> dict:
    return {
        "frame_metrics": {
            camera: [
                (
                    {"frame_idx": index, "skipped": True, "reason": "missing"}
                    if value is None
                    else {
                        "frame_idx": index,
                        "skipped": False,
                        "filtered_sam2_bbox_in_padded_render_bbox_ratio": value,
                    }
                )
                for index, value in enumerate(values)
            ]
            for camera, values in values_by_camera.items()
        }
    }


def _chamfer(values_by_camera: dict[str, list[float | None]]) -> dict:
    per_camera = {}
    pooled = []
    for camera, values in values_by_camera.items():
        rows = [
            {"frame_index": index, "distance_mm": value}
            for index, value in enumerate(values)
            if value is not None
        ]
        numeric = [float(row["distance_mm"]) for row in rows]
        pooled.extend(numeric)
        per_camera[camera] = {
            "per_frame": rows,
            "per_frame_mm": numeric,
            "mean_mm": sum(numeric) / len(numeric) if numeric else 0.0,
            "median_mm": statistics.median(numeric) if numeric else 0.0,
        }
    return {
        "per_camera": per_camera,
        "combined": {
            "mean_mm": statistics.fmean(pooled) if pooled else 0.0,
            "median_mm": statistics.median(pooled) if pooled else 0.0,
        },
    }


def _inputs(
    tmp_path: Path,
    *,
    human_silhouette: dict[str, list[float | None]],
    object_silhouette: dict[str, list[float | None]] | None = None,
    human_chamfer: dict[str, list[float | None]] | None = None,
    object_chamfer: dict[str, list[float | None]] | None = None,
) -> dict:
    object_silhouette = object_silhouette or {
        camera: [1.0] * len(values) for camera, values in human_silhouette.items()
    }
    human_chamfer = human_chamfer or {
        camera: [0.0] * len(values) for camera, values in human_silhouette.items()
    }
    object_chamfer = object_chamfer or {
        camera: [0.0] * len(values) for camera, values in human_silhouette.items()
    }
    roots = {
        "silhouette_human_dir": tmp_path / "silhouette_human",
        "silhouette_object_dir": tmp_path / "silhouette_object",
        "chamfer_human_dir": tmp_path / "chamfer_human",
        "chamfer_object_dir": tmp_path / "chamfer_object",
        "check_object_mask_dir": tmp_path / "mask",
    }
    _write(
        roots["silhouette_human_dir"] / "silhouette_mask_metrics.json",
        _silhouette(human_silhouette),
    )
    _write(
        roots["silhouette_object_dir"] / "silhouette_mask_metrics.json",
        _silhouette(object_silhouette),
    )
    _write(
        roots["chamfer_human_dir"] / "chamfer_metrics.json",
        _chamfer(human_chamfer),
    )
    _write(
        roots["chamfer_object_dir"] / "chamfer_metrics.json",
        _chamfer(object_chamfer),
    )
    # A failing manual-label diagnostic must not gate the sequence.
    _write(
        roots["check_object_mask_dir"] / "check_object_mask.json",
        {"status": "FAIL", "avg_containment": 0.1, "threshold": 0.8},
    )
    return {
        key: str(value) for key, value in roots.items()
    } | {"output_dir": str(tmp_path / "output")}


def test_short_runs_and_threshold_equality_pass(tmp_path):
    inputs = _inputs(
        tmp_path,
        human_silhouette={"front": [0.79] * 4 + [0.8] + [1.0] * 5},
        human_chamfer={"front": [51.0] * 4 + [50.0] + [0.0] * 5},
    )

    decision = ca.check_accuracy(**inputs)

    assert decision["status"] == "PASS"
    assert decision["failure_segments"] == []
    assert decision["object_mask_containment_diagnostic"]["diagnostic_only"] is True


def test_skipped_frames_break_runs_and_camera_runs_union(tmp_path):
    inputs = _inputs(
        tmp_path,
        human_silhouette={
            "front": [0.7] * 5 + [1.0] * 7,
            "left": [1.0] * 4 + [0.7] * 5 + [1.0] * 3,
        },
        object_chamfer={
            "front": [51.0, 51.0, None, 51.0, 51.0, 51.0, 51.0, 51.0]
            + [0.0] * 4,
            "left": [0.0] * 12,
        },
    )

    decision = ca.check_accuracy(
        **inputs,
        max_chamfer_object=100.0,
        max_silhouette_failure_coverage=1.0,
    )

    assert decision["status"] == "PASS"
    assert decision["failure_segment_count"] == 2
    silhouette = next(
        segment for segment in decision["failure_segments"]
        if segment["failure_category"] == ca.SILHOUETTE_CATEGORY
    )
    assert (silhouette["start_frame"], silhouette["end_frame"]) == (0, 9)
    assert silhouette["reason"] == (
        "Human Silhouette bounding box containment: failed in views [front, left]"
    )
    chamfer = next(
        segment for segment in decision["failure_segments"]
        if segment["failure_category"] == ca.CHAMFER_CATEGORY
    )
    assert (chamfer["start_frame"], chamfer["end_frame"]) == (3, 8)


def _separated_runs(total: int, run_count: int) -> list[float]:
    values = [1.0] * total
    for run_index in range(run_count):
        start = run_index * 6
        values[start : start + 5] = [0.0] * 5
    return values


def test_exact_segment_and_coverage_limits_pass(tmp_path):
    # Ten 5-frame runs separated by one frame cover exactly 50/100 frames.
    inputs = _inputs(
        tmp_path,
        human_silhouette={"front": _separated_runs(100, 10)},
    )

    decision = ca.check_accuracy(**inputs)

    assert decision["status"] == "PASS"
    assert decision["failure_segment_count"] == 10
    assert decision["failure_coverage"] == 0.5


def test_more_than_ten_segments_does_not_fail(tmp_path):
    inputs = _inputs(
        tmp_path,
        human_silhouette={"front": _separated_runs(200, 11)},
    )

    decision = ca.check_accuracy(**inputs)
    segments = json.loads(
        (Path(inputs["output_dir"]) / "failure_segments.json").read_text()
    )
    assert decision["status"] == "PASS"
    assert decision["failure_segment_count"] == 11
    assert len(segments) == 11
    assert decision["reason"] == ""


def test_more_than_coverage_limit_fails(tmp_path):
    inputs = _inputs(
        tmp_path,
        human_silhouette={"front": [0.0] * 51 + [1.0] * 49},
    )

    with pytest.raises(SystemExit):
        ca.check_accuracy(**inputs)

    decision = json.loads((Path(inputs["output_dir"]) / "check_accuracy.json").read_text())
    assert decision["failure_segment_count"] == 1
    assert decision["failure_coverage"] == 0.51
    assert decision["failed_accuracy_checks"] == [
        "human_silhouette_bbox_containment"
    ]


def test_metric_coverages_are_gated_independently(tmp_path):
    inputs = _inputs(
        tmp_path,
        human_silhouette={"front": [0.0] * 40 + [1.0] * 60},
        object_silhouette={"front": [1.0] * 60 + [0.0] * 40},
    )

    decision = ca.check_accuracy(**inputs)

    assert decision["status"] == "PASS"
    assert decision["failure_coverage"] == 0.8
    assert decision["metric_details"][
        "human_silhouette_bounding_box_containment"
    ]["failure_coverage"] == 0.4
    assert decision["metric_details"][
        "object_silhouette_bounding_box_containment"
    ]["failure_coverage"] == 0.4


def test_chamfer_pooled_median_boundary_and_hard_gate(tmp_path):
    passing = _inputs(
        tmp_path / "passing",
        human_silhouette={"front": [1.0] * 9},
        human_chamfer={"front": [40.0] * 5 + [100.0] * 4},
    )
    decision = ca.check_accuracy(**passing)
    details = decision["metric_details"]["human_chamfer_distance"]
    assert decision["status"] == "PASS"
    assert details["pooled_median_mm"] == 40.0
    assert details["gate_status"] == "PASS"

    failing = _inputs(
        tmp_path / "failing",
        human_silhouette={"front": [1.0] * 9},
        human_chamfer={"front": [40.0] * 4 + [41.0] * 5},
    )
    with pytest.raises(SystemExit):
        ca.check_accuracy(**failing)
    failed = json.loads(
        (Path(failing["output_dir"]) / "check_accuracy.json").read_text()
    )
    assert failed["failed_accuracy_checks"] == ["chamfer_human"]
    assert failed["failure_segments"] == []


def test_chamfer_segments_use_fifty_mm_but_do_not_gate(tmp_path):
    inputs = _inputs(
        tmp_path,
        human_silhouette={"front": [1.0] * 12},
        object_chamfer={"front": [51.0] * 5 + [50.0] + [0.0] * 6},
    )

    decision = ca.check_accuracy(**inputs)

    assert decision["status"] == "PASS"
    assert decision["failed_accuracy_checks"] == []
    assert decision["failure_segment_count"] == 1
    assert decision["failure_segments"][0]["start_frame"] == 0
    assert decision["failure_segments"][0]["end_frame"] == 5
    details = decision["metric_details"]["object_chamfer_distance"]
    assert details["segment_threshold_mm"] == 50.0
    assert details["gate_status"] == "PASS"


def test_inconsistent_combined_chamfer_is_an_error(tmp_path):
    inputs = _inputs(tmp_path, human_silhouette={"front": [1.0] * 10})
    path = Path(inputs["chamfer_human_dir"]) / "chamfer_metrics.json"
    payload = json.loads(path.read_text())
    payload["combined"]["median_mm"] = 123.0
    _write(path, payload)

    with pytest.raises(ValueError, match="combined.median_mm is inconsistent"):
        ca.check_accuracy(**inputs)


def test_rejects_compact_chamfer_without_source_indices(tmp_path):
    inputs = _inputs(tmp_path, human_silhouette={"front": [1.0] * 10})
    _write(
        Path(inputs["chamfer_human_dir"]) / "chamfer_metrics.json",
        {"per_camera": {"front": {"per_frame_mm": [1.0] * 10}}},
    )

    with pytest.raises(ValueError, match="source-indexed per_frame"):
        ca.check_accuracy(**inputs)
