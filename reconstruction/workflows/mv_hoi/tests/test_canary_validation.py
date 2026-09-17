from pathlib import Path
import sys


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration.canary_validation import (
    validate_accuracy_failure_evidence,
    validate_foundation_comparison_evidence,
)


def test_accuracy_failure_evidence_regenerates_limits_and_coverage():
    segments = [{
        "start_frame": 0, "end_frame": 6,
        "failure_category": "Chamfer distance",
        "reason": "Object Chamfer distance: failed in views [front]",
    }]
    report = {
        "schema": "v2d.mv_hoi.check_accuracy.v2", "status": "FAIL",
        "reason": "accuracy failure coverage 0.600000 > 0.500000",
        "total_frames": 10, "failure_segment_count": 1,
        "failure_coverage_frames": 6, "failure_coverage": 0.6,
        "failure_segments": segments,
        "thresholds": {
            "max_chamfer_object_mm": 40.0,
            "max_chamfer_human_mm": 40.0,
            "min_silhouette_bbox_containment": 0.8,
            "min_failure_run_frames": 5,
            "max_failure_segments": 10,
            "max_failure_coverage": 0.5,
        },
    }
    valid, detail = validate_accuracy_failure_evidence(
        report, segments, expected_thresholds=report["thresholds"],
    )
    assert valid is True
    assert "regenerated" in detail
    report["failure_coverage"] = 0.5
    assert validate_accuracy_failure_evidence(
        report, segments, expected_thresholds=report["thresholds"],
    )[0] is False


def test_v3_accuracy_failure_regenerates_independent_metric_gate():
    segments = [{
        "start_frame": 0, "end_frame": 6,
        "failure_category": "Silhouette bounding box containment",
        "reason": (
            "Object Silhouette bounding box containment: "
            "failed in views [back]"
        ),
    }]
    empty = {
        "failure_segments": 0, "failure_coverage_frames": 0,
        "failure_coverage": 0.0, "gate_status": "PASS",
    }
    report = {
        "schema": "v2d.mv_hoi.check_accuracy.v3", "status": "FAIL",
        "reason": (
            "Object silhouette bbox containment coverage "
            "0.600000 > 0.500000"
        ),
        "failed_accuracy_checks": [
            "object_silhouette_bbox_containment"
        ],
        "total_frames": 10, "failure_segment_count": 1,
        "failure_coverage_frames": 6, "failure_coverage": 0.6,
        "failure_segments": segments,
        "metric_details": {
            "human_silhouette_bounding_box_containment": {
                **empty, "gate_threshold": 0.5,
            },
            "human_chamfer_distance": {
                **empty, "pooled_median_mm": 10.0,
                "gate_threshold_mm": 40.0, "segment_threshold_mm": 50.0,
            },
            "object_silhouette_bounding_box_containment": {
                "failure_segments": 1, "failure_coverage_frames": 6,
                "failure_coverage": 0.6, "gate_status": "FAIL",
                "gate_threshold": 0.5,
            },
            "object_chamfer_distance": {
                **empty, "pooled_median_mm": 20.0,
                "gate_threshold_mm": 40.0, "segment_threshold_mm": 50.0,
            },
        },
        "thresholds": {
            "max_chamfer_object_mm": 40.0,
            "max_chamfer_human_mm": 40.0,
            "max_chamfer_segment_object_mm": 50.0,
            "max_chamfer_segment_human_mm": 50.0,
            "min_silhouette_bbox_containment": 0.8,
            "min_failure_run_frames": 5,
            "max_silhouette_failure_coverage": 0.5,
        },
    }

    valid, detail = validate_accuracy_failure_evidence(
        report, segments, expected_thresholds=report["thresholds"],
    )

    assert valid is True
    assert "independent metric decisions" in detail
    report["metric_details"][
        "object_silhouette_bounding_box_containment"
    ]["failure_coverage"] = 0.5
    assert validate_accuracy_failure_evidence(
        report, segments, expected_thresholds=report["thresholds"],
    )[0] is False


def test_foundation_comparison_failure_evidence_requires_provenance():
    report = {
        "schema": "v2d.mv_hoi.foundation_pose_comparison.v2",
        "status": "FAIL", "failures": ["valid_coverage"],
        "tolerances_sha256": "a" * 64,
    }
    assert validate_foundation_comparison_evidence(report)[0] is True
    report["failures"] = []
    assert validate_foundation_comparison_evidence(report)[0] is False
