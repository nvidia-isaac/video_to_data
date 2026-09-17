import sys
from pathlib import Path

import pytest


LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

import export_qc


def _trim(start=10, end=30, source=40):
    return {
        "schema": "v2d.mv_hoi.interaction_trim.v2",
        "decision_sha256": "trim-hash",
        "source_frame_count": source,
        "export_source_start_frame": start,
        "export_source_end_frame": end,
        "export_frame_count": end - start,
    }


def _containment(subject, start, end, identifier="segment"):
    category = export_qc.CONTAINMENT_CATEGORY
    return {
        "id": identifier,
        "start_frame": start,
        "end_frame": end,
        "failure_category": category,
        "reason": f"{subject} {category}: failed in views [front]",
    }


def test_independent_containment_and_equality_boundaries():
    decision = export_qc.evaluate_post_trim_qc(
        trim_manifest=_trim(),
        human_segments=[{"id": "human", "start_frame": 10, "end_frame": 20}],
        metric_segments=[
            _containment("Human", 10, 20, "hc"),
            _containment("Object", 10, 21, "oc"),
            {
                "id": "chamfer", "start_frame": 10, "end_frame": 30,
                "failure_category": "Chamfer distance",
                "reason": "Object Chamfer distance: failed in views [front]",
            },
        ],
        max_failure_annotations=1,
        max_failure_coverage=0.5,
        max_silhouette_failure_coverage=0.5,
    )
    by_name = {gate["name"]: gate for gate in decision["gates"]}
    assert by_name["human_qc_failure_count"]["status"] == "PASS"
    assert by_name["human_qc_failure_coverage"]["status"] == "PASS"
    assert by_name["human_silhouette_bbox_containment_coverage"]["status"] == "PASS"
    assert by_name["object_silhouette_bbox_containment_coverage"]["status"] == "FAIL"
    assert decision["status"] == "REJECTED"


def test_trim_clips_drops_and_rebases_before_gate():
    decision = export_qc.evaluate_post_trim_qc(
        trim_manifest=_trim(),
        human_segments=[
            {"id": "before", "start_frame": 0, "end_frame": 10},
            {"id": "cross", "start_frame": 5, "end_frame": 15},
            {"id": "inside", "start_frame": 20, "end_frame": 25},
            {"id": "after", "start_frame": 30, "end_frame": 35},
        ],
        metric_segments=[], max_failure_annotations=10,
        max_failure_coverage=0.5, max_silhouette_failure_coverage=0.5,
    )
    assert [(s["id"], s["start_frame"], s["end_frame"]) for s in
            decision["trimmed_human_segments"]] == [
        ("cross", 0, 5), ("inside", 10, 15),
    ]
    assert decision["status"] == "PASS"


def test_rejection_report_hash_and_malformed_refusal():
    decision = export_qc.evaluate_post_trim_qc(
        trim_manifest=_trim(),
        human_segments=[{"id": str(i), "start_frame": 10, "end_frame": 11}
                        for i in range(11)],
        metric_segments=[], max_failure_annotations=10,
        max_failure_coverage=1.0, max_silhouette_failure_coverage=1.0,
    )
    report = export_qc.build_rejection_report(
        decision=decision, trim_manifest=_trim(),
        human_segments=[], metric_segments=[], provenance={"request_id": 1},
    )
    assert export_qc.validate_rejection_report(report) == report
    report["gates"][0]["observed"] = 10
    with pytest.raises(ValueError, match="hash"):
        export_qc.validate_rejection_report(report)
