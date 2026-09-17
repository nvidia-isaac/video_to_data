import re
from pathlib import Path

import yaml


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]


def _load_workflow():
    raw = (WORKFLOW_ROOT / "osmo" / "mv_hoi_reconstruction.yaml").read_text()
    parsed = yaml.safe_load(re.sub(r"\{\{[^}]+\}\}", "placeholder", raw))
    return raw, parsed


def _task(workflow: dict, name: str) -> dict:
    return next(task for task in workflow["workflow"]["tasks"] if task["name"] == name)


def test_human_and_object_silhouette_metrics_feed_segment_accuracy():
    raw, workflow = _load_workflow()
    object_eval = _task(workflow, "eval_silhouette_mask_object")
    human_eval = _task(workflow, "eval_silhouette_mask_human")
    accuracy = _task(workflow, "check_accuracy")

    object_config = object_eval["files"][1]["contents"]
    assert "min_mask_pixels: placeholder" in object_config
    assert "min_bbox_containment: placeholder" in object_config
    assert "max_unexplained_sam2_ratio" not in object_config
    assert "min_bbox_component_pixels: placeholder" in object_config
    assert "min_bbox_component_fraction_of_largest: placeholder" in object_config
    assert "render_bbox_padding_pixels: placeholder" in object_config
    assert "render_bbox_padding_fraction: placeholder" in object_config
    assert "max_bad_frame_fraction: placeholder" in object_config
    assert "bad_run_length" not in object_config

    human_config = human_eval["files"][1]["contents"]
    assert "min_mask_pixels: placeholder" in human_config
    assert "min_bbox_component_pixels: placeholder" in human_config
    assert "min_bbox_component_fraction_of_largest: placeholder" in human_config
    assert "render_bbox_padding_pixels: placeholder" in human_config
    assert "render_bbox_padding_fraction: placeholder" in human_config
    assert "max_unexplained_sam2_ratio" not in human_config
    assert "bad_run_length" not in human_config

    assert {item["task"] for item in accuracy["inputs"] if "task" in item} >= {
        "eval_silhouette_mask_object", "eval_silhouette_mask_human",
        "eval_chamfer_object", "eval_chamfer_human", "export_soma",
        "estimate_ground_plane", "export_fused_pointcloud",
    }
    assert accuracy["inputs"][:5] == [
        {"task": "eval_chamfer_object"},
        {"task": "eval_chamfer_human"},
        {"task": "check_object_mask"},
        {"task": "eval_silhouette_mask_object"},
        {"task": "eval_silhouette_mask_human"},
    ]
    assert accuracy["inputs"][-3:] == [
        {"task": "export_soma"}, {"task": "estimate_ground_plane"},
        {"task": "export_fused_pointcloud"},
    ]
    accuracy_script = accuracy["files"][0]["contents"]
    assert "--silhouette_object_dir placeholder" in accuracy_script
    assert "--silhouette_human_dir placeholder" in accuracy_script
    assert "--min_failure_run_frames placeholder" in accuracy_script
    assert "--max_chamfer_segment_object placeholder" in accuracy_script
    assert "--max_chamfer_segment_human placeholder" in accuracy_script
    assert "--max_silhouette_failure_coverage placeholder" in accuracy_script
    assert "--max_failure_segments" not in accuracy_script

    for variable, default in (
        ("min_object_silhouette_bbox_containment", '"0.8"'),
        ("min_object_segment_pixels", '"10"'),
        ("min_object_bbox_component_pixels", '"3"'),
        ("min_object_bbox_component_fraction_of_largest", '"0.001"'),
        ("object_silhouette_render_bbox_padding_pixels", '"8"'),
        ("silhouette_render_bbox_padding_fraction", '"0.1"'),
        ("max_object_silhouette_bad_frame_fraction", '"0.05"'),
    ):
        assert f"{variable}: {default}" in raw


def test_object_silhouette_thresholds_propagate_from_dataset_config():
    config = yaml.safe_load((WORKFLOW_ROOT / "config.yaml").read_text())
    thresholds = config["datasets"]["sc_office_4exo_1"]["pipelines"][
        "mv_hoi_reconstruction"
    ]["workflows"]["reconstruction"]["qc_thresholds"]
    assert thresholds["min_object_silhouette_bbox_containment"] == 0.8
    assert thresholds["min_object_segment_pixels"] == 10
    assert thresholds["min_object_bbox_component_pixels"] == 3
    assert thresholds["min_object_bbox_component_fraction_of_largest"] == 0.001
    assert thresholds["object_silhouette_render_bbox_padding_pixels"] == 8
    assert thresholds["max_object_silhouette_bad_frame_fraction"] == 0.05

    submit_source = (WORKFLOW_ROOT / "orchestration" / "submit.py").read_text()
    for variable in (
        "min_object_silhouette_bbox_containment",
        "min_object_segment_pixels",
        "min_object_bbox_component_pixels",
        "min_object_bbox_component_fraction_of_largest",
        "object_silhouette_render_bbox_padding_pixels",
        "max_object_silhouette_bad_frame_fraction",
    ):
        assert f'set_vars["{variable}"]' in submit_source


def test_revalidation_uses_the_same_bbox_containment_gate():
    raw = (WORKFLOW_ROOT / "osmo" / "mv_hoi_revalidation.yaml").read_text()
    parsed = yaml.safe_load(re.sub(r"\{\{[^}]+\}\}", "placeholder", raw))
    object_eval = _task(parsed, "eval_silhouette_mask_object")
    object_config = object_eval["files"][1]["contents"]
    human_config = _task(parsed, "eval_silhouette_mask_human")["files"][1]["contents"]
    accuracy = _task(parsed, "check_accuracy")

    assert "min_mask_pixels: placeholder" in object_config
    assert "min_bbox_containment: placeholder" in object_config
    assert "min_bbox_component_pixels: placeholder" in object_config
    assert "min_bbox_component_fraction_of_largest: placeholder" in object_config
    assert "render_bbox_padding_pixels: placeholder" in object_config
    assert "render_bbox_padding_fraction: placeholder" in object_config
    assert "max_bad_frame_fraction: placeholder" in object_config
    assert "max_unexplained_sam2_ratio" not in object_config
    assert "min_bbox_component_pixels: placeholder" in human_config
    assert "render_bbox_padding_fraction: placeholder" in human_config
    assert {item["task"] for item in accuracy["inputs"] if "task" in item} >= {
        "eval_silhouette_mask_object", "eval_silhouette_mask_human",
        "eval_chamfer_object", "eval_chamfer_human",
    }
    assert not {
        "export_soma", "estimate_ground_plane", "export_fused_pointcloud",
    } & {item["task"] for item in accuracy["inputs"] if "task" in item}
    assert "--silhouette_human_dir placeholder" in accuracy["files"][0]["contents"]
    assert "min_object_silhouette_bbox_containment: 0.8" in raw
    assert "min_object_segment_pixels: 10" in raw
    assert "min_object_bbox_component_pixels: 3" in raw
    assert "min_object_bbox_component_fraction_of_largest: 0.001" in raw
    assert "object_silhouette_render_bbox_padding_pixels: 8" in raw
    assert "max_object_silhouette_bad_frame_fraction: 0.05" in raw
