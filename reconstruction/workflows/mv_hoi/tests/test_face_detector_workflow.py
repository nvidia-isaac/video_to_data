import re
from pathlib import Path

import yaml


WORKFLOW_DIR = Path(__file__).resolve().parents[1]


def _load_osmo_workflow(name: str):
    raw = (WORKFLOW_DIR / "osmo" / name).read_text()
    parsed = yaml.safe_load(re.sub(r"\{\{[^}]+\}\}", "placeholder", raw))
    return raw, parsed


def test_face_detector_follows_preprocess_in_normal_and_oneoff_workflows():
    for workflow_name in ("mv_preprocess.yaml", "mv_preprocess_oneoff.yaml"):
        raw, workflow = _load_osmo_workflow(workflow_name)
        tasks = workflow["workflow"]["tasks"]
        task_names = [task["name"] for task in tasks]
        assert task_names.index("face_detector") > task_names.index("mv_preprocess")
        face_task = next(task for task in tasks if task["name"] == "face_detector")
        assert face_task["image"] == (
            "nvcr.io/nvstaging/isaac-amr/mv_hoi_face_detector:placeholder"
        )
        assert face_task["inputs"] == [
            {"task": "mv_preprocess"},
            {"url": "placeholder/face_detector/"},
        ]
        assert face_task["outputs"] == [
            {"url": "placeholder/face_detector/"}
        ]
        assert "v2d_mv_hoi_data" not in raw
        assert "weights_base_url: ???" in raw
        assert "face_detection_yunet_2023mar.onnx" in raw
        assert "--rgb_dir {{input:0}}/images" in raw
        assert '--model_dir "$MODEL_DIR"' in raw
        assert 'rgb_path_template: "${rgb_dir}/{cam_name}.h5"' in raw
        assert (
            'output_image_path_template: "${output_dir}/images/{cam_name}.h5"'
            in raw
        )


def test_hoi_overlay_uses_anonymized_face_detector_videos():
    reconstruction_raw, reconstruction = _load_osmo_workflow(
        "mv_hoi_reconstruction.yaml"
    )
    reconstruction_overlay = next(
        task
        for task in reconstruction["workflow"]["tasks"]
        if task["name"] == "render_hoi_overlay"
    )
    assert reconstruction_overlay["inputs"][-1] == {"url": "placeholder"}
    assert "face_detector_url: ???" in reconstruction_raw
    assert "--rgb_dir {{input:3}}/videos" in reconstruction_raw
    assert 'rgb_path_template: "${rgb_dir}/{cam_name}.mp4"' in reconstruction_raw

    revalidation_raw, revalidation = _load_osmo_workflow(
        "mv_hoi_revalidation.yaml"
    )
    revalidation_overlay = next(
        task
        for task in revalidation["workflow"]["tasks"]
        if task["name"] == "render_hoi_overlay"
    )
    assert revalidation_overlay["inputs"][4] == {"task": "face_detector"}
    assert "--rgb_dir {{input:4}}/videos" in revalidation_raw
    assert 'rgb_path_template: "${rgb_dir}/{cam_name}.mp4"' in revalidation_raw


def test_revalidation_tasks_download_only_the_source_subtrees_they_consume():
    raw, workflow = _load_osmo_workflow("mv_hoi_revalidation.yaml")
    tasks = {
        task["name"]: task for task in workflow["workflow"]["tasks"]
    }

    # The controller validates all frozen CSS objects before submission,
    # reconciliation, and publication. Workflow source checks therefore mount
    # only the immutable inventory/configuration, while the final packager is
    # the sole task allowed to download the complete source prefix.
    # A bare placeholder remains legitimate for configuration_url. The counts
    # below ensure there is no additional bare source_url input.
    allowed_bare_urls = {
        "face_detector": 0,
        "foundation_pose": 1,
        "compare_foundation_pose": 1,
        "eval_chamfer_object": 0,
        "eval_silhouette_mask_object": 0,
        "render_hoi_overlay": 0,
        "check_accuracy": 0,
    }
    for name, allowed_count in allowed_bare_urls.items():
        assert tasks[name]["inputs"].count({"url": "placeholder"}) == allowed_count

    assert {"url": "placeholder/mv_preprocess/images/"} in tasks[
        "face_detector"
    ]["inputs"]
    assert {"url": "placeholder/foundation_stereo/"} in tasks[
        "foundation_pose"
    ]["inputs"]
    assert {"url": "placeholder/mv_preprocess/edex"} in tasks[
        "foundation_pose"
    ]["inputs"]
    assert raw.count("--camera_params_path {{input:1}}/edex") == 4
    assert {"url": "placeholder/sam2_object_masks/"} in tasks[
        "eval_silhouette_mask_object"
    ]["inputs"]
    assert "--rgb_dir {{input:4}}/videos" in raw

    # Heavy source downloads form a chain instead of hitting CSS in parallel.
    assert tasks["foundation_pose"]["inputs"][-1] == {"task": "face_detector"}
    assert tasks["eval_chamfer_object"]["inputs"][-1] == {
        "task": "compare_foundation_pose"
    }
    assert tasks["eval_silhouette_mask_object"]["inputs"][-1] == {
        "task": "eval_chamfer_object"
    }
    assert tasks["render_hoi_overlay"]["inputs"][-1] == {
        "task": "eval_silhouette_mask_object"
    }

    assert tasks["validate_source"]["inputs"] == [
        {"url": "placeholder"},
        {"url": "placeholder"},
    ]
    assert tasks["revalidate_source"]["inputs"][-1] == {"url": "placeholder"}
    assert all(
        item != {"url": "placeholder"}
        for item in tasks["revalidate_source"]["inputs"][:-1]
    )
    assert "--source-dir" not in tasks["validate_source"]["files"][0]["contents"]
    assert "--source-dir" not in tasks["revalidate_source"]["files"][0]["contents"]
    assert "--legacy-export-dir" not in tasks["validate_source"]["files"][0]["contents"]
    assert "--legacy-export-dir" not in tasks["revalidate_source"]["files"][0]["contents"]
