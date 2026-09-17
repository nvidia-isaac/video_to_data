import re
from pathlib import Path

import yaml


WORKFLOW_DIR = Path(__file__).resolve().parents[1]


def _load(name: str):
    raw = (WORKFLOW_DIR / "osmo" / name).read_text()
    parsed = yaml.safe_load(re.sub(r"\{\{[^}]+\}\}", "placeholder", raw))
    return raw, parsed


def _task(workflow: dict, name: str) -> dict:
    return next(task for task in workflow["workflow"]["tasks"] if task["name"] == name)


def test_all_weight_inputs_are_module_specific_css_urls():
    preprocess_raw, preprocess = _load("mv_preprocess.yaml")
    oneoff_raw, oneoff = _load("mv_preprocess_oneoff.yaml")
    reconstruction_raw, reconstruction = _load("mv_hoi_reconstruction.yaml")

    for raw in (preprocess_raw, oneoff_raw, reconstruction_raw):
        assert "v2d_mv_hoi_data" not in raw
        assert "weights_base_url: ???" in raw

    for workflow in (preprocess, oneoff):
        assert _task(workflow, "face_detector")["inputs"][1] == {
            "url": "placeholder/face_detector/"
        }

    expected_inputs = {
        "foundation_stereo": (1, "placeholder/foundation_stereo/"),
        "grounding_dino": (1, "placeholder/grounding_dino/"),
        "sam2_object_masks": (2, "placeholder/sam2/"),
        "foundation_pose": (4, "placeholder/foundation_pose/"),
        "detectron2": (1, "placeholder/detectron2/"),
        "sam2_human_masks": (2, "placeholder/sam2/"),
        "sam3d_body": (3, "placeholder/sam3d_body/"),
    }
    for task_name, (index, url) in expected_inputs.items():
        assert _task(reconstruction, task_name)["inputs"][index] == {"url": url}


def test_writable_caches_offline_dino_and_export_soma_dependencies():
    raw, workflow = _load("mv_hoi_reconstruction.yaml")
    assert "MODEL_DIR=/tmp/foundation_stereo_weights" in raw
    assert "get_versioned_engine_filename" in raw
    assert 'cp "{{input:1}}/$ENGINE_NAME" "$MODEL_DIR/$ENGINE_NAME"' in raw
    assert "No compatible provisioned TensorRT engine; building from ONNX" in raw
    assert "FOUNDATIONPOSE_ENGINE_CACHE_DIR=/tmp/foundation_pose_engine_cache" in raw
    assert "TORCH_HOME=/tmp/sam3d_torch_home" in raw
    assert 'tar -xzf "{{input:3}}/dinov3_repo.tar.gz"' in raw

    export_soma = _task(workflow, "export_soma")
    assert export_soma["inputs"] == [
        {"task": "sam3d_body"},
        {"url": "placeholder/sam3d_body/hf_home/"},
    ]
    export_script = export_soma["files"][0]["contents"]
    assert "--weights_dir" not in export_script
    assert 'export HF_HOME="placeholder"' in export_script
    assert "export HF_HUB_OFFLINE=1" in export_script
    assert "export XDG_CACHE_HOME=/tmp/soma_x_runtime_cache" in export_script
    assert 'SOMA_REPO="$HF_HOME/hub/models--nvidia--soma-x"' in export_script
    assert 'test -f "$SOMA_REPO/refs/main"' in export_script
    assert "SOMA_neutral.npz" in export_script
    assert "MHR/mhr_model_lod1.pt" in export_script
    assert "get_assets_dir" in export_script
    assert "tar -" not in export_script
    assert "/tmp/soma_hf_home" not in export_script


def test_every_inference_task_validates_its_required_weight_files():
    raw, _ = _load("mv_hoi_reconstruction.yaml")
    required = (
        "deployable_foundationstereo_small_576x960_v2.0.onnx",
        "groundingdino_swint_ogc.pth",
        "sam2.1_hiera_large.pt",
        "sam2.1_hiera_l.yaml",
        "refiner_net.onnx",
        "score_net.onnx",
        "cascade_mask_rcnn_vitdet_b/model_final_435fa9.pkl",
        "sam-3d-body-dinov3/model.ckpt",
        "sam-3d-body-dinov3/model_config.yaml",
        "sam-3d-body-dinov3/assets/mhr_model.pt",
        "dinov3_repo.tar.gz",
    )
    for path in required:
        assert path in raw
    assert "nvidia_tensorrt/deployable_v1.0" in raw
    assert "1.0.1_onnx" not in raw
