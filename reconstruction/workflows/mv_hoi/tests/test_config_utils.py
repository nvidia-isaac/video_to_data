import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import config_utils


def _dataset_cfg() -> dict:
    return {
        "mesh_base": "swift://host/AUTH/container/mesh",
        "pipelines": {
            config_utils.CALIBRATION_PIPELINE: {
                "input_path": "calibration",
                "output_path": "calibration_output",
                "max_concurrent": 7,
                "workflows": {
                    config_utils.CALIBRATION_WORKFLOW: {
                        "workflow_yaml": "osmo/mv_calibration.yaml",
                    },
                },
            },
            config_utils.PREPROCESS_PIPELINE: {
                "input_path": "data",
                "output_path": "data_output",
                "max_concurrent": 9,
                "workflows": {
                    config_utils.PREPROCESS_WORKFLOW: {
                        "workflow_yaml": "osmo/mv_preprocess.yaml",
                    },
                },
            },
            config_utils.RECON_PIPELINE: {
                "input_path": "data",
                "output_path": "data_output",
                "export_path": "data_export",
                "max_concurrent": 11,
                "workflows": {
                    config_utils.RECONSTRUCTION_WORKFLOW: {
                        "workflow_yaml": "osmo/mv_hoi_reconstruction.yaml",
                        "hitl_s3_base": "s3://bucket/path",
                    },
                    config_utils.EXPORT_WORKFLOW: {
                        "workflow_yaml": "osmo/mv_hoi_export.yaml",
                        "batch_size": 30,
                    },
                },
            },
        },
    }


def test_config_helpers_resolve_pipeline_paths_and_workflows():
    dataset_cfg = _dataset_cfg()

    assert (
        config_utils.get_pipeline_input_path(
            dataset_cfg,
            config_utils.CALIBRATION_PIPELINE,
        )
        == "calibration"
    )
    assert (
        config_utils.get_pipeline_output_path(
            dataset_cfg,
            config_utils.PREPROCESS_PIPELINE,
        )
        == "data_output"
    )
    assert (
        config_utils.get_pipeline_output_path(
            dataset_cfg,
            config_utils.RECON_PIPELINE,
        )
        == "data_output"
    )
    assert (
        config_utils.get_pipeline_export_path(dataset_cfg)
        == "data_export"
    )
    assert (
        config_utils.get_pipeline_max_concurrent(
            dataset_cfg,
            config_utils.PREPROCESS_PIPELINE,
        )
        == 9
    )
    assert (
        config_utils.get_pipeline_max_concurrent(
            dataset_cfg,
            config_utils.RECON_PIPELINE,
        )
        == 11
    )
    assert (
        config_utils.get_workflow_cfg(
            dataset_cfg,
            config_utils.RECON_PIPELINE,
            config_utils.EXPORT_WORKFLOW,
        )["workflow_yaml"]
        == "osmo/mv_hoi_export.yaml"
    )


def test_apply_test_mode_updates_outputs_but_not_inputs():
    dataset_cfg = _dataset_cfg()

    config_utils.apply_test_mode(dataset_cfg)

    assert (
        config_utils.get_pipeline_input_path(
            dataset_cfg,
            config_utils.CALIBRATION_PIPELINE,
        )
        == "calibration"
    )
    assert (
        config_utils.get_pipeline_output_path(
            dataset_cfg,
            config_utils.CALIBRATION_PIPELINE,
        )
        == "calibration_output_test"
    )
    assert (
        config_utils.get_pipeline_output_path(
            dataset_cfg,
            config_utils.PREPROCESS_PIPELINE,
        )
        == "data_output_test"
    )
    assert (
        config_utils.get_pipeline_output_path(
            dataset_cfg,
            config_utils.RECON_PIPELINE,
        )
        == "data_output_test"
    )
    assert config_utils.get_pipeline_export_path(dataset_cfg) == "data_export_test"
    assert dataset_cfg["mesh_base"].endswith("_test")
