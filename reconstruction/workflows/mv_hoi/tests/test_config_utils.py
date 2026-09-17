import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import config_utils


def _dataset_cfg() -> dict:
    return {
        "mesh_base": "swift://host/AUTH/container/mesh",
        "weights_base_url": "swift://host/AUTH/container/releases/test",
        "pipelines": {
            config_utils.CALIBRATION_PIPELINE: {
                "input_path": "calibration",
                "output_path": "calibration_output",
                "max_concurrent": 7,
                "workflows": {
                    config_utils.CALIBRATION_WORKFLOW: {
                        "workflow_yaml": "osmo/mv_calibration.yaml",
                        "calibration_setup": "stereo4_6x10_100mm_marker",
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
                "max_concurrent": 11,
                "workflows": {
                    config_utils.RECONSTRUCTION_WORKFLOW: {
                        "workflow_yaml": "osmo/mv_hoi_reconstruction.yaml",
                        "hitl_s3_base": "s3://bucket/path",
                    },
                },
            },
            config_utils.EXPORT_CONFIG_PIPELINE: {
                "input_path": "data_output",
                "output_path": "data_export_2",
                "workflows": {
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
        == "data_export_2"
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
            config_utils.EXPORT_CONFIG_PIPELINE,
            config_utils.EXPORT_WORKFLOW,
        )["workflow_yaml"]
        == "osmo/mv_hoi_export.yaml"
    )
    assert (
        config_utils.get_workflow_cfg(
            dataset_cfg,
            config_utils.CALIBRATION_PIPELINE,
            config_utils.CALIBRATION_WORKFLOW,
        )["calibration_setup"]
        == "stereo4_6x10_100mm_marker"
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
    assert (
        config_utils.get_pipeline_input_path(
            dataset_cfg,
            config_utils.EXPORT_CONFIG_PIPELINE,
        )
        == "data_output_test"
    )
    assert config_utils.get_pipeline_export_path(dataset_cfg) == "data_export_2_test"
    assert dataset_cfg["mesh_base"].endswith("_test")
    assert dataset_cfg["weights_base_url"] == (
        "swift://host/AUTH/container/releases/test"
    )


def test_cleanup_settings_support_bounded_asynchronous_mode():
    cleanup = {
        "enabled": True,
        "campaign_allowlist": ["campaign"],
        "automatic_export_cutoff": "2026-08-02T22:46:43Z",
        "mode": "asynchronous",
        "max_per_cycle": 40,
        "workers": 8,
    }
    config = {
        "datasets": {
            "dataset": {
                "pipelines": {
                    config_utils.EXPORT_CONFIG_PIPELINE: {
                        "cleanup_intermediates_after_export": cleanup,
                    },
                },
            },
        },
    }
    assert config_utils.get_cleanup_settings(
        config, "dataset", "campaign",
    ) == {
        "mode": "asynchronous",
        "limit": 40,
        "workers": 8,
        "completed_after": "2026-08-02T22:46:43Z",
    }
