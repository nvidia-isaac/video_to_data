"""Config helpers for MV HOI pipeline and OSMO workflow settings."""

from __future__ import annotations

from pathlib import Path

import yaml


CALIBRATION_PIPELINE = "mv_calibration"
PREPROCESS_PIPELINE = "mv_preprocess"
RECON_PIPELINE = "mv_hoi_reconstruction"
CALIBRATION_WORKFLOW = "calibration"
PREPROCESS_WORKFLOW = "preprocess"
RECONSTRUCTION_WORKFLOW = "reconstruction"
EXPORT_WORKFLOW = "export"


def load_config(script_dir: str | Path) -> dict:
    with open(Path(script_dir) / "config.yaml") as f:
        return yaml.safe_load(f)


def get_dataset_cfg(config: dict, dataset: str) -> dict:
    return config["datasets"][dataset]


def get_pipeline_cfg(dataset_cfg: dict, pipeline_type: str) -> dict:
    return dataset_cfg["pipelines"][pipeline_type]


def get_workflow_cfg(dataset_cfg: dict, pipeline_type: str, workflow_key: str) -> dict:
    return get_pipeline_cfg(dataset_cfg, pipeline_type)["workflows"][workflow_key]


def get_pipeline_input_path(dataset_cfg: dict, pipeline_type: str) -> str:
    return get_pipeline_cfg(dataset_cfg, pipeline_type)["input_path"]


def get_pipeline_output_path(dataset_cfg: dict, pipeline_type: str) -> str:
    return get_pipeline_cfg(dataset_cfg, pipeline_type)["output_path"]


def get_pipeline_export_path(dataset_cfg: dict, pipeline_type: str = RECON_PIPELINE) -> str:
    return get_pipeline_cfg(dataset_cfg, pipeline_type)["export_path"]


def get_pipeline_max_concurrent(dataset_cfg: dict, pipeline_type: str, default: int = 10) -> int:
    return int(get_pipeline_cfg(dataset_cfg, pipeline_type).get("max_concurrent", default))


def append_test_suffix(value: str) -> str:
    return value.rstrip("/") + "_test"


def apply_test_mode(dataset_cfg: dict) -> None:
    """In-place: append `_test` to output paths and test mesh path."""
    pipelines = dataset_cfg["pipelines"]
    pipelines[CALIBRATION_PIPELINE]["output_path"] = append_test_suffix(
        pipelines[CALIBRATION_PIPELINE]["output_path"]
    )
    if PREPROCESS_PIPELINE in pipelines:
        pipelines[PREPROCESS_PIPELINE]["output_path"] = append_test_suffix(
            pipelines[PREPROCESS_PIPELINE]["output_path"]
        )
    pipelines[RECON_PIPELINE]["output_path"] = append_test_suffix(
        pipelines[RECON_PIPELINE]["output_path"]
    )
    pipelines[RECON_PIPELINE]["export_path"] = append_test_suffix(
        pipelines[RECON_PIPELINE]["export_path"]
    )
    dataset_cfg["mesh_base"] = append_test_suffix(dataset_cfg["mesh_base"])
