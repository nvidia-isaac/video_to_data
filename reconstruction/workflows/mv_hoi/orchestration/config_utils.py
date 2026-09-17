"""Config helpers for MV HOI pipeline and OSMO workflow settings."""

from __future__ import annotations

from pathlib import Path
import os

import yaml


CALIBRATION_PIPELINE = "mv_calibration"
PREPROCESS_PIPELINE = "mv_preprocess"
RECON_PIPELINE = "mv_hoi_reconstruction"
EXPORT_PIPELINE = "mv_export"
EXPORT_CONFIG_PIPELINE = "mv_hoi_export"
REVALIDATION_PIPELINE = "mv_hoi_revalidation"
CALIBRATION_WORKFLOW = "calibration"
PREPROCESS_WORKFLOW = "preprocess"
RECONSTRUCTION_WORKFLOW = "reconstruction"
EXPORT_WORKFLOW = "export"
REVALIDATION_WORKFLOW = "revalidation"
REVALIDATION_EXPORT_RETRY_WORKFLOW = "export_retry"

PIPELINE_TO_STAGE = {
    CALIBRATION_PIPELINE: "calibration",
    PREPROCESS_PIPELINE: "preprocess",
    RECON_PIPELINE: "reconstruction",
    EXPORT_PIPELINE: "export",
}
STAGE_TO_PIPELINE = {stage: pipeline for pipeline, stage in PIPELINE_TO_STAGE.items()}


def load_config(script_dir: str | Path) -> dict:
    configured = os.environ.get("MV_HOI_CONFIG_PATH")
    path = Path(configured).expanduser() if configured else Path(script_dir) / "config.yaml"
    with open(path) as f:
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


def get_pipeline_export_path(dataset_cfg: dict) -> str:
    return get_pipeline_output_path(dataset_cfg, EXPORT_CONFIG_PIPELINE)


def get_legacy_export_path(dataset_cfg: dict) -> str:
    return get_pipeline_cfg(dataset_cfg, EXPORT_CONFIG_PIPELINE).get(
        "legacy_output_path", "data_export"
    )


def get_pipeline_max_concurrent(dataset_cfg: dict, pipeline_type: str, default: int = 10) -> int:
    return int(get_pipeline_cfg(dataset_cfg, pipeline_type).get("max_concurrent", default))


def get_cleanup_settings(
    config: dict, dataset: str, campaign_name: str,
) -> dict | None:
    """Return validated, campaign-scoped intermediate cleanup settings."""
    export_cfg = (
        get_dataset_cfg(config, dataset)
        .get("pipelines", {}).get(EXPORT_CONFIG_PIPELINE, {})
    )
    configured = export_cfg.get("cleanup_intermediates_after_export", False)
    if configured is False or configured is None:
        return None
    if not isinstance(configured, dict):
        raise ValueError(
            "cleanup_intermediates_after_export must use the bounded mapping format"
        )
    if not configured.get("enabled", False):
        return None
    allowlist = {str(item) for item in configured.get("campaign_allowlist", [])}
    if campaign_name not in allowlist:
        return None
    cutoff = configured.get("automatic_export_cutoff")
    if not cutoff:
        raise ValueError(
            "automatic cleanup requires automatic_export_cutoff to prevent "
            "an unbounded historical sweep"
        )
    mode = str(configured.get("mode", "inline")).lower()
    if mode not in ("inline", "asynchronous"):
        raise ValueError("cleanup mode must be inline or asynchronous")
    limit = int(configured.get("max_per_cycle", 2))
    workers = int(configured.get("workers", 2))
    if limit < 1 or workers < 1:
        raise ValueError("cleanup max_per_cycle and workers must be positive")
    return {
        "mode": mode,
        "limit": limit,
        "workers": workers,
        "completed_after": str(cutoff),
    }


def get_backlog_promotion_settings(config: dict, dataset: str) -> dict:
    """Return bounded per-cycle final-export publication settings."""
    export_cfg = (
        get_dataset_cfg(config, dataset)
        .get("pipelines", {}).get(EXPORT_CONFIG_PIPELINE, {})
    )
    configured = export_cfg.get("promotion", {})
    if not isinstance(configured, dict):
        raise ValueError("mv_hoi_export.promotion must be a mapping")
    limit = int(configured.get("max_per_cycle", 20))
    workers = int(configured.get("workers", 4))
    if limit < 1 or workers < 1:
        raise ValueError("promotion max_per_cycle and workers must be positive")
    return {"limit": limit, "workers": workers}


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
    reconstruction_output = pipelines[RECON_PIPELINE]["output_path"]
    pipelines[RECON_PIPELINE]["output_path"] = append_test_suffix(
        reconstruction_output
    )
    export_cfg = pipelines[EXPORT_CONFIG_PIPELINE]
    if export_cfg["input_path"] == reconstruction_output:
        export_cfg["input_path"] = pipelines[RECON_PIPELINE]["output_path"]
    else:
        export_cfg["input_path"] = append_test_suffix(export_cfg["input_path"])
    export_cfg["output_path"] = append_test_suffix(export_cfg["output_path"])
    dataset_cfg["mesh_base"] = append_test_suffix(dataset_cfg["mesh_base"])
