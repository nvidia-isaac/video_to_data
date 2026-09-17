from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from lib_mhr.refit import MHR_REFIT_ADAPTIVE_AUDIT_KEYS, MHR_REFIT_MODE_FULL, MHR_REFIT_MODE_TRANSLATION_UNIFORM_SCALE, MHRRefitConfig, mhr_refit_fixed_keys, mhr_refit_parameter_keys


def build_mhr_refit_metadata(config: MHRRefitConfig, errors: Mapping[str, list[np.ndarray]], *, enabled: bool) -> dict[str, Any] | None:
    if not enabled:
        return None
    required = ["initial_mean_error_m", "final_mean_error_m", "final_max_error_m"]
    if config.mode == MHR_REFIT_MODE_FULL and config.adaptive_rescue:
        required.extend(MHR_REFIT_ADAPTIVE_AUDIT_KEYS)
    if config.mode == MHR_REFIT_MODE_TRANSLATION_UNIFORM_SCALE:
        required.extend(("uniform_scale_direction_relative_error", "uniform_scale_final_relative_error"))
    missing = [key for key in required if not errors.get(key)]
    if missing:
        raise ValueError(f"Enabled MHR {config.mode} refit is missing error arrays: {missing}")
    concatenated = {key: np.concatenate(errors[key]) for key in required}
    for key, value in concatenated.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Enabled MHR {config.mode} refit has nonfinite {key}")
    for key in ("strong_adam_mask", "lbfgs_mask"):
        if key in concatenated and concatenated[key].dtype != np.dtype("bool"):
            raise TypeError(f"Enabled MHR adaptive refit requires Boolean {key}, got {concatenated[key].dtype}")
    optimized_parameters = mhr_refit_parameter_keys(config.mode)
    metadata = {
        "mode": config.mode,
        "optimized_parameters": list(optimized_parameters),
        "fixed_parameters": list(mhr_refit_fixed_keys(config.mode)),
        "iterations": config.iterations,
        "learning_rates": {key: config.learning_rate(key) for key in optimized_parameters},
        "sample_count": config.sample_count,
        "optimization_batch_size": config.optimization_batch_size,
        "parameter_prior_weight": config.parameter_prior_weight,
        "max_mean_error_increase_m": config.max_mean_error_increase_m,
        "mean_initial_vertex_error_m": float(concatenated["initial_mean_error_m"].mean()),
        "mean_final_vertex_error_m": float(concatenated["final_mean_error_m"].mean()),
        "max_final_vertex_error_m": float(concatenated["final_max_error_m"].max()),
    }
    if config.mode == MHR_REFIT_MODE_TRANSLATION_UNIFORM_SCALE:
        metadata.update({"uniform_scale_probe_epsilon": config.uniform_scale_probe_epsilon, "uniform_scale_direction_max_relative_error": config.uniform_scale_direction_max_relative_error, "uniform_scale_final_max_relative_error": config.uniform_scale_final_max_relative_error, "max_uniform_scale_direction_relative_error": float(concatenated["uniform_scale_direction_relative_error"].max()), "max_uniform_scale_final_relative_error": float(concatenated["uniform_scale_final_relative_error"].max())})
    if config.mode == MHR_REFIT_MODE_FULL and config.adaptive_rescue:
        metadata.update({"mean_baseline_final_vertex_error_m": float(concatenated["baseline_final_mean_error_m"].mean()), "max_baseline_final_vertex_error_m": float(concatenated["baseline_final_max_error_m"].max()), "mean_post_adam_vertex_error_m": float(concatenated["post_adam_mean_error_m"].mean()), "max_post_adam_vertex_error_m": float(concatenated["post_adam_max_error_m"].max()), "adaptive_rescue": {"enabled": True, "strong_adam": {"trigger_mean_error_m": config.strong_refit_mean_threshold_m, "iterations": config.strong_iterations, "learning_rate_multiplier": config.strong_learning_rate_multiplier, "parameter_prior_weight": config.strong_parameter_prior_weight, "cosine_final_learning_rate_ratio": config.strong_cosine_final_learning_rate_ratio, "frame_count": int(concatenated["strong_adam_mask"].sum())}, "lbfgs": {"trigger_mean_error_m": config.lbfgs_mean_threshold_m, "trigger_max_error_m": config.lbfgs_max_threshold_m, "iterations": config.lbfgs_iterations, "frame_count": int(concatenated["lbfgs_mask"].sum())}}})
    return metadata
