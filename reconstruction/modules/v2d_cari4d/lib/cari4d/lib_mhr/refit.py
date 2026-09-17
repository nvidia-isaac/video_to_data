from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Mapping


MHR_REFIT_KEYS = ("mhr_trans", "mhr_body_pose_cont", "mhr_shape", "mhr_scale")
MHR_FIXED_KEYS = ("mhr_global_rot6d", "mhr_hand", "mhr_face")
MHR_REFIT_MODE_FULL = "full"
MHR_REFIT_MODE_TRANSLATION_UNIFORM_SCALE = "translation_uniform_scale"
MHR_REFIT_MODES = (MHR_REFIT_MODE_FULL, MHR_REFIT_MODE_TRANSLATION_UNIFORM_SCALE)
MHR_REFIT_LEARNING_RATE_FIELDS = {
    "mhr_trans": "translation_learning_rate",
    "mhr_body_pose_cont": "body_pose_learning_rate",
    "mhr_shape": "shape_learning_rate",
    "mhr_scale": "scale_learning_rate",
}
MHR_REFIT_BASE_AUDIT_KEYS = ("initial_mean_error_m", "initial_max_error_m", "final_mean_error_m", "final_max_error_m")
MHR_REFIT_ADAPTIVE_AUDIT_KEYS = ("baseline_final_mean_error_m", "baseline_final_max_error_m", "post_adam_mean_error_m", "post_adam_max_error_m", "strong_adam_mask", "lbfgs_mask")
MHR_REFIT_UNIFORM_SCALE_AUDIT_KEYS = ("uniform_scale_alpha", "uniform_scale_direction", "uniform_scale_direction_relative_error", "uniform_scale_final_relative_error")


def mhr_refit_parameter_keys(mode: str) -> tuple[str, ...]:
    if mode == MHR_REFIT_MODE_FULL:
        return MHR_REFIT_KEYS
    if mode == MHR_REFIT_MODE_TRANSLATION_UNIFORM_SCALE:
        return ("mhr_trans", "mhr_scale")
    raise ValueError(f"mode must be one of {MHR_REFIT_MODES}, got {mode!r}")


def mhr_refit_fixed_keys(mode: str) -> tuple[str, ...]:
    optimized = set(mhr_refit_parameter_keys(mode))
    return tuple(key for key in MHR_REFIT_KEYS + MHR_FIXED_KEYS if key not in optimized)


def mhr_refit_audit_keys(config: "MHRRefitConfig") -> tuple[str, ...]:
    if config.mode == MHR_REFIT_MODE_TRANSLATION_UNIFORM_SCALE:
        return MHR_REFIT_BASE_AUDIT_KEYS + MHR_REFIT_UNIFORM_SCALE_AUDIT_KEYS
    if config.adaptive_rescue:
        return MHR_REFIT_BASE_AUDIT_KEYS + MHR_REFIT_ADAPTIVE_AUDIT_KEYS
    return MHR_REFIT_BASE_AUDIT_KEYS


@dataclass(frozen=True)
class MHRRefitConfig:
    mode: str = MHR_REFIT_MODE_FULL
    iterations: int = 500
    translation_learning_rate: float = 1e-3
    body_pose_learning_rate: float = 1e-3
    shape_learning_rate: float = 1e-2
    scale_learning_rate: float = 1e-2
    sample_count: int = 18439
    parameter_prior_weight: float = 1.0
    max_mean_error_increase_m: float = 1e-5
    adaptive_rescue: bool = True
    strong_refit_mean_threshold_m: float = 0.010
    strong_iterations: int = 1500
    strong_learning_rate_multiplier: float = 4.0
    strong_parameter_prior_weight: float = 0.0
    strong_cosine_final_learning_rate_ratio: float = 0.01
    lbfgs_mean_threshold_m: float = 0.003
    lbfgs_max_threshold_m: float = 0.010
    lbfgs_iterations: int = 200
    optimization_batch_size: int = 512
    uniform_scale_probe_epsilon: float = 1e-3
    uniform_scale_direction_max_relative_error: float = 0.05
    uniform_scale_final_max_relative_error: float = 0.02

    def learning_rate(self, key: str) -> float:
        return float(getattr(self, MHR_REFIT_LEARNING_RATE_FIELDS[key]))


@dataclass(frozen=True)
class MHRRefitResult:
    params: dict[str, Any]
    initial_mean_error_m: Any
    initial_max_error_m: Any
    final_mean_error_m: Any
    final_max_error_m: Any
    baseline_final_mean_error_m: Any | None = None
    baseline_final_max_error_m: Any | None = None
    post_adam_mean_error_m: Any | None = None
    post_adam_max_error_m: Any | None = None
    strong_adam_mask: Any | None = None
    lbfgs_mask: Any | None = None
    uniform_scale_alpha: Any | None = None
    uniform_scale_direction: Any | None = None
    uniform_scale_direction_relative_error: Any | None = None
    uniform_scale_final_relative_error: Any | None = None


def _validate_config(config: MHRRefitConfig) -> None:
    if config.mode not in MHR_REFIT_MODES:
        raise ValueError(f"mode must be one of {MHR_REFIT_MODES}, got {config.mode!r}")
    if config.iterations <= 0:
        raise ValueError(f"iterations must be positive, got {config.iterations}")
    for key in MHR_REFIT_KEYS:
        learning_rate = config.learning_rate(key)
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError(f"{MHR_REFIT_LEARNING_RATE_FIELDS[key]} must be positive, got {learning_rate}")
    if config.sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {config.sample_count}")
    if not math.isfinite(config.parameter_prior_weight) or config.parameter_prior_weight < 0:
        raise ValueError(f"parameter_prior_weight must be nonnegative, got {config.parameter_prior_weight}")
    if not math.isfinite(config.max_mean_error_increase_m) or config.max_mean_error_increase_m < 0:
        raise ValueError(f"max_mean_error_increase_m must be nonnegative, got {config.max_mean_error_increase_m}")
    if not isinstance(config.adaptive_rescue, bool):
        raise TypeError(f"adaptive_rescue must be bool, got {type(config.adaptive_rescue).__name__}")
    if not math.isfinite(config.strong_refit_mean_threshold_m) or config.strong_refit_mean_threshold_m < 0:
        raise ValueError(f"strong_refit_mean_threshold_m must be nonnegative, got {config.strong_refit_mean_threshold_m}")
    if config.strong_iterations <= 0:
        raise ValueError(f"strong_iterations must be positive, got {config.strong_iterations}")
    if not math.isfinite(config.strong_learning_rate_multiplier) or config.strong_learning_rate_multiplier <= 0:
        raise ValueError(f"strong_learning_rate_multiplier must be positive, got {config.strong_learning_rate_multiplier}")
    if not math.isfinite(config.strong_parameter_prior_weight) or config.strong_parameter_prior_weight < 0:
        raise ValueError(f"strong_parameter_prior_weight must be nonnegative, got {config.strong_parameter_prior_weight}")
    if not math.isfinite(config.strong_cosine_final_learning_rate_ratio) or not 0 < config.strong_cosine_final_learning_rate_ratio <= 1:
        raise ValueError(f"strong_cosine_final_learning_rate_ratio must be in (0, 1], got {config.strong_cosine_final_learning_rate_ratio}")
    if not math.isfinite(config.lbfgs_mean_threshold_m) or config.lbfgs_mean_threshold_m < 0:
        raise ValueError(f"lbfgs_mean_threshold_m must be nonnegative, got {config.lbfgs_mean_threshold_m}")
    if not math.isfinite(config.lbfgs_max_threshold_m) or config.lbfgs_max_threshold_m < 0:
        raise ValueError(f"lbfgs_max_threshold_m must be nonnegative, got {config.lbfgs_max_threshold_m}")
    if config.lbfgs_iterations <= 0:
        raise ValueError(f"lbfgs_iterations must be positive, got {config.lbfgs_iterations}")
    if config.optimization_batch_size <= 0:
        raise ValueError(f"optimization_batch_size must be positive, got {config.optimization_batch_size}")
    if not math.isfinite(config.uniform_scale_probe_epsilon) or config.uniform_scale_probe_epsilon <= 0:
        raise ValueError(f"uniform_scale_probe_epsilon must be positive, got {config.uniform_scale_probe_epsilon}")
    if not math.isfinite(config.uniform_scale_direction_max_relative_error) or config.uniform_scale_direction_max_relative_error < 0:
        raise ValueError(f"uniform_scale_direction_max_relative_error must be nonnegative, got {config.uniform_scale_direction_max_relative_error}")
    if not math.isfinite(config.uniform_scale_final_max_relative_error) or config.uniform_scale_final_max_relative_error < 0:
        raise ValueError(f"uniform_scale_final_max_relative_error must be nonnegative, got {config.uniform_scale_final_max_relative_error}")


def _clone_tensor_params(params: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    cloned = {}
    for key, value in params.items():
        if not torch.is_tensor(value):
            raise TypeError(f"{key} must be a torch.Tensor, got {type(value).__name__}")
        cloned[key] = value.detach().clone()
    return cloned


def _prepare_vertices_context(mhr_layer: Any, params: Mapping[str, Any]) -> Any | None:
    prepare = getattr(mhr_layer, "prepare_mhr_vertices", None)
    return prepare(params) if callable(prepare) else None


def _decode_vertices(mhr_layer: Any, params: Mapping[str, Any], context: Any | None = None) -> Any:
    decode = getattr(mhr_layer, "mhr_forward_vertices", None)
    return decode(params, context=context) if callable(decode) else mhr_layer.mhr_forward(params).vertices


def _vertex_errors(vertices: Any, target_vertices: Any) -> tuple[Any, Any]:
    import torch

    point_errors = torch.linalg.vector_norm(vertices - target_vertices, dim=-1)
    return point_errors.mean(dim=-1), point_errors.amax(dim=-1)


def _sample_vertex_indices(vertex_count: int, sample_count: int, device: Any) -> Any:
    import torch

    count = min(vertex_count, sample_count)
    if count == vertex_count:
        return torch.arange(vertex_count, device=device)
    return torch.linspace(0, vertex_count - 1, steps=count, device=device).round().long().unique()


def _full_objective(mhr_layer: Any, base_params: Mapping[str, Any], optimized_values: Mapping[str, Any], initial_values: Mapping[str, Any] | None, sampled_target: Any, sample_indices: Any, prior_weight: float, stage: str, *, context: Any | None = None, vertices_only: bool = True) -> tuple[Any, Any, Any]:
    import torch

    current_params = dict(base_params)
    current_params.update(optimized_values)
    vertices = _decode_vertices(mhr_layer, current_params, context=context) if vertices_only else mhr_layer.mhr_forward(current_params).vertices
    sampled_vertices = vertices.index_select(-2, sample_indices)
    squared_residual = (sampled_vertices - sampled_target).square().sum(dim=-1)
    objective = squared_residual.mean(dim=-1)
    if prior_weight:
        if initial_values is None:
            raise ValueError("initial_values are required when the parameter prior weight is nonzero")
        prior = sum((optimized_values[key] - initial_values[key]).square().flatten(start_dim=1).mean(dim=-1) for key in MHR_REFIT_KEYS)
        objective = objective + prior_weight * prior
    if not torch.isfinite(objective).all():
        raise FloatingPointError(f"MHR {stage} produced nonfinite per-frame objective")
    return objective, vertices, squared_residual


def _snapshot_best_values(best_values: dict[str, Any], optimized_values: Mapping[str, Any], improved: Any) -> None:
    import torch

    for key in MHR_REFIT_KEYS:
        mask = improved.reshape((improved.shape[0],) + (1,) * (optimized_values[key].ndim - 1))
        best_values[key] = torch.where(mask, optimized_values[key].detach(), best_values[key])


def _mean_vertex_error(vertices: Any, target: Any) -> Any:
    return (vertices - target).square().sum(dim=-1).sqrt().mean(dim=-1)


def _selection_error(vertices: Any, target: Any, squared_residual: Any, sample_indices: Any) -> Any:
    if sample_indices.numel() == target.shape[-2]:
        return squared_residual.sqrt().mean(dim=-1)
    return _mean_vertex_error(vertices, target)


def _run_full_adam(mhr_layer: Any, params: Mapping[str, Any], sampled_target: Any, selection_target: Any, sample_indices: Any, config: MHRRefitConfig, *, iterations: int, learning_rate_multiplier: float, prior_weight: float, cosine_final_learning_rate_ratio: float | None, keep_best: bool, stage: str, vertices_only: bool = True) -> dict[str, Any]:
    import torch

    initial_values = {key: params[key].detach().clone() for key in MHR_REFIT_KEYS} if prior_weight else None
    optimized_values = {key: params[key].detach().clone().requires_grad_(True) for key in MHR_REFIT_KEYS}
    context = _prepare_vertices_context(mhr_layer, params) if vertices_only else None
    optimizer = torch.optim.Adam([{"params": [optimized_values[key]], "lr": config.learning_rate(key) * learning_rate_multiplier} for key in MHR_REFIT_KEYS])
    scheduler = None
    if cosine_final_learning_rate_ratio is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_final_learning_rate_ratio + (1.0 - cosine_final_learning_rate_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(step, iterations) / iterations)))
    best_error = None
    best_values = None
    if keep_best:
        with torch.no_grad():
            _, vertices, squared_residual = _full_objective(mhr_layer, params, optimized_values, initial_values, sampled_target, sample_indices, prior_weight, stage, context=context, vertices_only=vertices_only)
            best_error = _selection_error(vertices, selection_target, squared_residual, sample_indices)
        best_error = best_error.detach().clone()
        best_values = {key: value.detach().clone() for key, value in optimized_values.items()}
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        objective, vertices, squared_residual = _full_objective(mhr_layer, params, optimized_values, initial_values, sampled_target, sample_indices, prior_weight, stage, context=context, vertices_only=vertices_only)
        if keep_best:
            error = _selection_error(vertices.detach(), selection_target, squared_residual.detach(), sample_indices)
            improved = error < best_error
            _snapshot_best_values(best_values, optimized_values, improved)
            best_error = torch.where(improved, error, best_error)
        objective.sum().backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    if keep_best:
        with torch.no_grad():
            current_params = dict(params)
            current_params.update(optimized_values)
            final_vertices = _decode_vertices(mhr_layer, current_params, context=context) if vertices_only else mhr_layer.mhr_forward(current_params).vertices
            if not torch.isfinite(final_vertices).all():
                raise FloatingPointError(f"MHR {stage} final decoded vertices contain nonfinite values")
            error = _mean_vertex_error(final_vertices, selection_target)
        improved = error < best_error
        _snapshot_best_values(best_values, optimized_values, improved)
        return {key: best_values[key].detach().clone() for key in MHR_REFIT_KEYS}
    return {key: optimized_values[key].detach().clone() for key in MHR_REFIT_KEYS}


def _run_full_lbfgs_frame(mhr_layer: Any, original_params: Mapping[str, Any], start_params: Mapping[str, Any], sampled_target: Any, selection_target: Any, sample_indices: Any, config: MHRRefitConfig, *, frame_index: int | None = None) -> dict[str, Any]:
    import torch

    initial_values = {key: original_params[key].detach().clone() for key in MHR_REFIT_KEYS} if config.strong_parameter_prior_weight else None
    optimized_values = {key: start_params[key].detach().clone().requires_grad_(True) for key in MHR_REFIT_KEYS}
    context = _prepare_vertices_context(mhr_layer, original_params)
    with torch.no_grad():
        _, vertices, squared_residual = _full_objective(mhr_layer, original_params, optimized_values, initial_values, sampled_target, sample_indices, config.strong_parameter_prior_weight, "L-BFGS rescue", context=context)
        best_error = _selection_error(vertices, selection_target, squared_residual, sample_indices).detach().clone()
    best_values = {key: value.detach().clone() for key, value in optimized_values.items()}
    optimizer = torch.optim.LBFGS(list(optimized_values.values()), lr=1.0, max_iter=config.lbfgs_iterations, max_eval=config.lbfgs_iterations * 2, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")

    def closure() -> Any:
        nonlocal best_error, best_values
        optimizer.zero_grad(set_to_none=True)
        objective, vertices, squared_residual = _full_objective(mhr_layer, original_params, optimized_values, initial_values, sampled_target, sample_indices, config.strong_parameter_prior_weight, "L-BFGS rescue", context=context)
        error = _selection_error(vertices.detach(), selection_target, squared_residual.detach(), sample_indices)
        improved = error < best_error
        _snapshot_best_values(best_values, optimized_values, improved)
        best_error = torch.where(improved, error, best_error)
        objective.sum().backward()
        return objective.sum()

    try:
        optimizer.step(closure)
    except FloatingPointError as exc:
        frame_label = f" for frame {frame_index}" if frame_index is not None else ""
        warnings.warn(f"MHR L-BFGS rescue encountered a nonfinite exploratory step{frame_label}; retaining the lowest-error finite iterate", RuntimeWarning)
        return best_values
    with torch.no_grad():
        _, vertices, squared_residual = _full_objective(mhr_layer, original_params, optimized_values, initial_values, sampled_target, sample_indices, config.strong_parameter_prior_weight, "L-BFGS rescue", context=context)
        error = _selection_error(vertices, selection_target, squared_residual, sample_indices)
    _snapshot_best_values(best_values, optimized_values, error < best_error)
    return best_values


def _select_batch(params: Mapping[str, Any], indices: Any) -> dict[str, Any]:
    return {key: value.index_select(0, indices).detach().clone() for key, value in params.items()}


def _merge_batch(params: dict[str, Any], selected_params: Mapping[str, Any], indices: Any) -> None:
    for key in MHR_REFIT_KEYS:
        params[key].index_copy_(0, indices, selected_params[key])


def _merge_better_batch(params: dict[str, Any], candidate_params: Mapping[str, Any], indices: Any, improved: Any) -> None:
    for key in MHR_REFIT_KEYS:
        current = params[key].index_select(0, indices)
        mask = improved.reshape((improved.shape[0],) + (1,) * (current.ndim - 1))
        params[key].index_copy_(0, indices, current.where(~mask, candidate_params[key]))


def _index_batches(indices: Any, batch_size: int):
    for start in range(0, int(indices.numel()), batch_size):
        yield indices[start:start + batch_size]


def _decode_vertices_batched(mhr_layer: Any, params: Mapping[str, Any], batch_size: int) -> Any:
    import torch

    frame_count = int(next(iter(params.values())).shape[0])
    indices = torch.arange(frame_count, device=next(iter(params.values())).device)
    decoded = []
    for batch_indices in _index_batches(indices, batch_size):
        selected = _select_batch(params, batch_indices)
        context = _prepare_vertices_context(mhr_layer, selected)
        decoded.append(_decode_vertices(mhr_layer, selected, context=context))
    return torch.cat(decoded, dim=0)


def _vertex_errors_batched(mhr_layer: Any, params: Mapping[str, Any], target_vertices: Any, batch_size: int, stage: str) -> tuple[Any, Any]:
    import torch

    indices = torch.arange(target_vertices.shape[0], device=target_vertices.device)
    means = []
    maxima = []
    with torch.no_grad():
        for batch_indices in _index_batches(indices, batch_size):
            selected = _select_batch(params, batch_indices)
            context = _prepare_vertices_context(mhr_layer, selected)
            vertices = _decode_vertices(mhr_layer, selected, context=context)
            selected_target = target_vertices.index_select(0, batch_indices)
            if vertices.shape != selected_target.shape:
                raise ValueError(f"decoded vertices shape {tuple(vertices.shape)} does not match target {tuple(selected_target.shape)}")
            if not torch.isfinite(vertices).all():
                raise FloatingPointError(f"MHR {stage} decoded vertices contain nonfinite values")
            mean_error, max_error = _vertex_errors(vertices, selected_target)
            means.append(mean_error)
            maxima.append(max_error)
    return torch.cat(means, dim=0), torch.cat(maxima, dim=0)


def _derive_uniform_scale_direction(mhr_layer: Any, params: Mapping[str, Any], initial_vertices: Any, sample_indices: Any, config: MHRRefitConfig) -> tuple[Any, Any]:
    import torch

    initial_sample = initial_vertices.index_select(-2, sample_indices)
    centered_initial = initial_sample - initial_sample.mean(dim=-2, keepdim=True)
    derivatives = []
    with torch.no_grad():
        for scale_index in range(params["mhr_scale"].shape[-1]):
            perturbed_params = dict(params)
            perturbed_scale = params["mhr_scale"].clone()
            perturbed_scale[..., scale_index] += config.uniform_scale_probe_epsilon
            perturbed_params["mhr_scale"] = perturbed_scale
            perturbed_vertices = mhr_layer.mhr_forward(perturbed_params).vertices.index_select(-2, sample_indices)
            derivative = (perturbed_vertices - initial_sample) / config.uniform_scale_probe_epsilon
            derivatives.append(derivative - derivative.mean(dim=-2, keepdim=True))
    jacobian = torch.stack(derivatives, dim=-1).flatten(start_dim=-3, end_dim=-2)
    target = centered_initial.flatten(start_dim=-2)
    jacobian_transpose = jacobian.transpose(-2, -1)
    normal_matrix = torch.matmul(jacobian_transpose, jacobian)
    normal_rhs = torch.matmul(jacobian_transpose, target.unsqueeze(-1))
    direction = torch.matmul(torch.linalg.pinv(normal_matrix, hermitian=True), normal_rhs).squeeze(-1)
    represented = torch.matmul(jacobian, direction.unsqueeze(-1)).squeeze(-1)
    relative_error = torch.linalg.vector_norm(represented - target, dim=-1) / torch.linalg.vector_norm(target, dim=-1).clamp_min(torch.finfo(target.dtype).eps)
    if not torch.isfinite(direction).all() or not torch.isfinite(relative_error).all():
        raise FloatingPointError("MHR uniform scale direction probe produced nonfinite values")
    failing = torch.nonzero(relative_error > config.uniform_scale_direction_max_relative_error, as_tuple=False).flatten()
    if failing.numel():
        first = failing[:10].detach().cpu().tolist()
        maximum = float(relative_error.amax().detach())
        raise RuntimeError(f"MHR scale space lacks a sufficiently accurate uniform scale direction for {failing.numel()} frames; maximum relative error {maximum:.6f} exceeds {config.uniform_scale_direction_max_relative_error:.6f}; first indices {first}")
    return direction.detach(), relative_error.detach()


def _uniform_scale_residual(initial_vertices: Any, final_vertices: Any) -> Any:
    import torch

    initial_centered = initial_vertices - initial_vertices.mean(dim=-2, keepdim=True)
    final_centered = final_vertices - final_vertices.mean(dim=-2, keepdim=True)
    scale = (initial_centered * final_centered).sum(dim=(-2, -1)) / initial_centered.square().sum(dim=(-2, -1)).clamp_min(torch.finfo(initial_vertices.dtype).eps)
    residual = final_centered - scale[..., None, None] * initial_centered
    return torch.linalg.vector_norm(residual.flatten(start_dim=-2), dim=-1) / torch.linalg.vector_norm(final_centered.flatten(start_dim=-2), dim=-1).clamp_min(torch.finfo(final_vertices.dtype).eps)


def refit_mhr_to_vertices(mhr_layer: Any, params: Mapping[str, Any], target_vertices: Any, *, config: MHRRefitConfig | None = None) -> MHRRefitResult:
    """Refit canonical MHR motion and identity parameters to target vertices."""

    import torch

    config = config or MHRRefitConfig()
    _validate_config(config)
    if not torch.is_tensor(target_vertices):
        raise TypeError(f"target_vertices must be a torch.Tensor, got {type(target_vertices).__name__}")
    if target_vertices.ndim < 3 or target_vertices.shape[-1] != 3:
        raise ValueError(f"target_vertices must have shape (..., V, 3), got {tuple(target_vertices.shape)}")
    if not torch.isfinite(target_vertices).all():
        raise ValueError("target_vertices must contain only finite values")

    fitted_params = _clone_tensor_params(params)
    original_params = _clone_tensor_params(params)
    for key in MHR_REFIT_KEYS + MHR_FIXED_KEYS:
        if key not in fitted_params:
            raise KeyError(f"MHR full refit requires {key}")
        if not fitted_params[key].dtype.is_floating_point:
            raise TypeError(f"{key} must use a floating-point dtype")
    reference = fitted_params[MHR_REFIT_KEYS[0]]
    for key in MHR_REFIT_KEYS + MHR_FIXED_KEYS:
        if fitted_params[key].device != reference.device or fitted_params[key].dtype != reference.dtype:
            raise ValueError(f"{key} must share device and dtype with {MHR_REFIT_KEYS[0]}")

    target_vertices = target_vertices.to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(target_vertices).all():
        raise ValueError("target_vertices became nonfinite after conversion to the parameter dtype")
    if config.mode == MHR_REFIT_MODE_FULL:
        initial_vertices = None
        initial_mean_error_m, initial_max_error_m = _vertex_errors_batched(mhr_layer, fitted_params, target_vertices, config.optimization_batch_size, f"{config.mode} refit initial")
    else:
        with torch.no_grad():
            initial_vertices = _decode_vertices_batched(mhr_layer, fitted_params, config.optimization_batch_size)
            if initial_vertices.shape != target_vertices.shape:
                raise ValueError(f"decoded vertices shape {tuple(initial_vertices.shape)} does not match target {tuple(target_vertices.shape)}")
            if not torch.isfinite(initial_vertices).all():
                raise FloatingPointError(f"MHR {config.mode} refit initial decoded vertices contain nonfinite values")
            initial_mean_error_m, initial_max_error_m = _vertex_errors(initial_vertices, target_vertices)

    sample_indices = _sample_vertex_indices(target_vertices.shape[-2], config.sample_count, target_vertices.device)
    sampled_target = target_vertices.index_select(-2, sample_indices)
    uniform_scale_alpha = None
    uniform_scale_direction = None
    uniform_scale_direction_relative_error = None
    baseline_final_mean_error_m = None
    baseline_final_max_error_m = None
    post_adam_mean_error_m = None
    post_adam_max_error_m = None
    strong_adam_mask = None
    lbfgs_mask = None
    if config.mode == MHR_REFIT_MODE_FULL:
        all_indices = torch.arange(target_vertices.shape[0], device=target_vertices.device)
        for batch_indices in _index_batches(all_indices, config.optimization_batch_size):
            selected_params = _select_batch(fitted_params, batch_indices)
            selected_sampled_target = sampled_target.index_select(0, batch_indices)
            selected_target = target_vertices.index_select(0, batch_indices)
            selected_values = _run_full_adam(mhr_layer, selected_params, selected_sampled_target, selected_target, sample_indices, config, iterations=config.iterations, learning_rate_multiplier=1.0, prior_weight=config.parameter_prior_weight, cosine_final_learning_rate_ratio=None, keep_best=True, stage="baseline full refit")
            selected_params.update(selected_values)
            _merge_batch(fitted_params, selected_params, batch_indices)
        baseline_final_mean_error_m, baseline_final_max_error_m = _vertex_errors_batched(mhr_layer, fitted_params, target_vertices, config.optimization_batch_size, "baseline full refit final")
        strong_adam_mask = torch.zeros_like(baseline_final_mean_error_m, dtype=torch.bool)
        lbfgs_mask = torch.zeros_like(baseline_final_mean_error_m, dtype=torch.bool)
        if config.adaptive_rescue:
            strong_adam_mask = baseline_final_mean_error_m > config.strong_refit_mean_threshold_m
            strong_indices = torch.nonzero(strong_adam_mask, as_tuple=False).flatten()
            for batch_indices in _index_batches(strong_indices, config.optimization_batch_size):
                original_selected = _select_batch(original_params, batch_indices)
                selected_target = sampled_target.index_select(0, batch_indices)
                selection_target = target_vertices.index_select(0, batch_indices)
                strong_values = _run_full_adam(mhr_layer, original_selected, selected_target, selection_target, sample_indices, config, iterations=config.strong_iterations, learning_rate_multiplier=config.strong_learning_rate_multiplier, prior_weight=config.strong_parameter_prior_weight, cosine_final_learning_rate_ratio=config.strong_cosine_final_learning_rate_ratio, keep_best=True, stage="strong Adam rescue")
                strong_params = dict(original_selected)
                strong_params.update(strong_values)
                with torch.no_grad():
                    context = _prepare_vertices_context(mhr_layer, strong_params)
                    strong_vertices = _decode_vertices(mhr_layer, strong_params, context=context)
                    strong_error = _mean_vertex_error(strong_vertices, selection_target)
                baseline_error = baseline_final_mean_error_m.index_select(0, batch_indices)
                _merge_better_batch(fitted_params, strong_params, batch_indices, strong_error < baseline_error)
            post_adam_mean_error_m, post_adam_max_error_m = _vertex_errors_batched(mhr_layer, fitted_params, target_vertices, config.optimization_batch_size, "strong Adam rescue")
            lbfgs_mask = strong_adam_mask & ((post_adam_mean_error_m > config.lbfgs_mean_threshold_m) | (post_adam_max_error_m > config.lbfgs_max_threshold_m))
            for frame_index in torch.nonzero(lbfgs_mask, as_tuple=False).flatten():
                selected_index = frame_index.reshape(1)
                original_frame = _select_batch(original_params, selected_index)
                start_frame = _select_batch(fitted_params, selected_index)
                target_frame = sampled_target.index_select(0, selected_index)
                selection_frame = target_vertices.index_select(0, selected_index)
                lbfgs_values = _run_full_lbfgs_frame(mhr_layer, original_frame, start_frame, target_frame, selection_frame, sample_indices, config, frame_index=int(frame_index))
                lbfgs_params = dict(original_frame)
                lbfgs_params.update(lbfgs_values)
                _merge_batch(fitted_params, lbfgs_params, selected_index)
        else:
            post_adam_mean_error_m = baseline_final_mean_error_m.detach().clone()
            post_adam_max_error_m = baseline_final_max_error_m.detach().clone()
    else:
        uniform_scale_direction, uniform_scale_direction_relative_error = _derive_uniform_scale_direction(mhr_layer, fitted_params, initial_vertices, sample_indices, config)
        initial_scale = fitted_params["mhr_scale"].detach().clone()
        initial_translation = fitted_params["mhr_trans"].detach().clone()
        uniform_scale_alpha = torch.zeros_like(initial_scale[..., 0], requires_grad=True)
        translation_delta = torch.zeros_like(initial_translation, requires_grad=True)
        optimizer = torch.optim.Adam([{"params": [uniform_scale_alpha], "lr": config.scale_learning_rate}, {"params": [translation_delta], "lr": config.translation_learning_rate}])
        for _ in range(config.iterations):
            optimizer.zero_grad(set_to_none=True)
            current_params = dict(fitted_params)
            current_params["mhr_scale"] = initial_scale + uniform_scale_alpha.unsqueeze(-1) * uniform_scale_direction
            current_params["mhr_trans"] = initial_translation + translation_delta
            sampled_vertices = mhr_layer.mhr_forward(current_params).vertices.index_select(-2, sample_indices)
            reconstruction_loss = (sampled_vertices - sampled_target).square().sum(dim=-1).mean(dim=-1).sum()
            prior_loss = uniform_scale_alpha.square().sum() + translation_delta.square().mean(dim=-1).sum()
            loss = reconstruction_loss + config.parameter_prior_weight * prior_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"MHR translation and uniform-scale refit produced nonfinite loss: {float(loss.detach())}")
            loss.backward()
            optimizer.step()
        fitted_params["mhr_scale"] = (initial_scale + uniform_scale_alpha.detach().unsqueeze(-1) * uniform_scale_direction).clone()
        fitted_params["mhr_trans"] = (initial_translation + translation_delta.detach()).clone()
        uniform_scale_alpha = uniform_scale_alpha.detach().clone()
    for key in MHR_REFIT_KEYS:
        if not torch.isfinite(fitted_params[key]).all():
            raise FloatingPointError(f"MHR {config.mode} refit produced nonfinite final parameters for {key}")
    with torch.no_grad():
        if config.mode == MHR_REFIT_MODE_FULL:
            final_vertices = None
            final_mean_error_m, final_max_error_m = _vertex_errors_batched(mhr_layer, fitted_params, target_vertices, config.optimization_batch_size, f"{config.mode} refit final")
        else:
            final_vertices = _decode_vertices_batched(mhr_layer, fitted_params, config.optimization_batch_size)
            if final_vertices.shape != target_vertices.shape:
                raise ValueError(f"final decoded vertices shape {tuple(final_vertices.shape)} does not match target {tuple(target_vertices.shape)}")
            if not torch.isfinite(final_vertices).all():
                raise FloatingPointError(f"MHR {config.mode} refit final decoded vertices contain nonfinite values")
            final_mean_error_m, final_max_error_m = _vertex_errors(final_vertices, target_vertices)
        if not torch.isfinite(final_mean_error_m).all() or not torch.isfinite(final_max_error_m).all():
            raise FloatingPointError("MHR full refit final errors contain nonfinite values")
        worsened = torch.nonzero(final_mean_error_m > initial_mean_error_m + config.max_mean_error_increase_m, as_tuple=False).flatten()
        if worsened.numel():
            raise RuntimeError(f"MHR {config.mode} refit worsened {worsened.numel()} frames by more than {config.max_mean_error_increase_m} m; first indices {worsened[:10].detach().cpu().tolist()}")
        uniform_scale_final_relative_error = None
        if config.mode == MHR_REFIT_MODE_TRANSLATION_UNIFORM_SCALE:
            uniform_scale_final_relative_error = _uniform_scale_residual(initial_vertices, final_vertices)
            failing = torch.nonzero(uniform_scale_final_relative_error > config.uniform_scale_final_max_relative_error, as_tuple=False).flatten()
            if failing.numel():
                maximum = float(uniform_scale_final_relative_error.amax().detach())
                raise RuntimeError(f"MHR constrained refit decoded nonuniform geometry for {failing.numel()} frames; maximum relative error {maximum:.6f} exceeds {config.uniform_scale_final_max_relative_error:.6f}; first indices {failing[:10].detach().cpu().tolist()}")
    return MHRRefitResult(params=fitted_params, initial_mean_error_m=initial_mean_error_m.detach(), initial_max_error_m=initial_max_error_m.detach(), final_mean_error_m=final_mean_error_m.detach(), final_max_error_m=final_max_error_m.detach(), baseline_final_mean_error_m=baseline_final_mean_error_m, baseline_final_max_error_m=baseline_final_max_error_m, post_adam_mean_error_m=post_adam_mean_error_m, post_adam_max_error_m=post_adam_max_error_m, strong_adam_mask=strong_adam_mask, lbfgs_mask=lbfgs_mask, uniform_scale_alpha=uniform_scale_alpha, uniform_scale_direction=uniform_scale_direction, uniform_scale_direction_relative_error=uniform_scale_direction_relative_error, uniform_scale_final_relative_error=uniform_scale_final_relative_error)


MHRIdentityRefitConfig = MHRRefitConfig
MHRIdentityRefitResult = MHRRefitResult
refit_mhr_identity_to_vertices = refit_mhr_to_vertices
