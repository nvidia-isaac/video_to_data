from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from lib_mhr.body_pose import BODY_CONT_INTERNAL_TRANSLATION_SLICE
from lib_mhr.collision_proxy import DEFAULT_MHR_COLLISION_PROXY_ASSET, MHR_COLLISION_PROXY_MODE_4000, MHR_COLLISION_PROXY_MODES, apply_mhr_collision_proxy, load_mhr_collision_proxy
from lib_mhr.contact import MHR_WRIST_INDICES, OBJECT_MESH_SCENE_LOADER_REVISION, load_object_mesh, sample_mesh_surface_points
from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
from lib_mhr.mhr_layer import MHRLayer, MHRLayerOutput, mhr70_to_coco17
from lib_mhr.postopt_crop import POSTOPT_CROP_CONTRACT, POSTOPT_RENDER_SIZE
from lib_mhr.schema import MHR_PARAM_DIMS
from lib_mhr.sdf import object_inside_human_penetration_loss, object_sdf_penetration_loss, validate_closed_triangle_mesh
from tools.pipeline_timing import PipelineTimer


STAGE_NAMES = {
    1: "object_only",
    2: "human_translation",
    3: "human_root_rotation",
    4: "human_hand_body",
    5: "object_inside_mhr_penetration",
}
STAGE_ORDER = tuple(sorted(STAGE_NAMES))
DEFAULT_HAND_SURFACE_SPEC = Path(__file__).resolve().parents[2] / "lib_mhr" / "assets" / "mhr_hand_surface_spec.npz"
MHR_DEFAULT_PENETRATION_WEIGHT = 2.0
MHR_DEFAULT_CONTACT_ACTIVATION_DISTANCE_M = 0.05
MHR_CONTACT_ACTIVATION_REVISION = "network-active-and-initial-surface-distance-lt-v1"
MHR_CONTACT_ACTIVATION_FRAME_CHUNK_SIZE = 64
MHR_BODY_ROTATION_CONTROL_COUNT = BODY_CONT_INTERNAL_TRANSLATION_SLICE.start
MHR_FOOT_DIAGNOSTIC_VERTICES_PER_SIDE = 128
MHR_PARITY_POSTOPT_CHECKPOINT_SCHEMA = "cari4d.mhr_postopt_smplh_parity.v10"
MHR_POSTOPT_BATCH_SAMPLING = "shuffled_contiguous_window_epoch_v1"
MHR_POSTOPT_FULL_CLIP_BATCH_SAMPLING = "full_clip_v1"
MHR_POSTOPT_FULL_CLIP_BATCH_SIZE = 0
MHR_POSTOPT_2D_OBSERVATION_KEYS = frozenset(("mhr_coco17_2d", "mhr_coco17_conf", "mhr_coco17_2d_full", "mhr_coco17_conf_full", "coco17_2d", "coco17_conf", "joints2d_coco17", "joints2d_coco17_conf", "j2d_source"))


@dataclass
class MHRPostOptConfig:
    """Configuration for native MHR post-CoCoNet optimization."""

    stage: int = 1
    iterations: int = 100
    device: str = "cuda"
    frame_start: int = 0
    frame_limit: int = 0
    object_surface_samples: int = 4096
    penetration_surface_samples: int = 6000
    penetration_frame_chunk_size: int = 8
    penetration_human_mesh_mode: str = MHR_COLLISION_PROXY_MODE_4000
    penetration_collision_proxy_path: str = str(DEFAULT_MHR_COLLISION_PROXY_ASSET)
    penetration_bbox_rejection: bool = True
    penetration_bbox_margin_m: float = 1e-6
    human_surface_samples: int = 512
    contact_topk: int = 64
    contact_sigmoid_threshold: float = 0.2
    contact_logit_temperature: float = 1.0
    contact_selection: str = "binary_logits"
    contact_activation_distance_m: float = MHR_DEFAULT_CONTACT_ACTIVATION_DISTANCE_M
    contact_distance_backend: str = "triangle_surface"
    hand_surface_spec_path: str = str(DEFAULT_HAND_SURFACE_SPEC)
    lr_obj_rot: float = 1e-3
    lr_obj_trans: float = 1e-3
    lr_mhr_trans: float = 1e-4
    lr_mhr_root: float = 1e-5
    lr_mhr_hand: float = 1e-5
    lr_mhr_body: float = 1e-5
    w_contact: float = 1.0
    w_obj_pose_prior: float = 0.02
    w_obj_smooth: float = 0.01
    w_static_object: float = 0.02
    w_mhr_trans_prior: float = 10.0
    w_mhr_root_prior: float = 1.0
    w_mhr_hand_prior: float = 0.1
    w_mhr_body_prior: float = 0.1
    w_mhr_smooth: float = 0.1
    w_sdf: float = 0.0
    w_pen: float = MHR_DEFAULT_PENETRATION_WEIGHT
    report_every: int = 10

    def checked(self) -> "MHRPostOptConfig":
        if self.stage not in STAGE_NAMES:
            raise ValueError(f"stage must be one of {sorted(STAGE_NAMES)}, got {self.stage}")
        if self.iterations < 0:
            raise ValueError(f"iterations must be non-negative, got {self.iterations}")
        if self.object_surface_samples <= 0:
            raise ValueError("object_surface_samples must be positive")
        if self.human_surface_samples <= 0:
            raise ValueError("human_surface_samples must be positive")
        if self.penetration_surface_samples <= 0:
            raise ValueError("penetration_surface_samples must be positive")
        if self.penetration_frame_chunk_size <= 0:
            raise ValueError("penetration_frame_chunk_size must be positive")
        if self.penetration_human_mesh_mode not in MHR_COLLISION_PROXY_MODES:
            raise ValueError(f"unknown penetration_human_mesh_mode {self.penetration_human_mesh_mode!r}")
        if self.penetration_bbox_margin_m < 0:
            raise ValueError(f"penetration_bbox_margin_m must be nonnegative, got {self.penetration_bbox_margin_m}")
        if self.contact_selection not in {"binary_logits", "legacy_soft_topk"}:
            raise ValueError(f"unknown contact_selection {self.contact_selection!r}")
        if not np.isfinite(self.contact_activation_distance_m) or self.contact_activation_distance_m <= 0:
            raise ValueError(f"contact_activation_distance_m must be finite and positive, got {self.contact_activation_distance_m}")
        if self.contact_distance_backend not in {"triangle_surface", "sampled_points"}:
            raise ValueError(f"unknown contact_distance_backend {self.contact_distance_backend!r}")
        return self


@dataclass
class MHRParityPostOptConfig:
    """MHR-native mapping of the public SMPL-H post-optimization protocol."""

    num_steps: int = 300
    batch_size: int = MHR_POSTOPT_FULL_CLIP_BATCH_SIZE
    device: str = "cuda"
    frame_start: int = 0
    frame_limit: int = 0
    lr: float = 1e-3
    warmup_steps: int = 30
    schedule_steps: int = 450
    object_surface_samples: int = 1000
    penetration_surface_samples: int = 1000
    penetration_point_reduction: str = "sum"
    penetration_human_mesh_mode: str = MHR_COLLISION_PROXY_MODE_4000
    penetration_collision_proxy_path: str = str(DEFAULT_MHR_COLLISION_PROXY_ASSET)
    penetration_bbox_rejection: bool = True
    penetration_bbox_margin_m: float = 1e-6
    hand_surface_mode: str = "all_vertices"
    hand_surface_spec_path: str = str(DEFAULT_HAND_SURFACE_SPEC)
    contact_activation_distance_m: float = MHR_DEFAULT_CONTACT_ACTIVATION_DISTANCE_M
    w_contact: float = 200.0
    w_silhouette: float = 0.002
    w_penetration: float = MHR_DEFAULT_PENETRATION_WEIGHT
    w_human_pose_prior: float = 200.0
    human_pose_prior_beta: float = 0.05
    w_temporal: float = 100.0
    w_object_translation_prior: float = 100.0
    penetration_start_fraction: float = 0.6
    symmetric_object: bool = False
    random_seed: int = 0
    report_every: int = 50
    diagnostics_every: int = 500
    save_every: int = 500
    checkpoint_path: str | None = None
    freeze_object_rotation: bool = True
    freeze_body_internal_translations: bool = True

    def checked(self) -> "MHRParityPostOptConfig":
        if self.num_steps < 0 or self.batch_size < 0:
            raise ValueError(f"num_steps and batch_size must be nonnegative, got {self.num_steps} and {self.batch_size}; batch_size=0 selects the full clip")
        if self.lr < 0 or self.warmup_steps < 0 or self.schedule_steps <= self.warmup_steps:
            raise ValueError(f"invalid optimizer schedule: lr={self.lr}, warmup={self.warmup_steps}, horizon={self.schedule_steps}")
        if self.object_surface_samples <= 0 or self.penetration_surface_samples <= 0:
            raise ValueError("object and penetration surface sample counts must be positive")
        if self.penetration_point_reduction not in {"mean", "sum"}:
            raise ValueError(f"unknown penetration point reduction {self.penetration_point_reduction!r}")
        if self.penetration_human_mesh_mode not in MHR_COLLISION_PROXY_MODES:
            raise ValueError(f"unknown penetration_human_mesh_mode {self.penetration_human_mesh_mode!r}")
        if self.penetration_bbox_margin_m < 0:
            raise ValueError(f"penetration_bbox_margin_m must be nonnegative, got {self.penetration_bbox_margin_m}")
        if self.hand_surface_mode not in {"all_vertices", "deterministic_256"}:
            raise ValueError(f"unknown hand_surface_mode {self.hand_surface_mode!r}")
        if not np.isfinite(self.contact_activation_distance_m) or self.contact_activation_distance_m <= 0:
            raise ValueError(f"contact_activation_distance_m must be finite and positive, got {self.contact_activation_distance_m}")
        if not 0.0 <= self.penetration_start_fraction <= 1.0:
            raise ValueError(f"penetration_start_fraction must be in [0,1], got {self.penetration_start_fraction}")
        if not np.isfinite(self.w_human_pose_prior) or self.w_human_pose_prior < 0:
            raise ValueError(f"w_human_pose_prior must be finite and nonnegative, got {self.w_human_pose_prior}")
        if not np.isfinite(self.human_pose_prior_beta) or self.human_pose_prior_beta <= 0:
            raise ValueError(f"human_pose_prior_beta must be finite and positive, got {self.human_pose_prior_beta}")
        if self.report_every <= 0 or self.diagnostics_every <= 0 or self.save_every < 0:
            raise ValueError(f"report_every and diagnostics_every must be positive and save_every must be nonnegative, got {self.report_every}, {self.diagnostics_every}, {self.save_every}")
        return self


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _should_run_postopt_diagnostics(step: int, start_step: int, num_steps: int, diagnostics_every: int) -> bool:
    if diagnostics_every <= 0:
        raise ValueError(f"diagnostics_every must be positive, got {diagnostics_every}")
    return step == start_step or step == num_steps or step % diagnostics_every == 0


def points_world_to_object(points_world: Any, obj_rot: Any, obj_t: Any) -> Any:
    if hasattr(points_world, "detach"):
        return (points_world - obj_t[..., None, :]) @ obj_rot
    return (np.asarray(points_world) - np.asarray(obj_t)[..., None, :]) @ np.asarray(obj_rot)


def object_sdf_penetration_from_world(
    points_world: Any,
    obj_rot: Any,
    obj_t: Any,
    sdf_fn: Callable[[Any], Any],
    *,
    weight: float = 1.0,
) -> Any:
    if weight == 0:
        if hasattr(points_world, "detach"):
            return points_world.sum() * 0
        return 0.0
    points_obj = points_world_to_object(points_world, obj_rot, obj_t)
    return object_sdf_penetration_loss(points_obj, sdf_fn, weight=weight)


def active_human_blocks_for_stage(stage: int) -> tuple[str, ...]:
    if stage < 2:
        return ()
    if stage == 2:
        return ("mhr_trans",)
    if stage == 3:
        return ("mhr_trans", "mhr_global_rot6d")
    return ("mhr_trans", "mhr_global_rot6d", "mhr_hand", "mhr_body_pose_cont")


def _torch() -> Any:
    import torch

    return torch


def _as_tensor(value: Any, *, device: Any, dtype: Any | None = None) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device)
        return tensor.to(dtype=dtype) if dtype is not None else tensor
    return torch.as_tensor(value, device=device, dtype=dtype if dtype is not None else torch.float32)


def _to_numpy(value: Any) -> np.ndarray:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _clone_value(value: Any) -> Any:
    if hasattr(value, "copy"):
        return value.copy()
    return copy.deepcopy(value)


def _slice_sequence(value: Any, frame_indices: np.ndarray) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [value[int(i)] for i in frame_indices]
    arr = np.asarray(value)
    if arr.ndim > 0 and arr.shape[0] >= int(frame_indices.max(initial=0)) + 1:
        return arr[frame_indices]
    return _clone_value(value)


def _select_frame_indices(total: int, cfg: MHRPostOptConfig) -> np.ndarray:
    start = max(0, int(cfg.frame_start))
    stop = total
    if cfg.frame_limit > 0:
        stop = min(total, start + int(cfg.frame_limit))
    if start >= stop:
        raise ValueError(f"empty frame selection start={start} stop={stop} total={total}")
    return np.arange(start, stop, dtype=np.int64)


def _copy_sliced_bundle(bundle: Mapping[str, Any], frame_indices: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in bundle.items():
        if key in {"pr", "in", "gt", "raw", "pr_initial", "observations"} and isinstance(value, Mapping):
            out[key] = {sub_key: _slice_sequence(sub_value, frame_indices) for sub_key, sub_value in value.items() if sub_key not in MHR_POSTOPT_2D_OBSERVATION_KEYS}
        elif key == "metadata" and isinstance(value, Mapping):
            out[key] = {sub_key: _clone_value(sub_value) for sub_key, sub_value in value.items() if sub_key not in MHR_POSTOPT_2D_OBSERVATION_KEYS}
        elif key in {"frames", "frame_meta", "K_rois", "observed_human_masks", "observed_object_masks"}:
            out[key] = _slice_sequence(value, frame_indices)
        else:
            out[key] = _clone_value(value)
    return out


def _skew(vec: Any) -> Any:
    torch = _torch()
    zero = torch.zeros_like(vec[..., 0])
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(*vec.shape[:-1], 3, 3)


def rotvec_to_rotmat(rotvec: Any) -> Any:
    """Differentiable Rodrigues map for small object-pose updates."""

    torch = _torch()
    theta2 = (rotvec * rotvec).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(torch.clamp(theta2, min=1e-12))
    small = theta2 < 1e-8
    a = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / torch.clamp(theta2, min=1e-12))
    k = _skew(rotvec)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(*rotvec.shape[:-1], 3, 3)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def pose_matrix(rot: Any, trans: Any) -> Any:
    torch = _torch()
    pose = torch.eye(4, device=rot.device, dtype=rot.dtype).expand(*rot.shape[:-2], 4, 4).clone()
    pose[..., :3, :3] = rot
    pose[..., :3, 3] = trans
    return pose


def _zero_like_tensor(reference: Any) -> Any:
    return reference.sum() * 0.0


def _acceleration_loss(value: Any) -> Any:
    if value.shape[0] < 3:
        return _zero_like_tensor(value)
    acc = value[:-2] - 2.0 * value[1:-1] + value[2:]
    return (acc * acc).mean()


def _public_acceleration_loss(value: Any) -> Any:
    """Match the public SMPL-H optimizer's sum-over-XYZ reduction."""

    if value.shape[0] < 3:
        return _zero_like_tensor(value)
    acc = value[:-2] - 2.0 * value[1:-1] + value[2:]
    return acc.square().sum(dim=-1).mean()


def _deterministic_sample_indices(count: int, target: int) -> np.ndarray:
    if count <= target:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=target, dtype=np.int64)


def sample_object_surface_vertices(vertices: Any, max_samples: int) -> np.ndarray:
    vertices_np = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if len(vertices_np) == 0:
        raise ValueError("object mesh has no vertices")
    return vertices_np[_deterministic_sample_indices(len(vertices_np), max_samples)]


def predicted_contact_weights(
    contact_logits: Any | None,
    *,
    contact_topk: int = 64,
    sigmoid_threshold: float = 0.2,
    temperature: float = 1.0,
    device: Any | None = None,
    dtype: Any | None = None,
    selection: str = "binary_logits",
) -> Any:
    torch = _torch()
    if contact_logits is None:
        raise KeyError("MHR post-optimization requires predicted contact logits")

    logits = _as_tensor(contact_logits, device=device or "cpu", dtype=dtype or torch.float32)
    if logits.ndim != 2 or logits.shape[-1] != len(MHR_WRIST_INDICES):
        raise ValueError(f"contact logits must have shape [T, 2], got {tuple(logits.shape)}")
    if selection == "binary_logits":
        return (logits > 0).to(dtype=logits.dtype)
    if selection != "legacy_soft_topk":
        raise ValueError(f"unknown contact selection {selection!r}")
    temp = max(float(temperature), 1e-6)
    weights = torch.sigmoid(logits / temp)
    gated = torch.where(weights >= float(sigmoid_threshold), weights, torch.zeros_like(weights))
    if float(gated.sum().detach().cpu()) == 0.0 and contact_topk > 0:
        flat = logits.reshape(-1)
        k = min(int(contact_topk), int(flat.numel()))
        top_idx = torch.topk(flat, k=k).indices
        gated = torch.zeros_like(flat)
        gated[top_idx] = 1.0
        gated = gated.reshape_as(logits)
    if float(gated.sum().detach().cpu()) == 0.0:
        gated = torch.ones_like(logits)
    return gated


def _point_triangle_surface_distances(points: Any, surface_vertices: Any, surface_faces: Any) -> Any:
    torch = _torch()
    from pytorch3d.loss.point_mesh_distance import point_face_distance

    points_flat = points.reshape(-1, 3).contiguous()
    faces = surface_faces.to(device=surface_vertices.device, dtype=torch.long)
    triangles = surface_vertices[faces].contiguous()
    first = torch.zeros(1, device=points_flat.device, dtype=torch.long)
    squared = point_face_distance(points_flat, first, triangles, first, int(points_flat.shape[0]))
    return torch.sqrt(torch.clamp(squared, min=1e-12)).reshape(points.shape[:-1])


def _initial_contact_activation(contact_weights: Any, hand_vertices_world: Any, object_rotation: Any, object_translation: Any, object_vertices: Any, object_faces: Any, distance_threshold_m: float) -> tuple[Any, Any, Any]:
    torch = _torch()
    if hand_vertices_world.ndim != 4 or hand_vertices_world.shape[1] != 2 or hand_vertices_world.shape[-1] != 3:
        raise ValueError(f"hand_vertices_world must have shape [T,2,S,3], got {tuple(hand_vertices_world.shape)}")
    if tuple(contact_weights.shape) != tuple(hand_vertices_world.shape[:2]):
        raise ValueError(f"contact weights must have shape {tuple(hand_vertices_world.shape[:2])}, got {tuple(contact_weights.shape)}")
    with torch.no_grad():
        hands_object = (hand_vertices_world - object_translation[:, None, None, :]) @ object_rotation[:, None, :, :]
        initial_distances_m = _point_triangle_surface_distances(hands_object, object_vertices, object_faces).min(dim=-1).values
        proximity_mask = initial_distances_m < float(distance_threshold_m)
        effective_weights = contact_weights * proximity_mask.to(dtype=contact_weights.dtype)
    return effective_weights.detach(), initial_distances_m.detach(), proximity_mask.detach()


def _contact_activation_summary(network_contact_weights: Any, initial_contact_distances_m: Any | None, initial_contact_proximity_mask: Any, effective_contact_weights: Any, *, distance_threshold_m: float, network_selection: str) -> dict[str, Any]:
    summary = {"revision": MHR_CONTACT_ACTIVATION_REVISION, "evaluation": "once_at_postopt_initialization_from_raw_coconet_prediction", "distance_threshold_m": float(distance_threshold_m), "distance_operator": "strictly_less_than", "network_selection": str(network_selection), "network_active_entries": int((network_contact_weights > 0).sum().detach().cpu()), "geometry_eligible_entries": int(initial_contact_proximity_mask.sum().detach().cpu()), "effective_active_entries": int((effective_contact_weights > 0).sum().detach().cpu())}
    if initial_contact_distances_m is not None:
        summary.update({"initial_distance_min_m": float(initial_contact_distances_m.min().detach().cpu()), "initial_distance_mean_m": float(initial_contact_distances_m.mean().detach().cpu()), "initial_distance_max_m": float(initial_contact_distances_m.max().detach().cpu())})
    return summary


def _nearest_surface_dist(points_obj: Any, surface_vertices: Any, chunk_size: int = 8192) -> Any:
    torch = _torch()
    points_flat = points_obj.reshape(-1, 3)
    dists = []
    for start in range(0, points_flat.shape[0], chunk_size):
        chunk = points_flat[start : start + chunk_size]
        dist2 = torch.cdist(chunk[None], surface_vertices[None]).square()[0].min(dim=-1).values
        dists.append(torch.sqrt(torch.clamp(dist2, min=1e-12)))
    return torch.cat(dists, dim=0).reshape(points_obj.shape[:-1])


def _extract_mhr_params(block: Mapping[str, Any], *, device: Any, dtype: Any) -> dict[str, Any]:
    params = {}
    for key in MHR_PARAM_DIMS:
        if key in block:
            params[key] = _as_tensor(block[key], device=device, dtype=dtype)
    missing = [key for key in ("mhr_global_rot6d", "mhr_trans", "mhr_body_pose_cont", "mhr_hand") if key not in params]
    if missing:
        raise KeyError(f"MHR prediction block is missing required params: {missing}")
    return params


def _layer_output_keypoints(output: Any) -> Any | None:
    if isinstance(output, MHRLayerOutput):
        return output.keypoints
    if isinstance(output, Mapping):
        return output.get("mhr_keypoints", output.get("keypoints"))
    return getattr(output, "keypoints", None)


def _layer_output_vertices(output: Any) -> Any | None:
    if isinstance(output, MHRLayerOutput):
        return output.vertices
    if isinstance(output, Mapping):
        return output.get("mhr_vertices", output.get("vertices"))
    return getattr(output, "vertices", None)


def _layer_output_joints(output: Any) -> Any | None:
    if isinstance(output, MHRLayerOutput):
        return output.joints
    if isinstance(output, Mapping):
        return output.get("mhr_joints", output.get("joints"))
    return getattr(output, "joints", None)


def _layer_output_coco17(output: Any) -> Any | None:
    if isinstance(output, MHRLayerOutput):
        return output.coco17
    if isinstance(output, Mapping):
        return output.get("mhr_coco17", output.get("coco17"))
    return getattr(output, "coco17", None)


def _layer_output_faces(output: Any) -> Any | None:
    if isinstance(output, MHRLayerOutput):
        return output.faces
    if isinstance(output, Mapping):
        return output.get("faces")
    return getattr(output, "faces", None)


class MHRPostOptimizer:
    """Stage-gated native MHR post-optimization from compact CoCoNet outputs."""

    def __init__(
        self,
        bundle: Mapping[str, Any],
        object_vertices: Any,
        cfg: MHRPostOptConfig | Mapping[str, Any] | None = None,
        *,
        object_faces: Any | None = None,
        mhr_layer: Any | None = None,
        sdf_fn: Callable[[Any], Any] | None = None,
        hand_surface_vertex_indices: Any | None = None,
        _contact_activation_state: Mapping[str, Any] | None = None,
    ) -> None:
        if cfg is None:
            cfg = MHRPostOptConfig()
        elif isinstance(cfg, Mapping):
            cfg = MHRPostOptConfig(**cfg)
        self.cfg = cfg.checked()
        torch = _torch()
        device_name = self.cfg.device
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.dtype = torch.float32

        total = len(bundle["pr"]["pose_abs"])
        self.frame_indices = _select_frame_indices(total, self.cfg)
        self.bundle = _copy_sliced_bundle(bundle, self.frame_indices)
        self.pr = self.bundle["pr"]
        self.ref = self.bundle.get("pr_initial", self.pr)
        self.gt = self.bundle.get("gt", {})
        self.object_faces = None if object_faces is None else np.asarray(object_faces)
        self.mhr_layer = mhr_layer
        self.sdf_fn = sdf_fn
        if self.cfg.stage >= 5 and self.cfg.w_sdf != 0 and self.sdf_fn is None:
            raise ValueError("w_sdf requires an explicit object signed-distance function")
        self.object_penetration_surface = None
        self.penetration_faces = None
        self.penetration_proxy_source_indices = None
        self.penetration_proxy_barycentric_weights = None
        if self.cfg.stage >= 5 and self.cfg.w_pen != 0:
            if self.object_faces is None:
                raise ValueError("w_pen requires object_faces for area-weighted object-surface sampling")
            self.object_penetration_surface = _as_tensor(sample_mesh_surface_points(object_vertices, self.object_faces, self.cfg.penetration_surface_samples, seed=0), device=self.device, dtype=self.dtype)
            if self.cfg.penetration_human_mesh_mode == MHR_COLLISION_PROXY_MODE_4000:
                source_faces = self.bundle.get("faces")
                if source_faces is None and self.mhr_layer is not None:
                    source_faces = _to_numpy(self.mhr_layer.mesh_faces(device=self.device))
                if source_faces is None:
                    raise RuntimeError("MHR collision-proxy penetration requires canonical MHR faces")
                proxy = load_mhr_collision_proxy(source_faces, self.cfg.penetration_collision_proxy_path)
                self.penetration_proxy_source_indices = _as_tensor(proxy.source_vertex_indices.copy(), device=self.device, dtype=_torch().long)
                self.penetration_proxy_barycentric_weights = _as_tensor(proxy.barycentric_weights.copy(), device=self.device, dtype=self.dtype)
                self.penetration_faces = _as_tensor(proxy.faces.copy(), device=self.device, dtype=_torch().long)

        self.object_vertices = _as_tensor(object_vertices, device=self.device, dtype=self.dtype)
        self.object_faces_tensor = None if self.object_faces is None else _as_tensor(self.object_faces, device=self.device, dtype=_torch().long)
        if self.cfg.w_contact != 0 and self.object_faces_tensor is None:
            raise ValueError("the initial contact proximity gate requires object_faces")
        self.object_surface = _as_tensor(sample_mesh_surface_points(object_vertices, self.object_faces, self.cfg.object_surface_samples, seed=0) if self.object_faces is not None else sample_object_surface_vertices(object_vertices, self.cfg.object_surface_samples), device=self.device, dtype=self.dtype)
        self.pose_init = _as_tensor(self.pr["pose_abs"], device=self.device, dtype=self.dtype)
        self.pose_ref = _as_tensor(self.ref.get("pose_abs", self.pr["pose_abs"]), device=self.device, dtype=self.dtype)
        self.obj_rot_delta = torch.zeros((len(self.frame_indices), 3), device=self.device, dtype=self.dtype, requires_grad=True)
        self.obj_trans_delta = torch.zeros((len(self.frame_indices), 3), device=self.device, dtype=self.dtype, requires_grad=True)

        self.params_init = _extract_mhr_params(self.pr, device=self.device, dtype=self.dtype)
        self.params_ref = _extract_mhr_params(self.ref, device=self.device, dtype=self.dtype)
        self.params_opt: dict[str, Any] = {}
        for key, value in self.params_init.items():
            tensor = value.clone().detach()
            tensor.requires_grad_(key in active_human_blocks_for_stage(self.cfg.stage))
            self.params_opt[key] = tensor

        if hand_surface_vertex_indices is None and self.cfg.w_contact != 0:
            faces = self.bundle.get("faces")
            if faces is None and self.mhr_layer is not None:
                faces = _to_numpy(self.mhr_layer.mesh_faces(device=self.device))
            if faces is None:
                raise RuntimeError("MHR hand-surface contact requires canonical MHR faces")
            spec = load_mhr_hand_surface_spec(self.cfg.hand_surface_spec_path, faces=faces)
            hand_surface_vertex_indices = np.stack([spec.vertex_indices[side][spec.sample_local_indices[side]] for side in range(2)], axis=0)
        if hand_surface_vertex_indices is None:
            self.hand_surface_vertex_indices = None
        else:
            hand_surface_vertex_indices = np.asarray(hand_surface_vertex_indices, dtype=np.int64)
            if hand_surface_vertex_indices.ndim != 2 or hand_surface_vertex_indices.shape[0] != 2 or hand_surface_vertex_indices.shape[1] == 0 or hand_surface_vertex_indices.min() < 0:
                raise ValueError(f"hand_surface_vertex_indices must have shape [2,S], got {hand_surface_vertex_indices.shape}")
            self.hand_surface_vertex_indices = _as_tensor(hand_surface_vertex_indices, device=self.device, dtype=_torch().long)

        self.network_contact_weights = predicted_contact_weights(
            self.pr.get("contact_logits"),
            contact_topk=self.cfg.contact_topk,
            sigmoid_threshold=self.cfg.contact_sigmoid_threshold,
            temperature=self.cfg.contact_logit_temperature,
            device=self.device,
            dtype=self.dtype,
            selection=self.cfg.contact_selection,
        )
        if self.cfg.w_contact == 0:
            self.initial_contact_distances_m = None
            self.initial_contact_proximity_mask = torch.zeros_like(self.network_contact_weights, dtype=torch.bool)
            self.contact_weights = torch.zeros_like(self.network_contact_weights)
        elif _contact_activation_state is None:
            self.contact_weights, self.initial_contact_distances_m, self.initial_contact_proximity_mask = self._compute_initial_contact_activation()
        else:
            self._restore_contact_activation_state(_contact_activation_state)

    def _compute_initial_contact_activation(self) -> tuple[Any, Any, Any]:
        if self.mhr_layer is None:
            raise RuntimeError("the initial contact proximity gate requires the parametric MHR layer")
        with _torch().no_grad():
            output = self.mhr_layer.mhr_forward(self.params_init)
            vertices = _layer_output_vertices(output)
            if vertices is None:
                raise RuntimeError("the initial contact proximity gate requires decoded MHR vertices")
            hand_vertices = vertices[:, self.hand_surface_vertex_indices, :]
        return _initial_contact_activation(self.network_contact_weights, hand_vertices, self.pose_init[:, :3, :3], self.pose_init[:, :3, 3], self.object_vertices, self.object_faces_tensor, self.cfg.contact_activation_distance_m)

    def _contact_activation_state(self) -> dict[str, Any]:
        return {"revision": MHR_CONTACT_ACTIVATION_REVISION, "distance_threshold_m": float(self.cfg.contact_activation_distance_m), "network_contact_weights": self.network_contact_weights.detach().clone(), "initial_contact_distances_m": None if self.initial_contact_distances_m is None else self.initial_contact_distances_m.detach().clone(), "initial_contact_proximity_mask": self.initial_contact_proximity_mask.detach().clone(), "effective_contact_weights": self.contact_weights.detach().clone()}

    def _restore_contact_activation_state(self, state: Mapping[str, Any]) -> None:
        if state.get("revision") != MHR_CONTACT_ACTIVATION_REVISION or float(state.get("distance_threshold_m", -1.0)) != float(self.cfg.contact_activation_distance_m):
            raise ValueError("contact activation state does not match the active configuration")
        network_weights = _as_tensor(state["network_contact_weights"], device=self.device, dtype=self.dtype)
        if tuple(network_weights.shape) != tuple(self.network_contact_weights.shape) or not _torch().equal(network_weights, self.network_contact_weights):
            raise ValueError("contact activation state does not match the active network predictions")
        self.initial_contact_distances_m = _as_tensor(state["initial_contact_distances_m"], device=self.device, dtype=self.dtype)
        self.initial_contact_proximity_mask = _as_tensor(state["initial_contact_proximity_mask"], device=self.device).bool()
        self.contact_weights = _as_tensor(state["effective_contact_weights"], device=self.device, dtype=self.dtype)

    def _optimizer(self) -> Any:
        torch = _torch()
        groups = [
            {"params": [self.obj_rot_delta], "lr": self.cfg.lr_obj_rot},
            {"params": [self.obj_trans_delta], "lr": self.cfg.lr_obj_trans},
        ]
        lr_by_key = {
            "mhr_trans": self.cfg.lr_mhr_trans,
            "mhr_global_rot6d": self.cfg.lr_mhr_root,
            "mhr_hand": self.cfg.lr_mhr_hand,
            "mhr_body_pose_cont": self.cfg.lr_mhr_body,
        }
        for key in active_human_blocks_for_stage(self.cfg.stage):
            if self.params_opt[key].requires_grad:
                groups.append({"params": [self.params_opt[key]], "lr": lr_by_key[key]})
        return torch.optim.Adam(groups)

    def object_pose(self) -> tuple[Any, Any, Any]:
        delta_rot = rotvec_to_rotmat(self.obj_rot_delta)
        rot = delta_rot @ self.pose_init[:, :3, :3]
        trans = self.pose_init[:, :3, 3] + self.obj_trans_delta
        return rot, trans, pose_matrix(rot, trans)

    def _decode_body(self) -> tuple[Any, Any | None, Any | None, Any | None]:
        if self.mhr_layer is None:
            raise RuntimeError("MHR post-optimization requires the parametric MHR layer; persisted vertex geometry is unsupported")

        if not any(value.requires_grad for value in self.params_opt.values()):
            with _torch().no_grad():
                output = self.mhr_layer.mhr_forward(self.params_opt)
        else:
            output = self.mhr_layer.mhr_forward(self.params_opt)
        keypoints = _layer_output_keypoints(output)
        if keypoints is None:
            raise RuntimeError("MHR layer output did not include keypoints")
        coco17 = _layer_output_coco17(output)
        if coco17 is None:
            coco17 = mhr70_to_coco17(keypoints)
        return keypoints, _layer_output_vertices(output), coco17, _layer_output_faces(output)

    def _hand_surface_points(self) -> Any:
        if self.hand_surface_vertex_indices is None:
            raise RuntimeError("MHR hand-surface indices were not initialized")
        _keypoints, vertices, _coco17, _faces = self._decode_body()
        if vertices is None:
            raise RuntimeError("MHR hand-surface contact requires decoded or cached MHR vertices")
        if int(self.hand_surface_vertex_indices.max()) >= vertices.shape[1]:
            raise ValueError(f"hand-surface index {int(self.hand_surface_vertex_indices.max())} exceeds {vertices.shape[1]} MHR vertices")
        return vertices[:, self.hand_surface_vertex_indices, :]

    def contact_loss(self, rot: Any, trans: Any) -> tuple[Any, Any]:
        hands_world = self._hand_surface_points()
        hands_obj = (hands_world - trans[:, None, None, :]) @ rot[:, None, :, :]
        if self.cfg.contact_distance_backend == "triangle_surface":
            if self.object_faces_tensor is None:
                raise ValueError("triangle_surface contact requires object_faces")
            point_distances = _point_triangle_surface_distances(hands_obj, self.object_vertices, self.object_faces_tensor)
        else:
            point_distances = _nearest_surface_dist(hands_obj, self.object_surface)
        distances = point_distances.min(dim=-1).values
        weights = self.contact_weights.to(device=distances.device, dtype=distances.dtype)
        loss = (distances * weights).sum() / torch_clamp_min(weights.sum(), 1.0)
        return loss, distances

    def object_prior_loss(self, rot: Any, trans: Any) -> Any:
        rot_ref = self.pose_ref[:, :3, :3]
        trans_ref = self.pose_ref[:, :3, 3]
        return ((rot - rot_ref) ** 2).mean() + ((trans - trans_ref) ** 2).mean()

    def object_smooth_loss(self, rot: Any, trans: Any) -> Any:
        return _acceleration_loss(trans) + 0.25 * _acceleration_loss(rot.reshape(rot.shape[0], -1))

    def static_object_loss(self, rot: Any, trans: Any) -> Any:
        frame_contact = self.network_contact_weights.max(dim=-1).values
        no_contact = (1.0 - torch_clamp_max(frame_contact, 1.0)).reshape(-1, 1)
        if float(no_contact.sum().detach().cpu()) == 0.0:
            return _zero_like_tensor(trans)
        rot_ref = self.pose_ref[:, :3, :3]
        trans_ref = self.pose_ref[:, :3, 3]
        loss_t = (((trans - trans_ref) ** 2).sum(dim=-1, keepdim=True) * no_contact).sum()
        loss_r = (((rot - rot_ref) ** 2).mean(dim=(-1, -2), keepdim=True) * no_contact[:, None]).sum()
        return (loss_t + loss_r) / torch_clamp_min(no_contact.sum(), 1.0)

    def mhr_prior_loss(self) -> Any:
        ref = self.params_ref
        loss = _zero_like_tensor(self.params_opt["mhr_trans"])
        if self.cfg.stage >= 2:
            loss = loss + ((self.params_opt["mhr_trans"] - ref["mhr_trans"]) ** 2).mean() * self.cfg.w_mhr_trans_prior
            loss = loss + _acceleration_loss(self.params_opt["mhr_trans"]) * self.cfg.w_mhr_smooth
        if self.cfg.stage >= 3:
            loss = loss + ((self.params_opt["mhr_global_rot6d"] - ref["mhr_global_rot6d"]) ** 2).mean() * self.cfg.w_mhr_root_prior
            loss = loss + _acceleration_loss(self.params_opt["mhr_global_rot6d"]) * self.cfg.w_mhr_smooth
        if self.cfg.stage >= 4:
            loss = loss + ((self.params_opt["mhr_hand"] - ref["mhr_hand"]) ** 2).mean() * self.cfg.w_mhr_hand_prior
            loss = loss + ((self.params_opt["mhr_body_pose_cont"] - ref["mhr_body_pose_cont"]) ** 2).mean() * self.cfg.w_mhr_body_prior
            loss = loss + _acceleration_loss(self.params_opt["mhr_hand"]) * self.cfg.w_mhr_smooth
            loss = loss + _acceleration_loss(self.params_opt["mhr_body_pose_cont"]) * self.cfg.w_mhr_smooth
        return loss

    def sdf_loss(self, rot: Any, trans: Any) -> Any:
        if self.cfg.stage < 5 or self.cfg.w_sdf == 0:
            return _zero_like_tensor(trans)
        keypoints, vertices, _coco17, _faces = self._decode_body()
        points = vertices if vertices is not None else keypoints[:, list(MHR_WRIST_INDICES), :]
        if points.ndim != 3:
            raise ValueError(f"human surface/keypoint tensor must be [T, N, 3], got {tuple(points.shape)}")
        if points.shape[1] > self.cfg.human_surface_samples:
            idx = _deterministic_sample_indices(points.shape[1], self.cfg.human_surface_samples)
            index = _as_tensor(idx, device=points.device).long()
            points = points.index_select(1, index)
        points_obj = (points - trans[:, None, :]) @ rot
        return object_sdf_penetration_loss(points_obj, self.sdf_fn, weight=self.cfg.w_sdf)

    def penetration_loss(self, rot: Any, trans: Any) -> Any:
        if self.cfg.stage < 5 or self.cfg.w_pen == 0:
            return _zero_like_tensor(trans)
        _keypoints, vertices, _coco17, faces = self._decode_body()
        if vertices is None or faces is None:
            raise RuntimeError("w_pen requires decoded MHR vertices and faces")
        if self.penetration_proxy_source_indices is not None:
            vertices = apply_mhr_collision_proxy(vertices, self.penetration_proxy_source_indices, self.penetration_proxy_barycentric_weights)
        elif self.penetration_faces is None:
            faces_np = _to_numpy(faces).astype(np.int64, copy=False)
            validate_closed_triangle_mesh(faces_np)
            self.penetration_faces = _as_tensor(faces_np, device=vertices.device, dtype=_torch().long)
        object_points_world = self.object_penetration_surface[None] @ rot.transpose(-1, -2) + trans[:, None, :]
        return object_inside_human_penetration_loss(vertices, self.penetration_faces, object_points_world, weight=self.cfg.w_pen, frame_chunk_size=self.cfg.penetration_frame_chunk_size, validate_face_indices=False, conservative_aabb_rejection=self.cfg.penetration_bbox_rejection, aabb_margin_m=self.cfg.penetration_bbox_margin_m)

    def loss(self) -> tuple[Any, dict[str, Any]]:
        rot, trans, _pose = self.object_pose()
        if self.cfg.w_contact != 0:
            contact, distances = self.contact_loss(rot, trans)
        else:
            contact = _zero_like_tensor(trans)
            distances = _torch().zeros((len(trans), 2), device=trans.device, dtype=trans.dtype)
        losses = {
            "loss_contact": contact * self.cfg.w_contact,
            "loss_obj_pose_prior": self.object_prior_loss(rot, trans) * self.cfg.w_obj_pose_prior,
            "loss_obj_smooth": self.object_smooth_loss(rot, trans) * self.cfg.w_obj_smooth,
            "loss_static_object": self.static_object_loss(rot, trans) * self.cfg.w_static_object,
            "loss_mhr_prior": self.mhr_prior_loss(),
            "loss_sdf": self.sdf_loss(rot, trans),
            "loss_pen": self.penetration_loss(rot, trans),
        }
        total = sum(losses.values())
        metrics = {
            **losses,
            "loss_total": total,
            "contact_distance_mean": distances.mean(),
            "contact_distance_weighted": contact,
            "contact_weight_sum": self.contact_weights.sum(),
            "network_contact_entries": (self.network_contact_weights > 0).sum(),
            "initial_proximity_entries": self.initial_contact_proximity_mask.sum(),
            "effective_contact_entries": (self.contact_weights > 0).sum(),
        }
        return total, metrics

    def run(self) -> dict[str, Any]:
        optimizer = self._optimizer()
        history: list[dict[str, float]] = []
        report_every = max(1, int(self.cfg.report_every))
        for step in range(self.cfg.iterations + 1):
            optimizer.zero_grad(set_to_none=True)
            total, metrics = self.loss()
            if step == 0 or step == self.cfg.iterations or step % report_every == 0:
                history.append({"iter": float(step), **{key: float(value.detach().cpu()) for key, value in metrics.items()}})
            if step == self.cfg.iterations:
                break
            total.backward()
            optimizer.step()
        return self.result(history)

    def result(self, history: Sequence[Mapping[str, float]]) -> dict[str, Any]:
        _rot, _trans, pose = self.object_pose()
        result = copy.deepcopy(self.bundle)
        result.setdefault("pr_initial", copy.deepcopy(self.ref))
        result["pr"] = copy.deepcopy(result["pr"])
        result["pr"]["pose_abs"] = _to_numpy(pose).astype(np.float32)
        result["pr"]["pose_abs_postopt"] = _to_numpy(pose).astype(np.float32)
        for key, value in self.params_opt.items():
            result["pr"][key] = _to_numpy(value).astype(np.float32)
        result["postopt"] = {
            "stage": int(self.cfg.stage),
            "stage_name": STAGE_NAMES[self.cfg.stage],
            "frame_indices": self.frame_indices.tolist(),
            "config": asdict(self.cfg),
            "history": [dict(item) for item in history],
            "contact_activation": _contact_activation_summary(self.network_contact_weights, self.initial_contact_distances_m, self.initial_contact_proximity_mask, self.contact_weights, distance_threshold_m=self.cfg.contact_activation_distance_m, network_selection=self.cfg.contact_selection),
        }
        return result


class MHRParityPostOptimizer:
    """One-optimizer MHR implementation of the public SMPL-H refinement protocol."""

    def __init__(self, bundle: Mapping[str, Any], object_vertices: Any, object_faces: Any, cfg: MHRParityPostOptConfig | Mapping[str, Any] | None = None, *, mhr_layer: Any) -> None:
        if cfg is None:
            cfg = MHRParityPostOptConfig()
        elif isinstance(cfg, Mapping):
            cfg = MHRParityPostOptConfig(**cfg)
        self.cfg = cfg.checked()
        torch = _torch()
        if mhr_layer is None:
            raise ValueError("SMPL-H-parity MHR post-optimization requires a differentiable MHR layer")
        device_name = self.cfg.device
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SMPL-H-parity MHR post-optimization requires CUDA")
        self.device, self.dtype, self.mhr_layer = torch.device(device_name), torch.float32, mhr_layer
        total = len(bundle["pr"]["pose_abs"])
        selection_cfg = MHRPostOptConfig(frame_start=self.cfg.frame_start, frame_limit=self.cfg.frame_limit)
        self.frame_indices = _select_frame_indices(total, selection_cfg)
        self.bundle = _copy_sliced_bundle(bundle, self.frame_indices)
        self.pr, self.observations = self.bundle["pr"], self.bundle.get("observations", {})
        self.params_fixed = _extract_mhr_params(self.pr, device=self.device, dtype=self.dtype)
        self.params_fixed = {key: value.detach().clone() for key, value in self.params_fixed.items()}
        self.body_pose_initial = self.params_fixed["mhr_body_pose_cont"].detach().clone()
        self.body_pose_rotation_initial = self.body_pose_initial[..., :MHR_BODY_ROTATION_CONTROL_COUNT]
        self.body_pose = (self.body_pose_rotation_initial if self.cfg.freeze_body_internal_translations else self.body_pose_initial).clone().requires_grad_(True)
        pose_init = _as_tensor(self.pr["pose_abs"], device=self.device, dtype=self.dtype)
        from pytorch3d.transforms import matrix_to_axis_angle

        self.object_rotation_initial = pose_init[:, :3, :3].detach().clone()
        self.object_axis = matrix_to_axis_angle(self.object_rotation_initial).detach().clone().requires_grad_(not self.cfg.freeze_object_rotation)
        self.object_translation = pose_init[:, :3, 3].detach().clone().requires_grad_(True)
        self.object_translation_initial = pose_init[:, :3, 3].detach().clone()
        self.object_vertices = _as_tensor(np.asarray(object_vertices, dtype=np.float32), device=self.device, dtype=self.dtype)
        self.object_faces = _as_tensor(np.asarray(object_faces, dtype=np.int64), device=self.device, dtype=torch.long)
        if self.object_vertices.ndim != 2 or self.object_vertices.shape[1] != 3 or self.object_faces.ndim != 2 or self.object_faces.shape[1] != 3:
            raise ValueError(f"invalid object mesh for MHR post-optimization: {tuple(self.object_vertices.shape)}, {tuple(self.object_faces.shape)}")
        surface = sample_mesh_surface_points(object_vertices, object_faces, self.cfg.object_surface_samples, seed=0)
        self.object_surface = _as_tensor(surface, device=self.device, dtype=self.dtype)
        penetration_surface = sample_mesh_surface_points(object_vertices, object_faces, self.cfg.penetration_surface_samples, seed=1)
        self.penetration_surface = _as_tensor(penetration_surface, device=self.device, dtype=self.dtype)
        self._initialize_frozen_object_geometry_cache()
        human_faces = _to_numpy(self.mhr_layer.mesh_faces(device=self.device)).astype(np.int64, copy=False)
        self.human_faces = _as_tensor(human_faces, device=self.device, dtype=torch.long)
        self.penetration_proxy_source_indices = None
        self.penetration_proxy_barycentric_weights = None
        self.penetration_human_faces = self.human_faces
        if self.cfg.penetration_human_mesh_mode == MHR_COLLISION_PROXY_MODE_4000:
            proxy = load_mhr_collision_proxy(human_faces, self.cfg.penetration_collision_proxy_path)
            self.penetration_proxy_source_indices = _as_tensor(proxy.source_vertex_indices.copy(), device=self.device, dtype=torch.long)
            self.penetration_proxy_barycentric_weights = _as_tensor(proxy.barycentric_weights.copy(), device=self.device, dtype=self.dtype)
            self.penetration_human_faces = _as_tensor(proxy.faces.copy(), device=self.device, dtype=torch.long)
        spec = load_mhr_hand_surface_spec(self.cfg.hand_surface_spec_path, faces=human_faces)
        if self.cfg.hand_surface_mode == "all_vertices":
            hand_indices = spec.vertex_indices
        else:
            hand_indices = np.stack([spec.vertex_indices[side][spec.sample_local_indices[side]] for side in range(2)], axis=0)
        self.hand_surface_vertex_indices = _as_tensor(hand_indices, device=self.device, dtype=torch.long)
        self.network_contact_mask = predicted_contact_weights(self.pr.get("contact_logits"), device=self.device, dtype=self.dtype, selection="binary_logits") if self.cfg.w_contact != 0 else torch.zeros((len(self.frame_indices), 2), device=self.device, dtype=self.dtype)
        if self.cfg.w_contact != 0:
            self.contact_mask, self.initial_contact_distances_m, self.initial_contact_proximity_mask = self._compute_initial_contact_activation()
        else:
            self.contact_mask = torch.zeros_like(self.network_contact_mask)
            self.initial_contact_distances_m = None
            self.initial_contact_proximity_mask = torch.zeros_like(self.network_contact_mask, dtype=torch.bool)
        self._initialize_pose_diagnostics()
        self.postopt_K_rois = self._required_observation("postopt_K_rois", self.observations, ndim=3) if self.cfg.w_silhouette != 0 else None
        self.human_mask = self._required_observation("postopt_human_mask", self.observations, ndim=3) if self.cfg.w_silhouette != 0 else None
        self.object_mask = self._required_observation("postopt_object_mask", self.observations, ndim=3) if self.cfg.w_silhouette != 0 else None
        if self.cfg.w_silhouette != 0:
            if str(self.observations.get("postopt_crop_contract", "")) != POSTOPT_CROP_CONTRACT:
                raise ValueError(f"SMPL-H-parity silhouette loss requires crop contract {POSTOPT_CROP_CONTRACT!r}")
            if tuple(self.human_mask.shape[-2:]) != (POSTOPT_RENDER_SIZE, POSTOPT_RENDER_SIZE) or tuple(self.object_mask.shape[-2:]) != (POSTOPT_RENDER_SIZE, POSTOPT_RENDER_SIZE):
                raise ValueError(f"SMPL-H-parity silhouette masks must be {POSTOPT_RENDER_SIZE}x{POSTOPT_RENDER_SIZE}, got {tuple(self.human_mask.shape[-2:])} and {tuple(self.object_mask.shape[-2:])}")
        self.glctx = None
        self.object_silhouette_tensors = None
        if self.cfg.w_silhouette != 0:
            import nvdiffrast.torch as dr

            self.glctx = dr.RasterizeCudaContext()
            self.object_silhouette_tensors = {"faces": self.object_faces.to(dtype=torch.int), "pos": self.object_vertices, "vertex_color": torch.ones_like(self.object_vertices)}
        self.optimizer = torch.optim.Adam(self._optimizer_parameter_groups(), lr=self.cfg.lr)
        from transformers import get_scheduler

        self.scheduler = get_scheduler(optimizer=self.optimizer, name="cosine", num_warmup_steps=self.cfg.warmup_steps, num_training_steps=self.cfg.schedule_steps)
        self.rng = np.random.default_rng(self.cfg.random_seed)
        self.batch_start_order = np.empty(0, dtype=np.int64)
        self.batch_start_cursor = 0
        self.start_step = 0
        self.history: list[dict[str, float]] = []
        self._load_checkpoint()

    def _required_observation(self, key: str, block: Mapping[str, Any], *, ndim: int) -> Any:
        if key not in block:
            raise KeyError(f"SMPL-H-parity MHR post-optimization requires {key}")
        value = _as_tensor(block[key], device=self.device, dtype=self.dtype)
        if value.ndim != ndim or value.shape[0] != len(self.frame_indices):
            raise ValueError(f"{key} must have {ndim} dimensions and {len(self.frame_indices)} frames, got {tuple(value.shape)}")
        return value

    def _compute_initial_contact_activation(self) -> tuple[Any, Any, Any]:
        torch = _torch()
        effective_chunks, distance_chunks, proximity_chunks = [], [], []
        with torch.no_grad():
            for start in range(0, len(self.frame_indices), MHR_CONTACT_ACTIVATION_FRAME_CHUNK_SIZE):
                indices = torch.arange(start, min(start + MHR_CONTACT_ACTIVATION_FRAME_CHUNK_SIZE, len(self.frame_indices)), device=self.device, dtype=torch.long)
                raw_params = {key: value.index_select(0, indices) for key, value in self.params_fixed.items()}
                vertices = _layer_output_vertices(self.mhr_layer.mhr_forward(raw_params))
                if vertices is None:
                    raise RuntimeError("the initial contact proximity gate requires decoded raw CoCoNet MHR vertices")
                effective, distances, proximity = _initial_contact_activation(self.network_contact_mask.index_select(0, indices), vertices[:, self.hand_surface_vertex_indices, :], self.object_rotation_initial.index_select(0, indices), self.object_translation_initial.index_select(0, indices), self.object_vertices, self.object_faces, self.cfg.contact_activation_distance_m)
                effective_chunks.append(effective)
                distance_chunks.append(distances)
                proximity_chunks.append(proximity)
        return torch.cat(effective_chunks, dim=0), torch.cat(distance_chunks, dim=0), torch.cat(proximity_chunks, dim=0)

    def _contact_activation_state(self) -> dict[str, Any]:
        return {"revision": MHR_CONTACT_ACTIVATION_REVISION, "distance_threshold_m": float(self.cfg.contact_activation_distance_m), "network_contact_mask": self.network_contact_mask.detach().clone(), "initial_contact_distances_m": None if self.initial_contact_distances_m is None else self.initial_contact_distances_m.detach().clone(), "initial_contact_proximity_mask": self.initial_contact_proximity_mask.detach().clone(), "effective_contact_mask": self.contact_mask.detach().clone()}

    def _validate_checkpoint_contact_activation(self, state: Mapping[str, Any]) -> None:
        torch = _torch()
        if state.get("revision") != MHR_CONTACT_ACTIVATION_REVISION or float(state.get("distance_threshold_m", -1.0)) != float(self.cfg.contact_activation_distance_m):
            raise ValueError("post-optimization checkpoint contact activation does not match the active configuration")
        for key, current in (("network_contact_mask", self.network_contact_mask), ("initial_contact_proximity_mask", self.initial_contact_proximity_mask), ("effective_contact_mask", self.contact_mask)):
            saved = _as_tensor(state[key], device=self.device, dtype=current.dtype)
            if tuple(saved.shape) != tuple(current.shape) or not torch.equal(saved, current):
                raise ValueError(f"post-optimization checkpoint {key} does not match the raw CoCoNet initialization")
        saved_distances = state.get("initial_contact_distances_m")
        if (saved_distances is None) != (self.initial_contact_distances_m is None):
            raise ValueError("post-optimization checkpoint initial contact distances do not match the active contact configuration")
        if saved_distances is not None:
            saved_distances = _as_tensor(saved_distances, device=self.device, dtype=self.dtype)
            if tuple(saved_distances.shape) != tuple(self.initial_contact_distances_m.shape) or not torch.allclose(saved_distances, self.initial_contact_distances_m, rtol=1e-6, atol=1e-7):
                raise ValueError("post-optimization checkpoint initial contact distances do not match the raw CoCoNet initialization")

    def _optimizer_parameter_groups(self) -> list[dict[str, Any]]:
        groups = [{"params": [self.object_translation], "lr": self.cfg.lr}, {"params": [self.body_pose], "lr": self.cfg.lr}]
        if not self.cfg.freeze_object_rotation:
            groups.insert(1, {"params": [self.object_axis], "lr": self.cfg.lr})
        return groups

    def _initialize_frozen_object_geometry_cache(self) -> None:
        self.object_vertices_rotated_initial = None
        self.object_surface_rotated_initial = None
        self.object_surface_acceleration_mean_initial = None
        self.object_surface_acceleration_square_mean_initial = None
        self.penetration_surface_rotated_initial = None
        if not self.cfg.freeze_object_rotation:
            return
        with _torch().no_grad():
            rotation_transpose = self.object_rotation_initial.transpose(-1, -2)
            self.object_vertices_rotated_initial = (self.object_vertices[None] @ rotation_transpose).detach()
            self.object_surface_rotated_initial = (self.object_surface[None] @ rotation_transpose).detach()
            surface_acceleration = self.object_surface_rotated_initial[:-2] - 2.0 * self.object_surface_rotated_initial[1:-1] + self.object_surface_rotated_initial[2:]
            self.object_surface_acceleration_mean_initial = surface_acceleration.mean(dim=1).detach()
            self.object_surface_acceleration_square_mean_initial = surface_acceleration.square().sum(dim=-1).mean(dim=1).detach()
            self.penetration_surface_rotated_initial = (self.penetration_surface[None] @ rotation_transpose).detach()

    def _load_checkpoint(self) -> None:
        if self.cfg.checkpoint_path is None or not Path(self.cfg.checkpoint_path).exists():
            return
        torch = _torch()
        payload = torch.load(self.cfg.checkpoint_path, map_location=self.device, weights_only=False)
        if payload.get("schema") != MHR_PARITY_POSTOPT_CHECKPOINT_SCHEMA or payload.get("config") != asdict(self.cfg):
            raise ValueError(f"post-optimization checkpoint identity does not match the active configuration: {self.cfg.checkpoint_path}")
        self._validate_checkpoint_contact_activation(payload["contact_activation"])
        self.object_translation.data.copy_(payload["object_translation"])
        if not self.cfg.freeze_object_rotation:
            self.object_axis.data.copy_(payload["object_axis"])
        checkpoint_body_pose = payload["mhr_body_pose_cont"]
        self.body_pose.data.copy_(checkpoint_body_pose[..., :MHR_BODY_ROTATION_CONTROL_COUNT] if self.cfg.freeze_body_internal_translations else checkpoint_body_pose)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.rng.bit_generator.state = payload["rng_state"]
        self.batch_start_order = np.asarray(payload["batch_start_order"], dtype=np.int64)
        self.batch_start_cursor = int(payload["batch_start_cursor"])
        self._validate_batch_sampler_state()
        self.start_step = int(payload["next_step"])
        self.history = [dict(item) for item in payload.get("history", [])]

    def _save_checkpoint(self, next_step: int) -> None:
        if self.cfg.checkpoint_path is None:
            return
        torch = _torch()
        path = Path(self.cfg.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        all_indices = torch.arange(len(self.frame_indices), device=self.device, dtype=torch.long)
        payload = {"schema": MHR_PARITY_POSTOPT_CHECKPOINT_SCHEMA, "config": asdict(self.cfg), "next_step": int(next_step), "object_translation": self.object_translation.detach(), "mhr_body_pose_cont": self._body_pose_for_indices(all_indices).detach(), "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(), "rng_state": self.rng.bit_generator.state, "batch_start_order": self.batch_start_order.copy(), "batch_start_cursor": int(self.batch_start_cursor), "contact_activation": self._contact_activation_state(), "history": self.history}
        if not self.cfg.freeze_object_rotation:
            payload["object_axis"] = self.object_axis.detach()
        torch.save(payload, temporary, pickle_protocol=4)
        temporary.replace(path)
        print("MHR_POSTOPT_CHECKPOINT " + json.dumps({"current": min(int(next_step), int(self.cfg.num_steps)), "total": int(self.cfg.num_steps), "unit": "optimizer steps"}, sort_keys=True), flush=True)

    def _validate_batch_sampler_state(self) -> None:
        length = len(self.frame_indices)
        batch_size = self._resolved_batch_size()
        window_count = length - batch_size + 1
        order = np.asarray(self.batch_start_order, dtype=np.int64)
        if order.ndim != 1 or len(order) not in {0, window_count}:
            raise ValueError(f"batch-start order must be empty or contain {window_count} starts, got {order.shape}")
        if len(order) and not np.array_equal(np.sort(order), np.arange(window_count, dtype=np.int64)):
            raise ValueError("batch-start order must be a permutation of every legal contiguous-window start")
        if not 0 <= int(self.batch_start_cursor) <= len(order):
            raise ValueError(f"batch-start cursor {self.batch_start_cursor} is outside [0,{len(order)}]")
        if not len(order) and int(self.batch_start_cursor) != 0:
            raise ValueError("an empty batch-start order requires cursor 0")
        self.batch_start_order = order

    def _resolved_batch_size(self) -> int:
        length = len(self.frame_indices)
        if length <= 0:
            raise ValueError("post-optimization requires at least one selected frame")
        return length if int(self.cfg.batch_size) == MHR_POSTOPT_FULL_CLIP_BATCH_SIZE else min(int(self.cfg.batch_size), length)

    def _batch_sampling_identity(self) -> str:
        return MHR_POSTOPT_FULL_CLIP_BATCH_SAMPLING if int(self.cfg.batch_size) == MHR_POSTOPT_FULL_CLIP_BATCH_SIZE else MHR_POSTOPT_BATCH_SAMPLING

    def _batch_indices(self) -> Any:
        length = len(self.frame_indices)
        batch_size = self._resolved_batch_size()
        window_count = length - batch_size + 1
        if self.batch_start_cursor >= len(self.batch_start_order):
            self.batch_start_order = np.asarray(self.rng.permutation(window_count), dtype=np.int64)
            self.batch_start_cursor = 0
        start = int(self.batch_start_order[self.batch_start_cursor])
        self.batch_start_cursor += 1
        return _torch().arange(start, start + batch_size, device=self.device, dtype=_torch().long)

    def _object_state(self, indices: Any, *, include_surface: bool = True) -> tuple[Any, Any, Any, Any]:
        rotation = self._object_rotation_for_indices(indices)
        translation = self.object_translation.index_select(0, indices)
        if self.cfg.freeze_object_rotation:
            surface = self.object_surface_rotated_initial.index_select(0, indices) + translation[:, None, :] if include_surface else None
            vertices = self.object_vertices_rotated_initial.index_select(0, indices) + translation[:, None, :]
        else:
            surface = self.object_surface[None] @ rotation.transpose(-1, -2) + translation[:, None, :]
            vertices = self.object_vertices[None] @ rotation.transpose(-1, -2) + translation[:, None, :]
        return rotation, translation, surface, vertices

    def _object_rotation_for_indices(self, indices: Any) -> Any:
        if self.cfg.freeze_object_rotation:
            return self.object_rotation_initial.index_select(0, indices)
        from pytorch3d.transforms import axis_angle_to_matrix
        return axis_angle_to_matrix(self.object_axis.index_select(0, indices))

    def _body_pose_for_indices(self, indices: Any) -> Any:
        body_pose = self.body_pose.index_select(0, indices)
        if not self.cfg.freeze_body_internal_translations:
            return body_pose
        initial = self.body_pose_initial.index_select(0, indices)
        return torch_cat((body_pose, initial[..., BODY_CONT_INTERNAL_TRANSLATION_SLICE]), dim=-1)

    def _decode_body(self, indices: Any) -> Any:
        params = {key: value.index_select(0, indices) for key, value in self.params_fixed.items()}
        params["mhr_body_pose_cont"] = self._body_pose_for_indices(indices)
        return self.mhr_layer.mhr_forward(params)

    def _initialize_pose_diagnostics(self) -> None:
        torch = _torch()
        joints_chunks, foot_index_chunks, foot_vertex_chunks = [], [], []
        batch_size = self._resolved_batch_size()
        with torch.no_grad():
            for start in range(0, len(self.frame_indices), batch_size):
                indices = torch.arange(start, min(start + batch_size, len(self.frame_indices)), device=self.device, dtype=torch.long)
                body = self._decode_body(indices)
                vertices, joints, keypoints, coco17 = _layer_output_vertices(body), _layer_output_joints(body), _layer_output_keypoints(body), _layer_output_coco17(body)
                if vertices is None or joints is None or keypoints is None:
                    raise RuntimeError("MHR pose diagnostics require decoded vertices, joints, and keypoints")
                if coco17 is None:
                    coco17 = mhr70_to_coco17(keypoints)
                if tuple(coco17.shape[-2:]) != (17, 3):
                    raise ValueError(f"MHR COCO17 diagnostics must have shape [T,17,3], got {tuple(coco17.shape)}")
                foot_count = min(MHR_FOOT_DIAGNOSTIC_VERTICES_PER_SIDE, int(vertices.shape[1]))
                foot_indices = torch.topk(torch.cdist(coco17[:, (15, 16)], vertices), k=foot_count, dim=-1, largest=False).indices
                flat_indices = foot_indices.reshape(len(indices), -1)
                foot_vertices = torch.gather(vertices, 1, flat_indices[..., None].expand(-1, -1, 3)).reshape(len(indices), 2, foot_count, 3)
                joints_chunks.append(joints.detach())
                foot_index_chunks.append(foot_indices.detach())
                foot_vertex_chunks.append(foot_vertices.detach())
        self.reference_joints = torch.cat(joints_chunks, dim=0)
        self.reference_foot_vertex_indices = torch.cat(foot_index_chunks, dim=0)
        self.reference_foot_vertices = torch.cat(foot_vertex_chunks, dim=0)

    def _human_pose_prior(self, indices: Any) -> Any:
        current = self.body_pose.index_select(0, indices)[..., :MHR_BODY_ROTATION_CONTROL_COUNT]
        reference = self.body_pose_rotation_initial.index_select(0, indices)
        return _torch().nn.functional.smooth_l1_loss(current, reference, beta=self.cfg.human_pose_prior_beta, reduction="mean")

    def _human_pose_prior_gradient_metrics(self, indices: Any) -> dict[str, Any]:
        torch = _torch()
        delta = self.body_pose.index_select(0, indices)[..., :MHR_BODY_ROTATION_CONTROL_COUNT] - self.body_pose_rotation_initial.index_select(0, indices)
        gradient = torch.where(delta.abs() < self.cfg.human_pose_prior_beta, delta / self.cfg.human_pose_prior_beta, delta.sign()) * (self.cfg.w_human_pose_prior / max(delta.numel(), 1))
        return {"human_pose_prior_gradient_abs_mean": gradient.abs().mean(), "human_pose_prior_gradient_abs_max": gradient.abs().max()}

    def _pose_drift_metrics(self, vertices: Any, joints: Any, indices: Any) -> dict[str, Any]:
        torch = _torch()
        delta = self.body_pose.index_select(0, indices)[..., :MHR_BODY_ROTATION_CONTROL_COUNT] - self.body_pose_rotation_initial.index_select(0, indices)
        joint_displacement = torch.linalg.vector_norm(joints - self.reference_joints.index_select(0, indices), dim=-1)
        foot_indices = self.reference_foot_vertex_indices.index_select(0, indices)
        flat_indices = foot_indices.reshape(len(indices), -1)
        foot_vertices = torch.gather(vertices, 1, flat_indices[..., None].expand(-1, -1, 3)).reshape_as(self.reference_foot_vertices.index_select(0, indices))
        foot_displacement = torch.linalg.vector_norm(foot_vertices - self.reference_foot_vertices.index_select(0, indices), dim=-1)
        return {
            "human_pose_parameter_abs_drift_mean": delta.abs().mean(),
            "human_pose_parameter_abs_drift_max": delta.abs().max(),
            "human_pose_parameter_l2_drift_mean": torch.linalg.vector_norm(delta, dim=-1).mean(),
            "decoded_joint_displacement_mean_m": joint_displacement.mean(),
            "decoded_joint_displacement_max_m": joint_displacement.max(),
            "foot_vertex_displacement_mean_m": foot_displacement.mean(),
            "foot_vertex_displacement_max_m": foot_displacement.max(),
        }

    def _full_pose_diagnostics(self) -> dict[str, float]:
        torch = _torch()
        mean_keys = ("human_pose_parameter_abs_drift_mean", "human_pose_parameter_l2_drift_mean", "decoded_joint_displacement_mean_m", "foot_vertex_displacement_mean_m")
        max_keys = ("human_pose_parameter_abs_drift_max", "decoded_joint_displacement_max_m", "foot_vertex_displacement_max_m")
        weighted_sums = {key: 0.0 for key in mean_keys}
        maxima = {key: 0.0 for key in max_keys}
        batch_size = self._resolved_batch_size()
        with torch.no_grad():
            for start in range(0, len(self.frame_indices), batch_size):
                indices = torch.arange(start, min(start + batch_size, len(self.frame_indices)), device=self.device, dtype=torch.long)
                body = self._decode_body(indices)
                vertices, joints = _layer_output_vertices(body), _layer_output_joints(body)
                if vertices is None or joints is None:
                    raise RuntimeError("MHR final pose diagnostics require decoded vertices and joints")
                metrics = self._pose_drift_metrics(vertices, joints, indices)
                for key in mean_keys:
                    weighted_sums[key] += float(metrics[key].detach().cpu()) * len(indices)
                for key in max_keys:
                    maxima[key] = max(maxima[key], float(metrics[key].detach().cpu()))
            all_indices = torch.arange(len(self.frame_indices), device=self.device, dtype=torch.long)
            prior_raw = float(self._human_pose_prior(all_indices).detach().cpu())
        return {"human_pose_prior_raw": prior_raw, "human_pose_prior_weighted": prior_raw * self.cfg.w_human_pose_prior, **{key: value / len(self.frame_indices) for key, value in weighted_sums.items()}, **maxima}

    def _contact_loss(self, vertices: Any, rotation: Any, translation: Any, indices: Any) -> tuple[Any, Any]:
        hands_world = vertices[:, self.hand_surface_vertex_indices, :]
        hands_object = (hands_world - translation[:, None, None, :]) @ rotation[:, None, :, :]
        point_distances = _point_triangle_surface_distances(hands_object, self.object_vertices, self.object_faces)
        distances = point_distances.min(dim=-1).values
        mask = self.contact_mask.index_select(0, indices).to(dtype=distances.dtype)
        return (distances.square() * mask).mean(), distances

    def _silhouette_loss(self, object_vertices_world: Any, indices: Any) -> Any:
        if self.cfg.w_silhouette == 0:
            return _zero_like_tensor(object_vertices_world)
        import Utils

        target = self.object_mask.index_select(0, indices)
        human = self.human_mask.index_select(0, indices)
        keep = (~(human > 0.5)) | (target > 0.5)
        height, width = target.shape[-2:]
        K = _to_numpy(self.postopt_K_rois.index_select(0, indices))
        color, _depth, _xyz = Utils.nvdiff_color_depth_render(K, self.glctx, self.object_silhouette_tensors, (height, width), object_vertices_world, depth_only=False)
        rendered = color.mean(dim=-1) * keep.to(dtype=color.dtype)
        return (rendered - target).square().sum(dim=(1, 2)).mean()

    def _temporal_loss(self, joints: Any, object_surface_world: Any, translation: Any, indices: Any | None = None) -> Any:
        human_acceleration = _public_acceleration_loss(joints)
        if self.cfg.symmetric_object:
            object_acceleration = _public_acceleration_loss(translation)
        elif self.cfg.freeze_object_rotation:
            if indices is None:
                raise ValueError("frozen object temporal acceleration requires frame indices")
            if len(indices) < 3:
                object_acceleration = _zero_like_tensor(translation)
            else:
                # _batch_indices constructs contiguous windows, so each leading frame indexes its cached second difference.
                rotation_acceleration_mean = self.object_surface_acceleration_mean_initial.index_select(0, indices[:-2])
                rotation_acceleration_square_mean = self.object_surface_acceleration_square_mean_initial.index_select(0, indices[:-2])
                translation_acceleration = translation[:-2] - 2.0 * translation[1:-1] + translation[2:]
                object_acceleration = (rotation_acceleration_square_mean + 2.0 * (rotation_acceleration_mean * translation_acceleration).sum(dim=-1) + translation_acceleration.square().sum(dim=-1)).mean()
        else:
            object_acceleration = _public_acceleration_loss(object_surface_world)
        return human_acceleration + object_acceleration

    def _penetration_loss(self, vertices: Any, rotation: Any, translation: Any, indices: Any, step: int) -> Any:
        if self.cfg.w_penetration == 0 or step <= self.cfg.penetration_start_fraction * self.cfg.num_steps:
            return _zero_like_tensor(vertices)
        if self.penetration_proxy_source_indices is not None:
            vertices = apply_mhr_collision_proxy(vertices, self.penetration_proxy_source_indices, self.penetration_proxy_barycentric_weights)
        if self.cfg.freeze_object_rotation:
            object_points_world = self.penetration_surface_rotated_initial.index_select(0, indices) + translation[:, None, :]
        else:
            object_points_world = self.penetration_surface[None] @ rotation.transpose(-1, -2) + translation[:, None, :]
        return object_inside_human_penetration_loss(vertices, self.penetration_human_faces, object_points_world, weight=1.0, frame_chunk_size=8, validate_face_indices=False, point_reduction=self.cfg.penetration_point_reduction, conservative_aabb_rejection=self.cfg.penetration_bbox_rejection, aabb_margin_m=self.cfg.penetration_bbox_margin_m)

    def loss(self, indices: Any, step: int, *, include_diagnostics: bool = True) -> tuple[Any, dict[str, Any]]:
        rotation, translation, object_surface_world, object_vertices_world = self._object_state(indices, include_surface=not self.cfg.freeze_object_rotation)
        body = self._decode_body(indices)
        vertices, joints, keypoints = _layer_output_vertices(body), _layer_output_joints(body), _layer_output_keypoints(body)
        if vertices is None or joints is None or keypoints is None:
            raise RuntimeError("MHR parity post-optimization requires decoded vertices, joints, and keypoints")
        if self.cfg.w_contact != 0:
            contact, distances = self._contact_loss(vertices, rotation, translation, indices)
        else:
            contact = _zero_like_tensor(vertices)
            distances = _torch().zeros((len(indices), 2), device=self.device, dtype=self.dtype)
        human_pose_prior_raw = self._human_pose_prior(indices)
        human_pose_prior_weighted = human_pose_prior_raw * self.cfg.w_human_pose_prior
        losses = {
            "loss_contact": contact * self.cfg.w_contact,
            "loss_silhouette": self._silhouette_loss(object_vertices_world, indices) * self.cfg.w_silhouette,
            "loss_penetration": self._penetration_loss(vertices, rotation, translation, indices, step) * self.cfg.w_penetration,
            "loss_human_pose_prior": human_pose_prior_weighted,
            "loss_temporal": self._temporal_loss(joints, object_surface_world, translation, indices) * self.cfg.w_temporal,
            "loss_object_translation_prior": (translation - self.object_translation_initial.index_select(0, indices)).square().sum(dim=-1).mean() * self.cfg.w_object_translation_prior,
        }
        total = sum(losses.values())
        diagnostics = {**self._human_pose_prior_gradient_metrics(indices), **self._pose_drift_metrics(vertices, joints, indices)} if include_diagnostics else {}
        return total, {**losses, "human_pose_prior_raw": human_pose_prior_raw, "human_pose_prior_weighted": human_pose_prior_weighted, **diagnostics, "loss_total": total, "contact_distance_mean": distances.mean(), "network_contact_entries": self.network_contact_mask.index_select(0, indices).sum(), "initial_proximity_entries": self.initial_contact_proximity_mask.index_select(0, indices).sum(), "contact_entries": self.contact_mask.index_select(0, indices).sum()}

    def run(self) -> dict[str, Any]:
        report_every = max(1, int(self.cfg.report_every))
        for step in range(self.start_step, self.cfg.num_steps + 1):
            indices = self._batch_indices()
            report = step == self.start_step or step == self.cfg.num_steps or step % report_every == 0
            diagnose = _should_run_postopt_diagnostics(step, self.start_step, self.cfg.num_steps, int(self.cfg.diagnostics_every))
            self.optimizer.zero_grad(set_to_none=True)
            total, metrics = self.loss(indices, step, include_diagnostics=diagnose)
            total.backward()
            gradient_metrics = {}
            if diagnose:
                selected_gradient = _torch().zeros_like(self.body_pose.index_select(0, indices)[..., :MHR_BODY_ROTATION_CONTROL_COUNT]) if self.body_pose.grad is None else self.body_pose.grad.index_select(0, indices)[..., :MHR_BODY_ROTATION_CONTROL_COUNT]
                gradient_metrics = {"body_pose_total_gradient_abs_mean": selected_gradient.abs().mean(), "body_pose_total_gradient_abs_max": selected_gradient.abs().max()}
            self.optimizer.step()
            self.scheduler.step()
            if report:
                self.history.append({"iter": float(step), "batch_start": float(indices[0]), "batch_size": float(len(indices)), "lr": float(self.scheduler.get_last_lr()[0]), **{key: float(value.detach().cpu()) for key, value in {**metrics, **gradient_metrics}.items()}})
                print("MHR_POSTOPT_PROGRESS " + json.dumps({"current": int(step), "total": int(self.cfg.num_steps), "unit": "optimizer steps"}, sort_keys=True), flush=True)
            if step == self.cfg.num_steps or (self.cfg.save_every > 0 and (step + 1) % self.cfg.save_every == 0):
                self._save_checkpoint(step + 1)
        return self.result()

    def result(self) -> dict[str, Any]:
        result = copy.deepcopy(self.bundle)
        all_indices = _torch().arange(len(self.frame_indices), device=self.device, dtype=_torch().long)
        pose = pose_matrix(self._object_rotation_for_indices(all_indices).detach(), self.object_translation.detach())
        result.setdefault("pr_initial", copy.deepcopy(result["pr"]))
        result["pr"] = copy.deepcopy(result["pr"])
        result["pr"]["pose_abs"] = _to_numpy(pose).astype(np.float32)
        result["pr"]["pose_abs_postopt"] = _to_numpy(pose).astype(np.float32)
        result["pr"]["mhr_body_pose_cont"] = _to_numpy(self._body_pose_for_indices(all_indices)).astype(np.float32)
        optimized_parameters = ["object_translation", "mhr_body_pose_cont_rotation_controls"] if self.cfg.freeze_body_internal_translations else ["object_translation", "mhr_body_pose_cont"]
        fixed_parameters = ["mhr_global_rot6d", "mhr_trans", "mhr_hand", "mhr_shape", "mhr_scale", "mhr_face"] + (["mhr_body_pose_cont_internal_translations"] if self.cfg.freeze_body_internal_translations else [])
        public_smplh_differences = ["MHR rotational body-pose parameterization with internal translations fixed", "hand-surface contact replaces SMPL-H wrist-joint contact", "contact eligibility is frozen from the raw CoCoNet pose and object geometry before optimization", "CoCoNet-anchored robust human-pose prior replaces detector-based 2D reprojection"]
        if self.cfg.freeze_object_rotation:
            fixed_parameters.insert(0, "object_rotation")
            public_smplh_differences.insert(0, "object rotation fixed at the CoCoNet prediction")
        else:
            optimized_parameters.insert(0, "object_rotation")
        result["postopt"] = {"mode": "smplh_parity", "checkpoint_schema": MHR_PARITY_POSTOPT_CHECKPOINT_SCHEMA, "batch_sampling": self._batch_sampling_identity(), "resolved_batch_size": self._resolved_batch_size(), "object_mesh_loader_revision": OBJECT_MESH_SCENE_LOADER_REVISION, "frame_indices": self.frame_indices.tolist(), "config": asdict(self.cfg), "history": self.history, "final_diagnostics": self._full_pose_diagnostics(), "optimized_parameters": optimized_parameters, "fixed_parameters": fixed_parameters, "human_pose_prior_reference": "immutable CoCoNet pr.mhr_body_pose_cont rotation controls at post-optimization entry", "human_pose_prior_control_count": MHR_BODY_ROTATION_CONTROL_COUNT, "foot_diagnostic_definition": f"{MHR_FOOT_DIAGNOSTIC_VERTICES_PER_SIDE} initial MHR vertices nearest each CoCoNet-predicted COCO17 ankle", "contact_definition": "minimum MHR hand-surface vertex to object triangle-surface distance", "contact_activation": _contact_activation_summary(self.network_contact_mask, self.initial_contact_distances_m, self.initial_contact_proximity_mask, self.contact_mask, distance_threshold_m=self.cfg.contact_activation_distance_m, network_selection="binary_logits"), "contact_training_consistency": "same hand vertices and object triangle surface as label generation; differentiable optimization omits discrete containment certification", "crop_contract": POSTOPT_CROP_CONTRACT, "silhouette_render_size": POSTOPT_RENDER_SIZE, "public_smplh_differences": public_smplh_differences}
        return result


def torch_clamp_min(value: Any, minimum: float) -> Any:
    torch = _torch()
    return torch.clamp(value, min=minimum)


def torch_clamp_max(value: Any, maximum: float) -> Any:
    torch = _torch()
    return torch.clamp(value, max=maximum)


def torch_cat(values: Sequence[Any], dim: int = 0) -> Any:
    return _torch().cat(tuple(values), dim=dim)


def run_postopt_stage(
    bundle: Mapping[str, Any],
    object_vertices: Any,
    cfg: MHRPostOptConfig,
    *,
    object_faces: Any | None = None,
    mhr_layer: Any | None = None,
    sdf_fn: Callable[[Any], Any] | None = None,
    hand_surface_vertex_indices: Any | None = None,
) -> dict[str, Any]:
    optimizer = MHRPostOptimizer(
        bundle,
        object_vertices,
        cfg,
        object_faces=object_faces,
        mhr_layer=mhr_layer,
        sdf_fn=sdf_fn,
        hand_surface_vertex_indices=hand_surface_vertex_indices,
    )
    return optimizer.run()


def run_postopt_smplh_parity(bundle: Mapping[str, Any], object_vertices: Any, object_faces: Any, cfg: MHRParityPostOptConfig | Mapping[str, Any] | None = None, *, mhr_layer: Any) -> dict[str, Any]:
    return MHRParityPostOptimizer(bundle, object_vertices, object_faces, cfg, mhr_layer=mhr_layer).run()


def run_postopt_all_stages(
    bundle: Mapping[str, Any],
    object_vertices: Any,
    cfg: MHRPostOptConfig,
    *,
    object_faces: Any | None = None,
    mhr_layer: Any | None = None,
    sdf_fn: Callable[[Any], Any] | None = None,
    iterations_per_stage: Sequence[int] | None = None,
    hand_surface_vertex_indices: Any | None = None,
) -> dict[str, Any]:
    current: Mapping[str, Any] = bundle
    stage_history = []
    contact_activation_state = None
    for index, stage in enumerate(STAGE_ORDER):
        iterations = cfg.iterations
        if iterations_per_stage is not None:
            iterations = int(iterations_per_stage[index])
        stage_cfg = replace(
            cfg,
            stage=stage,
            iterations=iterations,
            frame_start=cfg.frame_start if index == 0 else 0,
            frame_limit=cfg.frame_limit if index == 0 else 0,
        )
        optimizer = MHRPostOptimizer(
            current,
            object_vertices,
            stage_cfg,
            object_faces=object_faces,
            mhr_layer=mhr_layer,
            sdf_fn=sdf_fn,
            hand_surface_vertex_indices=hand_surface_vertex_indices,
            _contact_activation_state=contact_activation_state,
        )
        if contact_activation_state is None and stage_cfg.w_contact != 0:
            contact_activation_state = optimizer._contact_activation_state()
        current = optimizer.run()
        stage_history.append(current["postopt"])
    result = copy.deepcopy(current)
    result["postopt_all_stages"] = stage_history
    return result


def _load_bundle(path: str | Path) -> Mapping[str, Any]:
    torch = _torch()
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_object_vertices(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    mesh = load_object_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = None if getattr(mesh, "faces", None) is None else np.asarray(mesh.faces, dtype=np.int64)
    return vertices, faces


def _parse_iterations_per_stage(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    values = [int(item) for item in text.split(",")]
    if len(values) != len(STAGE_ORDER):
        raise ValueError(f"--iterations-per-stage needs {len(STAGE_ORDER)} comma-separated integers")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native MHR post-CoCoNet optimization.")
    parser.add_argument("--bundle", required=True, help="Compact MHR forward-viz .pth bundle.")
    parser.add_argument("--object-mesh", required=True, help="Object mesh used for contact and SDF terms.")
    parser.add_argument("--out", required=True, help="Output .pth path.")
    parser.add_argument("--mode", choices=("smplh_parity", "legacy_staged"), default="smplh_parity")
    parser.add_argument("--stage", default="all", help="'all' or one stage id 1..5.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--iterations-per-stage", default=None)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--object-surface-samples", type=int, default=None)
    parser.add_argument("--human-surface-samples", type=int, default=512)
    parser.add_argument("--penetration-surface-samples", type=int, default=None)
    parser.add_argument("--penetration-frame-chunk-size", type=int, default=8)
    parser.add_argument("--penetration-human-mesh-mode", choices=MHR_COLLISION_PROXY_MODES, default=MHR_COLLISION_PROXY_MODE_4000)
    parser.add_argument("--penetration-collision-proxy-path", default=str(DEFAULT_MHR_COLLISION_PROXY_ASSET))
    parser.add_argument("--penetration-bbox-rejection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--penetration-bbox-margin-m", type=float, default=1e-6)
    parser.add_argument("--contact-topk", type=int, default=64)
    parser.add_argument("--contact-activation-distance-m", type=float, default=MHR_DEFAULT_CONTACT_ACTIVATION_DISTANCE_M, help="Freeze contact eligibility at initialization when the raw CoCoNet hand-surface distance is strictly below this value.")
    parser.add_argument("--w-contact", type=float, default=None)
    parser.add_argument("--w-sdf", type=float, default=0.0)
    parser.add_argument("--w-pen", type=float, default=None)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--diagnostics-every", type=int, default=500)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=MHR_POSTOPT_FULL_CLIP_BATCH_SIZE, help="Frames optimized together; 0 uses every selected frame in the clip.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-silhouette", type=float, default=0.002)
    parser.add_argument("--w-human-pose-prior", type=float, default=200.0)
    parser.add_argument("--human-pose-prior-beta", type=float, default=0.05)
    parser.add_argument("--w-temporal", type=float, default=100.0)
    parser.add_argument("--w-object-translation-prior", type=float, default=100.0)
    parser.add_argument("--optimize-object-rotation", action="store_true", help="Optimize object rotation instead of retaining the CoCoNet prediction.")
    parser.add_argument("--symmetric-object", action="store_true")
    parser.add_argument("--postopt-checkpoint", default=None)
    parser.add_argument("--mhr-assets-root", default=None)
    parser.add_argument("--no-mhr-layer", action="store_true")
    return parser


def _run_cli(args: argparse.Namespace, profiler: PipelineTimer) -> None:
    setup_started = profiler.start()
    bundle = _load_bundle(args.bundle)
    object_vertices, object_faces = _load_object_vertices(args.object_mesh)
    mhr_layer = None
    if not args.no_mhr_layer:
        mhr_layer = MHRLayer.from_mhr_assets(mhr_assets_root=args.mhr_assets_root, device=args.device)
    profiler.record("setup", setup_started)
    profiler.update_metadata({"frames": len(bundle.get("frames", [])), "mode": args.mode, "requested_steps": int(args.num_steps) if args.mode == "smplh_parity" else int(sum(_parse_iterations_per_stage(args.iterations_per_stage))) if str(args.stage).lower() == "all" else int(args.iterations)})
    optimization_started = profiler.start()
    if args.mode == "smplh_parity":
        if mhr_layer is None:
            raise ValueError("--mode smplh_parity requires the MHR layer")
        if object_faces is None:
            raise ValueError("--mode smplh_parity requires object triangle faces")
        cfg = MHRParityPostOptConfig(num_steps=args.num_steps, batch_size=args.batch_size, device=args.device, frame_start=args.frame_start, frame_limit=args.frame_limit, lr=args.lr, warmup_steps=args.num_steps // 10, schedule_steps=int(args.num_steps * 1.5), object_surface_samples=1000 if args.object_surface_samples is None else args.object_surface_samples, penetration_surface_samples=1000 if args.penetration_surface_samples is None else args.penetration_surface_samples, penetration_human_mesh_mode=args.penetration_human_mesh_mode, penetration_collision_proxy_path=args.penetration_collision_proxy_path, penetration_bbox_rejection=args.penetration_bbox_rejection, penetration_bbox_margin_m=args.penetration_bbox_margin_m, contact_activation_distance_m=args.contact_activation_distance_m, w_contact=200.0 if args.w_contact is None else args.w_contact, w_silhouette=args.w_silhouette, w_penetration=MHR_DEFAULT_PENETRATION_WEIGHT if args.w_pen is None else args.w_pen, w_human_pose_prior=args.w_human_pose_prior, human_pose_prior_beta=args.human_pose_prior_beta, w_temporal=args.w_temporal, w_object_translation_prior=args.w_object_translation_prior, symmetric_object=args.symmetric_object, report_every=args.report_every, diagnostics_every=args.diagnostics_every, checkpoint_path=args.postopt_checkpoint, freeze_object_rotation=not args.optimize_object_rotation)
        result = run_postopt_smplh_parity(bundle, object_vertices, object_faces, cfg, mhr_layer=mhr_layer)
    else:
        cfg = MHRPostOptConfig(
            iterations=args.iterations,
            device=args.device,
            frame_start=args.frame_start,
            frame_limit=args.frame_limit,
            object_surface_samples=4096 if args.object_surface_samples is None else args.object_surface_samples,
            human_surface_samples=args.human_surface_samples,
            penetration_surface_samples=6000 if args.penetration_surface_samples is None else args.penetration_surface_samples,
            penetration_frame_chunk_size=args.penetration_frame_chunk_size,
            penetration_human_mesh_mode=args.penetration_human_mesh_mode,
            penetration_collision_proxy_path=args.penetration_collision_proxy_path,
            penetration_bbox_rejection=args.penetration_bbox_rejection,
            penetration_bbox_margin_m=args.penetration_bbox_margin_m,
            contact_topk=args.contact_topk,
            contact_activation_distance_m=args.contact_activation_distance_m,
            w_contact=1.0 if args.w_contact is None else args.w_contact,
            w_sdf=args.w_sdf,
            w_pen=MHR_DEFAULT_PENETRATION_WEIGHT if args.w_pen is None else args.w_pen,
            report_every=args.report_every,
        )
        iterations_per_stage = _parse_iterations_per_stage(args.iterations_per_stage)
        if str(args.stage).lower() == "all":
            result = run_postopt_all_stages(bundle, object_vertices, cfg, object_faces=object_faces, mhr_layer=mhr_layer, iterations_per_stage=iterations_per_stage)
        else:
            result = run_postopt_stage(bundle, object_vertices, replace(cfg, stage=int(args.stage)), object_faces=object_faces, mhr_layer=mhr_layer)
    profiler.record("optimization", optimization_started)

    output_started = profiler.start()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _torch().save(result, out_path, pickle_protocol=4)
    summary = result.get("postopt", {})
    if "postopt_all_stages" in result:
        summary = {"stages": [item["stage_name"] for item in result["postopt_all_stages"]]}
    print(json.dumps({"out": str(out_path), **summary}, indent=2))
    profiler.record("result_write_and_summary", output_started)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    with PipelineTimer("contact_guided_refinement") as profiler:
        _run_cli(args, profiler)


if __name__ == "__main__":
    main()
