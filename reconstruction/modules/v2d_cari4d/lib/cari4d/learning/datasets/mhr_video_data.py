from __future__ import annotations

import json
import os.path as osp
import pickle
import re
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Mapping

import h5py
import joblib
import numpy as np
import torch

from lib_mhr import COCO17_TO_MHR70, OBJECT_POSE_FRAME, OBJECT_SYMMETRY_MODE_FINITE, OBJECT_SYMMETRY_MODE_FULL_SO3, assert_mhr_schema, compute_mhr_delta, mhr70_to_coco17, object_poses_to_training_frame, object_rotations_translations_to_training_frame, object_symmetry_to_training_frame, resolve_object_pose_frame
from lib_mhr.camera_conventions import MHR_ROOT_JOINT_INDEX, mhr_init_translation_is_decode_consistent, mhr_root_rot6d_between_frames, mhr_translation_between_frames, validate_mhr_init_root_metadata
from lib_mhr.schema import MHR_OPTIONAL_PARAM_KEYS, MHR_PARAM_DIMS, MHR_REQUIRED_PARAM_KEYS
from learning.datasets.mhr_dataset_index import load_dataset_index
from learning.datasets.mhr_augmentation import MHR_INPUT_AUGMENTATION_DISABLED, MHR_INPUT_AUGMENTATION_KEEP_ALIGNED_DEPTH, resolve_mhr_input_augmentation_mode
from learning.datasets.mhr_input_materialization import MHR_ENCODED_DEPTHS_KEY, MHR_ENCODED_MASKS_KEY, MHR_ENCODED_RENDER_KEYS, MHR_ENCODED_RGBS_KEY, MHR_INPUT_MATERIALIZATION_GPU, MHR_NEUTRAL_HEIGHT_KEY, MHR_XYZ_ANCHOR_KEY, resolve_mhr_input_materialization_mode, resolve_mhr_xyz_anchor_type, select_mhr_xyz_anchor, validate_mhr_gpu_materialization_config
from learning.datasets.mhr_rank_local import MHR_LARGE_RENDER_KEYS, MHR_RENDER_REQUEST_KEY, MHRRenderRequest
from learning.datasets.mhr_tier_sampling import MHR_TIER_SAMPLING_REVISION, configured_tier_sampling_revision, stateless_clip_tiers, tier_sampling_contract
from learning.datasets.mhr_window_sampling import MHR_CANONICAL_OBJECT_NONEMPTY_FRAME_FRACTION_DEFAULT, MHR_CANONICAL_OBJECT_NONEMPTY_WINDOW_REVISION, MHR_WINDOW_SAMPLING_LEGACY, build_mhr_window_sampling_contract, load_mhr_interaction_trim, mhr_window_starts_and_strides, minimum_true_frame_fraction_window_mask
from prep.mhr_effective_masks import EFFECTIVE_MASK_WINDOW_VALIDITY_REVISION
from prep.mhr_export_utils import load_canonical_effective_mask_validity
from prep.mhr_foundationpose_training_tiers import FOUNDATIONPOSE_TRAINING_TIER_H5_PATHS, FOUNDATIONPOSE_TRAINING_TIER_REVISION
from prep.mhr_gt_object_visibility import GT_OBJECT_VISIBILITY_FIELDS, GT_OBJECT_VISIBILITY_H5_PATHS
from prep.mhr_packed_h5 import GEOMETRY_KEYS, OPTIONAL_AUX_KEYS, PARAM_KEYS, SCHEMA, SUPPORTED_SCHEMAS, TIER_SCHEMAS, TRAINING_TIER_SCHEMA, validate_packed_mhr_hand_surface_contacts, validate_packed_mhr_init_root
from prep.mhr_render_shards import MHR_RENDER_RECORD_MODE_LEGACY, MHR_RENDER_RECORD_MODE_TIER_ONLY
from render_h5_codec import load_pickled_dataset


MHR_PARAM_TARGET_SPECS = (
    ("mhr_global_rot6d", "delta_mhr_global_rot6d", "w_mhr_root_rot"),
    ("mhr_trans", "delta_mhr_trans", "w_mhr_trans"),
    ("mhr_body_pose_cont", "delta_mhr_body_pose_cont", "w_mhr_body_pose"),
    ("mhr_hand", "delta_mhr_hand", "w_mhr_hand"),
    ("mhr_shape", "delta_mhr_shape", "w_mhr_shape"),
    ("mhr_scale", "delta_mhr_scale", "w_mhr_scale"),
    ("mhr_face", "delta_mhr_face", "w_mhr_face"),
)
MHR_CAMERA_POINT_KEYS = ("mhr_joints", "mhr_keypoints", "mhr_coco17", "mhr_contact_closest_points_gt")
MHR_INTERNAL_ROOT_JOINT_KEY = "_mhr_root_joint"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _profile_start(profile: dict[str, Any] | None) -> float | None:
    return time.monotonic() if profile is not None else None


def _profile_add(profile: dict[str, Any] | None, key: str, started_at: float | None) -> None:
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + time.monotonic() - float(started_at)


def _parse_kid_filter(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        items = [item for item in re.split(r"[\s,]+", text) if item]
    elif isinstance(value, (int, np.integer)):
        items = [value]
    else:
        items = list(value)
    out = []
    for item in items:
        kid = int(item)
        if kid not in out:
            out.append(kid)
    return out or None


def _load_mapping(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _candidate_files(root: str | None, seq: str, suffixes: tuple[str, ...]) -> list[str]:
    if not root:
        return []
    return [osp.join(root, f"{seq}{suffix}") for suffix in suffixes]


def _first_existing(paths: list[str]) -> str | None:
    for item in paths:
        if osp.isfile(item):
            return item
    return None


def _extract_prefixed(data: Mapping[str, Any], suffix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in list(MHR_PARAM_DIMS) + ["mhr_joints", "mhr_keypoints", "mhr_joint_global_rots"]:
        suffixed = f"{key}{suffix}"
        if suffixed in data:
            out[key] = data[suffixed]
    return out


def _take_frame_axis(value: Any, start: int, indices: np.ndarray, *trailing_indices: Any) -> np.ndarray:
    absolute_indices = int(start) + np.asarray(indices, dtype=np.int64)
    if absolute_indices.ndim != 1 or absolute_indices.size == 0 or np.any(absolute_indices < 0):
        raise ValueError(f"Frame indices must be a nonempty nonnegative vector, got {absolute_indices}")
    order = np.argsort(absolute_indices, kind="stable")
    sorted_indices = absolute_indices[order]
    if len(np.unique(sorted_indices)) != len(sorted_indices):
        raise ValueError(f"Frame indices must be unique, got {absolute_indices}")
    if len(sorted_indices) == 1:
        selector = slice(int(sorted_indices[0]), int(sorted_indices[0]) + 1)
    else:
        differences = np.diff(sorted_indices)
        selector = slice(int(sorted_indices[0]), int(sorted_indices[-1] + differences[0]), int(differences[0])) if np.all(differences == differences[0]) else sorted_indices.tolist()
    result = np.asarray(value[(selector, *trailing_indices)]).copy()
    if result.shape[0] != len(absolute_indices):
        raise IndexError(f"Frame selection returned {result.shape[0]} values for {len(absolute_indices)} requested indices ending at {int(sorted_indices[-1])}")
    return result[np.argsort(order, kind="stable")]


def _extract_frame_slice(data: Mapping[str, Any], start: int, end: int, indices: np.ndarray, keys: set[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if keys is not None and key not in keys:
            continue
        if key in {"frames", "kids", "body_model", "faces"}:
            continue
        if key in {"obj_symmetry_tfs", "obj_symmetry_mode", "obj_symmetry_center"}:
            out[key] = value
            continue
        arr = np.asarray(value)
        if len(arr.shape) > 0 and arr.shape[0] >= end:
            out[key] = _take_frame_axis(arr, start, indices)
    return out


def _select_view(data: Mapping[str, Any], kids: list[int], kid: int) -> dict[str, Any]:
    if kid not in kids:
        return dict(data)
    view_idx = kids.index(kid)
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in {"obj_symmetry_tfs", "obj_symmetry_mode", "obj_symmetry_center"}:
            out[key] = value
            continue
        arr = np.asarray(value)
        if arr.ndim >= 2 and arr.shape[1] == len(kids) and (arr.ndim >= 3 or key in GT_OBJECT_VISIBILITY_FIELDS):
            out[key] = arr[:, view_idx].copy()
        else:
            out[key] = value
    return out


def _load_h5_pickle(node: Any) -> Any:
    return load_pickled_dataset(node)


def _add_mhr_coco17_fields(sample: dict[str, Any]) -> None:
    for suffix in ("init", "gt"):
        coco_key = f"mhr_coco17_{suffix}"
        keypoints_key = f"mhr_keypoints_{suffix}"
        if coco_key not in sample and keypoints_key in sample:
            keypoints = np.asarray(sample[keypoints_key])
            if keypoints.shape[-2] >= max(COCO17_TO_MHR70) + 1:
                sample[coco_key] = mhr70_to_coco17(keypoints).copy()


def _pose_matrix(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    rot = np.asarray(rot, dtype=np.float32)
    if rot.shape == (3, 3):
        pose[:3, :3] = rot
    elif rot.shape == (3,):
        from scipy.spatial.transform import Rotation as R

        pose[:3, :3] = R.from_rotvec(rot).as_matrix().astype(np.float32)
    else:
        raise ValueError(f"Unsupported object rotation shape: {rot.shape}")
    pose[:3, 3] = np.asarray(trans, dtype=np.float32)
    return pose


def _object_data_to_training_frame(data: Mapping[str, Any], storage_to_training: np.ndarray) -> dict[str, Any]:
    out = dict(data)
    for rotation_key, translation_key in (("obj_rot_gt", "obj_t_gt"), ("obj_rot_init", "obj_t_init")):
        if rotation_key not in out:
            continue
        if translation_key not in out:
            raise KeyError(f"Object pose requires both {rotation_key} and {translation_key}")
        rotations = np.asarray(out[rotation_key], dtype=np.float32)
        if rotations.shape[-1:] == (3,) and rotations.shape[-2:] != (3, 3):
            from scipy.spatial.transform import Rotation as R

            rotations = R.from_rotvec(rotations.reshape(-1, 3)).as_matrix().reshape(rotations.shape[:-1] + (3, 3)).astype(np.float32)
        rotations, translations = object_rotations_translations_to_training_frame(rotations, out[translation_key], storage_to_training)
        out[rotation_key], out[translation_key] = rotations, translations
    if "obj_rot" in out and "obj_angles" in out:
        raise KeyError("Object pose contains both obj_rot and obj_angles aliases")
    rotation_key = "obj_rot" if "obj_rot" in out else "obj_angles" if "obj_angles" in out else None
    if rotation_key is not None:
        if "obj_t" not in out:
            raise KeyError(f"Object pose requires both {rotation_key} and obj_t")
        rotations = np.asarray(out[rotation_key], dtype=np.float32)
        if rotation_key == "obj_angles":
            from scipy.spatial.transform import Rotation as R

            rotations = R.from_rotvec(rotations.reshape(-1, 3)).as_matrix().reshape(rotations.shape[:-1] + (3, 3)).astype(np.float32)
        rotations, translations = object_rotations_translations_to_training_frame(rotations, out["obj_t"], storage_to_training)
        out["obj_rot"], out["obj_t"] = rotations, translations
        out.pop("obj_angles", None)
    if "obj_symmetry_tfs" in out:
        transforms, center = object_symmetry_to_training_frame(out["obj_symmetry_tfs"], out.get("obj_symmetry_center", np.zeros(3, dtype=np.float32)), storage_to_training)
        out["obj_symmetry_tfs"], out["obj_symmetry_center"] = transforms, center
    out["obj_pose_storage_to_training_transform"] = np.asarray(storage_to_training, dtype=np.float32)
    return out


def _validate_world_to_camera(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite [4,4] matrix, got {matrix.shape}")
    rotation = matrix[:3, :3].astype(np.float64)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{label} must contain a proper rotation")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{label} must be a rigid homogeneous transform")
    return matrix


def _transform_points_between_frames(points: Any, transform: np.ndarray) -> np.ndarray:
    value = np.asarray(points)
    if value.ndim < 1 or value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError(f"MHR point field must be finite and end in dimension 3, got {value.shape}")
    return (value @ transform[:3, :3].T + transform[:3, 3]).astype(value.dtype, copy=False)


def _transform_mhr_mapping_between_frames(data: Mapping[str, Any], transform: np.ndarray, *, pivot_aware_translation: bool) -> dict[str, Any]:
    out = dict(data)
    root_joint = out.pop(MHR_INTERNAL_ROOT_JOINT_KEY, None)
    if "mhr_global_rot6d" in out:
        out["mhr_global_rot6d"] = mhr_root_rot6d_between_frames(out["mhr_global_rot6d"], transform)
    if "mhr_trans" in out:
        if pivot_aware_translation:
            if root_joint is None:
                raise KeyError("Pivot-aware MHR translation conversion requires the source root joint")
            out["mhr_trans"] = mhr_translation_between_frames(out["mhr_trans"], root_joint, transform)
        else:
            out["mhr_trans"] = _transform_points_between_frames(out["mhr_trans"], transform)
    for key in MHR_CAMERA_POINT_KEYS:
        if key in out:
            out[key] = _transform_points_between_frames(out[key], transform)
    if root_joint is not None:
        out[MHR_INTERNAL_ROOT_JOINT_KEY] = _transform_points_between_frames(root_joint, transform)
    return out


def _extract_mhr_root_joint(data: Mapping[str, Any], start: int, end: int, indices: np.ndarray, *, view_index: int | None = None) -> np.ndarray:
    if "mhr_joints" not in data:
        raise KeyError("MHR frame conversion requires mhr_joints to preserve the decoder root pivot")
    joints = _take_frame_axis(np.asarray(data["mhr_joints"]), start, indices)
    if joints.ndim == 4:
        if view_index is None or view_index < 0 or view_index >= joints.shape[1]:
            raise ValueError(f"View-specific MHR joints require a valid view index, got {view_index} for {joints.shape}")
        joints = joints[:, view_index]
    if joints.ndim != 3 or joints.shape[-1] != 3 or joints.shape[-2] <= MHR_ROOT_JOINT_INDEX:
        raise ValueError(f"MHR joints cannot provide root joint {MHR_ROOT_JOINT_INDEX}: {joints.shape}")
    return joints[:, MHR_ROOT_JOINT_INDEX].copy()


def _init_translation_is_decode_consistent(sequence_data: Mapping[str, Any], kid: int) -> bool:
    metadata = sequence_data.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError(f"MHR packed metadata must be a mapping, got {type(metadata).__name__}")
    if mhr_init_translation_is_decode_consistent(metadata, "MHR packed initialization"):
        return True
    for key in ("mhr_full_refit_by_camera", "mhr_translation_uniform_scale_refit_by_camera"):
        by_camera = metadata.get(key)
        if by_camera is not None and not isinstance(by_camera, Mapping):
            raise TypeError(f"{key} must be a mapping, got {type(by_camera).__name__}")
        if isinstance(by_camera, Mapping) and str(kid) in by_camera:
            return True
    per_view = metadata.get("mhr_init_metadata")
    if per_view is not None:
        if not isinstance(per_view, (list, tuple)) or len(per_view) != len(sequence_data["kids"]):
            raise ValueError(f"mhr_init_metadata must match camera ids {sequence_data['kids']}")
        view_index = [int(value) for value in sequence_data["kids"]].index(int(kid))
        return mhr_init_translation_is_decode_consistent(per_view[view_index], f"MHR initialization camera {kid}")
    init_metadata = sequence_data.get("init_metadata", metadata)
    return mhr_init_translation_is_decode_consistent(init_metadata, f"MHR initialization camera {kid}")


class _SampleIndexView(Sequence):
    def __init__(self, dataset: "MHRVideoDataset"):
        self.dataset = dataset

    def __len__(self) -> int:
        return self.dataset._sample_count

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [self.dataset._sample_tuple(item) for item in range(*index.indices(len(self)))]
        return self.dataset._sample_tuple(index)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Sequence):
            return list(self) == list(other)
        return False


class MHRVideoDataset:
    """Native MHR dataset contract for temporal refinement.

    Expected data can be stored either as one combined mapping with
    `mhr_*_init` and `mhr_*_gt` fields or as separate init/target mappings.
    The dataset emits canonical init/target fields plus `delta_mhr_*` blocks.
    """

    def __init__(self, cfg: Any, seqs: list[str], split: str = "val"):
        self.cfg = cfg
        self.seqs = list(seqs)
        self.split = split
        self.clip_len = int(_cfg_get(cfg, "clip_len", 1))
        self.window = int(_cfg_get(cfg, "window", self.clip_len))
        self.window_sampling_contract = build_mhr_window_sampling_contract(cfg, split)
        self.min_valid_frame_fraction = float(_cfg_get(cfg, "min_valid_frame_fraction", 0.0))
        if not 0.0 <= self.min_valid_frame_fraction < 1.0:
            raise ValueError(f"min_valid_frame_fraction must be in [0, 1), got {self.min_valid_frame_fraction}")
        self.packed_root = _cfg_get(cfg, "packed_root")
        self.mhr_gt_root = _cfg_get(cfg, "mhr_gt_root", self.packed_root)
        self.mhr_init_root = _cfg_get(cfg, "mhr_init_root", _cfg_get(cfg, "nlf_root"))
        self.mhr_pseudogt_root = _cfg_get(cfg, "mhr_pseudogt_root")
        self.packed_format = str(_cfg_get(cfg, "packed_format", "pickle")).lower()
        self.render_h5_root = _cfg_get(cfg, "render_root")
        self.require_geometry = bool(_cfg_get(cfg, "require_mhr_geometry", False))
        self.random_flip = bool(_cfg_get(cfg, "random_flip", False))
        self.body_model = _cfg_get(cfg, "body_model", "mhr")
        self.input_augmentation_mode = resolve_mhr_input_augmentation_mode(cfg)
        self.input_materialization_mode = resolve_mhr_input_materialization_mode(cfg)
        self.mhr_xyz_anchor_type = resolve_mhr_xyz_anchor_type(cfg)
        self.sample_kids = _parse_kid_filter(_cfg_get(cfg, "mhr_sample_kids"))
        self.h5_handle_cache_size = int(_cfg_get(cfg, "mhr_h5_handle_cache_size", 32))
        self.rank_local_preprocess = bool(_cfg_get(cfg, "mhr_rank_local_preprocess", False))
        self.foundationpose_tier_sampling = bool(_cfg_get(cfg, "mhr_foundationpose_tier_sampling", True))
        self.foundationpose_tier_sampling_revision = configured_tier_sampling_revision(cfg)
        self.tier_sampling_seed = _cfg_get(cfg, "seed")
        self._tier_sampling_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.require_foundationpose_training_tier_schema = bool(_cfg_get(cfg, "mhr_require_foundationpose_training_tier_schema", False))
        self.required_contact_revision = _cfg_get(cfg, "mhr_required_contact_revision")
        contact_supervision_enabled = int(_cfg_get(cfg, "cont_out_dim", -1)) == 2 and float(_cfg_get(cfg, "w_contact", 0.0)) != 0.0
        if contact_supervision_enabled and not self.required_contact_revision:
            raise ValueError("MHR hand-contact supervision requires mhr_required_contact_revision")
        self.dataset_index_path = _cfg_get(cfg, "mhr_dataset_index_path")
        self.interaction_trim_root = _cfg_get(cfg, "mhr_interaction_trim_root")
        self.effective_mask_root = _cfg_get(cfg, "mhr_effective_mask_root")
        self.min_canonical_object_nonempty_frame_fraction = float(_cfg_get(cfg, "mhr_min_canonical_object_nonempty_frame_fraction", MHR_CANONICAL_OBJECT_NONEMPTY_FRAME_FRACTION_DEFAULT))
        if not 0.0 <= self.min_canonical_object_nonempty_frame_fraction <= 1.0:
            raise ValueError(f"mhr_min_canonical_object_nonempty_frame_fraction must be in [0, 1], got {self.min_canonical_object_nonempty_frame_fraction}")
        self.supports_sampler_level_resume_skip = self.packed_format == "h5" and self.rank_local_preprocess and not self.random_flip
        self.minimal_batch_fields = bool(_cfg_get(cfg, "mhr_minimal_batch_fields", False))
        self.load_init_geometry = bool(_cfg_get(cfg, "mhr_load_init_geometry", False))
        if self.load_init_geometry:
            raise ValueError("mhr_load_init_geometry is unsupported because MHR vertices are decoded from parameters")
        self.mhr_cond_key = str(_cfg_get(cfg, "mhr_cond_key", "mhr_coco17_init"))
        self.object_pose_frame = str(_cfg_get(cfg, "mhr_object_pose_frame", OBJECT_POSE_FRAME))
        init_clip_keys = set(PARAM_KEYS)
        init_clip_keys.add(MHR_NEUTRAL_HEIGHT_KEY.removesuffix("_init"))
        if self.mhr_cond_key == "mhr_coco17_init":
            init_clip_keys.add("mhr_keypoints")
        elif self.mhr_cond_key.endswith("_init"):
            source_key = self.mhr_cond_key.removesuffix("_init")
            if source_key in set(GEOMETRY_KEYS + OPTIONAL_AUX_KEYS):
                init_clip_keys.add(source_key)
        self.init_clip_keys = init_clip_keys
        self.gt_param_keys = set(PARAM_KEYS)
        self.gt_output_param_keys = set(PARAM_KEYS)
        self.delta_output_keys = {delta_key for _param_key, delta_key, _weight_key in MHR_PARAM_TARGET_SPECS}
        if self.minimal_batch_fields and self.split == "train":
            active_specs = [spec for spec in MHR_PARAM_TARGET_SPECS if float(_cfg_get(cfg, spec[2], 0.0)) != 0]
            if float(_cfg_get(cfg, "w_mhr_v2v", 0.0)) != 0:
                self.gt_param_keys = set(PARAM_KEYS)
                self.gt_output_param_keys = set(PARAM_KEYS)
            else:
                self.gt_param_keys = {"mhr_global_rot6d", *(param_key for param_key, _delta_key, _weight_key in active_specs)}
                self.gt_output_param_keys = {"mhr_shape"} if any(param_key == "mhr_shape" for param_key, _delta_key, _weight_key in active_specs) else set()
            self.delta_output_keys = {delta_key for param_key, delta_key, _weight_key in active_specs if param_key != "mhr_shape"}
        self.gt_clip_keys = set(self.gt_param_keys)
        if self.require_geometry or self.split == "val" or float(_cfg_get(cfg, "w_mhr_joints", 0.0)) != 0:
            self.gt_clip_keys.update(("mhr_keypoints", "mhr_coco17"))
        if self.body_model != "mhr":
            raise ValueError(f"MHRVideoDataset requires body_model=mhr, got {self.body_model}")
        if self.object_pose_frame != OBJECT_POSE_FRAME:
            raise ValueError(f"MHRVideoDataset requires mhr_object_pose_frame={OBJECT_POSE_FRAME!r}, got {self.object_pose_frame!r}")
        if self.packed_format not in {"pickle", "h5"}:
            raise ValueError(f"packed_format must be pickle or h5, got {self.packed_format!r}")
        if self.h5_handle_cache_size < 1:
            raise ValueError(f"mhr_h5_handle_cache_size must be positive, got {self.h5_handle_cache_size}")
        if self.rank_local_preprocess and not self.render_h5_root:
            raise ValueError("mhr_rank_local_preprocess requires render_root")
        if self.input_materialization_mode == MHR_INPUT_MATERIALIZATION_GPU:
            if not self.rank_local_preprocess:
                raise ValueError("gpu_compact_v1 requires mhr_rank_local_preprocess=true")
            validate_mhr_gpu_materialization_config(cfg)
        if self.mhr_pseudogt_root:
            raise ValueError("mhr_pseudogt_root is unsupported; publish canonical MHR parameters into packed H5 and decode geometry online")
        if self.require_foundationpose_training_tier_schema and self.packed_format != "h5":
            raise ValueError("mhr_require_foundationpose_training_tier_schema requires packed_format=h5")
        if self.require_foundationpose_training_tier_schema and self.split == "train" and not self.foundationpose_tier_sampling:
            raise ValueError("Tier-only Daniel training requires mhr_foundationpose_tier_sampling=true")
        if self.effective_mask_root and not self.interaction_trim_root:
            raise ValueError("mhr_effective_mask_root requires mhr_interaction_trim_root to resolve source sequences")

        self.render_h5_handles: OrderedDict[str, Any] = OrderedDict()
        self.packed_h5_handles: OrderedDict[str, Any] = OrderedDict()
        self.input_viz_mesh_geometry: OrderedDict[str, tuple[np.ndarray, str, np.ndarray]] = OrderedDict()
        self.render_metadata: dict[str, dict[str, Any]] = {}
        self.render_processor: Any | None = None
        self.sequence_data: list[dict[str, Any]]
        if self.dataset_index_path:
            payload = load_dataset_index(self.dataset_index_path, cfg, self.seqs, split)
            self.sequence_data = payload["sequence_data"]
            self.render_metadata = payload["render_metadata"]
            self.sample_offsets = payload["sample_offsets"]
            self._sample_count = payload["sample_count"]
        else:
            self._scan_sequence_metadata()
        for data in self.sequence_data:
            if "object_pose_storage_to_training_transform" not in data or "object_mesh_to_training_transform" not in data:
                object_pose_frame = resolve_object_pose_frame(data.get("metadata", {}), assume_aligned_without_mesh=True)
                data["object_pose_storage_to_training_transform"] = object_pose_frame.storage_to_training
                data["object_mesh_to_training_transform"] = object_pose_frame.mesh_to_training
        uses_training_tiers = self.split == "train" and self.foundationpose_tier_sampling and any(data.get("foundationpose_training_tier_revision") == FOUNDATIONPOSE_TRAINING_TIER_REVISION for data in self.sequence_data)
        if uses_training_tiers and self.foundationpose_tier_sampling_revision != MHR_TIER_SAMPLING_REVISION:
            raise ValueError(f"Tier-only Daniel training requires sampling revision {MHR_TIER_SAMPLING_REVISION!r}, got {self.foundationpose_tier_sampling_revision!r}")
        self.data_sampling_revision = self.foundationpose_tier_sampling_revision if uses_training_tiers else None
        self.data_sampling_contract = tier_sampling_contract(self.data_sampling_revision, self.tier_sampling_seed)
        self.samples = _SampleIndexView(self)

    def _scan_sequence_metadata(self) -> None:
        self.sequence_data = []
        sample_offsets = [0]
        for seq in self.seqs:
            data = self._load_sequence(seq)
            frames = data["frames"]
            if self.render_h5_root:
                self.render_metadata[seq] = self._load_render_metadata(seq, data["kids"], frames)
                data["render_kids"] = self.render_metadata[seq]["render_kids"]
                packed_tiers = bool(data.get("foundationpose_training_tier_revision"))
                rendered_tiers = self.render_metadata[seq].get("foundationpose_training_tier_revision") == FOUNDATIONPOSE_TRAINING_TIER_REVISION
                if packed_tiers != rendered_tiers:
                    raise ValueError(f"Packed/render FoundationPose training-tier mismatch for {seq}: packed={packed_tiers}, rendered={rendered_tiers}")
                expected_record_mode = MHR_RENDER_RECORD_MODE_TIER_ONLY if packed_tiers else MHR_RENDER_RECORD_MODE_LEGACY
                if self.render_metadata[seq]["object_initialization_record_mode"] != expected_record_mode:
                    raise ValueError(f"Packed/render object initialization record mode mismatch for {seq}: expected={expected_record_mode!r}, actual={self.render_metadata[seq]['object_initialization_record_mode']!r}")
            sample_kids = self._sample_kids_for_sequence(data)
            valid_mask = data["frame_valid_mask"]
            interaction_trim = load_mhr_interaction_trim(self.interaction_trim_root, seq) if self.interaction_trim_root else None
            canonical_mask_frame_valid = None
            if self.effective_mask_root:
                mask_validity = load_canonical_effective_mask_validity(osp.join(self.interaction_trim_root, seq), self.effective_mask_root, sample_kids, frames)
                canonical_mask_frame_valid = mask_validity["frame_usable"]
                data["canonical_mask_frame_valid_mask"] = canonical_mask_frame_valid.astype(bool, copy=False)
                data["canonical_mask_window_validity_revision"] = EFFECTIVE_MASK_WINDOW_VALIDITY_REVISION
            starts, strides = mhr_window_starts_and_strides(frames, valid_mask, self.cfg, self.split, interaction_trim, required_frame_mask=canonical_mask_frame_valid)
            if self.effective_mask_root and self.split == "train":
                object_nonempty_by_camera = np.asarray(mask_validity["object_nonempty_by_camera"], dtype=bool)
                expected_shape = (len(sample_kids), len(frames))
                if object_nonempty_by_camera.shape != expected_shape:
                    raise ValueError(f"Canonical object-mask nonempty flags for {seq} must have shape {expected_shape}, got {object_nonempty_by_camera.shape}")
                keep = minimum_true_frame_fraction_window_mask(object_nonempty_by_camera, starts, strides, self.clip_len, self.min_canonical_object_nonempty_frame_fraction)
                starts, strides = starts[keep], strides[keep]
                data["canonical_object_nonempty_by_camera_mask"] = object_nonempty_by_camera
                data["canonical_object_nonempty_window_revision"] = MHR_CANONICAL_OBJECT_NONEMPTY_WINDOW_REVISION
            if interaction_trim is not None:
                data["interaction_trim"] = interaction_trim
            data["sample_starts"] = starts
            data["sample_strides"] = strides
            data["sample_kids"] = tuple(int(kid) for kid in sample_kids)
            self.sequence_data.append(data)
            sample_offsets.append(sample_offsets[-1] + len(starts) * len(sample_kids))
        self.sample_offsets = np.asarray(sample_offsets, dtype=np.int64)
        self._sample_count = int(self.sample_offsets[-1])

    def _sample_kids_for_sequence(self, data: Mapping[str, Any]) -> list[int]:
        available = [int(kid) for kid in (data.get("render_kids") or data["kids"])]
        if self.sample_kids is None:
            return available
        missing = [kid for kid in self.sample_kids if kid not in available]
        if missing:
            raise ValueError(f"Requested mhr_sample_kids={self.sample_kids} but available kids are {available}")
        return list(self.sample_kids)

    def _load_sequence(self, seq: str) -> dict[str, Any]:
        if self.packed_format == "h5":
            return self._load_sequence_h5(seq)
        combined_file = _first_existing(
            _candidate_files(self.mhr_gt_root, seq, ("_MHR-packed.pkl", "_mhr.pkl", "_GT-packed.pkl"))
        )
        if combined_file is None:
            raise FileNotFoundError(f"No MHR target file found for {seq} under {self.mhr_gt_root}")
        combined = _load_mapping(combined_file)
        combined_metadata = combined.get("metadata", {})
        if not isinstance(combined_metadata, Mapping):
            raise TypeError(f"MHR sequence {seq} metadata must be a mapping")

        if "init" in combined:
            gt = dict(combined.get("gt", {}))
            init = dict(combined["init"])
        else:
            gt = _extract_prefixed(combined, "_gt")
            init = _extract_prefixed(combined, "_init")
            if not gt:
                gt = {key: combined[key] for key in MHR_PARAM_DIMS if key in combined}
                for key in ("mhr_joints", "mhr_keypoints", "mhr_joint_global_rots"):
                    if key in combined:
                        gt[key] = combined[key]

        if not init:
            init_file = _first_existing(
                _candidate_files(self.mhr_init_root, seq, ("_params.pkl", "_mhr.pkl", "_MHR-init.pkl"))
            )
            if init_file is None:
                raise FileNotFoundError(f"No MHR init file found for {seq} under {self.mhr_init_root}")
            init = _load_mapping(init_file)

        init_metadata = init.get("metadata", combined_metadata) if isinstance(init, Mapping) else combined_metadata
        validate_mhr_init_root_metadata(init_metadata, f"MHR initialization for {seq}")
        validate_mhr_init_root_metadata(combined_metadata, f"MHR packed sequence {seq}")

        frames = combined.get("frames") or gt.get("frames") or init.get("frames")
        if frames is None:
            raise ValueError(f"MHR sequence {seq} has no frames field")
        kids = combined.get("kids") or gt.get("kids") or init.get("kids") or [_cfg_get(self.cfg, "cam_id", 0)]

        init = self._complete_and_validate(init, frames, kids, "init", seq)
        pose_validity = self._load_pose_validity_masks(combined, frames, seq)
        geometry_gt = None
        pseudogt_valid_mask = None
        gt = self._complete_and_validate(gt, frames, kids, "target", seq)

        out = {
            "seq": seq,
            "frames": list(frames),
            "kids": list(kids),
            "render_kids": None,
            "gt": gt,
            "init": init,
            "geometry_gt": geometry_gt,
            "pseudogt_valid_mask": pseudogt_valid_mask,
            "faces": combined.get("faces", gt.get("faces")),
            "metadata": combined.get("metadata", {}),
            "init_metadata": init_metadata,
            **pose_validity,
            "object": self._extract_object_data(combined),
        }
        object_pose_frame = resolve_object_pose_frame(out["metadata"], assume_aligned_without_mesh=True)
        out["object_pose_storage_to_training_transform"] = object_pose_frame.storage_to_training
        out["object_mesh_to_training_transform"] = object_pose_frame.mesh_to_training
        return out

    def _load_sequence_h5(self, seq: str) -> dict[str, Any]:
        path = osp.join(self.mhr_gt_root, f"{seq}_MHR-packed.h5")
        if not osp.isfile(path):
            raise FileNotFoundError(f"No canonical packed H5 found for {seq}: {path}")
        with h5py.File(path, "r") as handle:
            schema = str(handle.attrs.get("schema", ""))
            if schema not in SUPPORTED_SCHEMAS:
                raise ValueError(f"Canonical packed H5 for {seq} has unsupported schema {schema!r}")
            if not bool(handle.attrs.get("complete", False)):
                raise ValueError(f"Canonical packed H5 for {seq} is not complete")
            if str(handle.attrs.get("sequence", "")) != seq or str(handle.attrs.get("body_model", "")) != "mhr":
                raise ValueError(f"Canonical packed H5 identity mismatch for {seq}")
            if self.require_foundationpose_training_tier_schema and schema not in TIER_SCHEMAS:
                raise ValueError(f"Daniel training requires a tier-only packed schema, got {schema!r} for {seq}")
            metadata = validate_packed_mhr_init_root(handle, f"canonical packed H5 for {seq}")
            if self.required_contact_revision:
                validate_packed_mhr_hand_surface_contacts(handle, f"canonical packed H5 for {seq}", str(self.required_contact_revision))
            tier_revision = metadata.get("foundationpose_training_tier_revision")
            if schema in TIER_SCHEMAS and tier_revision != FOUNDATIONPOSE_TRAINING_TIER_REVISION:
                raise ValueError(f"Canonical packed H5 for {seq} has stale FoundationPose training-tier revision {tier_revision!r}")
            if schema not in TIER_SCHEMAS and tier_revision is not None:
                raise ValueError(f"Legacy canonical packed H5 for {seq} unexpectedly declares FoundationPose training tiers")
            frames = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in handle["frames"][()]]
            kids = [int(item) for item in handle["kids"][()]]
            frame_valid_mask = np.asarray(handle["frame_valid_mask"][()], dtype=np.float32)
            human_pose_valid_mask = np.asarray(handle["human_pose_valid_mask"][()], dtype=np.float32) if "human_pose_valid_mask" in handle else frame_valid_mask.copy()
            object_pose_valid_mask = np.asarray(handle["object_pose_valid_mask"][()], dtype=np.float32) if "object_pose_valid_mask" in handle else frame_valid_mask.copy()
        pose_validity = self._validate_pose_validity_masks({"human_pose_valid_mask": human_pose_valid_mask, "object_pose_valid_mask": object_pose_valid_mask, "frame_valid_mask": frame_valid_mask}, len(frames), seq)
        object_pose_frame = resolve_object_pose_frame(metadata, assume_aligned_without_mesh=True)
        return {"seq": seq, "frames": frames, "kids": kids, "render_kids": None, **pose_validity, "packed_h5_path": path, "packed_h5_schema": schema, "geometry_gt": None, "pseudogt_valid_mask": None, "foundationpose_training_tier_revision": tier_revision, "metadata": metadata, "object_pose_storage_to_training_transform": object_pose_frame.storage_to_training, "object_mesh_to_training_transform": object_pose_frame.mesh_to_training}

    @staticmethod
    def _load_frame_valid_mask(combined: Mapping[str, Any], frames: Any, seq: str) -> np.ndarray:
        return MHRVideoDataset._load_pose_validity_masks(combined, frames, seq)["frame_valid_mask"]

    @staticmethod
    def _validate_pose_validity_masks(masks: Mapping[str, Any], frame_count: int, seq: str) -> dict[str, np.ndarray]:
        validated = {}
        for key in ("human_pose_valid_mask", "object_pose_valid_mask", "frame_valid_mask"):
            mask = np.asarray(masks[key])
            if mask.shape != (frame_count,):
                raise ValueError(f"{key} for {seq} must have shape [{frame_count}], got {mask.shape}")
            if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
                raise ValueError(f"{key} for {seq} must contain only finite binary values")
            validated[key] = mask.astype(np.float32)
        if not np.array_equal(validated["frame_valid_mask"], validated["human_pose_valid_mask"] * validated["object_pose_valid_mask"]):
            raise ValueError(f"frame_valid_mask for {seq} must equal human_pose_valid_mask & object_pose_valid_mask")
        return validated

    @staticmethod
    def _load_pose_validity_masks(combined: Mapping[str, Any], frames: Any, seq: str) -> dict[str, np.ndarray]:
        frame_count = len(frames)
        if "frame_valid_mask" not in combined:
            frame_valid_mask = np.ones((frame_count,), dtype=np.float32)
        else:
            frame_valid_mask = combined["frame_valid_mask"]
        human_pose_valid_mask = combined.get("human_pose_valid_mask", frame_valid_mask)
        object_pose_valid_mask = combined.get("object_pose_valid_mask", frame_valid_mask)
        return MHRVideoDataset._validate_pose_validity_masks({"human_pose_valid_mask": human_pose_valid_mask, "object_pose_valid_mask": object_pose_valid_mask, "frame_valid_mask": frame_valid_mask}, frame_count, seq)

    def _discover_render_kids(self, seq: str, frames: Any) -> list[int] | None:
        if not self.render_h5_root:
            return None
        h5_path = osp.join(self.render_h5_root, f"{seq}_render.h5")
        if not osp.isfile(h5_path):
            raise FileNotFoundError(f"No render H5 found for {seq}: {h5_path}")
        first_frame = list(frames)[0]
        pattern = re.compile(rf"^{re.escape(seq)}\+{re.escape(first_frame)}_k(\d+)_input$")
        with h5py.File(h5_path, "r") as f:
            kids = sorted({int(match.group(1)) for key in f.keys() if (match := pattern.match(key))})
        if not kids:
            raise ValueError(f"Render H5 for {seq} has no input keys for frame {first_frame}")
        return kids

    @staticmethod
    def _discover_render_kids_from_handle(handle: h5py.File, seq: str, frames: Any) -> list[int]:
        first_frame = list(frames)[0]
        pattern = re.compile(rf"^{re.escape(seq)}\+{re.escape(first_frame)}_k(\d+)_input$")
        kids = sorted({int(match.group(1)) for key in handle.keys() if (match := pattern.match(key))})
        if not kids:
            raise ValueError(f"Render H5 for {seq} has no input keys for frame {first_frame}")
        return kids

    def _load_render_metadata(self, seq: str, packed_kids: Sequence[int], frames: Any) -> dict[str, Any]:
        path = osp.join(self.render_h5_root, f"{seq}_render.h5")
        key = f"{seq}_w2c"
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise KeyError(f"Render H5 for {seq} is missing {key}")
            payload = _load_h5_pickle(handle[key])
            render_kids = [int(kid) for kid in np.asarray(payload["kids"]).reshape(-1)] if isinstance(payload, Mapping) and "kids" in payload else self._discover_render_kids_from_handle(handle, seq, frames)
        if not isinstance(payload, Mapping):
            raise TypeError(f"Render H5 calibration for {seq} must be a mapping, got {type(payload).__name__}")
        required_keys = {"rot", "trans", "mesh_diameter", "trans_normalizer", "rot_normalizer"}
        missing_keys = sorted(required_keys - set(payload))
        if missing_keys:
            raise KeyError(f"Render H5 calibration for {seq} is missing {missing_keys}")
        rotations = np.asarray(payload.get("rot"), dtype=np.float32)
        translations = np.asarray(payload.get("trans"), dtype=np.float32)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or translations.shape != (len(rotations), 3):
            raise ValueError(f"Render H5 calibration for {seq} has invalid rotation/translation shapes {rotations.shape} and {translations.shape}")
        if "kids" in payload:
            calibration_kids = [int(kid) for kid in np.asarray(payload["kids"]).reshape(-1)]
            if len(calibration_kids) != len(rotations):
                raise ValueError(f"Render H5 calibration for {seq} has {len(calibration_kids)} camera IDs but {len(rotations)} transforms")
        else:
            required_kids = sorted({int(kid) for kid in list(packed_kids) + list(render_kids)})
            if required_kids and len(rotations) > max(required_kids):
                calibration_kids = list(range(len(rotations)))
            elif len(rotations) == 1 and len(render_kids) == 1:
                calibration_kids = [int(render_kids[0])]
            elif len(rotations) == len(packed_kids) and list(map(int, packed_kids)) == list(map(int, render_kids)):
                calibration_kids = list(map(int, packed_kids))
            else:
                raise ValueError(f"Render H5 calibration for {seq} lacks explicit camera IDs and cannot safely map {len(rotations)} transforms to packed kids {list(packed_kids)} and rendered kids {list(render_kids)}")
        if len(set(calibration_kids)) != len(calibration_kids):
            raise ValueError(f"Render H5 calibration for {seq} contains duplicate camera IDs {calibration_kids}")
        world_to_camera_by_kid = {}
        for index, kid in enumerate(calibration_kids):
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = rotations[index]
            transform[:3, 3] = translations[index]
            world_to_camera_by_kid[kid] = _validate_world_to_camera(transform, f"Render H5 world-to-camera transform for {seq}/k{kid}")
        missing = [int(kid) for kid in render_kids if int(kid) not in world_to_camera_by_kid]
        if missing:
            raise ValueError(f"Render H5 calibration for {seq} has no transforms for rendered camera IDs {missing}")
        mesh_diameter = float(payload["mesh_diameter"])
        trans_normalizer = np.asarray(payload["trans_normalizer"], dtype=np.float32)
        rot_normalizer = np.asarray(payload["rot_normalizer"], dtype=np.float32)
        if not np.isfinite(mesh_diameter) or mesh_diameter <= 0:
            raise ValueError(f"Render H5 calibration for {seq} has invalid mesh diameter {mesh_diameter}")
        if trans_normalizer.shape != (3,) or not np.isfinite(trans_normalizer).all():
            raise ValueError(f"Render H5 calibration for {seq} has invalid translation normalizer {trans_normalizer.shape}")
        if rot_normalizer.ndim != 0 or not np.isfinite(rot_normalizer).all():
            raise ValueError(f"Render H5 calibration for {seq} has invalid rotation normalizer {rot_normalizer.shape}")
        tier_revision = payload.get("foundationpose_training_tier_revision")
        if tier_revision is not None and tier_revision != FOUNDATIONPOSE_TRAINING_TIER_REVISION:
            raise ValueError(f"Render H5 for {seq} has stale FoundationPose training-tier revision {tier_revision!r}")
        record_mode = payload.get("object_initialization_record_mode")
        expected_record_mode = MHR_RENDER_RECORD_MODE_TIER_ONLY if tier_revision is not None else MHR_RENDER_RECORD_MODE_LEGACY
        if tier_revision is not None and record_mode != expected_record_mode:
            raise ValueError(f"Render H5 for {seq} must use object initialization record mode {expected_record_mode!r}, got {record_mode!r}")
        if tier_revision is None and record_mode not in (None, expected_record_mode):
            raise ValueError(f"Render H5 for {seq} must use legacy object initialization records, got {record_mode!r}")
        record_mode = expected_record_mode
        return {
            "render_kids": render_kids,
            "world_to_camera_by_kid": world_to_camera_by_kid,
            "mesh_diameter": mesh_diameter,
            "trans_normalizer": trans_normalizer,
            "rot_normalizer": rot_normalizer,
            "foundationpose_training_tier_revision": tier_revision,
            "object_initialization_record_mode": record_mode,
        }

    def _complete_and_validate(self, data: dict[str, Any], frames: Any, kids: Any, label: str, seq: str) -> dict[str, Any]:
        data = dict(data)
        data["body_model"] = "mhr"
        data["frames"] = list(frames)
        data["kids"] = list(kids)
        missing = [key for key in MHR_REQUIRED_PARAM_KEYS if key not in data]
        if missing:
            raise ValueError(f"MHR {label} data for {seq} is missing required fields: {missing}")
        for key in MHR_OPTIONAL_PARAM_KEYS:
            if key not in data:
                shape = np.asarray(data["mhr_trans"]).shape[:-1] + (MHR_PARAM_DIMS[key],)
                data[key] = np.zeros(shape, dtype=np.asarray(data["mhr_trans"]).dtype)
        assert_mhr_schema(data, require_geometry=self.require_geometry)
        return data

    @staticmethod
    def _extract_object_data(data: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if "obj_rot_gt" in data:
            out["obj_rot_gt"] = data["obj_rot_gt"]
        if "obj_t_gt" in data:
            out["obj_t_gt"] = data["obj_t_gt"]
        if "obj_rot_init" in data:
            out["obj_rot_init"] = data["obj_rot_init"]
        if "obj_t_init" in data:
            out["obj_t_init"] = data["obj_t_init"]
        if "obj_rot" in data:
            out["obj_rot"] = data["obj_rot"]
        elif "obj_angles" in data:
            out["obj_angles"] = data["obj_angles"]
        if "obj_t" in data:
            out["obj_t"] = data["obj_t"]
        elif "obj_trans" in data:
            out["obj_t"] = data["obj_trans"]
        if "obj_symmetry_tfs" in data:
            out["obj_symmetry_tfs"] = data["obj_symmetry_tfs"]
        if "obj_symmetry_mode" in data:
            out["obj_symmetry_mode"] = data["obj_symmetry_mode"]
        if "obj_symmetry_center" in data:
            out["obj_symmetry_center"] = data["obj_symmetry_center"]
        if "mhr_contact_dist_gt" in data:
            out["mhr_contact_dist_gt"] = data["mhr_contact_dist_gt"]
        for key in GT_OBJECT_VISIBILITY_FIELDS:
            if key in data:
                out[key] = data[key]
        return out

    def __len__(self) -> int:
        return self._sample_count

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError(f"MHR dataset epoch must be nonnegative, got {epoch}")
        self._tier_sampling_epoch.fill_(epoch)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["packed_h5_handles"] = OrderedDict()
        state["render_h5_handles"] = OrderedDict()
        state["input_viz_mesh_geometry"] = OrderedDict()
        state["render_processor"] = None
        state["samples"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.samples = _SampleIndexView(self)

    def close(self) -> None:
        for handle in self.packed_h5_handles.values():
            handle.close()
        for handle in self.render_h5_handles.values():
            handle.close()
        self.packed_h5_handles.clear()
        self.render_h5_handles.clear()
        getattr(self, "input_viz_mesh_geometry", {}).clear()

    def clone_for_rank_local_render(self) -> "MHRVideoDataset":
        state = self.__getstate__()
        state["sequence_data"] = []
        state["sample_offsets"] = np.zeros((1,), dtype=np.int64)
        state["_sample_count"] = 0
        clone = self.__class__.__new__(self.__class__)
        clone.__setstate__(state)
        return clone

    def _sample_descriptor(self, idx: int) -> tuple[int, int, int, int]:
        if idx < 0:
            idx += self._sample_count
        if idx < 0 or idx >= self._sample_count:
            raise IndexError(idx)
        seq_id = int(np.searchsorted(self.sample_offsets, idx, side="right") - 1)
        local_index = idx - int(self.sample_offsets[seq_id])
        seq_data = self.sequence_data[seq_id]
        kids = seq_data["sample_kids"]
        start_index, kid_index = divmod(local_index, len(kids))
        return seq_id, int(seq_data["sample_starts"][start_index]), int(seq_data["sample_strides"][start_index]), int(kids[kid_index])

    def _sample_tuple(self, idx: int) -> tuple[int, int, int] | tuple[int, int, int, int]:
        seq_id, start, temporal_stride, kid = self._sample_descriptor(idx)
        if self.window_sampling_contract["mode"] == MHR_WINDOW_SAMPLING_LEGACY:
            return seq_id, start, kid
        return seq_id, start, temporal_stride, kid

    def _ensure_packed_h5(self, seq_data: Mapping[str, Any]) -> h5py.File:
        seq = seq_data["seq"]
        if seq in self.packed_h5_handles:
            self.packed_h5_handles.move_to_end(seq)
            return self.packed_h5_handles[seq]
        while len(self.packed_h5_handles) >= self.h5_handle_cache_size:
            _, handle = self.packed_h5_handles.popitem(last=False)
            handle.close()
        if self.required_contact_revision:
            with h5py.File(seq_data["packed_h5_path"], "r") as handle:
                validate_packed_mhr_hand_surface_contacts(handle, f"canonical packed H5 for {seq}", str(self.required_contact_revision))
        self.packed_h5_handles[seq] = h5py.File(seq_data["packed_h5_path"], "r")
        return self.packed_h5_handles[seq]

    def _load_h5_clip(self, seq_data: Mapping[str, Any], start: int, end: int, indices: np.ndarray, kid: int, sample_index: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        handle = self._ensure_packed_h5(seq_data)
        kids = [int(item) for item in seq_data["kids"]]
        if kid not in kids:
            raise ValueError(f"Camera {kid} is absent from canonical packed H5 for {seq_data['seq']}: {kids}")
        view_index = kids.index(kid)
        gt = {}
        init = {}
        for key in self.gt_clip_keys:
            gt_path = f"gt/{key}"
            if gt_path in handle:
                gt[key] = _take_frame_axis(handle[gt_path], start, indices)
        for key in self.init_clip_keys:
            init_path = f"init/{key}"
            if init_path in handle:
                init[key] = _take_frame_axis(handle[init_path], start, indices, view_index)
        if "mhr_trans" in gt:
            gt_joints_path = "gt/mhr_joints"
            if gt_joints_path not in handle:
                raise KeyError(f"MHR packed H5 is missing {gt_joints_path} required for frame conversion")
            gt[MHR_INTERNAL_ROOT_JOINT_KEY] = _take_frame_axis(handle[gt_joints_path], start, indices, MHR_ROOT_JOINT_INDEX)
        if "mhr_trans" in init:
            init_joints_path = "init/mhr_joints"
            if init_joints_path not in handle:
                raise KeyError(f"MHR packed H5 is missing {init_joints_path} required for frame conversion")
            init[MHR_INTERNAL_ROOT_JOINT_KEY] = _take_frame_axis(handle[init_joints_path], start, indices, view_index, MHR_ROOT_JOINT_INDEX)
        object_data = {
            "obj_rot_gt": _take_frame_axis(handle["object/rot_gt"], start, indices), "obj_t_gt": _take_frame_axis(handle["object/trans_gt"], start, indices),
            "obj_symmetry_tfs": np.asarray(handle["object/symmetry_transforms"][()]).copy(), "obj_symmetry_mode": np.asarray(handle["object/symmetry_mode"][()]).copy(), "obj_symmetry_center": np.asarray(handle["object/symmetry_center"][()]).copy(), "mhr_contact_dist_gt": _take_frame_axis(handle["contact/distance_gt"], start, indices),
        }
        tier_valid_path = FOUNDATIONPOSE_TRAINING_TIER_H5_PATHS["fp_training_tier_valid"]
        top1_tier_path = FOUNDATIONPOSE_TRAINING_TIER_H5_PATHS["fp_training_top1_tier"]
        if tier_valid_path in handle:
            tier_valid = _take_frame_axis(handle[tier_valid_path], start, indices, view_index).astype(bool, copy=False)
            if self.split == "train" and self.foundationpose_tier_sampling:
                frame_indices = start + np.asarray(indices, dtype=np.int64)
                missing = np.flatnonzero(~tier_valid.any(axis=1))
                if len(missing):
                    raise ValueError(f"MHR packed H5 has no available FoundationPose training tier for {seq_data['seq']} frame={int(frame_indices[missing[0]])} camera={kid}")
                selected_tiers = stateless_clip_tiers(tier_valid, int(self.tier_sampling_seed), int(self._tier_sampling_epoch.item()), int(sample_index), str(seq_data["seq"]), int(kid), frame_indices)
            else:
                selected_tiers = _take_frame_axis(handle[top1_tier_path], start, indices, view_index).astype(np.int8, copy=False)
                if not np.isin(selected_tiers, (1, 2, 3)).all():
                    raise ValueError(f"MHR packed H5 has invalid deterministic top-1 tier IDs for {seq_data['seq']} camera={kid}")
                selected_valid = np.take_along_axis(tier_valid, (selected_tiers.astype(np.int64) - 1)[:, None], axis=1)[:, 0]
                if not selected_valid.all():
                    raise ValueError(f"MHR packed H5 has invalid deterministic top-1 tiers for {seq_data['seq']} camera={kid}")
            object_data["obj_init_tier_ids"] = selected_tiers
        elif self.require_foundationpose_training_tier_schema:
            raise ValueError(f"Tier-only Daniel packed H5 has no FoundationPose training tiers for {seq_data['seq']}")
        for key, path in GT_OBJECT_VISIBILITY_H5_PATHS.items():
            if path in handle:
                object_data[key] = _take_frame_axis(handle[path], start, indices, view_index)
        return gt, init, object_data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_id, start, temporal_stride, kid = self._sample_descriptor(idx)
        seq_data = self.sequence_data[seq_id]
        indices = np.arange(self.clip_len, dtype=np.int64) * temporal_stride
        end = start + int(indices[-1]) + 1
        if self.split == "train" and self.random_flip and np.random.uniform() > 0.5:
            indices = indices[::-1]

        kid_int = int(kid)
        kids = [int(x) for x in seq_data["kids"]]
        if self.packed_format == "h5":
            gt, init, obj = self._load_h5_clip(seq_data, start, end, indices, kid_int, int(idx))
            has_parameter_targets = True
        else:
            init = _select_view(_extract_frame_slice(seq_data["init"], start, end, indices, self.init_clip_keys), kids, kid_int)
            geometry_gt = seq_data.get("geometry_gt")
            if geometry_gt is None:
                gt = _extract_frame_slice(seq_data["gt"], start, end, indices, self.gt_clip_keys)
                has_parameter_targets = True
            else:
                gt = _extract_frame_slice(geometry_gt, start, end, indices, self.gt_clip_keys)
                has_parameter_targets = False
            obj = _select_view(_extract_frame_slice(seq_data["object"], start, end, indices), kids, kid_int)
            if "mhr_trans" in init:
                init[MHR_INTERNAL_ROOT_JOINT_KEY] = _extract_mhr_root_joint(seq_data["init"], start, end, indices, view_index=kids.index(kid_int))
            if "mhr_trans" in gt:
                gt_source = seq_data["gt"] if geometry_gt is None else geometry_gt
                gt[MHR_INTERNAL_ROOT_JOINT_KEY] = _extract_mhr_root_joint(gt_source, start, end, indices)
        obj = _object_data_to_training_frame(obj, seq_data["object_pose_storage_to_training_transform"])
        if self.render_h5_root:
            world_to_camera = self.render_metadata[seq_data["seq"]]["world_to_camera_by_kid"][kid_int]
            init = _transform_mhr_mapping_between_frames(init, world_to_camera, pivot_aware_translation=_init_translation_is_decode_consistent(seq_data, kid_int))
            gt = _transform_mhr_mapping_between_frames(gt, world_to_camera, pivot_aware_translation=True)
        delta = compute_mhr_delta(init, gt) if has_parameter_targets else {}
        mhr_xyz_anchor = select_mhr_xyz_anchor(init["mhr_trans"], init.pop(MHR_INTERNAL_ROOT_JOINT_KEY, None), self.cfg)
        gt.pop(MHR_INTERNAL_ROOT_JOINT_KEY, None)
        if self.minimal_batch_fields:
            delta = {key: value for key, value in delta.items() if key in self.delta_output_keys}

        frames = [f"{seq_data['seq']}/{seq_data['frames'][start + int(i)]}" for i in indices]
        out: dict[str, Any] = {}
        if not self.minimal_batch_fields:
            out.update({"body_model": "mhr", "seq_name": seq_data["seq"], "frames": frames, "image_files": frames, "kids": np.asarray(seq_data["kids"]), "frame_indices": indices.copy(), "temporal_stride": np.int16(temporal_stride)})
        for key, value in init.items():
            out[f"{key}_init"] = value
        out[MHR_XYZ_ANCHOR_KEY] = mhr_xyz_anchor
        for key, value in gt.items():
            if self.minimal_batch_fields and key in PARAM_KEYS and key not in self.gt_output_param_keys:
                continue
            out[f"{key}_gt"] = value
        out.update(delta)
        _add_mhr_coco17_fields(out)
        if self.mhr_cond_key == "mhr_coco17_init":
            out.pop("mhr_keypoints_init", None)
        if "mhr_coco17_gt" in out:
            out.pop("mhr_keypoints_gt", None)

        if "mhr_contact_dist_gt" in obj:
            out["contact_dist_gt"] = obj["mhr_contact_dist_gt"]
        for key in GT_OBJECT_VISIBILITY_FIELDS:
            if key in obj:
                out[key] = obj[key]
        if self.render_h5_root:
            if self.rank_local_preprocess:
                out.update({key: obj[key] for key in ("obj_rot_gt", "obj_t_gt", "obj_rot", "obj_t", "obj_symmetry_tfs", "obj_symmetry_mode", "obj_symmetry_center", "obj_init_tier_ids", "obj_pose_storage_to_training_transform") if key in obj})
                request_frame_names = tuple(str(seq_data["frames"][start + int(i)]) for i in indices)
                out[MHR_RENDER_REQUEST_KEY] = MHRRenderRequest(sequence_index=seq_id, start=start, indices=tuple(int(i) for i in indices), kid=kid_int, sequence_name=seq_data["seq"], frame_names=request_frame_names, temporal_stride=temporal_stride)
            else:
                render_context = dict(obj)
                render_context[MHR_XYZ_ANCHOR_KEY] = mhr_xyz_anchor
                out.update(self._load_render_fields(seq_data, start, indices, kid_int, render_context))
        else:
            out.update(obj)
        human_frame_mask = _take_frame_axis(seq_data["human_pose_valid_mask"], start, indices)
        object_frame_mask = _take_frame_axis(seq_data["object_pose_valid_mask"], start, indices)
        frame_mask = _take_frame_axis(seq_data["frame_valid_mask"], start, indices)
        if "frame_mask" in out:
            render_frame_mask = np.asarray(out["frame_mask"], dtype=np.float32)
            human_frame_mask *= render_frame_mask
            object_frame_mask *= render_frame_mask
            frame_mask *= render_frame_mask
        if float(frame_mask.sum()) <= 0:
            raise ValueError(f"MHR clip {seq_data['seq']} start={start} stride={temporal_stride} camera={kid_int} has no valid frames")
        out["human_frame_mask"] = human_frame_mask
        out["object_frame_mask"] = object_frame_mask
        out["frame_mask"] = frame_mask
        return out

    def _ensure_render_h5(self, seq: str) -> Any:
        if seq in self.render_h5_handles:
            self.render_h5_handles.move_to_end(seq)
            return self.render_h5_handles[seq]
        while len(self.render_h5_handles) >= self.h5_handle_cache_size:
            _, handle = self.render_h5_handles.popitem(last=False)
            handle.close()
        self.render_h5_handles[seq] = h5py.File(osp.join(self.render_h5_root, f"{seq}_render.h5"), "r")
        return self.render_h5_handles[seq]

    def _ensure_render_processor(self) -> Any:
        if self.render_processor is None:
            from learning.datasets.video_data import VideoDataProcessor

            processor_split = "train" if self.split == "train" and self.input_augmentation_mode != MHR_INPUT_AUGMENTATION_DISABLED else "val"
            depth_corruption_probability = 0.5 if self.input_augmentation_mode == MHR_INPUT_AUGMENTATION_KEEP_ALIGNED_DEPTH else None
            self.render_processor = VideoDataProcessor(self.cfg, self.seqs, processor_split, depth_corruption_probability=depth_corruption_probability)
            self.render_processor.render_h5_handles = self.render_h5_handles
        return self.render_processor

    def load_input_viz_source_inputs(self, sequence_name: str, camera_id: int, frame_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        if not self.render_h5_root:
            raise ValueError("MHR input-grid source inputs require render_root")
        if not frame_names:
            raise ValueError("MHR input-grid source inputs require at least one frame")
        handle = self._ensure_render_h5(str(sequence_name))
        backgrounds, observed_full_xyz = [], []
        spatial_shape = None
        for frame_name in frame_names:
            key = f"{sequence_name}+{frame_name}_k{int(camera_id)}_input"
            if key not in handle:
                raise KeyError(f"MHR input-grid source record is missing: {key}")
            payload = _load_h5_pickle(handle[key])
            missing = [field for field in ("rgbmB", "xyzB") if field not in payload]
            if missing:
                raise KeyError(f"MHR input-grid source record {key} is missing fields: {missing}")
            rgbm = np.asarray(payload["rgbmB"])
            xyz = np.asarray(payload["xyzB"])
            if rgbm.ndim != 3 or rgbm.shape[2] < 3 or rgbm.dtype != np.uint8:
                raise ValueError(f"MHR input-grid background {key} must be uint8 [H,W,C>=3], got {rgbm.shape} {rgbm.dtype}")
            if xyz.ndim != 3 or xyz.shape[2] != 3 or not np.issubdtype(xyz.dtype, np.floating):
                raise ValueError(f"MHR input-grid full XYZ {key} must be floating-point [H,W,3], got {xyz.shape} {xyz.dtype}")
            if xyz.shape[:2] != rgbm.shape[:2]:
                raise ValueError(f"MHR input-grid source RGB/XYZ shapes differ for {key}: {rgbm.shape[:2]} and {xyz.shape[:2]}")
            if spatial_shape is None:
                spatial_shape = rgbm.shape[:2]
            elif rgbm.shape[:2] != spatial_shape:
                raise ValueError(f"MHR input-grid source inputs have inconsistent spatial shapes: {spatial_shape} and {rgbm.shape[:2]} for {key}")
            backgrounds.append(np.ascontiguousarray(rgbm[:, :, :3].transpose(2, 0, 1)))
            observed_full_xyz.append(np.ascontiguousarray(xyz.transpose(2, 0, 1), dtype=np.float32))
        return np.stack(backgrounds, axis=0), np.stack(observed_full_xyz, axis=0)

    def load_input_viz_background_rgbs(self, sequence_name: str, camera_id: int, frame_names: Sequence[str]) -> np.ndarray:
        backgrounds, _ = self.load_input_viz_source_inputs(sequence_name, camera_id, frame_names)
        return backgrounds

    def load_input_viz_gt_mhr_params(self, sequence_name: str, camera_id: int, frame_names: Sequence[str]) -> dict[str, np.ndarray]:
        sequence_name = str(sequence_name)
        frame_names = [str(value) for value in frame_names]
        if not frame_names:
            raise ValueError("MHR input-grid ground-truth parameters require at least one frame")
        matches = [data for data in self.sequence_data if str(data["seq"]) == sequence_name]
        if len(matches) != 1:
            raise KeyError(f"Expected exactly one MHR sequence named {sequence_name!r}, found {len(matches)}")
        sequence_data = matches[0]
        source_frames = [str(value) for value in sequence_data["frames"]]
        if len(set(source_frames)) != len(source_frames):
            raise ValueError(f"MHR sequence {sequence_name} contains duplicate frame names")
        frame_lookup = {name: index for index, name in enumerate(source_frames)}
        missing_frames = [name for name in frame_names if name not in frame_lookup]
        if missing_frames:
            raise KeyError(f"MHR input-grid frames are absent from {sequence_name}: {missing_frames}")
        frame_indices = np.asarray([frame_lookup[name] for name in frame_names], dtype=np.int64)
        start, end = int(frame_indices.min()), int(frame_indices.max()) + 1
        offsets = frame_indices - start
        if self.packed_format == "h5":
            handle = self._ensure_packed_h5(sequence_data)
            missing_paths = [f"gt/{key}" for key in PARAM_KEYS if f"gt/{key}" not in handle]
            if "gt/mhr_joints" not in handle:
                missing_paths.append("gt/mhr_joints")
            if missing_paths:
                raise KeyError(f"MHR input-grid ground-truth source for {sequence_name} is missing {missing_paths}")
            params = {key: np.asarray(handle[f"gt/{key}"][start:end])[offsets].copy() for key in PARAM_KEYS}
            params[MHR_INTERNAL_ROOT_JOINT_KEY] = np.asarray(handle["gt/mhr_joints"][start:end, MHR_ROOT_JOINT_INDEX])[offsets].copy()
        else:
            source = sequence_data.get("geometry_gt") or sequence_data.get("gt")
            if not isinstance(source, Mapping):
                raise TypeError(f"MHR input-grid ground-truth source for {sequence_name} must be a mapping")
            missing_keys = [key for key in (*PARAM_KEYS, "mhr_joints") if key not in source]
            if missing_keys:
                raise KeyError(f"MHR input-grid ground-truth source for {sequence_name} is missing {missing_keys}")
            params = {key: np.asarray(source[key])[frame_indices].copy() for key in PARAM_KEYS}
            params[MHR_INTERNAL_ROOT_JOINT_KEY] = np.asarray(source["mhr_joints"])[frame_indices, MHR_ROOT_JOINT_INDEX].copy()
        if self.render_h5_root:
            world_to_camera_by_kid = self.render_metadata[sequence_name]["world_to_camera_by_kid"]
            if int(camera_id) not in world_to_camera_by_kid:
                raise KeyError(f"MHR input-grid camera {camera_id} is absent from render calibration for {sequence_name}")
            params = _transform_mhr_mapping_between_frames(params, world_to_camera_by_kid[int(camera_id)], pivot_aware_translation=True)
        params.pop(MHR_INTERNAL_ROOT_JOINT_KEY, None)
        for key, dimension in MHR_PARAM_DIMS.items():
            value = np.asarray(params[key])
            expected_shape = (len(frame_names), int(dimension))
            if value.shape != expected_shape or not np.isfinite(value).all():
                raise ValueError(f"MHR input-grid ground-truth parameter {key} for {sequence_name} must be finite with shape {expected_shape}, got {value.shape}")
        return params

    def load_input_viz_mesh_geometry(self, sequence_name: str) -> tuple[np.ndarray, str, np.ndarray]:
        sequence_name = str(sequence_name)
        cache = getattr(self, "input_viz_mesh_geometry", None)
        if cache is None:
            cache = self.input_viz_mesh_geometry = OrderedDict()
        if sequence_name in cache:
            cache.move_to_end(sequence_name)
            return cache[sequence_name]
        matches = [data for data in self.sequence_data if str(data["seq"]) == sequence_name]
        if len(matches) != 1:
            raise KeyError(f"Expected exactly one MHR sequence named {sequence_name!r}, found {len(matches)}")
        sequence_data = matches[0]
        if self.packed_format == "h5":
            handle = self._ensure_packed_h5(sequence_data)
            human_faces = np.asarray(handle["faces"][()], dtype=np.int32)
            metadata = validate_packed_mhr_init_root(handle, f"canonical packed H5 for {sequence_name}")
        else:
            human_faces = np.asarray(sequence_data.get("faces"), dtype=np.int32)
            metadata = sequence_data.get("metadata", {})
        object_source = metadata.get("object_symmetry_mesh_source") or metadata.get("mhr_contact_source") or metadata.get("object_mesh_file")
        if not object_source:
            raise KeyError(f"MHR sequence {sequence_name} metadata has no object mesh source")
        object_mesh_to_training_transform = sequence_data.get("object_mesh_to_training_transform")
        if object_mesh_to_training_transform is None:
            object_mesh_to_training_transform = resolve_object_pose_frame(metadata, assume_aligned_without_mesh=True).mesh_to_training
        object_mesh_to_pose_transform = np.asarray(object_mesh_to_training_transform, dtype=np.float32)
        if human_faces.ndim != 2 or human_faces.shape[1] != 3 or human_faces.size == 0:
            raise ValueError(f"MHR sequence {sequence_name} has invalid human faces {human_faces.shape}")
        if not osp.isfile(object_source):
            raise FileNotFoundError(f"MHR sequence {sequence_name} object mesh does not exist: {object_source}")
        while len(cache) >= 4:
            cache.popitem(last=False)
        cache[sequence_name] = human_faces, str(object_source), object_mesh_to_pose_transform
        return cache[sequence_name]

    def load_render_fields_into(self, request: MHRRenderRequest, object_data: Mapping[str, Any], destinations: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        if request.sequence_name is not None:
            if request.frame_names is None or len(request.frame_names) != self.clip_len:
                raise ValueError(f"Self-contained MHR render request has {0 if request.frame_names is None else len(request.frame_names)} frame names, expected {self.clip_len}")
            seq_data = {"seq": request.sequence_name, "frames": list(request.frame_names)}
            return self._load_render_fields(seq_data, 0, np.arange(self.clip_len, dtype=np.int64), request.kid, object_data, destinations=destinations)
        if request.sequence_index < 0 or request.sequence_index >= len(self.sequence_data):
            raise IndexError(f"MHR render request sequence index is out of range: {request.sequence_index}")
        if len(request.indices) != self.clip_len:
            raise ValueError(f"MHR render request has {len(request.indices)} indices, expected {self.clip_len}")
        seq_data = self.sequence_data[request.sequence_index]
        return self._load_render_fields(seq_data, request.start, np.asarray(request.indices, dtype=np.int64), request.kid, object_data, destinations=destinations)

    def load_render_fields_into_profiled(self, request: MHRRenderRequest, object_data: Mapping[str, Any], destinations: Mapping[str, torch.Tensor]) -> tuple[dict[str, Any], dict[str, Any]]:
        profile: dict[str, Any] = {}
        if request.sequence_name is not None:
            if request.frame_names is None or len(request.frame_names) != self.clip_len:
                raise ValueError(f"Self-contained MHR render request has {0 if request.frame_names is None else len(request.frame_names)} frame names, expected {self.clip_len}")
            seq_data = {"seq": request.sequence_name, "frames": list(request.frame_names)}
            output = self._load_render_fields(seq_data, 0, np.arange(self.clip_len, dtype=np.int64), request.kid, object_data, destinations=destinations, profile=profile)
        else:
            if request.sequence_index < 0 or request.sequence_index >= len(self.sequence_data):
                raise IndexError(f"MHR render request sequence index is out of range: {request.sequence_index}")
            if len(request.indices) != self.clip_len:
                raise ValueError(f"MHR render request has {len(request.indices)} indices, expected {self.clip_len}")
            seq_data = self.sequence_data[request.sequence_index]
            output = self._load_render_fields(seq_data, request.start, np.asarray(request.indices, dtype=np.int64), request.kid, object_data, destinations=destinations, profile=profile)
        return output, profile

    @staticmethod
    def _copy_render_value(destination: torch.Tensor, source: Any, key: str, frame_index: int) -> None:
        if torch.is_tensor(source):
            source_tensor = source.to(dtype=destination.dtype)
        else:
            source_array = np.asarray(source)
            source_tensor = torch.from_numpy(source_array).to(dtype=destination.dtype)
        if tuple(destination.shape) != tuple(source_tensor.shape):
            raise ValueError(f"Rank-local {key} frame {frame_index} has shape {tuple(source_tensor.shape)}, expected {tuple(destination.shape)}")
        destination.copy_(source_tensor)

    def _load_render_fields(
        self,
        seq_data: Mapping[str, Any],
        start: int,
        indices: np.ndarray,
        kid: int,
        object_data: Mapping[str, Any],
        destinations: Mapping[str, torch.Tensor] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sample_started_at = _profile_start(profile)
        seq = seq_data["seq"]
        h5_open_started_at = _profile_start(profile)
        h5 = self._ensure_render_h5(seq)
        _profile_add(profile, "render_h5_open_seconds", h5_open_started_at)
        processor = self._ensure_render_processor()
        processor.render_h5_handles = self.render_h5_handles

        metadata = self.render_metadata[seq]
        w2c_k = metadata["world_to_camera_by_kid"][kid]
        mesh_diameter = float(metadata["mesh_diameter"])
        rot_normalizer = np.asarray(metadata["rot_normalizer"], dtype=np.float32) if not self.minimal_batch_fields else None
        trans_normalizer = np.asarray(metadata["trans_normalizer"], dtype=np.float32)

        frame_names = [seq_data["frames"][start + int(i)] for i in indices]
        object_init_tiers = object_data.get("obj_init_tier_ids")
        if object_init_tiers is not None:
            object_init_tiers = np.asarray(object_init_tiers, dtype=np.int8)
            if object_init_tiers.shape != (len(frame_names),) or not np.isin(object_init_tiers, (1, 2, 3)).all():
                raise ValueError(f"obj_init_tier_ids must contain tiers 1, 2, or 3 with shape [{len(frame_names)}], got {object_init_tiers}")
        first_suffix = f"tier_{int(object_init_tiers[0])}" if object_init_tiers is not None else "perturb_0"
        first_key = f"{seq}+{frame_names[0]}_k{kid}_{first_suffix}"
        render_decode_started_at = _profile_start(profile)
        compact_materialization = self.input_materialization_mode == MHR_INPUT_MATERIALIZATION_GPU
        if compact_materialization and destinations is None:
            raise ValueError("gpu_compact_v1 requires fixed rank-local destinations")
        first_render_values = processor.extract_render_data_compact(first_key, seq) if compact_materialization else processor.extract_render_data(first_key, seq, mesh_diameter)
        _profile_add(profile, "render_record_decode_seconds", render_decode_started_at)
        if compact_materialization:
            _, pose_init_storage, rgb_render, render_data = first_render_values
            dmap_xyz_init = None
        else:
            _, dmap_xyz_init, pose_init_storage, rgb_render, render_data = first_render_values

        if destinations is not None:
            expected_destinations = MHR_ENCODED_RENDER_KEYS if compact_materialization else MHR_LARGE_RENDER_KEYS
            missing_destinations = sorted(set(expected_destinations) - set(destinations))
            unknown_destinations = sorted(set(destinations) - set(expected_destinations))
            if missing_destinations or unknown_destinations:
                raise KeyError(f"Rank-local render destinations mismatch: missing={missing_destinations}, unknown={unknown_destinations}")
        input_rgbs, input_xyz = [], []
        render_rgbs, render_xyz = [], []
        poses_gt, poses_perturbed = [], []
        delta_transl, delta_rot, K_rois = [], [], []
        poses_gt_symm = []

        obj_rot_gt = np.asarray(object_data.get("obj_rot_gt", object_data.get("obj_rot")), dtype=np.float32)
        obj_t_gt = np.asarray(object_data.get("obj_t_gt", object_data.get("obj_t")), dtype=np.float32)
        mhr_xyz_anchor = np.asarray(object_data.get(MHR_XYZ_ANCHOR_KEY), dtype=np.float32)
        obj_symmetry_tfs = np.asarray(object_data.get("obj_symmetry_tfs", np.eye(4, dtype=np.float32)[None]), dtype=np.float32)
        if obj_symmetry_tfs.ndim != 3 or obj_symmetry_tfs.shape[1:] != (4, 4):
            raise ValueError(f"obj_symmetry_tfs must have shape [N, 4, 4], got {obj_symmetry_tfs.shape}")
        obj_symmetry_mode = np.asarray(object_data.get("obj_symmetry_mode", OBJECT_SYMMETRY_MODE_FINITE))
        obj_symmetry_center = np.asarray(object_data.get("obj_symmetry_center", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        object_pose_storage_to_training = np.asarray(object_data.get("obj_pose_storage_to_training_transform", np.eye(4, dtype=np.float32)), dtype=np.float32)
        if obj_symmetry_mode.shape != () or int(obj_symmetry_mode) not in (OBJECT_SYMMETRY_MODE_FINITE, OBJECT_SYMMETRY_MODE_FULL_SO3):
            raise ValueError(f"obj_symmetry_mode must be a supported integer scalar, got {obj_symmetry_mode.shape} value={obj_symmetry_mode}")
        if obj_symmetry_center.shape != (3,) or not np.isfinite(obj_symmetry_center).all():
            raise ValueError(f"obj_symmetry_center must be finite with shape [3], got {obj_symmetry_center.shape}")
        if int(obj_symmetry_mode) == OBJECT_SYMMETRY_MODE_FULL_SO3:
            if not np.allclose(obj_symmetry_tfs, np.eye(4, dtype=np.float32)[None], rtol=0.0, atol=1e-6):
                raise ValueError("Full SO(3) object symmetry must contain only identity finite representatives")
            obj_symmetry_tfs = obj_symmetry_tfs[:1]
        if mhr_xyz_anchor.shape != (len(frame_names), 3) or not np.isfinite(mhr_xyz_anchor).all():
            raise ValueError(f"{MHR_XYZ_ANCHOR_KEY} must be finite with shape [{len(frame_names)},3], got {mhr_xyz_anchor.shape}")

        for i, frame_name in enumerate(frame_names):
            frame_key = f"{seq}+{frame_name}"
            if object_init_tiers is not None:
                render_key = f"{frame_key}_k{kid}_tier_{int(object_init_tiers[i])}"
                if render_key not in h5:
                    raise KeyError(f"Sampled FoundationPose training-tier render is missing: {render_key}")
                if i == 0:
                    if compact_materialization:
                        _, pose_init_storage, rgb_render, render_data = first_render_values
                    else:
                        _, dmap_xyz_init, pose_init_storage, rgb_render, render_data = first_render_values
                else:
                    render_decode_started_at = _profile_start(profile)
                    if compact_materialization:
                        _, pose_init_storage, rgb_render, render_data = processor.extract_render_data_compact(render_key, seq)
                    else:
                        _, dmap_xyz_init, pose_init_storage, rgb_render, render_data = processor.extract_render_data(render_key, seq, mesh_diameter)
                    _profile_add(profile, "render_record_decode_seconds", render_decode_started_at)
            elif _cfg_get(self.cfg, "pose_init_type", "copy-first") != "copy-first":
                render_key = f"{frame_key}_k{kid}_perturb_0"
                render_decode_started_at = _profile_start(profile)
                if compact_materialization:
                    _, pose_init_storage, rgb_render, render_data = processor.extract_render_data_compact(render_key, seq)
                else:
                    _, dmap_xyz_init, pose_init_storage, rgb_render, render_data = processor.extract_render_data(render_key, seq, mesh_diameter)
                _profile_add(profile, "render_record_decode_seconds", render_decode_started_at)
            pose_init = object_poses_to_training_frame(np.asarray(pose_init_storage, dtype=np.float32)[None], object_pose_storage_to_training)[0]

            input_key = f"{frame_key}_k{kid}_input"
            input_decode_started_at = _profile_start(profile)
            input_data = _load_h5_pickle(h5[input_key])
            _profile_add(profile, "input_record_decode_seconds", input_decode_started_at)
            process_input_started_at = _profile_start(profile)
            if compact_materialization:
                from learning.datasets.video_data import augment_observed_video_components

                input_depth = np.asarray(input_data["xyzB"][:, :, -1]).copy()
                input_rgb = np.asarray(input_data["rgbmB"][:, :, :3]).copy()
                if processor.split == "train":
                    input_depth, input_rgb = augment_observed_video_components(input_data, processor.augm_img)
                input_masks = np.asarray(input_data["rgbmB"][:, :, 3:5])
                render_depth = np.asarray(render_data["depth"])
                render_rgb_source = np.asarray(rgb_render)
                render_masks = np.asarray(render_data["mask_o"])
                if input_rgb.dtype != np.uint8 or render_rgb_source.dtype != np.uint8 or input_masks.dtype != np.uint8 or input_depth.dtype != np.float16 or render_depth.dtype != np.float16:
                    raise ValueError(f"gpu_compact_v1 encoded dtypes must be uint8 RGB/masks and float16 depth, got input_rgb={input_rgb.dtype} render_rgb={render_rgb_source.dtype} masks={input_masks.dtype} input_depth={input_depth.dtype} render_depth={render_depth.dtype}")
                if input_rgb.shape != render_rgb_source.shape or input_rgb.ndim != 3 or input_rgb.shape[2] != 3 or input_masks.shape != input_rgb.shape[:2] + (2,) or input_depth.shape != input_rgb.shape[:2] or render_depth.shape != input_rgb.shape[:2]:
                    raise ValueError(f"gpu_compact_v1 encoded shapes differ: input_rgb={input_rgb.shape} render_rgb={render_rgb_source.shape} masks={input_masks.shape} input_depth={input_depth.shape} render_depth={render_depth.shape}")
                if render_masks.ndim != 3 or render_masks.shape[2] < 2 or render_masks.dtype != np.bool_:
                    raise ValueError(f"gpu_compact_v1 render masks must be binary [H,W,C>=2], got {render_masks.shape} {render_masks.dtype}")
                render_object = render_masks[:, :, 0].astype(bool, copy=False)
                render_full_object = render_masks[:, :, 1].astype(bool, copy=False)
                render_full = np.mean(rgb_render, axis=-1) > 0.01
                render_human = render_full & (~render_object)
                encoded_rgbs = np.concatenate((input_rgb.transpose(2, 0, 1), render_rgb_source.transpose(2, 0, 1)), axis=0)
                encoded_depths = np.stack((input_depth, render_depth), axis=0)
                encoded_masks = np.stack((input_masks[:, :, 0], input_masks[:, :, 1], render_human.astype(np.uint8) * 255, render_object.astype(np.uint8) * 255, render_full_object.astype(np.uint8) * 255), axis=0)
                self._copy_render_value(destinations[MHR_ENCODED_RGBS_KEY][i], encoded_rgbs, MHR_ENCODED_RGBS_KEY, i)
                self._copy_render_value(destinations[MHR_ENCODED_DEPTHS_KEY][i], encoded_depths, MHR_ENCODED_DEPTHS_KEY, i)
                self._copy_render_value(destinations[MHR_ENCODED_MASKS_KEY][i], encoded_masks, MHR_ENCODED_MASKS_KEY, i)
            else:
                dmap_xyz, dmap_xyz_a, rgb = processor.process_input(dmap_xyz_init, i, input_data, mesh_diameter, mhr_xyz_anchor, pose_init, render_data, rgb_render, f"{seq}/{frame_name}", kid, augment_depth=self.split == "train" and self.input_augmentation_mode != MHR_INPUT_AUGMENTATION_DISABLED)
            _profile_add(profile, "input_augmentation_tensor_seconds", process_input_started_at)

            target_assembly_started_at = _profile_start(profile)
            pose_gt = w2c_k @ _pose_matrix(obj_rot_gt[i], obj_t_gt[i])
            pose_gt_symm = pose_gt[None] @ obj_symmetry_tfs

            if not compact_materialization:
                render_rgb = rgb_render.copy().transpose(2, 0, 1) / 255.0
                if destinations is None:
                    render_rgbs.append(render_rgb)
                    input_rgbs.append(rgb)
                    input_xyz.append(dmap_xyz)
                    render_xyz.append(dmap_xyz_a)
                else:
                    self._copy_render_value(destinations["render_rgbs"][i], render_rgb, "render_rgbs", i)
                    self._copy_render_value(destinations["input_rgbs"][i], rgb, "input_rgbs", i)
                    self._copy_render_value(destinations["input_xyz"][i], dmap_xyz, "input_xyz", i)
                    self._copy_render_value(destinations["render_xyz"][i], dmap_xyz_a, "render_xyz", i)
            poses_perturbed.append(pose_init.copy())
            poses_gt.append(pose_gt.astype(np.float32))
            poses_gt_symm.append(pose_gt_symm.astype(np.float32))
            delta_transl.append((pose_gt[:3, 3] - pose_init[:3, 3]).astype(np.float32))
            delta_rot.append((pose_gt[:3, :3] @ pose_init[:3, :3].T).astype(np.float32))
            K_rois.append(np.asarray(render_data["K_roi"], dtype=np.float32))
            _profile_add(profile, "target_assembly_copy_seconds", target_assembly_started_at)

        sample_finalize_started_at = _profile_start(profile)
        pose_perturbed = np.stack(poses_perturbed).astype(np.float32)
        poseA_norm = pose_perturbed.copy()
        L = len(frame_names)

        output = {
            "pose_gt": np.stack(poses_gt).astype(np.float32),
            "pose_gt_symm": np.stack(poses_gt_symm).astype(np.float32),
            "obj_symmetry_mode": np.uint8(obj_symmetry_mode),
            "obj_symmetry_center": obj_symmetry_center.astype(np.float32),
            "pose_perturbed": pose_perturbed,
            "delta_transl": np.stack(delta_transl).astype(np.float32),
            "delta_rot": np.stack(delta_rot).astype(np.float32),
            "mesh_diameter": np.full((L,), mesh_diameter, dtype=np.float32),
            "trans_normalizer": np.stack([trans_normalizer] * L, 0).astype(np.float32),
            "poseA_norm": poseA_norm.astype(np.float32),
            "K_rois": np.stack(K_rois, 0).astype(np.float32),
            "frame_mask": np.ones((L,), dtype=np.float32),
        }
        if object_init_tiers is not None:
            output["object_init_tier"] = object_init_tiers.astype(np.int64)
        if not self.minimal_batch_fields:
            output["rot_normalizer"] = np.stack([rot_normalizer] * L, 0).astype(np.float32)
            output["Krois"] = output["K_rois"]
        if not self.minimal_batch_fields or int(_cfg_get(self.cfg, "fp_err_dim", -1)) > 0:
            output["fp_error"] = np.zeros((L,), dtype=np.float32)
        if not self.minimal_batch_fields or int(_cfg_get(self.cfg, "visibility_dim", -1)) > 0:
            output["visibility"] = np.ones((L,), dtype=np.float32)
        if destinations is None:
            output.update({
                "input_rgbs": torch.stack(input_rgbs, 0).float(),
                "render_rgbs": np.stack(render_rgbs, axis=0).astype(np.float32),
                "input_xyz": torch.stack(input_xyz, 0).float(),
                "render_xyz": torch.stack(render_xyz, 0).float(),
            })
        _profile_add(profile, "sample_finalize_seconds", sample_finalize_started_at)
        if profile is not None:
            profile["frame_count"] = len(frame_names)
            profile["render_sample_total_seconds"] = time.monotonic() - float(sample_started_at)
        return output
