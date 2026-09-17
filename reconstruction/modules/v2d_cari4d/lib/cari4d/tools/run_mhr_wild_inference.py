from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import joblib
import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from tqdm import tqdm

import Utils
from learning.datasets.mhr_input_materialization import MHR_SPATIAL_SCALE_KEY, MHR_XYZ_ANCHOR_KEY, build_mhr_spatial_normalization_contract, build_mhr_xyz_anchor_contract, mhr_spatial_scale_from_height, prepare_mhr_spatial_batch, restore_mhr_spatial_normalization_contract, restore_mhr_xyz_anchor_contract, select_mhr_xyz_anchor
from learning.inference import WildVideoDataProcessor, forward_batch_mhr, object_pose_from_relative
from learning.models import get_model
from learning.training.mhr_losses import compose_mhr_output
from learning.training.mhr_supervision import resolve_mhr_inference_provenance
from learning.training.training_config import TrainTemporalRefinerConfig
from lib_mhr import MHRLayer, MHR_PARAM_DIMS, assert_mhr_schema, decode_mhr_vertices_numpy, load_mhr_result, mhr70_to_coco17, object_poses_to_training_frame, resolve_object_pose_frame
from lib_mhr.camera_conventions import MHR_ROOT_JOINT_INDEX
from lib_mhr.human_texture import MHR_PART_TEXTURE_REVISION, batched_vertex_normals, make_mhr_part_texture_tensors
from lib_mhr.object_texture import MHR_OBJECT_RENDER_MATERIAL_REVISION, concatenate_object_mesh_parts, load_object_mesh_parts, load_original_object_visual_tensors
from lib_mhr.postopt_crop import build_postopt_crop, stack_postopt_crops
from prep.mhr_effective_masks import path_identity
from prep.mhr_depth_h5 import DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE
from prep.mhr_depth_backend import MOGE2_MODEL_ID, MOGE2_MODEL_REVISION, MOGE2_SOURCE_COMMIT
from prep.mhr_wild_depth import DEFAULT_WILD_DEPTH_BACKEND
from prep.prepare_mhr_wild_export import WILD_EXPORT_SCHEMA
from prep.mhr_export_utils import MHR_CAMERA_NAMES, camera_calibration, effective_mask_path, frame_names, load_edex, mask_input_identity, read_depth_m, read_mask, read_rgb, resolve_object_mesh_path
from tools.mhr_input_renderer import build_input_payload_from_modalities, compose_human_object_render, render_textured_object_parts
from tools.pipeline_timing import PipelineTimer


WILD_INFERENCE_SCHEMA = "cari4d.mhr_wild_inference.v1"
WILD_INPUT_CACHE_SCHEMA = "cari4d.mhr_wild_input_cache.v1"
MHR_INIT_DECODE_BATCH_SIZE = 8
_WILD_CROP_WORKER_STATE: dict[str, Any] | None = None


def _directory_identity(path: Path) -> dict[str, Any]:
    entries = []
    for entry in sorted(value for value in path.rglob("*") if value.is_file()):
        stat = entry.stat()
        entries.append({"path": str(entry.relative_to(path)), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return {"path": str(path.resolve()), "entries": entries}


def _storage_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return path_identity(path) if path.is_file() else _directory_identity(path)


def _rgb_source_identity(export_seq: Path) -> dict[str, Any]:
    camera_name = MHR_CAMERA_NAMES[0]
    h5_path = export_seq / "images" / f"{camera_name}.h5"
    return _storage_identity(h5_path if h5_path.is_file() else export_seq / "images" / camera_name)


def _source_code_identity() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    paths = (repo / "Utils.py", repo / "tools/run_mhr_wild_inference.py", repo / "tools/mhr_input_renderer.py", repo / "prep/mhr_geometry_crop.py", repo / "learning/inference.py", repo / "lib_mhr/human_texture.py", repo / "lib_mhr/object_texture.py")
    return {str(path.relative_to(repo)): {"size": int(path.stat().st_size), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths}


def _depth_h5_provenance(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        method = handle.attrs.get("depth_alignment_method")
        identity = handle.attrs.get(DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE)
    if isinstance(method, (bytes, np.bytes_)):
        method = bytes(method).decode("utf-8")
    if isinstance(identity, (bytes, np.bytes_)):
        identity = bytes(identity).decode("utf-8")
    identity = None if identity is None else json.loads(str(identity))
    if identity is not None and not isinstance(identity, dict):
        raise TypeError(f"{DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE} must decode to an object: {path}")
    backend = None if identity is None else identity.get("depth_backend")
    if backend is None and isinstance(identity, dict) and "raw_unidepth" in identity:
        backend = "unidepth"
    if backend != DEFAULT_WILD_DEPTH_BACKEND:
        raise ValueError(f"Native MHR wild inference requires {DEFAULT_WILD_DEPTH_BACKEND} depth, got {backend!r}: {path}")
    expected_identity = {"depth_model_id": MOGE2_MODEL_ID, "depth_model_revision": MOGE2_MODEL_REVISION, "depth_source_commit": MOGE2_SOURCE_COMMIT}
    mismatches = [key for key, expected in expected_identity.items() if identity.get(key) != expected]
    if mismatches:
        raise ValueError(f"Depth H5 differs from the pinned MoGe 2 identity in {mismatches}: {path}")
    return {"backend": backend, "alignment_method": None if method is None else str(method), "alignment_input_identity": identity}


def _validate_wild_export_depth_provenance(metadata: Mapping[str, Any], marker_path: Path) -> None:
    expected_identity = {"depth_backend": DEFAULT_WILD_DEPTH_BACKEND, "depth_model_id": MOGE2_MODEL_ID, "depth_model_revision": MOGE2_MODEL_REVISION, "depth_source_commit": MOGE2_SOURCE_COMMIT}
    mismatches = [key for key, expected in expected_identity.items() if metadata.get(key) != expected]
    if mismatches:
        raise ValueError(f"Wild export differs from the pinned MoGe 2 identity in {mismatches}: {marker_path}")


def _input_cache_identity(export_seq: Path, depth_h5: Path, depth_provenance: Mapping[str, Any], mhr_init: Path, foundationpose_file: Path, cfg: Any, object_mesh_path: Path, object_pose_frame: Any, decoder_identity: Mapping[str, str], names: Sequence[str]) -> dict[str, Any]:
    object_asset = object_mesh_path.resolve().parent if object_mesh_path.is_file() else object_mesh_path.resolve()
    identity = {
        "schema": WILD_INPUT_CACHE_SCHEMA,
        "frames": list(names),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "rgb": _rgb_source_identity(export_seq),
        "depth": path_identity(depth_h5),
        "depth_provenance": dict(depth_provenance),
        "human_mask": mask_input_identity(export_seq, "human", 0),
        "object_mask": mask_input_identity(export_seq, "object", 0),
        "effective_mask_sidecar": path_identity(effective_mask_path(export_seq, 0)) if effective_mask_path(export_seq, 0).is_file() else None,
        "mhr_initialization": path_identity(mhr_init),
        "foundationpose": path_identity(foundationpose_file),
        "object_asset": _storage_identity(object_asset),
        "object_pose_frame": object_pose_frame.metadata(),
        "mhr_decoder": dict(decoder_identity),
        "mhr_xyz_anchor_contract": build_mhr_xyz_anchor_contract(cfg),
        "mhr_spatial_normalization_contract": build_mhr_spatial_normalization_contract(cfg),
        "human_material_revision": MHR_PART_TEXTURE_REVISION,
        "object_material_revision": MHR_OBJECT_RENDER_MATERIAL_REVISION,
        "source_code": _source_code_identity(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "value": identity}


def _initialize_wild_crop_worker(state: dict[str, Any]) -> None:
    import cv2

    global _WILD_CROP_WORKER_STATE
    cv2.setNumThreads(1)
    _WILD_CROP_WORKER_STATE = state


def _build_wild_crop_payload(state: Mapping[str, Any], index: int, frame_name: str) -> tuple[dict[str, Any], dict[str, float]]:
    timings: dict[str, float] = {}
    export_seq = Path(state["export_seq"])
    rgb_started = time.perf_counter()
    rgb = read_rgb(export_seq, 0, frame_name)
    timings["rgb_decode"] = time.perf_counter() - rgb_started
    depth_started = time.perf_counter()
    depth = read_depth_m(export_seq, 0, frame_name, depth_root=state["depth_h5"])
    timings["depth_decode"] = time.perf_counter() - depth_started
    mask_started = time.perf_counter()
    mask_h = read_mask(export_seq, "human", 0, frame_name).astype(np.uint8) * 255
    mask_o = read_mask(export_seq, "object", 0, frame_name).astype(np.uint8) * 255
    timings["mask_decode"] = time.perf_counter() - mask_started
    pose = state["object_poses"][index]
    object_vertices_camera = state["object_vertices"] @ pose[:3, :3].T + pose[:3, 3]
    rgbm, xyz, K_roi, bbox, diagnostics = build_input_payload_from_modalities(rgb, depth, mask_h, mask_o, state["K"], state["render_size"], state["init_vertices"][index], state["faces"], object_vertices_camera, state["object_faces"], human_pose_valid=True, object_pose_valid=True, timings=timings)
    record = {"rgbmB": rgbm, "xyzB": xyz, "K_roi": K_roi, "bbox": bbox, "crop_diagnostics": diagnostics, "observed_human_mask": rgbm[..., 3] > 127, "observed_object_mask": rgbm[..., 4] > 127}
    return record, timings


def _wild_crop_payload_worker(index: int, frame_name: str) -> tuple[dict[str, Any], dict[str, float]]:
    if _WILD_CROP_WORKER_STATE is None:
        raise RuntimeError("Wild crop worker was not initialized")
    return _build_wild_crop_payload(_WILD_CROP_WORKER_STATE, int(index), str(frame_name))


def validate_renderer_crop_size(input_resize: Sequence[int]) -> tuple[int, int]:
    render_size = tuple(int(value) for value in input_resize)
    if len(render_size) != 2 or render_size[0] != render_size[1]:
        raise ValueError(f"Renderer crop size must be square, got {render_size}")
    return render_size


def sliding_window_starts(frame_count: int, clip_len: int, stride: int) -> list[int]:
    frame_count, clip_len, stride = int(frame_count), int(clip_len), int(stride)
    if frame_count <= 0 or clip_len <= 0 or stride <= 0:
        raise ValueError(f"frame_count, clip_len, and stride must be positive, got {frame_count}, {clip_len}, {stride}")
    if frame_count < clip_len:
        raise ValueError(f"Wild inference needs at least {clip_len} frames, got {frame_count}")
    starts = list(range(0, frame_count - clip_len + 1, stride))
    terminal = frame_count - clip_len
    if starts[-1] != terminal:
        starts.append(terminal)
    return starts


def _load_config(path: str | Path) -> Any:
    base = OmegaConf.structured(TrainTemporalRefinerConfig)
    cfg = OmegaConf.merge(base, OmegaConf.load(str(path)))
    if str(cfg.body_model) != "mhr":
        raise ValueError(f"Wild native inference requires body_model=mhr, got {cfg.body_model!r}")
    if tuple(int(value) for value in cfg.input_resize) != (224, 224):
        raise ValueError(f"Checkpoint production input resolution must be 224x224, got {cfg.input_resize}")
    if int(cfg.fp_err_dim) > 0 or int(cfg.visibility_dim) > 0:
        raise ValueError("Wild inference does not fabricate ground-truth-only auxiliary features")
    cfg.no_wandb = True
    cfg.job = "test"
    cfg.load_training_state = False
    cfg.mhr_rank_local_preprocess = False
    cfg.render_root = "mhr-wild-runtime"
    return cfg


def _load_checkpoint_model(cfg: Any, checkpoint_path: Path, device: torch.device, *, wandb_run_path: str | None = None, offline_supervision_contract: bool = False) -> tuple[torch.nn.Module, int, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    xyz_anchor_contract = restore_mhr_xyz_anchor_contract(checkpoint, cfg)
    spatial_normalization_contract = restore_mhr_spatial_normalization_contract(checkpoint, cfg)
    provenance = resolve_mhr_inference_provenance(checkpoint, checkpoint_path, cfg, wandb_run_path=wandb_run_path, offline=offline_supervision_contract)
    provenance = {**provenance, "xyzAnchorContract": xyz_anchor_contract, "spatialNormalizationContract": spatial_normalization_contract}
    model = get_model(cfg)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint model state must be a mapping: {checkpoint_path}")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"Checkpoint architecture differs from resolved config: missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    return model, int(checkpoint.get("step", -1)), provenance


def _load_object_mesh(export_seq: Path) -> trimesh.Trimesh:
    return concatenate_object_mesh_parts(load_object_mesh_parts(resolve_object_mesh_path(export_seq)))


def _pose_vertices(vertices: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return vertices[None] @ poses[:, :3, :3].transpose(0, 2, 1) + poses[:, None, :3, 3]


class WildInputBuilder:
    def __init__(self, cfg: Any, export_seq: Path, depth_h5: Path, init: Mapping[str, Any], init_vertices: np.ndarray, object_poses: np.ndarray, K: np.ndarray, faces: np.ndarray, object_mesh: trimesh.Trimesh, mesh_diameter: float, mhr_spatial_scale: np.ndarray, device: torch.device, render_batch_size: int, *, crop_workers: int = 8, crop_buffer_count: int = 2, profiler: PipelineTimer | None = None, input_cache_path: Path | None = None, input_cache_identity: Mapping[str, Any] | None = None):
        import nvdiffrast.torch as dr

        self.cfg, self.export_seq, self.depth_h5, self.init, self.object_poses, self.K, self.faces = cfg, export_seq, depth_h5, init, object_poses, K, faces
        self.init_vertices = np.asarray(init_vertices, dtype=np.float32)
        init_joints = np.asarray(init["mhr_joints"], dtype=np.float32)
        if init_joints.ndim != 3 or init_joints.shape[0] != len(init["frames"]) or init_joints.shape[1] <= MHR_ROOT_JOINT_INDEX or init_joints.shape[2] != 3:
            raise ValueError(f"Wild MHR initialization cannot provide root joint {MHR_ROOT_JOINT_INDEX}: {init_joints.shape}")
        self.mhr_xyz_anchor = np.asarray(select_mhr_xyz_anchor(np.asarray(init["mhr_trans"], dtype=np.float32), init_joints[:, MHR_ROOT_JOINT_INDEX], cfg), dtype=np.float32)
        self.mhr_spatial_scale = np.asarray(mhr_spatial_scale, dtype=np.float32)
        if self.mhr_spatial_scale.shape != (len(init["frames"]),) or not np.isfinite(self.mhr_spatial_scale).all() or not (self.mhr_spatial_scale > 0).all():
            raise ValueError(f"Wild MHR spatial scale must be finite and positive [{len(init['frames'])}], got {self.mhr_spatial_scale.shape}")
        self.object_vertices = np.asarray(object_mesh.vertices, dtype=np.float32)
        self.object_faces = np.asarray(object_mesh.faces, dtype=np.int32)
        self.mesh_diameter, self.device, self.render_batch_size = float(mesh_diameter), device, int(render_batch_size)
        self.crop_workers, self.crop_buffer_count, self.profiler = int(crop_workers), int(crop_buffer_count), profiler
        if self.render_batch_size <= 0 or self.crop_workers <= 0 or self.crop_buffer_count <= 0:
            raise ValueError(f"render_batch_size, crop_workers, and crop_buffer_count must be positive, got {self.render_batch_size}, {self.crop_workers}, {self.crop_buffer_count}")
        self.processor = WildVideoDataProcessor(cfg)
        self.glctx = dr.RasterizeCudaContext()
        self.human_tensors = make_mhr_part_texture_tensors(faces, device)
        self.object_tensors = load_original_object_visual_tensors(resolve_object_mesh_path(export_seq), device)
        self.render_size = validate_renderer_crop_size(cfg.input_resize)
        self._crop_executor: ProcessPoolExecutor | None = None
        self._frame_cache: dict[tuple[int, int | None], dict[str, Any]] = {}
        self._frame_cache_hits = 0
        self._frame_cache_misses = 0
        self._persistent_cache_hits = 0
        self.input_cache_path = None if input_cache_path is None else Path(input_cache_path).resolve()
        self.input_cache_identity = None if input_cache_identity is None else dict(input_cache_identity)
        self._persistent_cache: h5py.File | None = None
        self._crop_state = {"export_seq": str(export_seq), "depth_h5": str(depth_h5), "init_vertices": self.init_vertices, "faces": self.faces, "object_vertices": self.object_vertices, "object_faces": self.object_faces, "object_poses": self.object_poses, "K": self.K, "render_size": self.render_size}
        self._open_persistent_cache()

    def _open_persistent_cache(self) -> None:
        if self.input_cache_path is None or not self.input_cache_path.is_file():
            return
        handle = h5py.File(self.input_cache_path, "r")
        schema = handle.attrs.get("schema", "")
        schema = schema.decode("utf-8") if isinstance(schema, bytes) else str(schema)
        if schema != WILD_INPUT_CACHE_SCHEMA:
            handle.close()
            raise ValueError(f"Wild input cache schema differs at {self.input_cache_path}: {schema!r}")
        expected_identity = json.dumps(self.input_cache_identity, sort_keys=True, separators=(",", ":"))
        observed_identity = handle["identity_json"][()] if "identity_json" in handle else ""
        observed_identity = observed_identity.decode("utf-8") if isinstance(observed_identity, bytes) else str(observed_identity)
        if observed_identity != expected_identity:
            handle.close()
            raise ValueError(f"Wild input cache identity differs at {self.input_cache_path}")
        if not bool(handle.attrs.get("complete", False)):
            handle.close()
            return
        frame_count = len(self.init["frames"])
        expected = {"input_rgbs": np.dtype("float32"), "render_rgbs": np.dtype("float32"), "input_xyz": np.dtype("float32"), "render_xyz": np.dtype("float32"), "K_rois": np.dtype("float32"), "bboxes": np.dtype("float32"), "observed_human_masks": np.dtype("bool"), "observed_object_masks": np.dtype("bool")}
        for key, dtype in expected.items():
            if key not in handle or not isinstance(handle[key], h5py.Dataset) or handle[key].shape[0] != frame_count or handle[key].dtype != dtype:
                handle.close()
                raise ValueError(f"Wild input cache dataset {key!r} differs at {self.input_cache_path}")
        self._persistent_cache = handle

    def _ensure_crop_executor(self) -> ProcessPoolExecutor | None:
        if self.crop_workers == 1:
            return None
        if self._crop_executor is None:
            started = self.profiler.start() if self.profiler is not None else 0.0
            self._crop_executor = ProcessPoolExecutor(max_workers=self.crop_workers, mp_context=get_context("spawn"), initializer=_initialize_wild_crop_worker, initargs=(self._crop_state,))
            if self.profiler is not None:
                self.profiler.record("crop_pool_create", started)
        return self._crop_executor

    def close(self) -> None:
        if self._crop_executor is not None:
            self._crop_executor.shutdown(wait=True, cancel_futures=True)
            self._crop_executor = None
        if self._persistent_cache is not None:
            self._persistent_cache.close()
            self._persistent_cache = None

    def _cache_context(self, indices: np.ndarray) -> int | None:
        if str(self.cfg.trans_ref_type) == "frame":
            return None
        if str(self.cfg.trans_ref_type) == "1st-frame":
            return int(indices[0])
        raise ValueError(f"Unknown translation reference type: {self.cfg.trans_ref_type}")

    def _cache_key(self, index: int, indices: np.ndarray) -> tuple[int, int | None]:
        return int(index), self._cache_context(indices)

    def _persistent_items(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        if self._persistent_cache is None:
            raise RuntimeError("Persistent wild input cache is not open")
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            return []
        started = self.profiler.start() if self.profiler is not None else 0.0
        handle = self._persistent_cache
        values = {key: handle[key][indices] for key in ("input_rgbs", "render_rgbs", "input_xyz", "render_xyz", "K_rois", "bboxes", "observed_human_masks", "observed_object_masks")}
        items = []
        for local in range(len(indices)):
            items.append({"input_rgbs": torch.from_numpy(values["input_rgbs"][local]), "render_rgbs": torch.from_numpy(values["render_rgbs"][local]), "input_xyz": torch.from_numpy(values["input_xyz"][local]), "render_xyz": torch.from_numpy(values["render_xyz"][local]), "record": {"K_roi": values["K_rois"][local], "bbox": values["bboxes"][local], "observed_human_mask": values["observed_human_masks"][local], "observed_object_mask": values["observed_object_masks"][local], "crop_diagnostics": {}}})
        if self.profiler is not None:
            self.profiler.record("persistent_input_cache_read", started)
        self._persistent_cache_hits += len(indices)
        return items

    def _submit_crop_chunk(self, indices: np.ndarray) -> list[Future[tuple[dict[str, Any], dict[str, float]]]] | list[tuple[dict[str, Any], dict[str, float]]]:
        names = [str(self.init["frames"][index]) for index in indices]
        executor = self._ensure_crop_executor()
        if executor is None:
            return [_build_wild_crop_payload(self._crop_state, int(index), name) for index, name in zip(indices, names)]
        return [executor.submit(_wild_crop_payload_worker, int(index), name) for index, name in zip(indices, names)]

    def _resolve_crop_chunk(self, pending: list[Future[tuple[dict[str, Any], dict[str, float]]]] | list[tuple[dict[str, Any], dict[str, float]]]) -> list[dict[str, Any]]:
        started = self.profiler.start() if self.profiler is not None else 0.0
        resolved = [value.result() if isinstance(value, Future) else value for value in pending]
        if self.profiler is not None:
            self.profiler.record("crop_payload_wait", started)
            for _, timings in resolved:
                for phase, elapsed in timings.items():
                    self.profiler.record_elapsed(f"crop_worker_{phase}", elapsed)
        return [record for record, _ in resolved]

    def _render_initialization(self, indices: np.ndarray, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        render_started = self.profiler.start() if self.profiler is not None else 0.0
        init_vertices = self.init_vertices[indices]
        results = []
        height, width = self.render_size[1], self.render_size[0]
        for start in range(0, len(indices), self.render_batch_size):
            stop = min(start + self.render_batch_size, len(indices))
            human = dict(self.human_tensors)
            human["pos"] = torch.as_tensor(init_vertices[start:stop], device=self.device, dtype=torch.float)
            human["vnormals"] = torch.as_tensor(batched_vertex_normals(init_vertices[start:stop], self.faces), device=self.device, dtype=torch.float)
            count = stop - start
            identity = torch.eye(4, device=self.device, dtype=torch.float).reshape(1, 4, 4).repeat(count, 1, 1)
            K_batch = np.stack([np.asarray(records[index]["K_roi"], dtype=np.float32) for index in range(start, stop)], axis=0)
            bbox = torch.tensor([[0.0, 0.0, float(width), float(height)]], device=self.device).repeat(count, 1)
            human_rgb, human_depth, _ = Utils.nvdiffrast_render(K=K_batch, H=height, W=width, ob_in_cams=identity, glctx=self.glctx, context="cuda", get_normal=False, mesh_tensors=human, output_size=(height, width), bbox2d=bbox, use_light=True, extra={})
            object_rgb, object_depth = render_textured_object_parts(self.object_tensors, self.object_poses[indices[start:stop]], K_batch, height, width, self.glctx, (height, width), bbox2d=bbox)
            rgb, depth = compose_human_object_render(human_rgb, human_depth, object_rgb, object_depth)
            rgb_np = (rgb.detach().cpu().numpy() * 255.0).astype(np.uint8)
            depth_np = depth.detach().cpu().numpy().astype(np.float32)
            object_depth_np = object_depth.detach().cpu().numpy().astype(np.float32)
            for local in range(count):
                visible = (object_depth_np[local] <= depth_np[local]) & (object_depth_np[local] > 0)
                full = object_depth_np[local] > 0
                results.append({"rgba": rgb_np[local], "depth": depth_np[local], "pose": self.object_poses[indices[start + local]], "K_roi": K_batch[local], "bbox": np.asarray(records[start + local]["bbox"], dtype=np.float32), "mask_o": np.stack([visible, full], axis=-1)})
        if self.profiler is not None:
            self.profiler.record("initialization_render", render_started)
        return results

    def _materialize_chunk(self, chunk_indices: np.ndarray, records: Sequence[Mapping[str, Any]], window_indices: np.ndarray) -> None:
        renders = self._render_initialization(chunk_indices, records)
        process_started = self.profiler.start() if self.profiler is not None else 0.0
        xyz_anchor = self.mhr_xyz_anchor[window_indices]
        local_by_index = {int(index): local for local, index in enumerate(window_indices)}
        for index, record, render in zip(chunk_indices, records, renders):
            local = local_by_index[int(index)]
            xyz_init = torch.from_numpy(Utils.depth2xyzmap(render["depth"], render["K_roi"])).permute(2, 0, 1).float()
            xyz_b, xyz_a, rgb_b = self.processor.process_input(xyz_init, local, record, self.mesh_diameter, xyz_anchor, render["pose"], render, render["rgba"])
            key = self._cache_key(int(index), window_indices)
            self._frame_cache[key] = {"input_rgbs": rgb_b, "render_rgbs": torch.from_numpy(render["rgba"].transpose(2, 0, 1) / 255.0).float(), "input_xyz": xyz_b, "render_xyz": xyz_a, "record": dict(record)}
            self._frame_cache_misses += 1
        if self.profiler is not None:
            self.profiler.record("process_input", process_started)

    def _materialize_missing(self, missing: np.ndarray, window_indices: np.ndarray) -> None:
        chunks = [missing[start:start + self.render_batch_size] for start in range(0, len(missing), self.render_batch_size)]
        pending: dict[int, list[Any]] = {}
        for chunk_index in range(min(self.crop_buffer_count, len(chunks))):
            pending[chunk_index] = self._submit_crop_chunk(chunks[chunk_index])
        for chunk_index, chunk_indices in enumerate(chunks):
            records = self._resolve_crop_chunk(pending.pop(chunk_index))
            refill = chunk_index + self.crop_buffer_count
            if refill < len(chunks):
                pending[refill] = self._submit_crop_chunk(chunks[refill])
            self._materialize_chunk(chunk_indices, records, window_indices)

    def build(self, indices: np.ndarray) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        indices = np.asarray(indices, dtype=np.int64)
        missing = []
        persistent_indices = []
        for index in indices:
            key = self._cache_key(int(index), indices)
            if key in self._frame_cache:
                self._frame_cache_hits += 1
            elif self._persistent_cache is not None:
                persistent_indices.append(int(index))
            else:
                missing.append(int(index))
        if persistent_indices:
            for index, item in zip(persistent_indices, self._persistent_items(persistent_indices)):
                self._frame_cache[self._cache_key(index, indices)] = item
        if missing:
            self._materialize_missing(np.asarray(missing, dtype=np.int64), indices)
        stack_started = self.profiler.start() if self.profiler is not None else 0.0
        items = [self._frame_cache[self._cache_key(int(index), indices)] for index in indices]
        records = [item["record"] for item in items]
        pose = self.object_poses[indices].astype(np.float32)
        pose_norm = pose.copy()
        batch: dict[str, torch.Tensor] = {
            "input_rgbs": torch.stack([item["input_rgbs"] for item in items])[None], "render_rgbs": torch.stack([item["render_rgbs"] for item in items])[None], "input_xyz": torch.stack([item["input_xyz"] for item in items])[None], "render_xyz": torch.stack([item["render_xyz"] for item in items])[None],
            "pose_perturbed": torch.from_numpy(pose)[None], "poseA_norm": torch.from_numpy(pose_norm)[None], "mesh_diameter": torch.full((1, len(indices)), self.mesh_diameter, dtype=torch.float32), "frame_mask": torch.ones((1, len(indices)), dtype=torch.float32),
            MHR_SPATIAL_SCALE_KEY: torch.from_numpy(self.mhr_spatial_scale[indices])[None], MHR_XYZ_ANCHOR_KEY: torch.from_numpy(self.mhr_xyz_anchor[indices])[None],
        }
        for key in MHR_PARAM_DIMS:
            batch[f"{key}_init"] = torch.from_numpy(np.asarray(self.init[key], dtype=np.float32)[indices])[None]
        coco17 = mhr70_to_coco17(np.asarray(self.init["mhr_keypoints"], dtype=np.float32)[indices])
        batch["mhr_coco17_init"] = torch.from_numpy(coco17)[None]
        prepare_mhr_spatial_batch(batch, self.cfg, None)
        if self.profiler is not None:
            self.profiler.record("input_stack", stack_started)
            self.profiler.update_metadata({"crop_workers": self.crop_workers, "crop_buffer_count": self.crop_buffer_count, "render_batch_size": self.render_batch_size, "frame_cache_hits": self._frame_cache_hits, "frame_cache_misses": self._frame_cache_misses, "persistent_input_cache_hits": self._persistent_cache_hits})
        return batch, records

    def publish_persistent_cache(self) -> Path | None:
        if self.input_cache_path is None or self._persistent_cache is not None:
            return self.input_cache_path
        if self._cache_context(np.asarray([0], dtype=np.int64)) is not None:
            raise ValueError("Persistent wild input caching requires trans_ref_type=frame")
        frame_count = len(self.init["frames"])
        missing = [index for index in range(frame_count) if (index, None) not in self._frame_cache]
        if missing:
            raise ValueError(f"Cannot publish wild input cache with {len(missing)} missing frames")
        started = self.profiler.start() if self.profiler is not None else 0.0
        path = self.input_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(f".{path.name}.lock")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if path.is_file():
                with h5py.File(path, "r") as existing:
                    identity_json = existing["identity_json"][()] if "identity_json" in existing else ""
                    identity_json = identity_json.decode("utf-8") if isinstance(identity_json, bytes) else str(identity_json)
                    expected_identity = json.dumps(self.input_cache_identity, sort_keys=True, separators=(",", ":"))
                    if bool(existing.attrs.get("complete", False)) and identity_json == expected_identity:
                        return path
            temporary.unlink(missing_ok=True)
            try:
                first = self._frame_cache[(0, None)]
                with h5py.File(temporary, "w") as handle:
                    handle.attrs["schema"] = WILD_INPUT_CACHE_SCHEMA
                    handle.attrs["identity_sha256"] = str(self.input_cache_identity["sha256"])
                    handle.attrs["complete"] = False
                    handle.create_dataset("identity_json", data=json.dumps(self.input_cache_identity, sort_keys=True, separators=(",", ":")), dtype=h5py.string_dtype(encoding="utf-8"))
                    for key in ("input_rgbs", "render_rgbs", "input_xyz", "render_xyz"):
                        value = first[key].detach().cpu().numpy()
                        handle.create_dataset(key, shape=(frame_count, *value.shape), dtype=value.dtype)
                    height, width = first["record"]["observed_human_mask"].shape
                    handle.create_dataset("K_rois", shape=(frame_count, 3, 3), dtype="float32")
                    handle.create_dataset("bboxes", shape=(frame_count, 4), dtype="float32")
                    handle.create_dataset("observed_human_masks", shape=(frame_count, height, width), dtype="bool")
                    handle.create_dataset("observed_object_masks", shape=(frame_count, height, width), dtype="bool")
                    for start in range(0, frame_count, 16):
                        stop = min(start + 16, frame_count)
                        items = [self._frame_cache[(index, None)] for index in range(start, stop)]
                        for key in ("input_rgbs", "render_rgbs", "input_xyz", "render_xyz"):
                            handle[key][start:stop] = np.stack([item[key].detach().cpu().numpy() for item in items])
                        handle["K_rois"][start:stop] = np.stack([item["record"]["K_roi"] for item in items]).astype(np.float32)
                        handle["bboxes"][start:stop] = np.stack([item["record"]["bbox"] for item in items]).astype(np.float32)
                        handle["observed_human_masks"][start:stop] = np.stack([item["record"]["observed_human_mask"] for item in items]).astype(bool)
                        handle["observed_object_masks"][start:stop] = np.stack([item["record"]["observed_object_mask"] for item in items]).astype(bool)
                    handle.flush()
                    handle.attrs.modify("complete", True)
                    handle.flush()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        if self.profiler is not None:
            self.profiler.record("persistent_input_cache_write", started)
        return path


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _numpy(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def write_wild_inference_bundle(bundle: Mapping[str, Any], output: str | Path) -> Path:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(bundle, temporary, pickle_protocol=4)
    os.replace(temporary, output)
    return output


def run_mhr_wild_inference(export_seq: str | Path, depth_h5: str | Path, mhr_init: str | Path, foundationpose_file: str | Path, config_file: str | Path, checkpoint_file: str | Path, output: str | Path, *, stride: int = 96, render_batch_size: int = 32, crop_workers: int = 8, crop_buffer_count: int = 2, input_cache: str | Path | None = None, use_input_cache: bool = True, device_name: str = "cuda", overwrite: bool = False, wandb_run_path: str | None = None, offline_supervision_contract: bool = False, profiler: PipelineTimer | None = None) -> Path:
    setup_started = profiler.start() if profiler is not None else 0.0
    export_seq, depth_h5, mhr_init, foundationpose_file, config_file, checkpoint_file, output = map(lambda value: Path(value).resolve(), (export_seq, depth_h5, mhr_init, foundationpose_file, config_file, checkpoint_file, output))
    marker_path = export_seq / "wild_export.json"
    for path in (marker_path, depth_h5, mhr_init, foundationpose_file, config_file, checkpoint_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if not torch.cuda.is_available() and str(device_name).startswith("cuda"):
        raise RuntimeError("Wild MHR inference requires CUDA")
    device = torch.device(device_name)
    cfg = _load_config(config_file)
    wild_metadata = json.loads(marker_path.read_text())
    if not isinstance(wild_metadata, Mapping) or wild_metadata.get("schema") != WILD_EXPORT_SCHEMA:
        raise ValueError(f"Wild export must be regenerated with schema {WILD_EXPORT_SCHEMA!r}: {marker_path}")
    _validate_wild_export_depth_provenance(wild_metadata, marker_path)
    object_pose_frame = resolve_object_pose_frame(wild_metadata)
    depth_provenance = _depth_h5_provenance(depth_h5)
    export_depth_backend = wild_metadata.get("depth_backend")
    if str(export_depth_backend) != str(depth_provenance["backend"]):
        raise ValueError(f"Wild export camera backend {export_depth_backend!r} differs from aligned depth backend {depth_provenance['backend']!r}")
    clip_len = int(cfg.clip_len)
    init = load_mhr_result(mhr_init)
    assert_mhr_schema(init, require_geometry=True)
    names = frame_names(export_seq, 0)
    if [str(value) for value in init["frames"]] != names:
        raise ValueError("Wild RGB export and direct MHR initialization timelines differ")
    fp = joblib.load(foundationpose_file)
    if [str(value) for value in fp.get("frames", [])] != names:
        raise ValueError("Wild RGB export and FoundationPose timelines differ")
    object_poses = np.asarray(fp["obj_pose_world"], dtype=np.float32)
    if object_poses.shape != (len(names), 4, 4) or not np.isfinite(object_poses).all():
        raise ValueError(f"FoundationPose object poses must be finite [{len(names)},4,4], got {object_poses.shape}")
    object_poses = object_poses_to_training_frame(object_poses, object_pose_frame.storage_to_training)
    K, world_to_camera = camera_calibration(load_edex(export_seq), 0)
    if not np.allclose(world_to_camera, np.eye(4), atol=1e-6):
        raise ValueError("Wild one-camera export must use camera space as world space")
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ.get("MHR_ASSETS_ROOT"), device=str(device))
    init_vertices = decode_mhr_vertices_numpy(layer, {key: np.asarray(init[key], dtype=np.float32) for key in MHR_PARAM_DIMS}, batch_size=MHR_INIT_DECODE_BATCH_SIZE)
    with torch.inference_mode():
        neutral_height = _numpy(layer.neutral_height({"mhr_shape": np.asarray(init["mhr_shape"], dtype=np.float32), "mhr_scale": np.asarray(init["mhr_scale"], dtype=np.float32)})).astype(np.float32)
    faces_value = layer.mesh_faces(device=device)
    faces = _numpy(faces_value).astype(np.int32)
    object_mesh = _load_object_mesh(export_seq)
    object_mesh.apply_transform(object_pose_frame.mesh_to_training)
    np.random.seed(0)
    mesh_diameter = float(Utils.compute_mesh_diameter(model_pts=init_vertices[0], n_sample=8000))
    model, checkpoint_step, inference_provenance = _load_checkpoint_model(cfg, checkpoint_file, device, wandb_run_path=wandb_run_path, offline_supervision_contract=offline_supervision_contract)
    mhr_spatial_scale = np.asarray(mhr_spatial_scale_from_height(neutral_height, cfg), dtype=np.float32)
    cache_identity_started = profiler.start() if profiler is not None else 0.0
    cache_identity = _input_cache_identity(export_seq, depth_h5, depth_provenance, mhr_init, foundationpose_file, cfg, resolve_object_mesh_path(export_seq), object_pose_frame, layer.decoder_identity(), names) if use_input_cache else None
    if profiler is not None:
        profiler.record("input_cache_identity", cache_identity_started)
    input_cache_path = None
    if use_input_cache:
        input_cache_path = Path(input_cache).resolve() if input_cache is not None else depth_h5.parent / f"{export_seq.name}_coconet-input-{cache_identity['sha256'][:16]}.h5"
    builder = WildInputBuilder(cfg, export_seq, depth_h5, init, init_vertices, object_poses, K, faces, object_mesh, mesh_diameter, mhr_spatial_scale, device, render_batch_size, crop_workers=crop_workers, crop_buffer_count=crop_buffer_count, profiler=profiler, input_cache_path=input_cache_path, input_cache_identity=cache_identity)
    starts = sliding_window_starts(len(names), clip_len, stride)
    assigned = np.zeros(len(names), dtype=bool)
    pred_params = {key: np.empty((len(names), dim), dtype=np.float32) for key, dim in MHR_PARAM_DIMS.items()}
    raw = {}
    pose_pred = np.empty((len(names), 4, 4), dtype=np.float32)
    contact_logits = np.empty((len(names), int(cfg.cont_out_dim)), dtype=np.float32) if int(cfg.cont_out_dim) > 0 else None
    K_rois = np.empty((len(names), 3, 3), dtype=np.float32)
    bboxes = np.empty((len(names), 4), dtype=np.float32)
    observed_human_masks = np.empty((len(names), *tuple(int(value) for value in cfg.input_resize[::-1])), dtype=bool)
    observed_object_masks = np.empty_like(observed_human_masks)
    supervision_contract = inference_provenance["supervisionContract"]
    if profiler is not None:
        profiler.record("setup", setup_started)
        profiler.update_metadata({"frames": len(names), "sliding_windows": len(starts), "window_length": clip_len, "window_stride": int(stride)})
    progress = tqdm(starts, desc="CoCoNet wild sliding windows")
    try:
        with torch.inference_mode():
            for start in progress:
                indices = np.arange(start, start + clip_len, dtype=np.int64)
                input_started = profiler.start() if profiler is not None else 0.0
                batch_cpu, records = builder.build(indices)
                if profiler is not None:
                    profiler.record("window_input_materialization", input_started)
                transfer_started = profiler.start() if profiler is not None and device.type != "cuda" else 0.0
                transfer_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if profiler is not None and device.type == "cuda" else None
                if transfer_events is not None:
                    transfer_events[0].record()
                batch = _to_device(batch_cpu, device)
                if transfer_events is not None:
                    transfer_events[1].record()
                elif profiler is not None:
                    profiler.record("host_to_gpu_transfer", transfer_started)
                model_and_collection_started = profiler.start() if profiler is not None else 0.0
                model_started = profiler.start() if profiler is not None and device.type != "cuda" else 0.0
                model_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if profiler is not None and device.type == "cuda" else None
                if model_events is not None:
                    model_events[0].record()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(cfg.enable_amp)):
                    output_model = forward_batch_mhr(batch, model, supervision_contract)
                if model_events is not None:
                    model_events[1].record()
                model_elapsed = profiler.elapsed(model_started) if profiler is not None and model_events is None else 0.0
                prediction_output = {key: value.float() if torch.is_tensor(value) else value for key, value in output_model.items()}
                composed = compose_mhr_output(prediction_output, batch)
                object_pred = object_pose_from_relative(batch, cfg, batch["pose_perturbed"], prediction_output["rot"].reshape(-1, prediction_output["rot"].shape[-1]), prediction_output["trans"].reshape(-1, 3))[0]
                local_keep = ~assigned[indices]
                target = indices[local_keep]
                for key in MHR_PARAM_DIMS:
                    pred_params[key][target] = _numpy(composed[key][0])[local_keep]
                pose_pred[target] = _numpy(object_pred)[local_keep]
                if contact_logits is not None:
                    if "contact" not in prediction_output:
                        raise KeyError("Configured contact head did not produce contact logits")
                    contact_logits[target] = _numpy(prediction_output["contact"]).reshape(1, clip_len, -1)[0][local_keep]
                for key, value in prediction_output.items():
                    if key in {"rot", "trans"} or key.startswith("delta_mhr_"):
                        raw.setdefault(key, np.empty((len(names), _numpy(value).reshape(1, clip_len, -1).shape[-1]), dtype=np.float32))[target] = _numpy(value).reshape(1, clip_len, -1)[0][local_keep]
                K_rois[target] = np.stack([records[index]["K_roi"] for index in np.flatnonzero(local_keep)], axis=0)
                bboxes[target] = np.stack([records[index]["bbox"] for index in np.flatnonzero(local_keep)], axis=0)
                selected_records = [records[index] for index in np.flatnonzero(local_keep)]
                observed_human_masks[target] = np.stack([np.asarray(record["observed_human_mask"], dtype=bool) for record in selected_records], axis=0)
                observed_object_masks[target] = np.stack([np.asarray(record["observed_object_mask"], dtype=bool) for record in selected_records], axis=0)
                assigned[target] = True
                if profiler is not None:
                    combined_elapsed = profiler.elapsed(model_and_collection_started)
                    transfer_elapsed = 0.0
                    if transfer_events is not None:
                        transfer_elapsed = transfer_events[0].elapsed_time(transfer_events[1]) / 1000.0
                        profiler.record_elapsed("host_to_gpu_transfer", transfer_elapsed)
                    if model_events is not None:
                        model_elapsed = model_events[0].elapsed_time(model_events[1]) / 1000.0
                    profiler.record_elapsed("window_model_execution", model_elapsed)
                    profiler.record_elapsed("window_output_collection", max(combined_elapsed - model_elapsed - transfer_elapsed, 0.0))
                print("MHR_WILD_INFERENCE_PROGRESS " + json.dumps({"current": int(assigned.sum()), "total": len(names), "unit": "frames"}, sort_keys=True), flush=True)
        builder.publish_persistent_cache()
    finally:
        builder.close()
    if not assigned.all():
        raise RuntimeError(f"Sliding-window inference left {int((~assigned).sum())} frames unassigned")
    observations_started = profiler.start() if profiler is not None else 0.0
    postopt_crops = [build_postopt_crop(read_mask(export_seq, "human", 0, name), read_mask(export_seq, "object", 0, name), K) for name in names]
    postopt_observations = stack_postopt_crops(postopt_crops, K)
    if profiler is not None:
        profiler.record("postopt_observation_build", observations_started)
    init_params = {key: np.asarray(init[key], dtype=np.float32).copy() for key in MHR_PARAM_DIMS}
    pr = {"pose_abs": pose_pred, "pose_abs_1st_delta": pose_pred.copy(), **pred_params}
    if contact_logits is not None:
        pr["contact_logits"] = contact_logits
    bundle = {
        "schema": WILD_INFERENCE_SCHEMA, "frames": names, "frame_meta": [{"frame": name, "src_frame": int(name), "kid": 0} for name in names], "kid": 0,
        "checkpoint": {"path": str(checkpoint_file), "step": checkpoint_step}, "mhr_inference": inference_provenance, "config": OmegaConf.to_container(cfg, resolve=True), "mesh_diameter": np.full(len(names), mesh_diameter, dtype=np.float32), "mhr_neutral_height_init": neutral_height, MHR_SPATIAL_SCALE_KEY: mhr_spatial_scale, "K_rois": K_rois, "bboxes": bboxes, "faces": faces,
        "pr": pr, "pr_initial": copy.deepcopy(pr), "in": {"pose_abs": object_poses.copy(), **init_params}, "gt": {}, "observations": {"human_mask": observed_human_masks, "object_mask": observed_object_masks, **postopt_observations}, "raw": raw,
        "metadata": {"ground_truth_used": False, "window_length": clip_len, "window_stride": int(stride), "overlap_policy": "first_occurrence", "depth_source": str(depth_h5), "depth_backend": depth_provenance["backend"], "depth_alignment_method": depth_provenance["alignment_method"], "mhr_init_source": str(mhr_init), "foundationpose_source": str(foundationpose_file), "object_mesh": str(resolve_object_mesh_path(export_seq)), "mhr_xyz_anchor_contract": inference_provenance["xyzAnchorContract"], "mhr_spatial_normalization_contract": inference_provenance["spatialNormalizationContract"], "materialized_input_cache": None if input_cache_path is None else str(input_cache_path), "materialized_input_cache_identity": None if cache_identity is None else cache_identity["sha256"], **object_pose_frame.metadata()},
    }
    write_started = profiler.start() if profiler is not None else 0.0
    result = write_wild_inference_bundle(bundle, output)
    if profiler is not None:
        profiler.record("bundle_write", write_started)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native MHR CoCoNet inference on a one-camera in-the-wild export without ground truth.")
    parser.add_argument("export_seq")
    parser.add_argument("--depth-h5", required=True)
    parser.add_argument("--mhr-init", required=True)
    parser.add_argument("--foundationpose-file", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--render-batch-size", type=int, default=32)
    parser.add_argument("--crop-workers", type=int, default=8)
    parser.add_argument("--crop-buffer-count", type=int, default=2)
    parser.add_argument("--input-cache", default=None)
    parser.add_argument("--no-input-cache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-run-path", default=None, help="Explicit entity/project/run_id for legacy checkpoints.")
    parser.add_argument("--offline-supervision-contract", action="store_true", help="Use checkpoint-embedded W&B identity and supervision contract without querying W&B.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with PipelineTimer("coconet_wild_inference") as profiler:
        print(run_mhr_wild_inference(args.export_seq, args.depth_h5, args.mhr_init, args.foundationpose_file, args.config, args.checkpoint, args.output, stride=args.stride, render_batch_size=args.render_batch_size, crop_workers=args.crop_workers, crop_buffer_count=args.crop_buffer_count, input_cache=args.input_cache, use_input_cache=not args.no_input_cache, device_name=args.device, overwrite=args.overwrite, wandb_run_path=args.wandb_run_path, offline_supervision_contract=args.offline_supervision_contract, profiler=profiler))


if __name__ == "__main__":
    main()
