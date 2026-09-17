from __future__ import annotations

import atexit
import json
import multiprocessing as mp
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from accelerate.utils import send_to_device
from learning.datasets.mhr_input_materialization import MHR_ENCODED_DEPTHS_KEY, MHR_ENCODED_MASKS_KEY, MHR_ENCODED_RGBS_KEY, MHR_INPUT_MATERIALIZATION_GPU, MHR_XYZ_ANCHOR_KEY, materialize_mhr_gpu_inputs, resolve_mhr_input_materialization_mode, validate_mhr_gpu_materialization_config
from learning.datasets.mhr_loader_config import rank_local_slot_count
from learning.datasets.mhr_sample_metadata import MHR_SAMPLE_METADATA_KEY
from learning.training.rank_timing_log import TimingWindow


MHR_RENDER_REQUEST_KEY = "_mhr_render_request"
MHR_LARGE_RENDER_KEYS = ("input_rgbs", "render_rgbs", "input_xyz", "render_xyz")
MHR_RENDER_SOURCE_KEYS = ("obj_rot_gt", "obj_t_gt", "obj_rot", "obj_t", "obj_symmetry_tfs", "obj_symmetry_mode", "obj_symmetry_center", "obj_rot_init", "obj_t_init", "obj_init_tier_ids", "mhr_contact_dist_gt", "mhr_contact_closest_points_gt", "obj_pose_storage_to_training_transform")
MHR_RENDER_RETAINED_SOURCE_KEYS = ("obj_symmetry_mode", "obj_symmetry_center")
MHR_RENDER_DIAGNOSTIC_KEYS = ("rot_normalizer", "Krois")
_PROCESS_DATASET = None
_PROCESS_BUFFERS = None
_PROCESS_THREAD_LIMITER = None
_PROCESS_PROFILE_REMAINING = 0


@dataclass(frozen=True)
class MHRRenderRequest:
    sequence_index: int
    start: int
    indices: tuple[int, ...]
    kid: int
    sequence_name: str | None = None
    frame_names: tuple[str, ...] | None = None
    temporal_stride: int = 1


@dataclass(frozen=True)
class _RenderWorkerResult:
    render_fields: dict[str, Any]
    timing: dict[str, Any]


def _close_process_dataset() -> None:
    global _PROCESS_DATASET
    if _PROCESS_DATASET is not None:
        _PROCESS_DATASET.close()
        _PROCESS_DATASET = None


def _initialize_process_worker(dataset: Any, buffers: list[dict[str, torch.Tensor]], torch_threads: int, profile_samples: int) -> None:
    import cv2
    from threadpoolctl import threadpool_limits

    global _PROCESS_DATASET, _PROCESS_BUFFERS, _PROCESS_THREAD_LIMITER, _PROCESS_PROFILE_REMAINING
    cv2.setNumThreads(1)
    _PROCESS_THREAD_LIMITER = threadpool_limits(limits=1, user_api="blas")
    torch.set_num_threads(int(torch_threads))
    torch.set_num_interop_threads(1)
    _PROCESS_DATASET = dataset
    _PROCESS_BUFFERS = buffers
    _PROCESS_PROFILE_REMAINING = int(profile_samples)
    atexit.register(_close_process_dataset)


def _process_render_sample(slot_index: int, sample_index: int, request: MHRRenderRequest, object_data: Mapping[str, np.ndarray]) -> _RenderWorkerResult:
    global _PROCESS_PROFILE_REMAINING
    if _PROCESS_DATASET is None or _PROCESS_BUFFERS is None:
        raise RuntimeError("Rank-local process worker was not initialized")
    destinations = {key: value[sample_index] for key, value in _PROCESS_BUFFERS[slot_index].items()}
    started_at = time.monotonic()
    if _PROCESS_PROFILE_REMAINING > 0 and hasattr(_PROCESS_DATASET, "load_render_fields_into_profiled"):
        _PROCESS_PROFILE_REMAINING -= 1
        render_fields, timing = _PROCESS_DATASET.load_render_fields_into_profiled(request, object_data, destinations)
    else:
        render_fields = _PROCESS_DATASET.load_render_fields_into(request, object_data, destinations)
        timing = {}
    timing = dict(timing)
    timing["worker_total_seconds"] = time.monotonic() - started_at
    return _RenderWorkerResult(render_fields=render_fields, timing=timing)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _xyz_channel_count(cfg: Any) -> int:
    if not bool(_cfg_get(cfg, "add_ho_mask", False)):
        return 3
    mask_type = str(_cfg_get(cfg, "mask_encode_type", "stack"))
    channels = {"stack": 2, "stack-occ": 2, "hum-obj-fullobj": 3, "obj-fullobj": 2, "one-channel": 1}
    if mask_type not in channels:
        raise ValueError(f"Unknown mask_encode_type for rank-local MHR buffers: {mask_type}")
    return 3 + channels[mask_type]


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise ValueError(f"Rank-local preprocessing requires CPU labels, got {value.device}")
        return value.detach().numpy()
    return np.asarray(value)


class _RenderContextPool:
    def __init__(self, dataset: Any):
        self.dataset = dataset
        self.local = threading.local()
        self.contexts = []
        self.lock = threading.Lock()

    def get(self) -> Any:
        context = getattr(self.local, "context", None)
        if context is None:
            context = self.dataset.clone_for_rank_local_render()
            self.local.context = context
            with self.lock:
                self.contexts.append(context)
        return context

    def close(self) -> None:
        for context in self.contexts:
            context.close()
        self.contexts.clear()


class _RenderBufferSlot:
    def __init__(self, cfg: Any, max_batch_size: int, pin_memory: bool, backend: str):
        input_resize = tuple(int(item) for item in _cfg_get(cfg, "input_resize"))
        if len(input_resize) != 2 or min(input_resize) <= 0:
            raise ValueError(f"input_resize must contain two positive values, got {input_resize}")
        clip_len = int(_cfg_get(cfg, "clip_len"))
        if clip_len <= 0 or max_batch_size <= 0:
            raise ValueError(f"clip_len and max_batch_size must be positive, got {clip_len} and {max_batch_size}")
        width, height = input_resize
        materialization_mode = resolve_mhr_input_materialization_mode(cfg)
        if materialization_mode == MHR_INPUT_MATERIALIZATION_GPU:
            validate_mhr_gpu_materialization_config(cfg)
            specs = {
                MHR_ENCODED_RGBS_KEY: ((max_batch_size, clip_len, 6, height, width), torch.uint8),
                MHR_ENCODED_DEPTHS_KEY: ((max_batch_size, clip_len, 2, height, width), torch.float16),
                MHR_ENCODED_MASKS_KEY: ((max_batch_size, clip_len, 5, height, width), torch.uint8),
            }
        else:
            xyz_channels = _xyz_channel_count(cfg)
            specs = {
                "input_rgbs": ((max_batch_size, clip_len, 3, height, width), torch.float32),
                "render_rgbs": ((max_batch_size, clip_len, 3, height, width), torch.float32),
                "input_xyz": ((max_batch_size, clip_len, xyz_channels, height, width), torch.float32),
                "render_xyz": ((max_batch_size, clip_len, xyz_channels, height, width), torch.float32),
            }
        if backend == "process":
            self.worker_buffers = {key: torch.empty(shape, dtype=dtype).share_memory_() for key, (shape, dtype) in specs.items()}
            self.buffers = {key: torch.empty(shape, dtype=dtype, pin_memory=True) for key, (shape, dtype) in specs.items()} if pin_memory else self.worker_buffers
        else:
            self.buffers = {key: torch.empty(shape, dtype=dtype, pin_memory=pin_memory) for key, (shape, dtype) in specs.items()}
            self.worker_buffers = self.buffers
        self.transfer_event = None

    def wait_for_reuse(self) -> None:
        if self.transfer_event is not None:
            self.transfer_event.synchronize()
            self.transfer_event = None

    def sample_destinations(self, sample_index: int) -> dict[str, torch.Tensor]:
        return {key: value[sample_index] for key, value in self.worker_buffers.items()}

    def prepare_transfer(self, batch_size: int) -> None:
        if self.buffers is self.worker_buffers:
            return
        for key in self.worker_buffers:
            self.buffers[key][:batch_size].copy_(self.worker_buffers[key][:batch_size])

    def batch_views(self, batch_size: int) -> dict[str, torch.Tensor]:
        return {key: value[:batch_size] for key, value in self.buffers.items()}

    def record_transfer(self, device: torch.device) -> None:
        if device.type != "cuda":
            return
        self.transfer_event = torch.cuda.Event()
        self.transfer_event.record(torch.cuda.current_stream(device=device))


@dataclass
class _PendingBatch:
    compact_batch: dict[str, Any]
    slot: _RenderBufferSlot
    futures: list[Future]
    batch_size: int
    sample_metadata: list[dict[str, Any]]
    submitted_at: float
    cpu_future: Future | None = None
    profile_records: list[dict[str, Any]] = field(default_factory=list)
    worker_wait_seconds: float = 0.0
    finalize_seconds: float = 0.0


class _MHRRankLocalIterator:
    def __init__(self, loader: "MHRRankLocalDataLoader", base_dataloader: Any | None = None):
        self.loader = loader
        self.base_iterator = iter(loader.base_dataloader if base_dataloader is None else base_dataloader)
        self.total_batches = len(loader.base_dataloader if base_dataloader is None else base_dataloader)
        self.generation = loader.next_iterator_generation()
        self.completed_batches = 0
        self.next_slot = 0
        self.pending = deque()
        item = self._submit_next()
        if item is not None:
            self.pending.append(item)
        self.loader.emit_preprocess_progress(self.generation, self.completed_batches, self.total_batches, force=True)

    def _submit_next(self) -> _PendingBatch | None:
        try:
            compact_batch = next(self.base_iterator)
        except StopIteration:
            return None
        pending = self.loader._submit(compact_batch, self.next_slot)
        self.next_slot = (self.next_slot + 1) % len(self.loader.slots)
        return pending

    def __iter__(self):
        return self

    def __next__(self):
        if not self.pending:
            raise StopIteration
        current = self.pending.popleft()
        batch = self.loader._resolve(current)
        self.completed_batches += 1
        self.loader.emit_preprocess_progress(self.generation, self.completed_batches, self.total_batches, force=self.completed_batches == 1)
        target_pending = len(self.loader.slots) - 1
        while len(self.pending) < target_pending:
            replacement = self._submit_next()
            if replacement is None:
                break
            self.pending.append(replacement)
        return batch


class MHRRankLocalDataLoader:
    def __init__(self, base_dataloader: Any, dataset: Any, cfg: Any, max_batch_size: int, worker_count: int, device: torch.device, pin_memory: bool = True):
        if int(worker_count) < 1:
            raise ValueError(f"Rank-local preprocessing requires at least one worker, got {worker_count}")
        self.base_dataloader = base_dataloader
        self.dataset = dataset
        self.cfg = cfg
        self.supports_sampler_level_resume_skip = bool(getattr(dataset, "supports_sampler_level_resume_skip", False))
        self.max_batch_size = int(max_batch_size)
        self.device = torch.device(device)
        self.pin_memory = bool(pin_memory)
        self.materialization_mode = resolve_mhr_input_materialization_mode(cfg)
        self.backend = str(_cfg_get(cfg, "mhr_rank_local_preprocess_backend", "thread"))
        if self.backend not in {"thread", "process"}:
            raise ValueError(f"mhr_rank_local_preprocess_backend must be thread or process, got {self.backend!r}")
        if self.pin_memory and not torch.cuda.is_available():
            raise RuntimeError("Pinned rank-local buffers require CUDA")
        self.context_pool = _RenderContextPool(dataset) if self.backend == "thread" else None
        slot_count = rank_local_slot_count(cfg, worker_count)
        self.slots = [_RenderBufferSlot(cfg, self.max_batch_size, self.pin_memory, self.backend) for _ in range(slot_count)]
        profile_samples = int(_cfg_get(cfg, "mhr_rank_local_profile_samples_per_worker", 0) or 0)
        if profile_samples < 0:
            raise ValueError(f"mhr_rank_local_profile_samples_per_worker must be nonnegative, got {profile_samples}")
        if self.backend == "thread":
            self.executor = ThreadPoolExecutor(max_workers=int(worker_count), thread_name_prefix="mhr-render")
        else:
            torch_threads = int(_cfg_get(cfg, "mhr_rank_local_worker_torch_threads", 4))
            if torch_threads < 1:
                raise ValueError(f"mhr_rank_local_worker_torch_threads must be positive, got {torch_threads}")
            shared_buffers = [slot.worker_buffers for slot in self.slots]
            render_dataset = dataset.clone_for_rank_local_render()
            self.executor = ProcessPoolExecutor(max_workers=int(worker_count), mp_context=mp.get_context("spawn"), initializer=_initialize_process_worker, initargs=(render_dataset, shared_buffers, torch_threads, profile_samples))
        self.finalizer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mhr-finalize")
        self.iterator_generation = 0
        self.last_progress_emit_at = 0.0
        self.process_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.preprocess_phase = f"{str(getattr(dataset, 'split', 'data'))}_preprocessing"
        self.timing_window = TimingWindow()
        self.closed = False

    def next_iterator_generation(self) -> int:
        generation = self.iterator_generation
        self.iterator_generation += 1
        return generation

    def emit_preprocess_progress(self, generation: int, current: int, total: int, force: bool = False) -> None:
        if self.process_rank != 0:
            return
        now = time.monotonic()
        if not force and now - self.last_progress_emit_at < 30.0:
            return
        payload = {"phase": self.preprocess_phase, "generation": int(generation), "current": int(current), "total": int(total), "unit": "batches", "observedAtUnix": time.time()}
        print("MHR_PREPROCESS_PROGRESS " + json.dumps(payload, sort_keys=True), flush=True)
        self.last_progress_emit_at = now

    def __len__(self) -> int:
        return len(self.base_dataloader)

    def __iter__(self):
        if self.closed:
            raise RuntimeError("MHRRankLocalDataLoader is closed")
        return _MHRRankLocalIterator(self)

    def iter_from_base_dataloader(self, base_dataloader: Any):
        if self.closed:
            raise RuntimeError("MHRRankLocalDataLoader is closed")
        return _MHRRankLocalIterator(self, base_dataloader)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.base_dataloader, name)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        if hasattr(self.base_dataloader, "set_epoch"):
            self.base_dataloader.set_epoch(epoch)

    @staticmethod
    def _object_data(compact_batch: Mapping[str, Any], sample_index: int) -> dict[str, np.ndarray]:
        keys = ("obj_rot_gt", "obj_t_gt", "obj_rot", "obj_t", "obj_symmetry_tfs", "obj_symmetry_mode", "obj_symmetry_center", "obj_init_tier_ids", MHR_XYZ_ANCHOR_KEY, "obj_pose_storage_to_training_transform")
        return {key: _to_numpy(compact_batch[key][sample_index]) for key in keys if key in compact_batch}

    def _load_sample(self, request: MHRRenderRequest, compact_batch: Mapping[str, Any], sample_index: int, destinations: Mapping[str, torch.Tensor]) -> _RenderWorkerResult:
        if self.context_pool is None:
            raise RuntimeError("Thread render context is unavailable for process backend")
        context = self.context_pool.get()
        started_at = time.monotonic()
        render_fields = context.load_render_fields_into(request, self._object_data(compact_batch, sample_index), destinations)
        return _RenderWorkerResult(render_fields=render_fields, timing={"worker_total_seconds": time.monotonic() - started_at})

    def _submit(self, compact_batch: Mapping[str, Any], slot_index: int) -> _PendingBatch:
        if not isinstance(compact_batch, Mapping):
            raise TypeError(f"Rank-local preprocessing expects a mapping batch, got {type(compact_batch).__name__}")
        compact_batch = dict(compact_batch)
        if MHR_RENDER_REQUEST_KEY not in compact_batch:
            raise KeyError(f"Rank-local MHR batch is missing {MHR_RENDER_REQUEST_KEY}")
        requests = compact_batch.pop(MHR_RENDER_REQUEST_KEY)
        if not isinstance(requests, list) or not all(isinstance(request, MHRRenderRequest) for request in requests):
            raise TypeError(f"{MHR_RENDER_REQUEST_KEY} must be an ordered list of MHRRenderRequest values")
        batch_size = len(requests)
        if batch_size < 1 or batch_size > self.max_batch_size:
            raise ValueError(f"Rank-local batch size must be in [1, {self.max_batch_size}], got {batch_size}")
        sample_metadata = [{"sequence_name": request.sequence_name, "sequence_index": request.sequence_index, "window_start": request.start, "temporal_stride": request.temporal_stride, "camera_id": request.kid, "frame_indices": list(request.indices), "frame_names": None if request.frame_names is None else list(request.frame_names)} for request in requests]
        slot = self.slots[slot_index]
        if self.backend == "thread":
            slot.wait_for_reuse()
            futures = [self.executor.submit(self._load_sample, request, compact_batch, index, slot.sample_destinations(index)) for index, request in enumerate(requests)]
        else:
            futures = [self.executor.submit(_process_render_sample, slot_index, index, request, self._object_data(compact_batch, index)) for index, request in enumerate(requests)]
        pending = _PendingBatch(compact_batch=compact_batch, slot=slot, futures=futures, batch_size=batch_size, sample_metadata=sample_metadata, submitted_at=time.monotonic())
        pending.cpu_future = self.finalizer.submit(self._finalize_cpu, pending)
        return pending

    def _finalize_cpu(self, pending: _PendingBatch) -> dict[str, Any]:
        from learning.datasets import mhr_collate

        worker_wait_started_at = time.monotonic()
        worker_results = [future.result() for future in pending.futures]
        pending.worker_wait_seconds = time.monotonic() - worker_wait_started_at
        pending.profile_records = [result.timing for result in worker_results]
        finalize_started_at = time.monotonic()
        render_samples = [result.render_fields for result in worker_results]
        render_batch = mhr_collate(render_samples)
        if "frame_mask" not in pending.compact_batch or "frame_mask" not in render_batch:
            raise KeyError("Rank-local MHR preprocessing requires compact and rendered frame_mask values")
        render_frame_mask = render_batch.pop("frame_mask")
        for key in ("human_frame_mask", "object_frame_mask", "frame_mask"):
            if key not in pending.compact_batch:
                raise KeyError(f"Rank-local MHR preprocessing requires compact {key}")
            pending.compact_batch[key] = pending.compact_batch[key] * render_frame_mask
        for key in MHR_RENDER_RETAINED_SOURCE_KEYS:
            if key in pending.compact_batch and key in render_batch:
                compact_value = _to_numpy(pending.compact_batch[key])
                render_value = _to_numpy(render_batch[key])
                if compact_value.shape != render_value.shape or not np.array_equal(compact_value, render_value):
                    raise ValueError(f"Rank-local rendered {key} differs from the authoritative compact batch")
                render_batch.pop(key)
        collisions = sorted(set(pending.compact_batch) & set(render_batch))
        if collisions:
            raise KeyError(f"Rank-local rendered metadata collides with compact batch keys: {collisions}")
        pending.compact_batch.update(render_batch)
        for key in MHR_RENDER_SOURCE_KEYS:
            if key not in MHR_RENDER_RETAINED_SOURCE_KEYS:
                pending.compact_batch.pop(key, None)
        if bool(_cfg_get(self.cfg, "mhr_minimal_batch_fields", False)):
            for key in MHR_RENDER_DIAGNOSTIC_KEYS:
                pending.compact_batch.pop(key, None)
            if int(_cfg_get(self.cfg, "fp_err_dim", -1)) <= 0:
                pending.compact_batch.pop("fp_error", None)
            if int(_cfg_get(self.cfg, "visibility_dim", -1)) <= 0:
                pending.compact_batch.pop("visibility", None)
        if self.backend == "process":
            pending.slot.wait_for_reuse()
        pending.slot.prepare_transfer(pending.batch_size)
        pending.compact_batch.update(pending.slot.batch_views(pending.batch_size))
        pending.finalize_seconds = time.monotonic() - finalize_started_at
        return pending.compact_batch

    def _resolve(self, pending: _PendingBatch) -> dict[str, Any]:
        if pending.cpu_future is None:
            raise RuntimeError("Rank-local batch has no CPU finalization future")
        resolve_started_at = time.monotonic()
        cpu_batch = pending.cpu_future.result()
        cpu_ready_at = time.monotonic()
        batch = send_to_device(cpu_batch, self.device, non_blocking=self.pin_memory and self.device.type == "cuda")
        if self.materialization_mode == MHR_INPUT_MATERIALIZATION_GPU:
            if self.device.type != "cuda":
                raise RuntimeError("gpu_compact_v1 requires a CUDA rank-local target device")
            batch = materialize_mhr_gpu_inputs(batch, self.cfg)
        transfer_enqueued_at = time.monotonic()
        batch[MHR_SAMPLE_METADATA_KEY] = pending.sample_metadata
        pending.slot.record_transfer(self.device)
        self.timing_window.add("batch_cpu_total", cpu_ready_at - pending.submitted_at)
        self.timing_window.add("resolve_wait", cpu_ready_at - resolve_started_at)
        self.timing_window.add("host_to_device_enqueue", transfer_enqueued_at - cpu_ready_at)
        self.timing_window.add("worker_wait", pending.worker_wait_seconds)
        self.timing_window.add("batch_finalize", pending.finalize_seconds)
        for record in pending.profile_records:
            for key, value in record.items():
                if key.endswith("_seconds"):
                    self.timing_window.add(key, float(value))
        return batch

    def consume_timing_summary(self) -> dict[str, dict[str, float | int]]:
        return self.timing_window.summary(reset=True)

    def close(self) -> None:
        if self.closed:
            return
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.finalizer.shutdown(wait=True, cancel_futures=False)
        for slot in self.slots:
            slot.wait_for_reuse()
        if self.context_pool is not None:
            self.context_pool.close()
        self.closed = True
