"""Load one shared training checkpoint on rank 0 and distribute it exactly."""

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class _TensorSlot:
    index: int
    shape: tuple
    dtype: torch.dtype


def _split_tensors(value, tensors):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError(f"Distributed checkpoint loading requires dense strided tensors, got layout={value.layout}")
        slot = _TensorSlot(index=len(tensors), shape=tuple(value.shape), dtype=value.dtype)
        tensors.append(value)
        return slot
    if isinstance(value, OrderedDict):
        return OrderedDict((key, _split_tensors(item, tensors)) for key, item in value.items())
    if type(value) is dict:
        return {key: _split_tensors(item, tensors) for key, item in value.items()}
    if type(value) is list:
        return [_split_tensors(item, tensors) for item in value]
    if type(value) is tuple:
        return tuple(_split_tensors(item, tensors) for item in value)
    return value


def _restore_tensors(value, tensors):
    if isinstance(value, _TensorSlot):
        tensor = tensors[value.index]
        if tuple(tensor.shape) != value.shape or tensor.dtype != value.dtype:
            raise ValueError(f"Checkpoint tensor slot {value.index} does not match metadata: expected {value.shape}/{value.dtype}, got {tuple(tensor.shape)}/{tensor.dtype}")
        return tensor
    if isinstance(value, OrderedDict):
        return OrderedDict((key, _restore_tensors(item, tensors)) for key, item in value.items())
    if type(value) is dict:
        return {key: _restore_tensors(item, tensors) for key, item in value.items()}
    if type(value) is list:
        return [_restore_tensors(item, tensors) for item in value]
    if type(value) is tuple:
        return tuple(_restore_tensors(item, tensors) for item in value)
    return value


def _backend_name(distributed):
    return str(distributed.get_backend()).lower().split(".")[-1]


def _broadcast_object(value, rank, device, distributed):
    values = [value if rank == 0 else None]
    kwargs = {"src": 0}
    if _backend_name(distributed) == "nccl":
        kwargs["device"] = device
    distributed.broadcast_object_list(values, **kwargs)
    return values[0]


def _broadcast_tensor(source, slot, rank, device, distributed, max_chunk_bytes):
    target = source if rank == 0 else torch.empty(slot.shape, dtype=slot.dtype, device="cpu")
    if target.numel() == 0:
        return target
    source_flat = target.contiguous().view(-1) if rank == 0 else None
    target_flat = target.view(-1) if rank != 0 else None
    elements_per_chunk = max(1, int(max_chunk_bytes) // target.element_size())
    use_cuda = _backend_name(distributed) == "nccl"
    for start in range(0, target.numel(), elements_per_chunk):
        stop = min(start + elements_per_chunk, target.numel())
        if use_cuda:
            chunk = source_flat[start:stop].to(device=device, non_blocking=False) if rank == 0 else torch.empty(stop - start, dtype=slot.dtype, device=device)
            distributed.broadcast(chunk, src=0)
            if rank != 0:
                target_flat[start:stop].copy_(chunk.cpu())
        else:
            chunk = source_flat[start:stop] if rank == 0 else target_flat[start:stop]
            distributed.broadcast(chunk, src=0)
    return target


def _checkpoint_stats(tensors):
    return {"tensor_count": len(tensors), "tensor_bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors)}


def load_checkpoint_on_rank_zero(path, accelerator, max_chunk_bytes=256 * 1024 * 1024, load_fn=torch.load, distributed=dist):
    """Read ``path`` once, then reconstruct the same CPU checkpoint on every rank."""

    max_chunk_bytes = int(max_chunk_bytes)
    if max_chunk_bytes <= 0:
        raise ValueError(f"max_chunk_bytes must be positive, got {max_chunk_bytes}")
    world_size = int(accelerator.num_processes)
    rank = int(accelerator.process_index)
    if world_size == 1:
        checkpoint = load_fn(path, map_location="cpu", weights_only=False)
        tensors = []
        _split_tensors(checkpoint, tensors)
        return checkpoint, {**_checkpoint_stats(tensors), "source_rank": 0, "world_size": 1}
    if not distributed.is_available() or not distributed.is_initialized():
        raise RuntimeError(f"Distributed checkpoint loading requires an initialized process group for world_size={world_size}")
    checkpoint = None
    source_tensors = []
    status = None
    if rank == 0:
        try:
            checkpoint = load_fn(path, map_location="cpu", weights_only=False)
            skeleton = _split_tensors(checkpoint, source_tensors)
            status = {"ok": True, "skeleton": skeleton, **_checkpoint_stats(source_tensors)}
        except Exception as exc:
            status = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    status = _broadcast_object(status, rank, accelerator.device, distributed)
    if not status["ok"]:
        raise RuntimeError(f"Rank-0 checkpoint load failed for {path}: {status['error_type']}: {status['error']}")
    tensor_slots = []

    def collect_slots(value):
        if isinstance(value, _TensorSlot):
            tensor_slots.append(value)
        elif isinstance(value, (dict, OrderedDict)):
            for item in value.values():
                collect_slots(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect_slots(item)

    collect_slots(status["skeleton"])
    if len(tensor_slots) != int(status["tensor_count"]):
        raise RuntimeError(f"Checkpoint tensor metadata count mismatch: expected {status['tensor_count']}, found {len(tensor_slots)}")
    if [slot.index for slot in tensor_slots] != list(range(len(tensor_slots))):
        raise RuntimeError("Checkpoint tensor metadata indices are not contiguous and ordered")
    received_tensors = []
    for slot in tensor_slots:
        source = source_tensors[slot.index] if rank == 0 else None
        received_tensors.append(_broadcast_tensor(source, slot, rank, accelerator.device, distributed, max_chunk_bytes))
    if rank != 0:
        checkpoint = _restore_tensors(status["skeleton"], received_tensors)
    return checkpoint, {"tensor_count": int(status["tensor_count"]), "tensor_bytes": int(status["tensor_bytes"]), "source_rank": 0, "world_size": world_size}
