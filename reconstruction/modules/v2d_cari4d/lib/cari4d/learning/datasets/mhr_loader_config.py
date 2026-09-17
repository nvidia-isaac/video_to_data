from collections.abc import Mapping
from typing import Any


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def rank_local_slot_count(cfg: Any, worker_count: int) -> int:
    worker_count = int(worker_count)
    if worker_count < 1:
        raise ValueError(f"rank-local worker_count must be positive, got {worker_count}")
    configured_slots = _cfg_get(cfg, "mhr_rank_local_preprocess_slots")
    if configured_slots is not None:
        slot_count = int(configured_slots)
        if slot_count < 2:
            raise ValueError(f"rank-local preprocessing requires at least two reusable slots, got {slot_count}")
        return slot_count
    prefetch_factor = _cfg_get(cfg, "mhr_rank_local_preprocess_prefetch_factor")
    if prefetch_factor is None:
        slot_count = 2
    else:
        prefetch_factor = int(prefetch_factor)
        if prefetch_factor < 1:
            raise ValueError(f"rank-local prefetch factor must be positive, got {prefetch_factor}")
        slot_count = worker_count * prefetch_factor
    if slot_count < 2:
        raise ValueError(f"rank-local preprocessing requires at least two reusable slots, got {slot_count}")
    return slot_count
