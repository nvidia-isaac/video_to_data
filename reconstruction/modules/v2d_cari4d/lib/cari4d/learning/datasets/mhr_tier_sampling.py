from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


MHR_TIER_SAMPLING_REVISION = "cari4d.mhr_tier_sampling.stateless.v1"
LEGACY_TIER_SAMPLING_MIGRATION_STEP_ENV = "MHR_LEGACY_TIER_SAMPLING_MIGRATION_STEP"
_HASH_PERSON = b"mhr-tier-v1"
_UINT64_RANGE = 1 << 64


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def configured_tier_sampling_revision(cfg: Any) -> str | None:
    if str(_cfg_get(cfg, "body_model", "smpl")) != "mhr":
        return None
    if not bool(_cfg_get(cfg, "mhr_foundationpose_tier_sampling", True)):
        return None
    return str(_cfg_get(cfg, "mhr_foundationpose_tier_sampling_revision", MHR_TIER_SAMPLING_REVISION))


def tier_sampling_contract(revision: str | None, seed: int | None) -> dict[str, Any] | None:
    if revision is None:
        return None
    if seed is None:
        raise ValueError("Stateless FoundationPose tier sampling requires a configured training seed")
    return {"revision": str(revision), "seed": int(seed)}


def _hash_value(parts: Sequence[str], counter: int) -> int:
    digest = hashlib.blake2b(digest_size=8, person=_HASH_PERSON)
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    digest.update(struct.pack("<I", int(counter)))
    return int.from_bytes(digest.digest(), "little", signed=False)


def stateless_available_tier(available_mask: np.ndarray, seed: int, epoch: int, sample_index: int, sequence_name: str, camera_id: int, frame_index: int) -> int:
    available = np.flatnonzero(np.asarray(available_mask, dtype=bool)) + 1
    if len(available) == 0:
        raise ValueError("FoundationPose tier selection requires at least one available tier")
    parts = (MHR_TIER_SAMPLING_REVISION, str(int(seed)), str(int(epoch)), str(int(sample_index)), str(sequence_name), str(int(camera_id)), str(int(frame_index)))
    acceptance_limit = (_UINT64_RANGE // len(available)) * len(available)
    counter = 0
    while True:
        value = _hash_value(parts, counter)
        if value < acceptance_limit:
            return int(available[value % len(available)])
        counter += 1


def stateless_clip_tiers(tier_valid: np.ndarray, seed: int, epoch: int, sample_index: int, sequence_name: str, camera_id: int, frame_indices: Sequence[int]) -> np.ndarray:
    tier_valid = np.asarray(tier_valid, dtype=bool)
    frame_indices = [int(index) for index in frame_indices]
    if tier_valid.ndim != 2 or tier_valid.shape[1] != 3:
        raise ValueError(f"FoundationPose tier-valid mask must have shape [T,3], got {tier_valid.shape}")
    if len(frame_indices) != len(tier_valid):
        raise ValueError(f"FoundationPose frame-index count {len(frame_indices)} does not match tier-valid length {len(tier_valid)}")
    return np.asarray([stateless_available_tier(mask, seed, epoch, sample_index, sequence_name, camera_id, frame_index) for mask, frame_index in zip(tier_valid, frame_indices)], dtype=np.int8)


def validate_resume_tier_sampling_contract(checkpoint_contract: Mapping[str, Any] | None, current_contract: Mapping[str, Any] | None, checkpoint_step: int) -> bool:
    if current_contract is None:
        if checkpoint_contract is not None:
            raise ValueError(f"Checkpoint uses data sampling contract {dict(checkpoint_contract)!r}, but the current run has no tier-sampling contract")
        return False
    current_contract = dict(current_contract)
    if current_contract.get("revision") != MHR_TIER_SAMPLING_REVISION:
        raise ValueError(f"Unsupported current MHR tier-sampling revision {current_contract.get('revision')!r}")
    if set(current_contract) != {"revision", "seed"}:
        raise ValueError(f"Current MHR tier-sampling contract has unsupported fields: {current_contract!r}")
    if checkpoint_contract is not None:
        checkpoint_contract = dict(checkpoint_contract)
        if checkpoint_contract != current_contract:
            raise ValueError(f"Checkpoint data sampling contract {checkpoint_contract!r} does not match current contract {current_contract!r}")
        return False
    approved_step = os.environ.get(LEGACY_TIER_SAMPLING_MIGRATION_STEP_ENV)
    if approved_step is None or int(approved_step) != int(checkpoint_step):
        raise ValueError(f"Checkpoint step {checkpoint_step} predates stateless tier sampling; set {LEGACY_TIER_SAMPLING_MIGRATION_STEP_ENV}={checkpoint_step} for the explicitly approved one-time migration")
    return True
