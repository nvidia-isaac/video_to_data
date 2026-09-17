from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from lib_mhr.schema import MHR_PARAM_DIMS


MHR_GEOMETRY_AUTHORITY_PARAMETER = "certified_parameters"
MHR_GEOMETRY_AUTHORITY_CACHED = "cached_geometry"
MHR_GEOMETRY_AUTHORITY_REVISION = "cari4d.mhr_geometry_authority.v1"
MHR_GEOMETRY_RECORD_GT = "gt"
MHR_GEOMETRY_RECORD_INIT = "init"
MHR_DECODER_IDENTITY_KEYS = ("mhr_model_sha256", "mhr_buffer_sha256", "mhr_decoder_revision")
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class MHRGeometryAuthority:
    mode: str
    decoder_identity: Mapping[str, str] | None = None

    @classmethod
    def certified_parameters(cls, decoder_identity: Mapping[str, str]) -> "MHRGeometryAuthority":
        return cls(mode=MHR_GEOMETRY_AUTHORITY_PARAMETER, decoder_identity=decoder_identity)

    @classmethod
    def runtime_parameters(cls) -> "MHRGeometryAuthority":
        return cls(mode=MHR_GEOMETRY_AUTHORITY_PARAMETER)


def geometry_authority_from_metadata(metadata: Mapping[str, Any]) -> MHRGeometryAuthority:
    mode = str(metadata.get("mhr_geometry_authority", MHR_GEOMETRY_AUTHORITY_CACHED))
    if mode == MHR_GEOMETRY_AUTHORITY_CACHED:
        return MHRGeometryAuthority.runtime_parameters()
    if mode != MHR_GEOMETRY_AUTHORITY_PARAMETER:
        raise ValueError(f"Unsupported MHR geometry authority {mode!r}")
    if metadata.get("mhr_geometry_authority_revision") != MHR_GEOMETRY_AUTHORITY_REVISION:
        raise ValueError(f"Parameter-authoritative MHR geometry requires revision {MHR_GEOMETRY_AUTHORITY_REVISION!r}")
    return MHRGeometryAuthority.certified_parameters({key: metadata[key] for key in MHR_DECODER_IDENTITY_KEYS})


def _shape(value: Any, label: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"{label} must expose a shape")
    return tuple(int(dim) for dim in shape)


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor"


def _require_floating_finite(value: Any, label: str) -> None:
    if _is_torch_tensor(value):
        import torch

        if not value.dtype.is_floating_point:
            raise TypeError(f"{label} must have floating dtype, got {value.dtype}")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{label} contains non-finite values")
        return
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{label} must have floating dtype, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")


def _normalize_decoder_identity(identity: Mapping[str, str] | None, label: str) -> dict[str, str]:
    if not isinstance(identity, Mapping):
        raise TypeError(f"{label} must be a mapping")
    missing = [key for key in MHR_DECODER_IDENTITY_KEYS if key not in identity]
    if missing:
        raise KeyError(f"{label} is missing required fields {missing}")
    normalized = {str(key): str(value) for key, value in identity.items()}
    for key in MHR_DECODER_IDENTITY_KEYS[:2]:
        if _SHA256_PATTERN.fullmatch(normalized[key]) is None:
            raise ValueError(f"{label}/{key} must be a 64-character SHA-256 digest")
        normalized[key] = normalized[key].lower()
    if not normalized["mhr_decoder_revision"].strip():
        raise ValueError(f"{label}/mhr_decoder_revision must be non-empty")
    return normalized


def normalize_mhr_decoder_identity(identity: Mapping[str, str] | None, label: str = "MHR decoder identity") -> dict[str, str]:
    return _normalize_decoder_identity(identity, label)


def _validate_record_kind(record_kind: str) -> int:
    if record_kind == MHR_GEOMETRY_RECORD_GT:
        return 1
    if record_kind == MHR_GEOMETRY_RECORD_INIT:
        return 2
    raise ValueError(f"Unsupported MHR geometry record kind {record_kind!r}")


def _concatenate(values: list[Any], label: str) -> Any:
    first_is_torch = _is_torch_tensor(values[0])
    if any(_is_torch_tensor(value) != first_is_torch for value in values[1:]):
        raise TypeError(f"{label} decoder batches returned mixed array types")
    if first_is_torch:
        import torch

        return torch.cat(values, dim=0)
    return np.concatenate(values, axis=0)


class MHRGeometryProvider:
    """Decode MHR vertices from authoritative parameters.

    Parameter-authoritative records are decoded in bounded flat batches after
    their decoder identity matches this provider. Legacy records without a
    decoder certificate are decoded with the configured runtime decoder and
    never read cached vertices.
    """

    def __init__(self, mhr_layer: Any | None = None, *, decoder_identity: Mapping[str, str] | None = None, max_batch_size: int = 256):
        if (mhr_layer is None) != (decoder_identity is None):
            raise ValueError("mhr_layer and decoder_identity must be provided together")
        if mhr_layer is not None and not callable(getattr(mhr_layer, "mhr_forward_vertices", None)):
            raise TypeError("mhr_layer must expose callable mhr_forward_vertices(params)")
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be a positive integer, got {max_batch_size!r}")
        self.mhr_layer = mhr_layer
        self.decoder_identity = _normalize_decoder_identity(decoder_identity, "provider decoder identity") if decoder_identity is not None else None
        self.max_batch_size = max_batch_size

    def vertices(self, record: Mapping[str, Any], *, record_kind: str, authority: MHRGeometryAuthority) -> Any:
        if not isinstance(record, Mapping):
            raise TypeError(f"MHR {record_kind} record must be a mapping")
        leading_rank = _validate_record_kind(record_kind)
        if not isinstance(authority, MHRGeometryAuthority):
            raise TypeError("authority must be an MHRGeometryAuthority")
        if authority.mode == MHR_GEOMETRY_AUTHORITY_PARAMETER:
            if self.mhr_layer is None or self.decoder_identity is None:
                raise RuntimeError("Parameter-authoritative geometry requires a configured MHR decoder")
            if authority.decoder_identity is not None:
                record_identity = _normalize_decoder_identity(authority.decoder_identity, f"{record_kind} decoder identity")
                mismatches = {key: (self.decoder_identity.get(key), value) for key, value in record_identity.items() if self.decoder_identity.get(key) != value}
                missing_provider_keys = sorted(set(self.decoder_identity) - set(record_identity))
                if mismatches or missing_provider_keys:
                    raise ValueError(f"{record_kind} decoder identity does not match provider: mismatches={mismatches}, missing={missing_provider_keys}")
            return self._decoded_vertices(record, record_kind, leading_rank)
        raise ValueError(f"Unsupported MHR geometry authority {authority.mode!r}")

    def _decoded_vertices(self, record: Mapping[str, Any], record_kind: str, leading_rank: int) -> Any:
        params: dict[str, Any] = {}
        leading_shape = None
        for key, dim in MHR_PARAM_DIMS.items():
            if key not in record:
                raise KeyError(f"Parameter-authoritative {record_kind} record is missing {key}")
            value = record[key]
            shape = _shape(value, f"{record_kind}/{key}")
            if len(shape) != leading_rank + 1 or shape[-1] != dim:
                expected = f"[T,{dim}]" if record_kind == MHR_GEOMETRY_RECORD_GT else f"[T,K,{dim}]"
                raise ValueError(f"{record_kind}/{key} must have shape {expected}, got {shape}")
            if leading_shape is None:
                leading_shape = shape[:-1]
                if any(size <= 0 for size in leading_shape):
                    raise ValueError(f"{record_kind} parameters have empty leading dimensions {leading_shape}")
            elif shape[:-1] != leading_shape:
                raise ValueError(f"{record_kind}/{key} leading shape {shape[:-1]} does not match {leading_shape}")
            _require_floating_finite(value, f"{record_kind}/{key}")
            params[key] = value
        assert leading_shape is not None
        flat_count = math.prod(leading_shape)
        flat_params = {key: params[key].reshape(flat_count, dim) for key, dim in MHR_PARAM_DIMS.items()}
        decoded_batches = []
        point_count = None
        for start in range(0, flat_count, self.max_batch_size):
            stop = min(start + self.max_batch_size, flat_count)
            batch = {key: value[start:stop] for key, value in flat_params.items()}
            decoded = self.mhr_layer.mhr_forward_vertices(batch)
            shape = _shape(decoded, f"decoded {record_kind} vertices")
            if len(shape) != 3 or shape[0] != stop - start or shape[-1] != 3 or shape[-2] <= 0:
                raise ValueError(f"Decoded {record_kind} vertices must have shape [{stop - start},V,3], got {shape}")
            if point_count is None:
                point_count = shape[-2]
            elif shape[-2] != point_count:
                raise ValueError(f"Decoded {record_kind} vertex count changed between batches: {point_count} then {shape[-2]}")
            _require_floating_finite(decoded, f"decoded {record_kind} vertices")
            decoded_batches.append(decoded)
        vertices = _concatenate(decoded_batches, f"decoded {record_kind} vertices")
        return vertices.reshape(*leading_shape, point_count, 3)


def decode_mhr_vertices_numpy(mhr_layer: Any, params: Mapping[str, Any], *, batch_size: int = 256) -> np.ndarray:
    first = np.asarray(params.get("mhr_trans"))
    if first.ndim == 2:
        record_kind = MHR_GEOMETRY_RECORD_GT
    elif first.ndim == 3:
        record_kind = MHR_GEOMETRY_RECORD_INIT
    else:
        raise ValueError(f"MHR parameters must have [T,D] or [T,K,D] shape, got mhr_trans={first.shape}")
    identity = mhr_layer.decoder_identity()
    provider = MHRGeometryProvider(mhr_layer, decoder_identity=identity, max_batch_size=batch_size)
    value = provider.vertices(params, record_kind=record_kind, authority=MHRGeometryAuthority.runtime_parameters())
    if _is_torch_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)
