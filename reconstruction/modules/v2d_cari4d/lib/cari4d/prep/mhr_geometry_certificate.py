from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from lib_mhr.geometry_provider import MHR_DECODER_IDENTITY_KEYS, MHR_GEOMETRY_AUTHORITY_PARAMETER, MHR_GEOMETRY_AUTHORITY_REVISION, MHRGeometryAuthority, MHRGeometryProvider, normalize_mhr_decoder_identity
from lib_mhr.schema import MHR_PARAM_DIMS


MHR_GEOMETRY_CERTIFICATE_REVISION = "cari4d.mhr_geometry_certificate.v3"
MHR_GEOMETRY_CERTIFICATE_LEGACY_REVISIONS = frozenset(("cari4d.mhr_geometry_certificate.v1", "cari4d.mhr_geometry_certificate.v2"))
MHR_GEOMETRY_CERTIFICATE_METHOD = "parameter_payload_sha256_and_decoder_identity"
MHR_GEOMETRY_CERTIFICATE_TOLERANCE_M = 2e-6
MHR_RECONSTRUCTIBLE_GEOMETRY_FIELDS = ("mhr_vertices",)
MHR_GEOMETRY_REFERENCE_CACHED = "cached_vertex_migration_comparison"
MHR_GEOMETRY_REFERENCE_PARAMETERS = "parameter_native_decode"
MHR_GEOMETRY_LEGACY_CERTIFICATE_FIELDS = ("mhr_geometry_vertex_l2_tolerance_m", "mhr_geometry_gt_vertex_l2_max_error_m", "mhr_geometry_init_vertex_l2_max_error_m", "mhr_geometry_gt_vertex_reference", "mhr_geometry_init_vertex_reference")


def _float32_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype("float32"):
        raise TypeError(f"{label} must be float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def parameter_payload_sha256(gt: Mapping[str, Any], init: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for record_name, record in (("gt", gt), ("init", init)):
        for key, dim in MHR_PARAM_DIMS.items():
            if key not in record:
                raise KeyError(f"{record_name} is missing required MHR parameter {key}")
            value = _float32_array(record[key], f"{record_name}/{key}")
            if value.ndim < 2 or value.shape[-1] != dim:
                raise ValueError(f"{record_name}/{key} must end in dimension {dim}, got {value.shape}")
            contiguous = np.ascontiguousarray(value.astype("<f4", copy=False))
            digest.update(record_name.encode("ascii") + b"\0" + key.encode("ascii") + b"\0")
            digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    if value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor":
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _certify_record(record: Mapping[str, Any], record_kind: str, provider: MHRGeometryProvider, authority: MHRGeometryAuthority) -> tuple[float, str]:
    first_param = _float32_array(record["mhr_trans"], f"{record_kind}/mhr_trans")
    camera_count = 1 if record_kind == "gt" else first_param.shape[1]
    frames_per_batch = max(1, provider.max_batch_size // camera_count)
    max_error = 0.0
    for start in range(0, first_param.shape[0], frames_per_batch):
        stop = min(start + frames_per_batch, first_param.shape[0])
        batch = {key: np.asarray(record[key][start:stop]) for key in MHR_PARAM_DIMS}
        _to_numpy(provider.vertices(batch, record_kind=record_kind, authority=authority)).astype(np.float32, copy=False)
    return max_error, MHR_GEOMETRY_REFERENCE_PARAMETERS


def stamp_parameter_authoritative_packed(packed: Mapping[str, Any], decoder_identity: Mapping[str, str]) -> dict[str, Any]:
    gt, init = packed.get("gt"), packed.get("init")
    if not isinstance(gt, Mapping) or not isinstance(init, Mapping):
        raise TypeError("MHR geometry certification requires gt and init mappings")
    identity = normalize_mhr_decoder_identity(decoder_identity)
    metadata = dict(packed.get("metadata", {}))
    for key in MHR_GEOMETRY_LEGACY_CERTIFICATE_FIELDS:
        metadata.pop(key, None)
    metadata.update({"mhr_geometry_authority": MHR_GEOMETRY_AUTHORITY_PARAMETER, "mhr_geometry_authority_revision": MHR_GEOMETRY_AUTHORITY_REVISION, "mhr_geometry_certificate_revision": MHR_GEOMETRY_CERTIFICATE_REVISION, "mhr_geometry_certificate_method": MHR_GEOMETRY_CERTIFICATE_METHOD, "mhr_geometry_parameter_sha256": parameter_payload_sha256(gt, init), "mhr_geometry_omitted_fields": list(MHR_RECONSTRUCTIBLE_GEOMETRY_FIELDS), **identity})
    output = dict(packed)
    output["metadata"] = metadata
    output["gt"] = {key: value for key, value in gt.items() if key not in MHR_RECONSTRUCTIBLE_GEOMETRY_FIELDS}
    output["init"] = {key: value for key, value in init.items() if key not in MHR_RECONSTRUCTIBLE_GEOMETRY_FIELDS}
    return output


def certify_parameter_authoritative_packed(packed: Mapping[str, Any], mhr_layer: Any, *, tolerance_m: float = MHR_GEOMETRY_CERTIFICATE_TOLERANCE_M, max_batch_size: int = 256) -> dict[str, Any]:
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError(f"MHR geometry certificate tolerance must be finite and positive, got {tolerance_m}")
    gt, init = packed.get("gt"), packed.get("init")
    if not isinstance(gt, Mapping) or not isinstance(init, Mapping):
        raise TypeError("MHR geometry certification requires gt and init mappings")
    decoder_identity = mhr_layer.decoder_identity()
    authority = MHRGeometryAuthority.certified_parameters(decoder_identity)
    provider = MHRGeometryProvider(mhr_layer, decoder_identity=decoder_identity, max_batch_size=max_batch_size)
    certified = {name: _certify_record(record, name, provider, authority) for name, record in (("gt", gt), ("init", init))}
    errors = {name: value[0] for name, value in certified.items()}
    references = {name: value[1] for name, value in certified.items()}
    if max(errors.values()) > tolerance_m:
        raise ValueError(f"MHR parameter reconstruction exceeds {tolerance_m:.9g} m: gt={errors['gt']:.9g}, init={errors['init']:.9g}")
    return stamp_parameter_authoritative_packed(packed, decoder_identity)


def validate_parameter_authoritative_certificate(metadata: Mapping[str, Any], gt: Mapping[str, Any], init: Mapping[str, Any]) -> dict[str, Any]:
    revision = str(metadata.get("mhr_geometry_certificate_revision", ""))
    if revision != MHR_GEOMETRY_CERTIFICATE_REVISION and revision not in MHR_GEOMETRY_CERTIFICATE_LEGACY_REVISIONS:
        raise ValueError(f"Unsupported parameter-authoritative MHR geometry certificate revision {revision!r}")
    expected = {"mhr_geometry_authority": MHR_GEOMETRY_AUTHORITY_PARAMETER, "mhr_geometry_authority_revision": MHR_GEOMETRY_AUTHORITY_REVISION, "mhr_geometry_omitted_fields": list(MHR_RECONSTRUCTIBLE_GEOMETRY_FIELDS)}
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ValueError(f"Invalid parameter-authoritative MHR geometry certificate: {mismatches}")
    decoder_identity = normalize_mhr_decoder_identity({key: metadata.get(key, "") for key in MHR_DECODER_IDENTITY_KEYS}, "MHR geometry certificate decoder identity")
    expected_hash = parameter_payload_sha256(gt, init)
    if metadata.get("mhr_geometry_parameter_sha256") != expected_hash:
        raise ValueError("MHR geometry parameter payload does not match its certificate")
    if revision == MHR_GEOMETRY_CERTIFICATE_REVISION:
        if metadata.get("mhr_geometry_certificate_method") != MHR_GEOMETRY_CERTIFICATE_METHOD:
            raise ValueError(f"MHR geometry certificate must use method {MHR_GEOMETRY_CERTIFICATE_METHOD!r}")
        stale = sorted(key for key in MHR_GEOMETRY_LEGACY_CERTIFICATE_FIELDS if key in metadata)
        if stale:
            raise ValueError(f"MHR geometry certificate retains obsolete decode-comparison fields: {stale}")
        return {"decoder_identity": decoder_identity, "parameter_sha256": expected_hash, "method": MHR_GEOMETRY_CERTIFICATE_METHOD}
    tolerance = float(metadata.get("mhr_geometry_vertex_l2_tolerance_m", np.nan))
    errors = {name: float(metadata.get(f"mhr_geometry_{name}_vertex_l2_max_error_m", np.nan)) for name in ("gt", "init")}
    if not np.isfinite(tolerance) or tolerance <= 0 or any(not np.isfinite(value) or value < 0 or value > tolerance for value in errors.values()):
        raise ValueError(f"MHR geometry certificate has invalid tolerance/errors: tolerance={tolerance}, errors={errors}")
    references = {name: str(metadata.get(f"mhr_geometry_{name}_vertex_reference", MHR_GEOMETRY_REFERENCE_CACHED if revision == "cari4d.mhr_geometry_certificate.v1" else "")) for name in ("gt", "init")}
    allowed_references = {MHR_GEOMETRY_REFERENCE_CACHED, MHR_GEOMETRY_REFERENCE_PARAMETERS}
    if any(value not in allowed_references for value in references.values()):
        raise ValueError(f"MHR geometry certificate has invalid vertex references: {references}")
    if any(references[name] == MHR_GEOMETRY_REFERENCE_PARAMETERS and errors[name] != 0.0 for name in references):
        raise ValueError(f"Parameter-native MHR geometry certificates must report zero comparison error: {errors}")
    return {"decoder_identity": decoder_identity, "parameter_sha256": expected_hash, "tolerance_m": tolerance, "errors_m": errors, "references": references}


def restamp_parameter_authoritative_certificate(metadata: Mapping[str, Any], gt: Mapping[str, Any], init: Mapping[str, Any], *, decoder_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
    identity = decoder_identity or {key: metadata.get(key, "") for key in MHR_DECODER_IDENTITY_KEYS}
    updated = stamp_parameter_authoritative_packed({"metadata": metadata, "gt": gt, "init": init}, identity)["metadata"]
    validate_parameter_authoritative_certificate(updated, gt, init)
    return updated
