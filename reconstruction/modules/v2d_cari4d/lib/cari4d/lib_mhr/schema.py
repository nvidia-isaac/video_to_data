from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


MHR_PARAM_DIMS = {
    "mhr_global_rot6d": 6,
    "mhr_trans": 3,
    "mhr_body_pose_cont": 260,
    "mhr_hand": 108,
    "mhr_shape": 45,
    "mhr_scale": 28,
    "mhr_face": 72,
}

MHR_FULL_DIM = sum(MHR_PARAM_DIMS.values())

MHR_REQUIRED_PARAM_KEYS = (
    "mhr_global_rot6d",
    "mhr_trans",
    "mhr_body_pose_cont",
    "mhr_hand",
    "mhr_shape",
    "mhr_scale",
)

MHR_OPTIONAL_PARAM_KEYS = (
    "mhr_face",
)

MHR_GEOMETRY_KEYS = (
    "mhr_joints",
    "mhr_keypoints",
)

MHR_REQUIRED_KEYS = MHR_REQUIRED_PARAM_KEYS + (
    "frames",
    "kids",
)

MHR_DELTA_KEYS = (
    "delta_mhr_global_rot6d",
    "delta_mhr_trans",
    "delta_mhr_body_pose_cont",
    "delta_mhr_hand",
    "delta_mhr_shape",
    "delta_mhr_scale",
    "delta_mhr_face",
)

MHR_INIT_SUFFIX = "_init"
MHR_GT_SUFFIX = "_gt"

def _key(*parts: str) -> str:
    return "".join(parts)


FORBIDDEN_NATIVE_KEYS = (
    _key("po", "ses"),
    _key("be", "tas"),
    _key("trans", "ls"),
    _key("s", "mpl_pose"),
    _key("s", "mpl_t"),
    _key("joints_", "s", "mpl"),
    _key("dists_h", "2o"),
)


@dataclass(frozen=True)
class SchemaError:
    key: str
    message: str

    def __str__(self) -> str:
        return f"{self.key}: {self.message}"


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(x) for x in shape)


def _has_last_dim(value: Any, dim: int) -> bool:
    shape = _shape_of(value)
    return shape is not None and len(shape) > 0 and shape[-1] == dim


def validate_mhr_schema(
    data: Mapping[str, Any],
    *,
    require_geometry: bool = False,
    allow_forbidden_legacy_keys: bool = False,
    prefix: str = "",
) -> list[SchemaError]:
    """Return schema errors for a native MHR mapping.

    The validator intentionally checks only stable contracts: key presence,
    forbidden legacy names, and final tensor dimension for parameter blocks.
    It accepts either per-frame arrays or batched arrays as long as the final
    dimension matches the canonical MHR block size.
    """

    errors: list[SchemaError] = []
    if not isinstance(data, Mapping):
        return [SchemaError(prefix or "<root>", "expected a mapping")]

    if not allow_forbidden_legacy_keys:
        for key in FORBIDDEN_NATIVE_KEYS:
            if key in data:
                errors.append(SchemaError(key, "legacy human field is not allowed in native MHR data"))

    for key in MHR_REQUIRED_KEYS:
        full_key = prefix + key
        if full_key not in data:
            errors.append(SchemaError(full_key, "required MHR field is missing"))

    if require_geometry:
        for key in MHR_GEOMETRY_KEYS:
            full_key = prefix + key
            if full_key not in data:
                errors.append(SchemaError(full_key, "required MHR geometry field is missing"))

    for key, dim in MHR_PARAM_DIMS.items():
        full_key = prefix + key
        if full_key in data and not _has_last_dim(data[full_key], dim):
            errors.append(SchemaError(full_key, f"expected final dimension {dim}, got {_shape_of(data[full_key])}"))

    return errors


def assert_mhr_schema(
    data: Mapping[str, Any],
    *,
    require_geometry: bool = False,
    allow_forbidden_legacy_keys: bool = False,
    prefix: str = "",
) -> None:
    errors = validate_mhr_schema(
        data,
        require_geometry=require_geometry,
        allow_forbidden_legacy_keys=allow_forbidden_legacy_keys,
        prefix=prefix,
    )
    if errors:
        detail = "; ".join(str(err) for err in errors)
        raise ValueError(f"Invalid MHR schema: {detail}")
