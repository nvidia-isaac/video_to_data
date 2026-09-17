from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping

from .camera_conventions import validate_mhr_init_root_metadata
from .delta import compose_mhr_delta
from .schema import MHR_PARAM_DIMS


def _key(*parts: str) -> str:
    return "".join(parts)


FORBIDDEN_RESULT_KEYS = (
    _key("s", "mpl_pose"),
    _key("s", "mpl_t"),
    _key("be", "tas"),
    _key("po", "ses"),
    _key("trans", "ls"),
)


MHR_RESULT_PARAM_KEYS = tuple(MHR_PARAM_DIMS.keys())
MHR_RESULT_GEOM_KEYS = (
    "mhr_joints",
    "mhr_keypoints",
    "mhr_joint_global_rots",
)
MHR_RESULT_KEYS = MHR_RESULT_PARAM_KEYS + MHR_RESULT_GEOM_KEYS + (
    "body_model",
    "obj_rot",
    "obj_t",
    "frames",
    "kids",
    "faces",
    "metadata",
)


def omit_reconstructible_mhr_vertices(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: omit_reconstructible_mhr_vertices(value) if isinstance(value, Mapping) else value for key, value in data.items() if key != "mhr_vertices"}


def assert_no_legacy_result_keys(data: Mapping[str, Any]) -> None:
    offenders = [key for key in FORBIDDEN_RESULT_KEYS if key in data]
    if offenders:
        raise ValueError(f"Native MHR result contains legacy fields: {offenders}")


def prepare_mhr_result(
    init_params: Mapping[str, Any],
    *,
    delta_output: Mapping[str, Any] | None = None,
    layer_output: Mapping[str, Any] | None = None,
    obj_rot: Any | None = None,
    obj_t: Any | None = None,
    frames: list[str] | None = None,
    kids: list[int] | None = None,
) -> dict[str, Any]:
    result = compose_mhr_delta(init_params, delta_output or {})
    if layer_output:
        result.update({key: value for key, value in layer_output.items() if key != "mhr_vertices"})
    if obj_rot is not None:
        result["obj_rot"] = obj_rot
    if obj_t is not None:
        result["obj_t"] = obj_t
    if frames is not None:
        result["frames"] = list(frames)
    elif "frames" in init_params:
        result["frames"] = list(init_params["frames"])
    if kids is not None:
        result["kids"] = list(kids)
    elif "kids" in init_params:
        result["kids"] = list(init_params["kids"])
    if "metadata" in init_params:
        result["metadata"] = dict(init_params["metadata"])

    result = {key: value for key, value in result.items() if key in MHR_RESULT_KEYS}
    assert_no_legacy_result_keys(result)
    validate_mhr_init_root_metadata(result.get("metadata", {}), "prepared MHR result")
    return result


def save_mhr_result(path: str | Path, data: Mapping[str, Any]) -> None:
    assert_no_legacy_result_keys(data)
    if "mhr_vertices" in data:
        raise ValueError("Persisted MHR results must omit reconstructible mhr_vertices")
    validate_mhr_init_root_metadata(data.get("metadata", {}), f"MHR result {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as f:
            pickle.dump(dict(data), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_mhr_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    assert_no_legacy_result_keys(data)
    data.pop("mhr_vertices", None)
    validate_mhr_init_root_metadata(data.get("metadata", {}), f"MHR result {path}")
    return data
