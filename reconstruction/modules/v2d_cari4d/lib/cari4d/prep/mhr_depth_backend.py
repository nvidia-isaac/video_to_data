from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


MONOCULAR_DEPTH_BACKENDS = ("unidepth", "moge2")
DEFAULT_MONOCULAR_DEPTH_BACKEND = "moge2"
UNIDEPTH_MODEL_ID = "lpiccinelli/unidepth-v2-vitl14"
MOGE2_MODEL_ID = "Ruicheng/moge-2-vitl-normal"
MOGE2_MODEL_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"
MOGE2_SOURCE_COMMIT = "925b8ed835a7a9cdb7578ba15c658a0afc969030"


def normalize_monocular_depth_backend(value: str) -> str:
    backend = str(value).strip().lower()
    backend = {"moge": "moge2", "moge-2-vitl-normal": "moge2", "unidepth-v2-vitl14": "unidepth"}.get(backend, backend)
    if backend not in MONOCULAR_DEPTH_BACKENDS:
        raise ValueError(f"Monocular depth backend must be one of {MONOCULAR_DEPTH_BACKENDS}, got {value!r}")
    return backend


def resolve_monocular_depth_model(backend: str, model_id: str | None = None, model_revision: str | None = None) -> tuple[str, str | None]:
    backend = normalize_monocular_depth_backend(backend)
    if backend == "unidepth":
        return model_id or UNIDEPTH_MODEL_ID, model_revision
    return model_id or MOGE2_MODEL_ID, model_revision or MOGE2_MODEL_REVISION


def monocular_depth_identity(backend: str = DEFAULT_MONOCULAR_DEPTH_BACKEND, model_id: str | None = None, model_revision: str | None = None) -> dict[str, Any]:
    backend = normalize_monocular_depth_backend(backend)
    resolved_model_id, resolved_model_revision = resolve_monocular_depth_model(backend, model_id, model_revision)
    return {"backend": backend, "model_id": resolved_model_id, "model_revision": resolved_model_revision, "source_commit": MOGE2_SOURCE_COMMIT if backend == "moge2" else None}


def stored_monocular_depth_identity(alignment_input_identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if alignment_input_identity is None or "monocular_depth" not in alignment_input_identity:
        return monocular_depth_identity()
    value = alignment_input_identity["monocular_depth"]
    if not isinstance(value, Mapping):
        raise TypeError("Depth alignment input identity monocular_depth must be an object")
    expected_keys = {"backend", "model_id", "model_revision", "source_commit"}
    if set(value) != expected_keys:
        raise ValueError(f"Depth alignment input identity monocular_depth keys differ: expected={sorted(expected_keys)}, actual={sorted(value)}")
    backend = normalize_monocular_depth_backend(str(value["backend"]))
    expected = monocular_depth_identity(backend, str(value["model_id"]), None if value["model_revision"] is None else str(value["model_revision"]))
    if dict(value) != expected:
        raise ValueError(f"Depth alignment input identity contains inconsistent monocular depth provenance: {dict(value)}")
    return expected


def depth_source_for_backend(backend: str) -> str:
    return f"{normalize_monocular_depth_backend(backend)}_aligned_to_export_depthimage"


def _decode_json_attribute(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    decoded = json.loads(str(value))
    if not isinstance(decoded, dict):
        raise TypeError("Depth alignment input identity must decode to an object")
    return decoded


def monocular_depth_identity_from_depth_h5(path: str | Path) -> dict[str, Any]:
    from prep.mhr_depth_h5 import DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE

    with h5py.File(Path(path), "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"Depth H5 is incomplete: {path}")
        identity = _decode_json_attribute(handle.attrs.get(DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE))
    return stored_monocular_depth_identity(identity)


def depth_source_from_depth_h5(path: str | Path) -> str:
    return depth_source_for_backend(monocular_depth_identity_from_depth_h5(path)["backend"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve monocular-depth provenance used by MHR preprocessing.")
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("source")
    source.add_argument("--depth-h5", required=True)
    identity = commands.add_parser("identity")
    identity.add_argument("--backend", choices=MONOCULAR_DEPTH_BACKENDS, default=DEFAULT_MONOCULAR_DEPTH_BACKEND)
    identity.add_argument("--model-id", default=None)
    identity.add_argument("--model-revision", default=None)
    args = parser.parse_args()
    if args.command == "source":
        print(depth_source_from_depth_h5(args.depth_h5))
    else:
        print(json.dumps(monocular_depth_identity(args.backend, args.model_id, args.model_revision), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
