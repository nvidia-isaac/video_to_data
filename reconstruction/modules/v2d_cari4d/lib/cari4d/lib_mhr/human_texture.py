from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np


MHR_PART_TEXTURE_REVISION = "mhr-native-lbs-part-palette-v1"
MHR_MODEL_RELATIVE_PATH = Path("checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
MHR_PART_NAMES = ("torso", "head", "left_upper_arm", "left_lower_arm", "left_hand", "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_leg", "left_lower_leg", "left_foot", "right_upper_leg", "right_lower_leg", "right_foot")
MHR_PART_PALETTE = np.asarray(((0.70, 0.70, 0.72), (0.95, 0.80, 0.35), (0.35, 0.70, 0.95), (0.20, 0.50, 0.85), (0.10, 0.30, 0.65), (0.95, 0.45, 0.45), (0.82, 0.25, 0.30), (0.60, 0.10, 0.20), (0.45, 0.80, 0.50), (0.25, 0.65, 0.35), (0.10, 0.45, 0.25), (0.75, 0.55, 0.90), (0.58, 0.36, 0.78), (0.40, 0.22, 0.62)), dtype=np.float32)


def mhr_faces_sha256(faces: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(faces, dtype="<i4").tobytes()).hexdigest()


def _resolve_mhr_model_path(mhr_model_path: str | Path | None) -> Path:
    if mhr_model_path is not None:
        path = Path(mhr_model_path)
    else:
        assets_root = os.environ.get("MHR_ASSETS_ROOT")
        root = Path(assets_root) if assets_root else Path(__file__).resolve().parents[1] / "sam-3d-body"
        path = root / MHR_MODEL_RELATIVE_PATH
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MHR model asset not found: {path}")
    return path


def _joint_part_index(name: str) -> int:
    if name in ("body_world", "root") or name.startswith("c_spine"):
        return 0
    if name.startswith(("c_neck", "c_head", "c_jaw", "c_teeth", "c_tongue", "l_eye", "r_eye")):
        return 1
    for prefix, upper_arm, lower_arm, hand, upper_leg, lower_leg, foot in (("l", 2, 3, 4, 8, 9, 10), ("r", 5, 6, 7, 11, 12, 13)):
        if name == f"{prefix}_clavicle" or name.startswith(f"{prefix}_uparm"):
            return upper_arm
        if name.startswith(f"{prefix}_lowarm") or name == f"{prefix}_wrist_twist":
            return lower_arm
        if name == f"{prefix}_wrist" or name.startswith(tuple(f"{prefix}_{finger}" for finger in ("thumb", "index", "middle", "ring", "pinky"))):
            return hand
        if name.startswith(f"{prefix}_upleg"):
            return upper_leg
        if name.startswith(f"{prefix}_lowleg"):
            return lower_leg
        if name in (f"{prefix}_foot", f"{prefix}_talocrural", f"{prefix}_subtalar", f"{prefix}_transversetarsal", f"{prefix}_ball"):
            return foot
    raise ValueError(f"MHR joint has no native part assignment: {name!r}")


@lru_cache(maxsize=2)
def _load_mhr_part_weights(mhr_model_path: str) -> np.ndarray:
    import torch

    model = torch.jit.load(mhr_model_path, map_location="cpu")
    joint_names = [str(name) for name in model.get_joint_names()]
    lbs_indices, lbs_weights = model.get_lbsw()
    lbs_indices = np.asarray(lbs_indices.to(torch.int64).cpu(), dtype=np.int64)
    lbs_weights = np.asarray(lbs_weights.cpu(), dtype=np.float32)
    if lbs_indices.ndim != 2 or lbs_weights.shape != lbs_indices.shape or lbs_indices.shape[1] == 0 or not np.isfinite(lbs_weights).all():
        raise ValueError(f"invalid MHR skinning arrays: indices={lbs_indices.shape}, weights={lbs_weights.shape}")
    if lbs_indices.min() < 0 or lbs_indices.max() >= len(joint_names) or np.any(lbs_weights < 0.0) or not np.allclose(lbs_weights.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("MHR skinning weights or joint indices are invalid")
    joint_parts = np.asarray([_joint_part_index(name) for name in joint_names], dtype=np.int32)
    part_weights = np.zeros((len(lbs_indices), len(MHR_PART_NAMES)), dtype=np.float32)
    vertices = np.arange(len(lbs_indices), dtype=np.int64)
    for slot in range(lbs_indices.shape[1]):
        np.add.at(part_weights, (vertices, joint_parts[lbs_indices[:, slot]]), lbs_weights[:, slot])
    if not np.allclose(part_weights.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("MHR native part weights do not sum to one")
    return part_weights


def load_mhr_part_texture_arrays(faces: np.ndarray, mhr_model_path: str | Path | None = None) -> dict[str, np.ndarray]:
    faces = np.asarray(faces, dtype=np.int32)
    part_weights = _load_mhr_part_weights(str(_resolve_mhr_model_path(mhr_model_path)))
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0 or faces.min() < 0 or faces.max() + 1 != len(part_weights):
        raise ValueError(f"faces are incompatible with the {len(part_weights)}-vertex MHR skinning model: {faces.shape}")
    face_parts = np.argmax(part_weights[faces].sum(axis=1), axis=1)
    face_uv = np.stack(((face_parts.astype(np.float32) + 0.5) / len(MHR_PART_NAMES), np.full(len(face_parts), 0.5, dtype=np.float32)), axis=1)
    uv = np.repeat(face_uv, 3, axis=0)
    uv_idx = np.arange(len(faces) * 3, dtype=np.int32).reshape(len(faces), 3)
    return {"faces": faces.copy(), "tex": MHR_PART_PALETTE[None].copy(), "uv": uv, "uv_idx": uv_idx}


def make_mhr_part_texture_tensors(faces: np.ndarray, device: str | object = "cuda", mhr_model_path: str | Path | None = None) -> dict[str, object]:
    import torch

    material = load_mhr_part_texture_arrays(faces, mhr_model_path=mhr_model_path)
    return {
        "faces": torch.as_tensor(material["faces"], device=device, dtype=torch.int),
        "tex": torch.as_tensor(material["tex"], device=device, dtype=torch.float)[None],
        "uv": torch.as_tensor(material["uv"], device=device, dtype=torch.float),
        "uv_idx": torch.as_tensor(material["uv_idx"], device=device, dtype=torch.int),
    }


def batched_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 3 or vertices.shape[2] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"vertices must be finite [B,V,3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0 or faces.min() < 0 or faces.max() >= vertices.shape[1]:
        raise ValueError(f"faces are incompatible with {vertices.shape[1]} vertices: {faces.shape}")
    normals = np.zeros_like(vertices, dtype=np.float32)
    for frame_index, frame_vertices in enumerate(vertices):
        triangle_normals = np.cross(frame_vertices[faces[:, 1]] - frame_vertices[faces[:, 0]], frame_vertices[faces[:, 2]] - frame_vertices[faces[:, 0]])
        for corner in range(3):
            np.add.at(normals[frame_index], faces[:, corner], triangle_normals)
    lengths = np.linalg.norm(normals, axis=2, keepdims=True)
    return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)
