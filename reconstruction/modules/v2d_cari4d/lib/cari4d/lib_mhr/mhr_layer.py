from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


MHR_COMPACT_BUFFER_FILENAME = "mhr_buffers.pt"
MHR_DECODER_REVISION = "cari4d.mhr_layer.vertices.v1"
MHR_NEUTRAL_HEIGHT_DECODE_BATCH_SIZE = 128
MHR_HEAD_PREFIX = "head_pose."
MHR_BUFFER_KEYS = (
    "joint_rotation",
    "scale_mean",
    "scale_comps",
    "faces",
    "hand_pose_mean",
    "hand_pose_comps",
    "hand_joint_idxs_left",
    "hand_joint_idxs_right",
    "keypoint_mapping",
    "right_wrist_coords",
    "root_coords",
    "local_to_world_wrist",
    "nonhand_param_idxs",
)

COCO17_TO_MHR70 = (
    0,   # nose
    1,   # left_eye
    2,   # right_eye
    3,   # left_ear
    4,   # right_ear
    5,   # left_shoulder
    6,   # right_shoulder
    7,   # left_elbow
    8,   # right_elbow
    62,  # left_wrist
    41,  # right_wrist
    9,   # left_hip
    10,  # right_hip
    11,  # left_knee
    12,  # right_knee
    13,  # left_ankle
    14,  # right_ankle
)


@dataclass
class MHRLayerOutput:
    vertices: Any
    joints: Any | None = None
    keypoints: Any | None = None
    coco17: Any | None = None
    faces: Any | None = None
    joint_global_rots: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "mhr_vertices": self.vertices,
        }
        if self.joints is not None:
            out["mhr_joints"] = self.joints
        if self.keypoints is not None:
            out["mhr_keypoints"] = self.keypoints
        if self.coco17 is not None:
            out["mhr_coco17"] = self.coco17
        if self.faces is not None:
            out["faces"] = self.faces
        if self.joint_global_rots is not None:
            out["mhr_joint_global_rots"] = self.joint_global_rots
        return out


@dataclass(frozen=True)
class MHRVerticesContext:
    batch_shape: tuple[int, ...]
    flat_count: int
    device: Any
    dtype: Any
    head: Any
    global_rot: Any
    hand: Any
    face: Any
    flip: Any


def mhr70_to_coco17(keypoints: Any) -> Any:
    """Select COCO17 body landmarks from SAM3D/MHR70 keypoints."""

    if keypoints is None:
        return None
    if keypoints.shape[-2] < max(COCO17_TO_MHR70) + 1:
        raise ValueError(f"MHR70 keypoints need at least 70 points, got shape {keypoints.shape}")
    return keypoints[..., list(COCO17_TO_MHR70), :]


class MHRLayer:
    """Parameter-to-mesh MHR layer.

    A parametric backend is required; persisted vertices are never accepted as
    a substitute for decoding canonical MHR parameters.
    """

    def __init__(self, backend: Callable[[Mapping[str, Any]], Mapping[str, Any] | MHRLayerOutput] | None = None,
                 faces: Any | None = None):
        self.backend = backend
        self.faces = faces

    def decoder_identity(self) -> dict[str, str]:
        identity = getattr(self.backend, "decoder_identity", None)
        if not callable(identity):
            raise RuntimeError("MHRLayer backend does not expose a reproducible decoder identity")
        return identity()

    @classmethod
    def from_mhr_assets(
        cls,
        *,
        mhr_assets_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        buffer_path: str | Path | None = None,
        mhr_model_path: str | Path | None = None,
        device: str | None = None,
    ) -> "MHRLayer":
        return cls(
            backend=MHRParametricBackend(
                mhr_assets_root=mhr_assets_root,
                checkpoint_path=checkpoint_path,
                buffer_path=buffer_path,
                mhr_model_path=mhr_model_path,
                device=device,
            )
        )

    def mhr_forward(self, params: Mapping[str, Any]) -> MHRLayerOutput:
        """Decode MHR parameters into mesh vertices and keypoints."""

        if self.backend is not None:
            out = self.backend(params)
            if isinstance(out, MHRLayerOutput):
                return out
            keypoints = out.get("mhr_keypoints")
            return MHRLayerOutput(
                vertices=out["mhr_vertices"],
                joints=out.get("mhr_joints"),
                keypoints=keypoints,
                coco17=out.get("mhr_coco17", mhr70_to_coco17(keypoints)),
                faces=out.get("faces", self.faces),
                joint_global_rots=out.get("mhr_joint_global_rots"),
            )

        raise RuntimeError("MHRLayer requires a parametric decoder backend")

    def prepare_mhr_vertices(self, params: Mapping[str, Any]) -> Any | None:
        if self.backend is not None and callable(getattr(self.backend, "prepare_vertices_context", None)):
            return self.backend.prepare_vertices_context(params)
        return None

    def mhr_forward_vertices(self, params: Mapping[str, Any], *, context: Any | None = None) -> Any:
        if self.backend is not None and callable(getattr(self.backend, "vertices_only", None)):
            return self.backend.vertices_only(params, context=context)
        return self.mhr_forward(params).vertices

    def neutral_height(self, params: Mapping[str, Any]) -> Any:
        if self.backend is None or not callable(getattr(self.backend, "neutral_height", None)):
            raise RuntimeError("MHRLayer backend does not expose neutral human height")
        return self.backend.neutral_height(params)

    def mesh_faces(self, *, device: Any | None = None) -> Any:
        if self.faces is None:
            if self.backend is None or not callable(getattr(self.backend, "mesh_faces", None)):
                raise RuntimeError("MHRLayer requires canonical mesh faces or a backend that exposes mesh_faces")
            self.faces = self.backend.mesh_faces(device=device)
        return self.faces

    def __call__(self, params: Mapping[str, Any]) -> MHRLayerOutput:
        return self.mhr_forward(params)


class MHRParametricBackend:
    """Differentiable MHR parameter decoder for canonical CARI4D MHR params.

    This uses the MHR head assets stored under the SAM3D-body checkout, but it
    does not run SAM3D image inference.
    """

    def __init__(
        self,
        *,
        mhr_assets_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        buffer_path: str | Path | None = None,
        mhr_model_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.mhr_assets_root = (
            Path(mhr_assets_root) if mhr_assets_root is not None else self.repo_root / "sam-3d-body"
        )
        ckpt_root = self.mhr_assets_root / "checkpoints" / "sam-3d-body-dinov3"
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else ckpt_root / "model.ckpt"
        self.buffer_path = Path(buffer_path) if buffer_path is not None else ckpt_root / MHR_COMPACT_BUFFER_FILENAME
        self.mhr_model_path = Path(mhr_model_path) if mhr_model_path is not None else ckpt_root / "assets" / "mhr_model.pt"
        self.device = device
        self.head = None

    @staticmethod
    def _sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(block_bytes):
                digest.update(block)
        return digest.hexdigest()

    def decoder_identity(self) -> dict[str, str]:
        buffer_source = self.buffer_path if self.buffer_path.exists() else self.checkpoint_path
        missing = [str(path) for path in (self.mhr_model_path, buffer_source) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Cannot identify MHR decoder because assets are missing: {missing}")
        return {"mhr_model_sha256": self._sha256_file(self.mhr_model_path), "mhr_buffer_sha256": self._sha256_file(buffer_source), "mhr_decoder_revision": MHR_DECODER_REVISION}

    @staticmethod
    def _is_sparse_tensor(value: Any) -> bool:
        return bool(getattr(value, "is_sparse", False))

    @staticmethod
    def _normalize_buffer_state(state: Mapping[str, Any]) -> dict[str, Any]:
        buffers = {}
        for key, value in state.items():
            dst_key = key[len(MHR_HEAD_PREFIX):] if key.startswith(MHR_HEAD_PREFIX) else key
            if dst_key not in MHR_BUFFER_KEYS:
                continue
            if MHRParametricBackend._is_sparse_tensor(value):
                value = value.to_dense()
            buffers[dst_key] = value
        return buffers

    @classmethod
    def _extract_mhr_head_buffers(cls, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
        state = checkpoint.get("state_dict", checkpoint)
        return cls._normalize_buffer_state(state)

    @classmethod
    def _load_mhr_buffer_file(cls, buffer_path: Path, torch: Any) -> dict[str, Any]:
        payload = torch.load(buffer_path, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"MHR buffer file must contain a state dict mapping: {buffer_path}")
        return cls._normalize_buffer_state(payload)

    @classmethod
    def _load_mhr_checkpoint_buffers(cls, checkpoint_path: Path, torch: Any) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return cls._extract_mhr_head_buffers(checkpoint)

    @staticmethod
    def _load_head_state(head: Any, loadable: Mapping[str, Any]):
        head_state = head.state_dict()
        shaped = {}
        for key, value in loadable.items():
            if key in head_state and tuple(head_state[key].shape) == tuple(value.shape):
                shaped[key] = value
        missing, unexpected = head.load_state_dict(shaped, strict=False)
        missing_critical = sorted(set(MHR_BUFFER_KEYS).intersection(missing))
        if missing_critical:
            raise RuntimeError(f"SAM3D MHR backend missing required buffers: {missing_critical}")
        if unexpected:
            raise RuntimeError(f"Unexpected SAM3D MHR buffers: {unexpected}")

    def _ensure_head(self, device: Any):
        import os
        import sys

        import torch

        if self.head is not None:
            self.head = self.head.to(device)
            return self.head

        if not self.mhr_assets_root.exists():
            raise FileNotFoundError(f"MHR assets root not found: {self.mhr_assets_root}")
        if not self.mhr_model_path.exists():
            raise FileNotFoundError(f"MHR model asset not found: {self.mhr_model_path}")
        if not self.buffer_path.exists() and not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "MHR buffers not found. Expected compact buffer file "
                f"{self.buffer_path} or fallback checkpoint {self.checkpoint_path}"
            )

        sys.path.insert(0, str(self.mhr_assets_root))
        os.environ.setdefault("MOMENTUM_ENABLED", "0")
        from sam_3d_body.models.heads.mhr_head import MHRHead

        head = MHRHead(input_dim=1, mhr_model_path=str(self.mhr_model_path))
        if self.buffer_path.exists():
            loadable = self._load_mhr_buffer_file(self.buffer_path, torch)
        else:
            loadable = self._load_mhr_checkpoint_buffers(self.checkpoint_path, torch)
        self._load_head_state(head, loadable)
        head.eval()
        self.head = head.to(device)
        return self.head

    def mesh_faces(self, *, device: Any | None = None) -> Any:
        return self._ensure_head(device or self.device or "cpu").faces

    @staticmethod
    def _is_torch_tensor(value: Any) -> bool:
        return value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor"

    @staticmethod
    def _param(params: Mapping[str, Any], key: str, shape: tuple[int, ...], device: Any, dtype: Any):
        import torch

        value = params.get(key)
        if value is None:
            return torch.zeros(*shape, device=device, dtype=dtype)
        if MHRParametricBackend._is_torch_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _zero_body_cont(shape: tuple[int, ...], device: Any, dtype: Any):
        import torch

        from sam_3d_body.models.modules.mhr_utils import compact_model_params_to_cont_body

        model_params = torch.zeros(*shape[:-1], 133, device=device, dtype=dtype)
        return compact_model_params_to_cont_body(model_params)

    def _vertices_context(self, params: Mapping[str, Any], *, detach_fixed: bool) -> MHRVerticesContext:
        import torch

        import roma
        from lib_mhr.rotations import rot6d_to_rotmat

        ref = params.get("mhr_global_rot6d", params.get("mhr_trans"))
        if ref is None:
            raise KeyError("MHRLayer requires mhr_global_rot6d or mhr_trans")
        if self._is_torch_tensor(ref):
            device = ref.device if self.device is None else torch.device(self.device)
            dtype = ref.dtype if ref.dtype.is_floating_point else torch.float32
        else:
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(self.device or default_device)
            dtype = torch.float32
        global_rot6d = self._param(params, "mhr_global_rot6d", (*ref.shape[:-1], 6), device, dtype)
        batch_shape = tuple(global_rot6d.shape[:-1])
        flat_count = math.prod(batch_shape) if batch_shape else 1
        head = self._ensure_head(device)
        hand = self._param(params, "mhr_hand", (*batch_shape, 108), device, dtype).reshape(flat_count, 108)
        face = self._param(params, "mhr_face", (*batch_shape, 72), device, dtype).reshape(flat_count, 72)
        global_rot = roma.rotmat_to_euler("ZYX", rot6d_to_rotmat(global_rot6d.reshape(flat_count, 6)))
        if detach_fixed:
            global_rot = global_rot.detach()
            hand = hand.detach()
            face = face.detach()
        flip = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)
        return MHRVerticesContext(batch_shape=batch_shape, flat_count=flat_count, device=device, dtype=dtype, head=head, global_rot=global_rot, hand=hand, face=face, flip=flip)

    def prepare_vertices_context(self, params: Mapping[str, Any]) -> MHRVerticesContext:
        return self._vertices_context(params, detach_fixed=True)

    def _mutable_vertices_inputs(self, params: Mapping[str, Any], context: MHRVerticesContext) -> tuple[Any, Any, Any, Any]:
        from sam_3d_body.models.modules.mhr_utils import compact_cont_to_model_params_body

        trans = self._param(params, "mhr_trans", (*context.batch_shape, 3), context.device, context.dtype).reshape(context.flat_count, 3)
        if "mhr_body_pose_cont" in params:
            body_cont = self._param(params, "mhr_body_pose_cont", (*context.batch_shape, 260), context.device, context.dtype)
        else:
            body_cont = self._zero_body_cont((*context.batch_shape, 260), context.device, context.dtype)
        body_pose_params = compact_cont_to_model_params_body(body_cont.reshape(context.flat_count, 260))
        shape = self._param(params, "mhr_shape", (*context.batch_shape, 45), context.device, context.dtype).reshape(context.flat_count, 45)
        scale = self._param(params, "mhr_scale", (*context.batch_shape, 28), context.device, context.dtype).reshape(context.flat_count, 28)
        return trans, body_pose_params, shape, scale

    def vertices_only(self, params: Mapping[str, Any], *, context: MHRVerticesContext | None = None) -> Any:
        context = context or self._vertices_context(params, detach_fixed=False)
        ref = params.get("mhr_global_rot6d", params.get("mhr_trans"))
        if ref is None or tuple(ref.shape[:-1]) != context.batch_shape:
            raise ValueError(f"MHR vertices context batch shape {context.batch_shape} does not match parameters")
        trans, body_pose_params, shape, scale = self._mutable_vertices_inputs(params, context)
        vertices = context.head.mhr_forward(global_trans=trans.new_zeros(trans.shape), global_rot=context.global_rot, body_pose_params=body_pose_params, hand_pose_params=context.hand, scale_params=scale, shape_params=shape, expr_params=context.face, return_keypoints=False, return_joint_coords=False, return_model_params=False, return_joint_rotations=False)
        vertices = vertices * context.flip + trans[:, None, :]
        return vertices.reshape(*context.batch_shape, *vertices.shape[1:])

    def neutral_height(self, params: Mapping[str, Any]) -> Any:
        import torch

        shape_value, scale_value = params.get("mhr_shape"), params.get("mhr_scale")
        if shape_value is None or scale_value is None:
            raise KeyError("neutral MHR height requires mhr_shape and mhr_scale")
        shape = shape_value if self._is_torch_tensor(shape_value) else torch.as_tensor(shape_value)
        scale = scale_value if self._is_torch_tensor(scale_value) else torch.as_tensor(scale_value)
        if shape.shape[:-1] != scale.shape[:-1] or shape.shape[-1] != 45 or scale.shape[-1] != 28:
            raise ValueError(f"neutral MHR height requires matching [...,45] shape and [...,28] scale tensors, got {tuple(shape.shape)} and {tuple(scale.shape)}")
        device = shape.device if self.device is None else torch.device(self.device)
        shape = shape.to(device=device, dtype=torch.float32)
        scale = scale.to(device=device, dtype=torch.float32)
        identity = torch.cat((shape.reshape(-1, 45), scale.reshape(-1, 28)), dim=-1)
        if not bool(torch.isfinite(identity).all()):
            raise ValueError("neutral MHR height identity parameters must be finite")
        unique_identity, inverse = torch.unique(identity, dim=0, return_inverse=True)
        head = self._ensure_head(device)
        heights = []
        for start in range(0, len(unique_identity), MHR_NEUTRAL_HEIGHT_DECODE_BATCH_SIZE):
            current = unique_identity[start:start + MHR_NEUTRAL_HEIGHT_DECODE_BATCH_SIZE]
            count = len(current)
            vertices = head.mhr_forward(global_trans=current.new_zeros((count, 3)), global_rot=current.new_zeros((count, 3)), body_pose_params=current.new_zeros((count, 133)), hand_pose_params=current.new_zeros((count, 108)), scale_params=current[:, 45:], shape_params=current[:, :45], expr_params=current.new_zeros((count, 72)), return_keypoints=False, return_joint_coords=False, return_model_params=False, return_joint_rotations=False)
            heights.append(vertices[:, :, 1].amax(dim=1) - vertices[:, :, 1].amin(dim=1))
        unique_heights = torch.cat(heights, dim=0)
        if not bool(torch.isfinite(unique_heights).all()) or not bool((unique_heights > 0).all()):
            raise ValueError("decoded neutral MHR height must be finite and positive")
        return unique_heights[inverse].reshape(shape.shape[:-1])

    def __call__(self, params: Mapping[str, Any]) -> MHRLayerOutput:
        import torch

        context = self._vertices_context(params, detach_fixed=False)
        trans, body_pose_params, shape, scale = self._mutable_vertices_inputs(params, context)
        verts, keypoints, joints, joint_global_rots = context.head.mhr_forward(
            global_trans=torch.zeros_like(trans),
            global_rot=context.global_rot,
            body_pose_params=body_pose_params,
            hand_pose_params=context.hand,
            scale_params=scale,
            shape_params=shape,
            expr_params=context.face,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=False,
            return_joint_rotations=True,
        )

        verts = verts * context.flip + trans[:, None, :]
        joints = joints * context.flip + trans[:, None, :]
        keypoints70 = keypoints[:, :70] * context.flip + trans[:, None, :]
        coco17 = mhr70_to_coco17(keypoints70)

        def restore(value):
            return value.reshape(*context.batch_shape, *value.shape[1:])

        return MHRLayerOutput(
            vertices=restore(verts),
            joints=restore(joints),
            keypoints=restore(keypoints70),
            coco17=restore(coco17),
            faces=context.head.faces,
            joint_global_rots=restore(joint_global_rots),
        )
