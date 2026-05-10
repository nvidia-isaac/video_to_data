"""v2d-side adapter for the gr00t MotionBricks planner bundle.

Loads the torch.package archive at
``planner/assets/models/motionbricks_planner.pkg`` and exposes
``MotionInferenceAgent`` with a contract identical to the legacy MFM-based
agent so ``g1_planner.py`` only changes its import line. All
``torch.package`` workarounds (dataclass exec ordering, bundled ``groot.*``
namespace strings) are contained in this file.
"""

from __future__ import annotations

import contextlib
import dataclasses as _dc
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from torch.package import PackageImporter

from robotic_grounding.planner.motionbricks.qpos import (
    DEFAULT_SEED_XML,
    HAND_ROOT_TO_WRIST_OFFSET_LOCAL_LEFT,
    HAND_ROOT_TO_WRIST_OFFSET_LOCAL_RIGHT,
    LOWER_BODY_BODY_INDICES_ISAACLAB,
    apply_hand_root_to_wrist_offset,
    build_seed_qpos,
    chunk_boundary_centers,
    features_to_qpos,
    qpos_to_body_world,
    smooth_qpos_at_boundaries,
    smooth_qpos_global,
)

# ---------------------------------------------------------------------------
# Section 1: torch.package compatibility shim
# ---------------------------------------------------------------------------
# `@dataclass`-defined types inside a torch.package archive crash on unpickle
# because the package importer exec's the module body before registering it
# in ``sys.modules``. Python's ``dataclasses._is_type`` does
# ``sys.modules.get(cls.__module__).__dict__`` to detect the KW_ONLY sentinel
# and raises ``AttributeError: NoneType``. Returning ``False`` from the
# patched ``_is_type`` is the correct fallback (the annotation is not
# ``KW_ONLY``). Fixed upstream in CPython 3.12.


@contextlib.contextmanager
def _torch_package_compat() -> Iterator[None]:
    """Patch ``dataclasses._is_type`` for one torch.package load.

    ``@dataclass``-defined types inside a torch.package'd module crash on
    unpickle without this; the patch is reverted on context exit.
    """
    orig = _dc._is_type

    def patched(
        annotation: Any,
        cls: type,
        a_module: Any,
        a_type: Any,
        is_type_predicate: Any,
    ) -> bool:
        if sys.modules.get(cls.__module__) is None:
            return False
        return orig(annotation, cls, a_module, a_type, is_type_predicate)

    _dc._is_type = patched
    try:
        yield
    finally:
        _dc._is_type = orig


# ---------------------------------------------------------------------------
# Section 2: Bundle loader
# ---------------------------------------------------------------------------
# The bundled module namespace appears here only; callers receive a clean
# dict and never see torch.package internals.

_BUNDLE_NAMESPACE = "groot.rl.trl.inference.motionbricks_e2e"
_SUPPORTED_BUNDLE_SCHEMA = 1

_PKG_PATH = Path(__file__).parent / "assets" / "models" / "planner.pkg"


def _load_bundle(
    device: str,
) -> tuple[PackageImporter, dict[str, Any], dict[str, Any]]:
    """Open the .pkg and import the bundled pipeline + transforms helpers.

    Returns the importer (kept alive so module imports stay valid), the
    unpickled bundle dict, and a dict of helper modules.
    """
    if not _PKG_PATH.exists():
        raise FileNotFoundError(
            f"MotionBricks planner bundle not found at {_PKG_PATH}. "
            "Run `git lfs pull` to fetch the binary asset."
        )

    with _torch_package_compat():
        importer = PackageImporter(str(_PKG_PATH))
        bundle = importer.load_pickle("bundle", "bundle.pkl")

    schema = int(bundle.get("schema_version", -1))
    if schema != _SUPPORTED_BUNDLE_SCHEMA:
        raise RuntimeError(
            f"Bundle at {_PKG_PATH} has schema_version={schema}, "
            f"but this adapter expects {_SUPPORTED_BUNDLE_SCHEMA}. "
            "Re-export the bundle from gr00t with the matching exporter."
        )

    # Hide the bundled-namespace import strings behind a private dict; the
    # public class never references them again.
    bundle_modules = {
        "pipeline": importer.import_module(f"{_BUNDLE_NAMESPACE}.pipeline"),
        "transforms": importer.import_module(f"{_BUNDLE_NAMESPACE}.transforms"),
    }

    bundle["root_model"].to(device).eval()
    bundle["pose_model"].to(device).eval()

    return importer, bundle, bundle_modules


# ---------------------------------------------------------------------------
# Section 3: Lower-body-fixed chunked AR loop
# ---------------------------------------------------------------------------
# Vendored from the bundled ``run_chunked_autoregressive_inference`` so we
# can override the lower-body body transforms at each chunk's pose seed and
# prediction tail. With this in place, the model sees a static lower body
# in its autoregressive context and produces upper-body motion consistent
# with stationary legs.


def _override_lower_body_in_place(
    transforms: np.ndarray, static_pose: np.ndarray
) -> None:
    """Pin lower-body body slots in ``transforms`` to ``static_pose``.

    Both arrays are ``(F, num_bodies * 9)`` packed; only the bodies in
    ``LOWER_BODY_BODY_INDICES_ISAACLAB`` are overwritten in place.
    """
    num_bodies = transforms.shape[-1] // 9
    F = transforms.shape[0]
    view = transforms.reshape(F, num_bodies, 9)
    static_view = static_pose[:F].reshape(F, num_bodies, 9)
    for idx in LOWER_BODY_BODY_INDICES_ISAACLAB:
        view[:, idx] = static_view[:, idx]


def _run_chunked_lower_body_fixed(
    pipeline_module: Any,
    transforms_module: Any,
    root_model: Any,
    pose_model: Any,
    *,
    root_ee_input: np.ndarray,
    pose_ee_input: np.ndarray | None,
    derive_pose_ee_from_pred_root: bool,
    seed_root: np.ndarray,
    seed_pose: np.ndarray,
    static_pose_shared: np.ndarray,
    chunk_tokens: int,
    overlap_tokens: int,
    device: str,
) -> dict[str, Any]:
    """Chunked AR with the lower-body bodies pinned across chunks.

    Mirrors the bundled chunked AR loop but, after each per-chunk inference,
    overrides the lower-body slots in the predicted body transforms with
    the static-seed values before the next chunk reads its seed. The model
    therefore sees stationary legs in its context window every chunk.
    """
    nfpt = root_model.num_frames_per_token
    max_tokens = min(
        root_model._max_tokens,
        getattr(pose_model, "_max_tokens", root_model._max_tokens),
    )
    chunk_tokens = min(max(chunk_tokens, 2), max_tokens)
    overlap_tokens = max(1, min(overlap_tokens, chunk_tokens - 1))
    overlap_frames = overlap_tokens * nfpt
    chunk_frames = chunk_tokens * nfpt
    stride_frames = chunk_frames - overlap_frames

    total_frames = (root_ee_input.shape[0] // nfpt) * nfpt
    if total_frames < nfpt:
        raise ValueError(f"Need at least {nfpt} frames, got {root_ee_input.shape[0]}")

    root_dim = root_model.root_dim
    pose_dim = pose_model.pose_dim
    root_accum = np.zeros((total_frames, root_dim), dtype=np.float32)
    pose_accum = np.zeros((total_frames, pose_dim), dtype=np.float32)
    weight_accum = np.zeros((total_frames, 1), dtype=np.float32)

    previous_result: dict[str, np.ndarray] | None = None
    previous_range: tuple[int, int] | None = None
    chunk_infos: list[dict[str, Any]] = []
    root_ee_key = pipeline_module._model_ee_key(root_model)
    pose_ee_key = pipeline_module._model_ee_key(pose_model)
    use_chunk_local_frame = (
        pipeline_module._model_root_key(root_model)
        == pipeline_module.HEADING_ROOT_XY_KEY
    )

    start = 0
    while start < total_frames:
        end = min(start + chunk_frames, total_frames)
        end = start + ((end - start) // nfpt) * nfpt
        if end - start < nfpt:
            break

        window_num_tokens = (end - start) // nfpt
        root_ee_window = root_ee_input[start:end]
        pose_ee_window = pose_ee_input[start:end] if pose_ee_input is not None else None

        if previous_result is None:
            start_root_shared = seed_root[:nfpt].copy()
            start_pose_shared = seed_pose[:nfpt].copy()
        else:
            prev_start, _ = previous_range
            local_start = start - prev_start
            local_end = local_start + nfpt
            if local_start < 0 or local_end > previous_result["pred_frames"]:
                raise RuntimeError(
                    f"Chunk {start}:{end} outside previous range {previous_range}"
                )
            start_root_shared = previous_result["pred_root"][
                local_start:local_end
            ].copy()
            start_pose_shared = previous_result["pred_joints"][
                local_start:local_end
            ].copy()

        # Pin the lower body in the AR seed so the model sees stationary legs.
        _override_lower_body_in_place(
            start_pose_shared, static_pose_shared[start : start + nfpt]
        )

        if use_chunk_local_frame:
            chunk_origin_xy = np.asarray(start_root_shared[0, :2], dtype=np.float32)
            chunk_heading_mat = transforms_module._heading_matrix_from_pose_frame(
                start_pose_shared[0]
            ).astype(np.float32)
            start_root = transforms_module._root_xy_shared_to_local(
                start_root_shared, chunk_origin_xy, chunk_heading_mat
            )
            start_pose = transforms_module._packed_transforms_change_frame(
                start_pose_shared,
                chunk_heading_mat,
                inverse_heading=True,
                positions_are_root_relative=True,
            )
            root_ee_window_model = transforms_module._packed_transforms_change_frame(
                root_ee_window,
                chunk_heading_mat,
                origin_xy=(
                    chunk_origin_xy
                    if root_ee_key == pipeline_module.ROOT_TARGET_EE_KEY
                    else None
                ),
                inverse_heading=True,
                positions_are_root_relative=root_ee_key
                != pipeline_module.ROOT_TARGET_EE_KEY,
            )
            pose_ee_window_model = (
                transforms_module._packed_transforms_change_frame(
                    pose_ee_window,
                    chunk_heading_mat,
                    origin_xy=(
                        chunk_origin_xy
                        if pose_ee_key == pipeline_module.ROOT_TARGET_EE_KEY
                        else None
                    ),
                    inverse_heading=True,
                    positions_are_root_relative=pose_ee_key
                    != pipeline_module.ROOT_TARGET_EE_KEY,
                )
                if pose_ee_window is not None
                else None
            )
        else:
            chunk_origin_xy = np.zeros(2, dtype=np.float32)
            chunk_heading_mat = np.eye(3, dtype=np.float32)
            start_root = start_root_shared
            start_pose = start_pose_shared
            root_ee_window_model = root_ee_window
            pose_ee_window_model = pose_ee_window

        result = pipeline_module.run_inference(
            root_model,
            pose_model,
            root_ee_input=root_ee_window_model,
            pose_ee_input=pose_ee_window_model,
            derive_pose_ee_from_pred_root=derive_pose_ee_from_pred_root,
            num_tokens=window_num_tokens,
            start_root=start_root,
            start_pose=start_pose,
            device=device,
        )

        local_n = min(result["pred_frames"], end - start)
        pred_root_shared = (
            transforms_module._root_xy_local_to_shared(
                result["pred_root"][:local_n], chunk_origin_xy, chunk_heading_mat
            )
            if use_chunk_local_frame
            else result["pred_root"][:local_n]
        )
        pred_joints_shared = (
            transforms_module._packed_transforms_change_frame(
                result["pred_joints"][:local_n],
                chunk_heading_mat,
                inverse_heading=False,
                positions_are_root_relative=True,
            )
            if use_chunk_local_frame
            else result["pred_joints"][:local_n]
        )

        # Overwrite lower-body bodies in the prediction so the next chunk's
        # AR seed (read from previous_result) is also lower-body-static.
        _override_lower_body_in_place(
            pred_joints_shared, static_pose_shared[start : start + local_n]
        )

        weights = pipeline_module._triangular_chunk_weights(local_n)[:, None]
        root_accum[start : start + local_n] += pred_root_shared * weights
        pose_accum[start : start + local_n] += pred_joints_shared * weights
        weight_accum[start : start + local_n] += weights

        chunk_infos.append(
            {
                "start_frame": start,
                "end_frame": start + local_n,
                "num_tokens": int(np.ceil(local_n / nfpt)),
                "window_num_tokens": int(window_num_tokens),
                "predicted_num_tokens": int(result["pred_num_tokens"]),
                "chunk_num_tokens_mode": "forced",
                "chunk_origin_xy": chunk_origin_xy.tolist(),
                "chunk_local_frame_enabled": bool(use_chunk_local_frame),
            }
        )

        previous_result = {
            "pred_root": pred_root_shared,
            "pred_joints": pred_joints_shared,
            "pred_frames": local_n,
        }
        previous_range = (start, start + local_n)
        if start + local_n >= total_frames:
            break
        start += stride_frames

    valid = weight_accum[:, 0] > 0
    if not np.all(valid):
        last_valid = int(np.nonzero(valid)[0][-1]) + 1
        root_accum = root_accum[:last_valid]
        pose_accum = pose_accum[:last_valid]
        weight_accum = weight_accum[:last_valid]

    pred_root = root_accum / np.clip(weight_accum, 1e-8, None)
    pred_joints = pose_accum / np.clip(weight_accum, 1e-8, None)
    # Final pass: enforce lower-body-static in the blended output.
    _override_lower_body_in_place(
        pred_joints, static_pose_shared[: pred_joints.shape[0]]
    )

    return {
        "pred_root": pred_root,
        "pred_joints": pred_joints,
        "pred_num_tokens": int(np.ceil(pred_root.shape[0] / nfpt)),
        "pred_frames": int(pred_root.shape[0]),
        "chunk_infos": chunk_infos,
    }


# ---------------------------------------------------------------------------
# Section 4: Half-stride second pass
# ---------------------------------------------------------------------------


def _blend_two_passes(
    pred_a: dict[str, Any],
    pred_b: dict[str, Any],
    *,
    offset_b: int,
    total_frames: int,
) -> dict[str, Any]:
    """Average two AR passes that share the same shared-frame indexing.

    Pass A covers ``[0, len_a)``; pass B covers ``[offset_b, offset_b + len_b)``
    in the same first-frame heading frame. Each pass is already a smooth
    chunked-AR output; the half-stride offset ensures that frames near a
    chunk boundary in one pass sit mid-chunk in the other, so a per-frame
    mean smooths out residual seam artifacts.
    """
    F = min(
        total_frames,
        max(pred_a["pred_frames"], offset_b + pred_b["pred_frames"]),
    )
    accum_root = np.zeros((F, pred_a["pred_root"].shape[-1]), dtype=np.float32)
    accum_pose = np.zeros((F, pred_a["pred_joints"].shape[-1]), dtype=np.float32)
    count = np.zeros((F, 1), dtype=np.float32)

    n_a = min(pred_a["pred_frames"], F)
    accum_root[:n_a] += pred_a["pred_root"][:n_a]
    accum_pose[:n_a] += pred_a["pred_joints"][:n_a]
    count[:n_a] += 1.0

    n_b = min(pred_b["pred_frames"], F - offset_b)
    if n_b > 0:
        accum_root[offset_b : offset_b + n_b] += pred_b["pred_root"][:n_b]
        accum_pose[offset_b : offset_b + n_b] += pred_b["pred_joints"][:n_b]
        count[offset_b : offset_b + n_b] += 1.0

    count = np.clip(count, 1e-8, None)
    return {
        "pred_root": accum_root / count,
        "pred_joints": accum_pose / count,
        "pred_num_tokens": pred_a.get("pred_num_tokens", 0),
        "pred_frames": int(F),
        "chunk_infos": (pred_a.get("chunk_infos") or [])
        + (pred_b.get("chunk_infos") or []),
    }


# ---------------------------------------------------------------------------
# Section 5: Public agent
# ---------------------------------------------------------------------------


class MotionInferenceAgent:
    """Whole-body motion inference: EE targets → full-body qpos.

    Drop-in replacement for the legacy MFM-based agent. Public method
    ``infer_from_ee_positions`` carries the same signature so
    ``g1_planner.py`` only needs to swap its import line.

    Inputs are MuJoCo z-up world frame; output qpos layout is
    ``[pos(3), xyzw_quat(4), joints(29)]`` — same as legacy.
    """

    def __init__(self, device: str = "cuda") -> None:
        """Load the bundle and place all weights on ``device``."""
        self._device = device
        self._importer, self._bundle, self._mods = _load_bundle(device)
        self._root_model = self._bundle["root_model"]
        self._pose_model = self._bundle["pose_model"]
        self._kin = self._bundle["humanoid_kinematics"]
        self._body_reorder = self._bundle["body_reorder"]

    @property
    def nfpt(self) -> int:
        """Number of frames per MotionBricks token."""
        return int(self._bundle["nfpt"])

    @property
    def model_fps(self) -> int:
        """Training-time frame rate, returned for legacy parity only.

        MotionBricks training data is at 25 fps. v2d's planner orchestrator
        resamples upstream and trims downstream, so this is for logging only.
        """
        return 25

    @property
    def max_tokens(self) -> int:
        """Maximum tokens in a single chunked-AR window."""
        return int(self._bundle["max_tokens"])

    def infer_from_ee_positions(
        self,
        root_pos: np.ndarray,
        root_wxyz: np.ndarray,
        left_ee_pos: np.ndarray,
        left_ee_quat_wxyz: np.ndarray,
        right_ee_pos: np.ndarray,
        right_ee_quat_wxyz: np.ndarray,
        root_height_override: float | None = None,
        z_offset: float = 0.0,
        src_fps: float | None = None,
        max_chunk_tokens: int = 6,
        overlap_tokens: int = 3,
        modes: tuple[str, ...] = ("autoregressive",),
        smooth: bool = True,
        half_stride_blend: bool = True,
        left_hand_root_offset_local: tuple[float, float, float] | None = (
            HAND_ROOT_TO_WRIST_OFFSET_LOCAL_LEFT
        ),
        right_hand_root_offset_local: tuple[float, float, float] | None = (
            HAND_ROOT_TO_WRIST_OFFSET_LOCAL_RIGHT
        ),
        fix_lower_body: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Run whole-body inference from end-effector targets.

        Inputs are MuJoCo z-up world-frame trajectories. Output qpos layout
        matches the legacy contract: ``[pos(3), xyzw_quat(4), joints(29)]``.

        Args:
            root_pos: ``(T, 3)`` desired root position trajectory.
            root_wxyz: ``(T, 4)`` desired root quaternion (wxyz).
            left_ee_pos: ``(T, 3)`` desired left wrist position.
            left_ee_quat_wxyz: ``(T, 4)`` desired left wrist quaternion (wxyz).
            right_ee_pos: ``(T, 3)`` desired right wrist position.
            right_ee_quat_wxyz: ``(T, 4)`` desired right wrist quaternion (wxyz).
            root_height_override: Optional fixed Z to apply to ``root_pos``.
            z_offset: Optional additive Z offset on top of the input root Z.
            src_fps: Source FPS, accepted for legacy parity but unused —
                v2d orchestrator handles resampling upstream.
            max_chunk_tokens: Chunk length in MotionBricks tokens.
            overlap_tokens: Overlap between consecutive chunks, used for
                triangular-weight blending and AR seeding of the next chunk.
            modes: Inference modes; currently only ``"autoregressive"`` is
                produced. Kept as a tuple for legacy parity.
            smooth: Accepted for legacy parity; the bundled pipeline applies
                its own boundary blending so this flag is a no-op.
            half_stride_blend: Accepted for legacy parity; chunked AR
                already overlap-blends so this flag is a no-op.
            left_hand_root_offset_local: Local-frame offset from the V2P
                hand-root body (palm_link) to wrist_yaw_link, applied to
                ``left_ee_pos`` before canonicalization. Pass ``None`` to
                skip when the inputs are already at wrist_yaw_link.
            right_hand_root_offset_local: Right-side counterpart.
            fix_lower_body: When ``True``, run a custom AR loop that pins
                the lower-body bodies (hips/knees/ankles) to the static
                seed pose at every chunk's AR context. The model sees
                stationary legs in its history and produces upper-body
                motion consistent with that constraint.

        Returns:
            ``{"autoregressive": {"qpos": (T, 36), "chunk_kf_info": [...]}}``
            with qpos in MuJoCo xyzw-quaternion layout.
        """
        del src_fps, half_stride_blend  # parity-only kwargs

        if root_height_override is not None:
            root_pos = root_pos.copy()
            root_pos[:, 2] = root_height_override
        if z_offset != 0.0:
            root_pos = root_pos.copy()
            root_pos[:, 2] += z_offset

        # V2P wrist positions track the hand-root body (palm_link / hand_C_MC).
        # The model expects wrist_yaw_link positions, which are offset along the
        # forearm. Convert before canonicalization.
        if left_hand_root_offset_local is not None:
            left_ee_pos = apply_hand_root_to_wrist_offset(
                left_ee_pos, left_ee_quat_wxyz, left_hand_root_offset_local
            )
        if right_hand_root_offset_local is not None:
            right_ee_pos = apply_hand_root_to_wrist_offset(
                right_ee_pos, right_ee_quat_wxyz, right_hand_root_offset_local
            )

        T = root_pos.shape[0]

        # Heading canonicalization expects a body trajectory derived from
        # MuJoCo FK on a qpos. Synthesize a static-pose qpos and FK it so
        # the bundled helpers see the same coordinate convention used at
        # training time.
        seed_qpos, seed_joint_names = build_seed_qpos(
            num_frames=T, root_height=float(root_pos[0, 2])
        )
        body_pos_w, body_wxyz_w = qpos_to_body_world(
            seed_qpos, seed_joint_names, DEFAULT_SEED_XML
        )

        transforms = self._mods["transforms"]
        gt_joint_transforms, gt_root_xy = transforms._canonicalize_heading_transforms(
            body_pos_w=body_pos_w,
            body_wxyz_w=body_wxyz_w,
            root_pos_w=seed_qpos[:, :3],
            root_wxyz_w=seed_qpos[:, 3:7],
        )
        root_ee_transforms, pose_ee_transforms, _ee_marker_pos = (
            transforms._canonicalize_ee_targets(
                left_pos_w=left_ee_pos,
                left_wxyz_w=left_ee_quat_wxyz,
                right_pos_w=right_ee_pos,
                right_wxyz_w=right_ee_quat_wxyz,
                root_pos_w=seed_qpos[:, :3],
                root_wxyz_w=seed_qpos[:, 3:7],
            )
        )

        root_target_ee_key = self._bundle["root_ee_key"].endswith(
            "hand_ee_target_transforms_nonflat"
        )
        root_ee_input = root_ee_transforms if root_target_ee_key else pose_ee_transforms
        derive_pose = self._bundle["derive_pose_ee_from_pred_root"]
        pose_ee_input = None if derive_pose else pose_ee_transforms

        pipeline = self._mods["pipeline"]
        nfpt = int(self._bundle["nfpt"])
        chunk_frames = int(max_chunk_tokens) * nfpt
        half_stride = chunk_frames // 2

        def _run_pass(start_offset: int) -> dict[str, Any]:
            ee_in = root_ee_input[start_offset:]
            pose_ee_in = (
                pose_ee_input[start_offset:] if pose_ee_input is not None else None
            )
            if fix_lower_body:
                static_pass = gt_joint_transforms[start_offset:]
                return _run_chunked_lower_body_fixed(
                    pipeline,
                    self._mods["transforms"],
                    self._root_model,
                    self._pose_model,
                    root_ee_input=ee_in,
                    pose_ee_input=pose_ee_in,
                    derive_pose_ee_from_pred_root=derive_pose,
                    seed_root=gt_root_xy[start_offset:],
                    seed_pose=gt_joint_transforms[start_offset:],
                    static_pose_shared=static_pass,
                    chunk_tokens=int(max_chunk_tokens),
                    overlap_tokens=int(overlap_tokens),
                    device=self._device,
                )
            return pipeline.run_chunked_autoregressive_inference(
                self._root_model,
                self._pose_model,
                root_ee_input=ee_in,
                seed_root=gt_root_xy[start_offset:],
                seed_pose=gt_joint_transforms[start_offset:],
                pose_ee_input=pose_ee_in,
                derive_pose_ee_from_pred_root=derive_pose,
                chunk_tokens=int(max_chunk_tokens),
                overlap_tokens=int(overlap_tokens),
                chunk_num_tokens_mode="forced",
                device=self._device,
            )

        result_a = _run_pass(0)
        if half_stride >= nfpt and root_ee_input.shape[0] - half_stride >= nfpt:
            result_b = _run_pass(half_stride)
            result = _blend_two_passes(
                result_a,
                result_b,
                offset_b=half_stride,
                total_frames=root_ee_input.shape[0],
            )
        else:
            result = result_a

        qpos_36 = features_to_qpos(
            result["pred_joints"],
            result["pred_root"],
            self._kin,
            self._body_reorder,
        )

        # Damp chunk-overlap oscillation: a mild global Hamming pass first,
        # then targeted boundary smoothing with a wider window.
        if smooth:
            qpos_36 = smooth_qpos_global(qpos_36, nfpt=nfpt)
            boundaries = chunk_boundary_centers(result.get("chunk_infos") or [])
            if boundaries:
                qpos_36 = smooth_qpos_at_boundaries(qpos_36, boundaries, nfpt=nfpt)

        # Repack root quat from wxyz → xyzw to match the legacy contract.
        qpos_xyzw = qpos_36.copy()
        qpos_xyzw[:, 3:7] = qpos_36[:, [4, 5, 6, 3]]

        chunk_kf_info = result.get("chunk_infos", [])
        return {
            "autoregressive": {
                "qpos": qpos_xyzw.astype(np.float32),
                "chunk_kf_info": chunk_kf_info,
            }
        }
