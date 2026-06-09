# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Viser-based inspection tool for motion_v1 parquets and live retargeting.

Two entry points share one scene graph:

- ``ViserPlayback(motion_file=...)`` loads a parquet and exposes a Frame
  slider + Play/Pause/FPS/Loop/Step controls. Used by
  ``scripts/replay_viser.py``.
- ``ViserPlayback.for_live_retarget(server, pin_model, ...)`` attaches to an
  already-created ``viser.ViserServer``, exposing only the per-frame draw
  surface. Used by ``scripts/retarget/soma_to_g1.py`` to stream
  in-progress frames.

The shared scene primitives (object mesh + ``/object``/``/head``/``/root``
frames + per-side contact spheres) live in private builders on the class so
both modes render the same handle paths. Parquet replay has no body mesh on
disk, so the body overlay is parquet-mode's one missing piece; live retarget
feeds it through ``LiveFrameState.body_vertices``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

from robotic_grounding.motion_schema import MotionData, load_motion_data_parquet
from robotic_grounding.retarget.pinocchio_viser_visualizer import ViserVisualizer
from robotic_grounding.retarget.robot_config import load_robot_config
from robotic_grounding.retarget.whole_body_kinematics import (
    ConfigDrivenWholeBodyKinematics,
)

# Silence the aborted-handshake tracebacks the `websockets` layer inside
# viser logs at INFO/WARNING when a browser tab reconnects mid-session or
# when a connection is dropped before the handshake finishes. They are
# cosmetic (viser reopens the connection and the scene re-syncs on the
# next heartbeat), but they look scary in the retarget/playback terminal.
# Bump each logger to ERROR so only genuine faults surface.
for _name in ("websockets", "websockets.server", "websockets.asyncio.server"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# NOTE: We deliberately do NOT import from
# ``robotic_grounding.tasks.scene_utils.replay_data``: that package's
# ``__init__.py`` pulls in ``apply_scene_config`` which imports Isaac Lab,
# and Isaac Lab refuses to load without a running ``omni.physics``. Viser
# playback is a lightweight tool that must work in plain Python envs, so we
# normalize ``MotionData`` inline here.

# Repo root used to resolve repo-relative mesh paths carried in the parquet.
# Mirrors soma_to_g1.py's REPO_ROOT so both agree on the anchor.
_REPO_ROOT = Path(__file__).resolve().parents[4]


# =============================================================================
# Per-frame state for live (non-parquet) callers
# =============================================================================


@dataclass
class LiveFrameState:
    """Per-frame pose bundle for ``ViserPlayback.display(...)``.

    Every field is optional. Only the fields provided get written to the
    scene; absent fields leave the corresponding handle untouched. This lets
    retargeters grow their overlay incrementally without touching the
    module.
    """

    # Pinocchio configuration vector. Must align with the model the playback
    # was constructed against (free-flyer base + joints in model order).
    q: np.ndarray | None = None

    # Object pose (world frame).
    object_pos: np.ndarray | None = None
    object_wxyz: np.ndarray | None = None

    # Anchor frames — useful during retargeting for head/root debug.
    head_pos: np.ndarray | None = None
    head_wxyz: np.ndarray | None = None
    root_pos: np.ndarray | None = None
    root_wxyz: np.ndarray | None = None

    # Per-side contact marker state. ``contact_wrists[i]`` is a length-3
    # xyz; ``contact_active[i]`` is a float in [0, 1] (threshold 0.5). The
    # ordering follows ``hand_sides`` passed at construction time.
    contact_wrists: list[np.ndarray] | None = None
    contact_active: list[float] | None = None

    # Optional body mesh overlay. Vertices expected in world frame with the
    # same ground alignment as the robot. Currently unused by the active
    # SOMA retarget path; kept as a forward-compatible field.
    body_vertices: np.ndarray | None = None

    # Optional IK-target frames for retargeter debug. Keys are frame names
    # used to namespace the handle under ``/targets/<name>``.
    ik_target_poses: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )


# =============================================================================
# Helpers
# =============================================================================


def _to_np(value: Any) -> np.ndarray | None:
    """Coerce torch tensors / sequences to ndarray; passthrough None."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    try:
        return np.asarray(value)
    except (TypeError, ValueError):
        return None


def _axis_angle_to_wxyz(aa: np.ndarray) -> np.ndarray:
    """Convert (T, 3) axis-angle rotations to (T, 4) wxyz quaternions."""
    return np.asarray(
        [R.from_rotvec(v).as_quat(scalar_first=True) for v in aa],
        dtype=np.float32,
    )


def _wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    """Reorder a (4,) wxyz quaternion to (4,) xyzw for Pinocchio's q layout."""
    return np.asarray([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64)


def _reorder_to_pinocchio(
    parquet_joint_names: list[str],
    pin_joint_names: list[str],
) -> np.ndarray | None:
    """Index array mapping parquet joint order to Pinocchio's model order.

    Returns ``None`` when the orders already match, letting the caller skip
    the permutation step.
    """
    if list(parquet_joint_names) == list(pin_joint_names):
        return None
    name_to_idx = {n: i for i, n in enumerate(pin_joint_names)}
    missing = [n for n in parquet_joint_names if n not in name_to_idx]
    if missing:
        raise ValueError(
            f"Parquet joint(s) {missing!r} not present in Pinocchio model; "
            f"model has {len(pin_joint_names)} joints."
        )
    return np.array([name_to_idx[n] for n in parquet_joint_names], dtype=np.int64)


def _resolve_object_mesh_path(md: MotionData) -> str | None:
    """Resolve the first object mesh path from ``MotionData`` to an absolute path.

    Parquets since the motion_v1 refactor store repo-relative paths; older
    files may store absolute ones. We handle both. Returns ``None`` when no
    mesh is listed.
    """
    if not md.object_mesh_paths:
        return None
    raw = md.object_mesh_paths[0]
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return str(p)


@dataclass
class _ParquetArrays:
    """Normalized per-frame arrays plucked from a ``MotionData`` at load time.

    Replaces the ``replay_data._motion_v1_to_replay`` adapter so this module
    stays free of the Isaac-Lab-dependent ``tasks.scene_utils`` package.
    """

    layout: str  # "single_robot" | "dual_hand"
    num_frames: int
    fps: float

    # Single-robot fields (None when dual-hand).
    robot_root_position: np.ndarray | None = None
    robot_root_wxyz: np.ndarray | None = None
    robot_joint_names: list[str] = field(default_factory=list)
    robot_joint_positions: np.ndarray | None = None

    # Dual-hand fields (None when single-robot).
    left_wrist_position: np.ndarray | None = None
    left_wrist_wxyz: np.ndarray | None = None
    right_wrist_position: np.ndarray | None = None
    right_wrist_wxyz: np.ndarray | None = None

    # Object root trajectory (either layout).
    object_root_position: np.ndarray | None = None
    object_root_wxyz: np.ndarray | None = None


def _parse_motion_data(md: MotionData) -> _ParquetArrays:
    """Normalize ``MotionData`` into ndarray slices used by parquet replay.

    Duplicates the minimal logic from
    ``tasks.scene_utils.replay_data._motion_v1_to_replay`` but stops short
    of importing that module (see file header for the rationale).
    """
    root_pos = _to_np(md.robot_root_position)
    root_wxyz = _to_np(md.robot_root_wxyz)
    joint_pos = _to_np(md.robot_joint_positions)

    obj_root_pos = _to_np(md.object_root_position)
    obj_root_aa = _to_np(md.object_root_axis_angle)
    obj_root_wxyz: np.ndarray | None = None
    if (
        obj_root_pos is not None
        and obj_root_aa is not None
        and obj_root_pos.ndim == 2
        and obj_root_pos.shape[1] == 3
        and obj_root_aa.ndim == 2
        and obj_root_aa.shape[1] == 3
    ):
        obj_root_wxyz = _axis_angle_to_wxyz(obj_root_aa.astype(np.float64))
        obj_root_pos = obj_root_pos.astype(np.float32)
    else:
        obj_root_pos = None

    has_joints = (
        bool(md.robot_joint_names)
        and joint_pos is not None
        and joint_pos.size > 0
        and root_pos is not None
        and root_wxyz is not None
    )
    if has_joints:
        assert root_pos is not None
        assert root_wxyz is not None
        assert joint_pos is not None
        return _ParquetArrays(
            layout="single_robot",
            num_frames=int(root_pos.shape[0]),
            fps=float(md.fps),
            robot_root_position=root_pos.astype(np.float32),
            robot_root_wxyz=root_wxyz.astype(np.float32),
            robot_joint_names=list(md.robot_joint_names),
            robot_joint_positions=joint_pos.astype(np.float32),
            object_root_position=obj_root_pos,
            object_root_wxyz=obj_root_wxyz,
        )

    # Dual-hand: derive wrist pose from ee_pose_w + ee_link_names.
    ee_pose = _to_np(md.ee_pose_w)
    if ee_pose is None or ee_pose.ndim != 3 or ee_pose.shape[1] < 2:
        raise ValueError(
            "motion_v1 file has no whole-body joint state and fewer than 2 EEs; "
            "cannot build a viser playback trajectory."
        )
    left_idx, right_idx = 0, 1
    for i, name in enumerate(md.ee_link_names or []):
        lname = (name or "").lower()
        if "left" in lname:
            left_idx = i
        elif "right" in lname:
            right_idx = i
    ee_pose_np = ee_pose.astype(np.float32)
    return _ParquetArrays(
        layout="dual_hand",
        num_frames=int(ee_pose_np.shape[0]),
        fps=float(md.fps),
        left_wrist_position=ee_pose_np[:, left_idx, 0:3],
        left_wrist_wxyz=ee_pose_np[:, left_idx, 3:7],
        right_wrist_position=ee_pose_np[:, right_idx, 0:3],
        right_wrist_wxyz=ee_pose_np[:, right_idx, 3:7],
        object_root_position=obj_root_pos,
        object_root_wxyz=obj_root_wxyz,
    )


# =============================================================================
# Main class
# =============================================================================


class ViserPlayback:
    """Shared viser scene graph with two drivers: parquet trajectory + live stream.

    Use ``ViserPlayback(motion_file=...)`` for parquet replay; call ``run()``
    to enter the GUI-driven tick loop. Use
    ``ViserPlayback.for_live_retarget(...)`` to reuse the scene primitives
    inside a retargeter; drive with ``display(LiveFrameState(...))`` per
    frame and skip ``run()``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        motion_file: str,
        port: int = 8080,
        device: str = "cpu",
        start_paused: bool = False,
        start_frame: int = 0,
    ) -> None:
        """Open viser on ``port`` and pre-build the scene from the parquet.

        Args:
            motion_file: Path to a motion_v1 parquet partition directory or file.
            port: Viser HTTP port. Default 8080 matches the retargeter.
            device: torch device passed to the motion reader.
            start_paused: Open the viewer paused so the user can scrub.
            start_frame: Initial frame shown after the scene is built.
        """
        self._mode: str = "parquet"
        self._server = viser.ViserServer(host="0.0.0.0", port=port)

        self._md: MotionData = load_motion_data_parquet(motion_file, device=device)
        self._parquet: _ParquetArrays | None = _parse_motion_data(self._md)
        self._num_frames: int = int(self._parquet.num_frames)
        self._fps: float = float(self._parquet.fps) or 30.0

        # Robot / wrist handles
        self._pin_viz: ViserVisualizer | None = None
        self._pin_model: pin.Model | None = None
        self._pin_joint_names: list[str] = []
        self._reorder: np.ndarray | None = None
        self._left_wrist_frame: Any = None
        self._right_wrist_frame: Any = None

        # Static scene handles, built lazily below.
        self._object_frame: Any = None
        self._object_mesh: Any = None
        self._head_frame: Any = None
        self._root_frame: Any = None
        self._contact_handles: dict[str, Any] = {}
        self._ik_target_handles: dict[str, Any] = {}
        self._body_mesh_handle: Any = None

        # Parquet-mode GUI handles, populated by ``_build_gui``.
        self._gui_frame: Any = None
        self._gui_play: Any = None
        self._gui_fps: Any = None
        self._gui_loop: Any = None

        # Parquet-mode: infer side list from hand_sides (fall back to both).
        self._hand_sides: list[str] = list(self._md.hand_sides or ["left", "right"])

        self._build_object_scene(_resolve_object_mesh_path(self._md))
        self._build_anchor_frames()
        self._build_contact_markers(self._hand_sides)

        if self._parquet.layout == "single_robot":
            self._build_pinocchio_visualizer_for_parquet()
        else:
            self._build_dual_hand_wrist_frames()

        self._build_gui(
            start_paused=start_paused,
            start_frame=max(0, min(int(start_frame), self._num_frames - 1)),
        )

        # Initial draw so the scene isn't blank until first play tick.
        assert self._gui_frame is not None  # just built by _build_gui
        self.set_frame(int(self._gui_frame.value))

    @classmethod
    def for_live_retarget(
        cls,
        server: viser.ViserServer,
        pin_model: pin.Model,
        pin_visual_model: Any = None,
        pin_collision_model: Any = None,
        object_mesh_path: str | None = None,
        hand_sides: tuple[str, ...] = ("left", "right"),
    ) -> "ViserPlayback":
        """Construct a no-parquet instance that shares the scene graph.

        The caller owns the ``viser.ViserServer`` and drives ``display(...)``
        each retargeted frame. No trajectory, no GUI slider, no tick loop.

        Args:
            server: Externally-created viser server.
            pin_model: Pinocchio kinodynamic model (free-flyer base expected).
            pin_visual_model: Optional pinocchio visual geometry model.
            pin_collision_model: Optional pinocchio collision geometry model.
            object_mesh_path: Optional object mesh file; a frame is still
                created even when the mesh is absent so ``display`` can
                write the pose without a lazy-add.
            hand_sides: Sides to pre-build contact marker handles for.
        """
        self = cls.__new__(cls)
        self._mode = "live"
        self._server = server

        self._md = MotionData()  # placeholder; live mode reads nothing from it
        self._parquet = None
        self._num_frames = 0
        self._fps = 0.0

        self._pin_model = pin_model
        self._pin_joint_names = [str(n) for n in pin_model.names]
        self._pin_viz = ViserVisualizer(
            viser_server=server,
            model=pin_model,
            visual_model=pin_visual_model,
            collision_model=pin_collision_model,
        )
        self._reorder = None
        self._left_wrist_frame = None
        self._right_wrist_frame = None

        self._object_frame = None
        self._object_mesh = None
        self._head_frame = None
        self._root_frame = None
        self._contact_handles = {}
        self._ik_target_handles = {}
        self._body_mesh_handle = None
        self._hand_sides = list(hand_sides)

        self._gui_frame = None
        self._gui_play = None
        self._gui_fps = None
        self._gui_loop = None

        self._build_object_scene(object_mesh_path)
        self._build_anchor_frames()
        self._build_contact_markers(self._hand_sides)
        return self

    # ------------------------------------------------------------------
    # Private scene builders
    # ------------------------------------------------------------------

    def _build_object_scene(self, mesh_path: str | None) -> None:
        """Create the ``/object`` frame and load the mesh when available."""
        self._object_frame = self._server.scene.add_frame(
            "/object",
            wxyz=(1, 0, 0, 0),
            position=(0, 0, 0),
            axes_length=0.018,
            axes_radius=0.0008,
        )
        if mesh_path is None:
            return
        p = Path(mesh_path)
        if not p.is_file():
            print(f"[viser_playback] object mesh not found at {p}; skipping.")
            return
        mesh = trimesh.load(str(p))
        self._object_mesh = self._server.scene.add_mesh_trimesh(
            "/object/mesh", mesh, position=(0, 0, 0), wxyz=(1, 0, 0, 0)
        )

    def _build_anchor_frames(self) -> None:
        """Create ``/head`` and ``/root`` anchor frames (empty/origin-seeded)."""
        self._head_frame = self._server.scene.add_frame(
            "/head",
            wxyz=(1, 0, 0, 0),
            position=(0, 0, 0),
            axes_length=0.1,
            axes_radius=0.002,
        )
        self._root_frame = self._server.scene.add_frame(
            "/root",
            wxyz=(1, 0, 0, 0),
            position=(0, 0, 0),
            axes_length=0.1,
            axes_radius=0.002,
        )

    def _build_contact_markers(self, sides: list[str]) -> None:
        """Add hidden per-side contact spheres under ``/contacts/{side}``."""
        colors = {
            "left": np.array([230, 60, 60]),
            "right": np.array([60, 200, 80]),
        }
        for side in sides:
            handle = self._server.scene.add_icosphere(
                name=f"/contacts/{side}",
                position=np.array([0.0, 0.0, 0.0]),
                radius=0.04,
                color=colors.get(side, np.array([200, 200, 200])),
            )
            handle.visible = False
            self._contact_handles[side] = handle

    def _build_pinocchio_visualizer_for_parquet(self) -> None:
        """Load the G1 whole-body Pinocchio model and wrap in ViserVisualizer.

        Only runs for single-robot parquets. Uses the canonical
        ``ConfigDrivenWholeBodyKinematics`` (the same kinematics backend
        ``soma_to_g1.py`` runs) loaded from the in-repo G1 config so
        replay matches retarget. Only parquets whose ``source_dataset``
        matches the active G1 config's source (currently ``soma``)
        replay through this path; parquets produced against other
        source schemas are rejected here rather than silently replayed
        against a mismatched joint mapping.
        """
        # Hard guard for mismatched parquets. The SOMA G1 config expects
        # a specific joint layout; parquets produced from a different
        # source schema have a different joint count + ordering and
        # would either crash in ``_reorder_to_pinocchio`` below or
        # silently render with joints in the wrong positions. Fail fast
        # with a clear message instead.
        src = (self._md.source_dataset or "").strip().lower()
        if src and src != "soma":
            raise ValueError(
                f"viser_playback: cannot replay parquet with "
                f"source_dataset={src!r}; only 'soma' is supported. "
                f"Re-retarget via scripts/retarget/soma_to_g1.py or "
                f"point at a soma-produced partition."
            )
        config = load_robot_config("g1")
        kin = ConfigDrivenWholeBodyKinematics(config=config)
        self._pin_model = kin.robot.model
        self._pin_joint_names = [str(n) for n in self._pin_model.names]
        self._pin_viz = ViserVisualizer(
            viser_server=self._server,
            model=kin.robot.model,
            visual_model=kin.robot.visual_model,
            collision_model=kin.robot.collision_model,
        )
        assert self._parquet is not None and self._parquet.layout == "single_robot"
        parquet_joint_names = list(self._parquet.robot_joint_names)
        # Drop the first 2 entries in pin_joint_names (universe + free-flyer)
        # so names align with the movable-joint portion of q.
        movable_pin_joint_names = self._pin_joint_names[2:]
        self._reorder = _reorder_to_pinocchio(
            parquet_joint_names, movable_pin_joint_names
        )

    def _build_dual_hand_wrist_frames(self) -> None:
        """Dual-hand replay: two wrist frames, no articulation in this pass."""
        self._left_wrist_frame = self._server.scene.add_frame(
            "/left_wrist",
            wxyz=(1, 0, 0, 0),
            position=(0, 0, 0),
            axes_length=0.08,
            axes_radius=0.003,
        )
        self._right_wrist_frame = self._server.scene.add_frame(
            "/right_wrist",
            wxyz=(1, 0, 0, 0),
            position=(0, 0, 0),
            axes_length=0.08,
            axes_radius=0.003,
        )

    def _build_gui(self, start_paused: bool, start_frame: int) -> None:
        """Playback folder: Frame slider + transport + FPS + Loop + Step + Reset."""
        with self._server.gui.add_folder("Playback"):
            self._gui_frame = self._server.gui.add_slider(
                "Frame",
                min=0,
                max=max(self._num_frames - 1, 0),
                step=1,
                initial_value=start_frame,
            )
            self._gui_play = self._server.gui.add_checkbox(
                "Play", initial_value=not start_paused
            )
            fps_max = max(int(round(self._fps)), 120)
            self._gui_fps = self._server.gui.add_slider(
                "FPS",
                min=1,
                max=fps_max,
                step=1,
                initial_value=int(np.clip(round(self._fps), 1, fps_max)),
            )
            self._gui_loop = self._server.gui.add_checkbox("Loop", initial_value=True)
            step_back = self._server.gui.add_button("Step -1")
            step_fwd = self._server.gui.add_button("Step +1")
            reset = self._server.gui.add_button("Reset")

        @self._gui_frame.on_update
        def _(_event: Any) -> None:
            # Slider scrub: redraw immediately. Scrubbing while Play is on is
            # allowed; the tick loop will advance from the new position on
            # its next iteration.
            self.set_frame(int(self._gui_frame.value))

        @step_back.on_click
        def _(_event: Any) -> None:
            self._gui_frame.value = max(int(self._gui_frame.value) - 1, 0)

        @step_fwd.on_click
        def _(_event: Any) -> None:
            self._gui_frame.value = min(
                int(self._gui_frame.value) + 1, self._num_frames - 1
            )

        @reset.on_click
        def _(_event: Any) -> None:
            self._gui_frame.value = 0

    # ------------------------------------------------------------------
    # Per-frame rendering
    # ------------------------------------------------------------------

    def set_frame(self, t: int) -> None:
        """Parquet-mode draw: render frame ``t`` of the loaded trajectory."""
        if self._mode != "parquet":
            raise RuntimeError(
                "set_frame is parquet-mode only; use display(LiveFrameState) "
                "with ViserPlayback.for_live_retarget."
            )
        t = int(np.clip(t, 0, self._num_frames - 1))

        assert self._parquet is not None
        if self._parquet.layout == "single_robot":
            self._draw_single_robot(t)
        else:
            self._draw_dual_hand(t)

        self._draw_parquet_object(t)
        self._draw_parquet_contacts(t)

    def _draw_single_robot(self, t: int) -> None:
        assert self._parquet is not None
        assert self._pin_viz is not None
        arr = self._parquet
        assert arr.robot_root_position is not None
        assert arr.robot_root_wxyz is not None
        assert arr.robot_joint_positions is not None
        root_pos = np.asarray(arr.robot_root_position[t], dtype=np.float64)
        root_wxyz = np.asarray(arr.robot_root_wxyz[t], dtype=np.float64)
        joints_parquet = np.asarray(arr.robot_joint_positions[t], dtype=np.float64)
        if self._reorder is not None:
            movable_count = len(self._pin_joint_names) - 2
            joints_pin = np.zeros(movable_count, dtype=np.float64)
            joints_pin[self._reorder] = joints_parquet
        else:
            joints_pin = joints_parquet
        q = np.concatenate([root_pos, _wxyz_to_xyzw(root_wxyz), joints_pin])
        self._pin_viz.display(q)

    def _draw_dual_hand(self, t: int) -> None:
        assert self._parquet is not None
        arr = self._parquet
        if (
            self._left_wrist_frame is not None
            and arr.left_wrist_position is not None
            and arr.left_wrist_wxyz is not None
        ):
            self._left_wrist_frame.position = np.asarray(
                arr.left_wrist_position[t], dtype=np.float64
            )
            self._left_wrist_frame.wxyz = np.asarray(
                arr.left_wrist_wxyz[t], dtype=np.float64
            )
        if (
            self._right_wrist_frame is not None
            and arr.right_wrist_position is not None
            and arr.right_wrist_wxyz is not None
        ):
            self._right_wrist_frame.position = np.asarray(
                arr.right_wrist_position[t], dtype=np.float64
            )
            self._right_wrist_frame.wxyz = np.asarray(
                arr.right_wrist_wxyz[t], dtype=np.float64
            )

    def _draw_parquet_object(self, t: int) -> None:
        if self._parquet is None or self._object_frame is None:
            return
        pos = self._parquet.object_root_position
        wxyz = self._parquet.object_root_wxyz
        if pos is None or wxyz is None:
            return
        self._object_frame.position = np.asarray(pos[t], dtype=np.float64)
        self._object_frame.wxyz = np.asarray(wxyz[t], dtype=np.float64)

    def _draw_parquet_contacts(self, t: int) -> None:
        for side in self._hand_sides:
            handle = self._contact_handles.get(side)
            if handle is None:
                continue
            wrist = getattr(self._md, f"{side}_wrist_position", None)
            active = getattr(self._md, f"{side}_hand_contact_active", None)
            if wrist is None or active is None:
                handle.visible = False
                continue
            tt = min(t, wrist.shape[0] - 1)
            handle.position = np.asarray(wrist[tt].cpu().numpy(), dtype=np.float64)
            handle.visible = float(active[tt]) > 0.5

    def display(
        self,
        state: LiveFrameState,
        body_model: Any | None = None,
    ) -> None:
        """Live-mode draw: apply whatever fields ``state`` provides.

        Args:
            state: Per-frame pose bundle (all fields optional).
            body_model: Optional source body model (e.g. a
                ``retarget.read_soma.SOMA`` instance) used to draw the human
                overlay from ``state.body_vertices``. It must expose
                ``visualize(server, vertices=...)``. Passing it from the
                caller avoids re-loading the body model here; the retargeter
                already holds one. When omitted, the overlay is skipped.
        """
        if state.q is not None and self._pin_viz is not None:
            self._pin_viz.display(np.asarray(state.q, dtype=np.float64))

        if state.object_pos is not None and self._object_frame is not None:
            self._object_frame.position = np.asarray(state.object_pos, dtype=np.float64)
        if state.object_wxyz is not None and self._object_frame is not None:
            self._object_frame.wxyz = np.asarray(state.object_wxyz, dtype=np.float64)

        if state.head_pos is not None and self._head_frame is not None:
            self._head_frame.position = np.asarray(state.head_pos, dtype=np.float64)
        if state.head_wxyz is not None and self._head_frame is not None:
            self._head_frame.wxyz = np.asarray(state.head_wxyz, dtype=np.float64)

        if state.root_pos is not None and self._root_frame is not None:
            self._root_frame.position = np.asarray(state.root_pos, dtype=np.float64)
        if state.root_wxyz is not None and self._root_frame is not None:
            self._root_frame.wxyz = np.asarray(state.root_wxyz, dtype=np.float64)

        if state.contact_wrists is not None and state.contact_active is not None:
            for side_idx, side in enumerate(self._hand_sides):
                handle = self._contact_handles.get(side)
                if handle is None or side_idx >= len(state.contact_wrists):
                    continue
                wrist_xyz = state.contact_wrists[side_idx]
                handle.position = np.asarray(wrist_xyz, dtype=np.float64)
                active = (
                    state.contact_active[side_idx]
                    if side_idx < len(state.contact_active)
                    else 0.0
                )
                handle.visible = float(active) > 0.5

        if state.body_vertices is not None and body_model is not None:
            body_model.visualize(self._server, vertices=state.body_vertices)

        if state.ik_target_poses:
            self._update_ik_target_frames(state.ik_target_poses)

    def _update_ik_target_frames(
        self,
        poses: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Lazily create and update ``/targets/<name>`` frames."""
        for name, (pos, wxyz) in poses.items():
            handle = self._ik_target_handles.get(name)
            if handle is None:
                handle = self._server.scene.add_frame(
                    f"/targets/{name}",
                    position=np.asarray(pos, dtype=np.float64),
                    wxyz=np.asarray(wxyz, dtype=np.float64),
                    axes_length=0.05,
                    axes_radius=0.003,
                )
                self._ik_target_handles[name] = handle
            else:
                handle.position = np.asarray(pos, dtype=np.float64)
                handle.wxyz = np.asarray(wxyz, dtype=np.float64)

    # ------------------------------------------------------------------
    # Parquet-mode tick loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Block on the playback tick loop until the process is killed.

        Advances the ``Frame`` slider in real time when ``Play`` is on.
        Scrubbing the slider during playback seeks without pausing; hitting
        the end stops (or wraps, with ``Loop`` checked).
        """
        if self._mode != "parquet":
            raise RuntimeError("run() is parquet-mode only.")
        if self._num_frames <= 0:
            return
        assert self._gui_frame is not None
        assert self._gui_play is not None
        assert self._gui_fps is not None
        assert self._gui_loop is not None

        last_wall = time.time()
        try:
            while True:
                time.sleep(1.0 / 120.0)
                if not self._gui_play.value:
                    last_wall = time.time()
                    continue
                now = time.time()
                dt = max(now - last_wall, 0.0)
                last_wall = now
                new_val = float(self._gui_frame.value) + dt * float(self._gui_fps.value)
                if new_val >= self._num_frames:
                    if self._gui_loop.value:
                        new_val = new_val % self._num_frames
                    else:
                        new_val = float(self._num_frames - 1)
                        self._gui_play.value = False
                # Assigning to the slider fires its on_update -> set_frame.
                self._gui_frame.value = int(new_val)
        except KeyboardInterrupt:
            print("[viser_playback] exiting on Ctrl+C")

    # Helpful for callers that want to keep the server alive without using
    # the tick loop (e.g. retargeters that close the process themselves).
    @property
    def server(self) -> viser.ViserServer:
        """The underlying ``viser.ViserServer`` instance."""
        return self._server
