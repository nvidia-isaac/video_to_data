# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Kinematic playback of retargeted motion data in Isaac Lab.

Supports both whole-body (G1) and dual floating-hand (Sharpa/Dex3) schemas.
Robot and object are teleported each sim step — no physics forces act on them.
Object collision geometry is disabled to avoid interpenetration artifacts from
the retargeted IK solution.  Playback loops by default; use ``--no-loop`` to
stop at the last frame.

If the motion_v1 parquet carries per-side wrist positions + binary
``{left,right}_hand_contact_active`` masks, a colored sphere (red=left,
green=right) is drawn at the corresponding wrist while the mask is on.

Usage:
    python scripts/replay_motion.py \
        --motion_file source/robotic_grounding/robotic_grounding/assets/human_motion_data/nvhuman_g1_processed/sequence_id=<seq>/robot_name=g1

    python scripts/replay_motion.py --motion_file <path> --speed 0.5
    python scripts/replay_motion.py --motion_file <path> --no-loop
    python scripts/replay_motion.py --motion_file <path> --headless
"""

import argparse
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay retargeted motion from Parquet.")
parser.add_argument(
    "--motion_file", type=str, required=True, help="Parquet partition dir."
)
parser.add_argument(
    "--speed", type=float, default=1.0, help="Playback speed multiplier."
)
parser.add_argument(
    "--no-loop",
    action="store_true",
    help="Stop at the last frame instead of looping (default: loop).",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--start_frame",
    type=int,
    default=0,
    help="First frame of the source motion to play (inclusive).",
)
parser.add_argument(
    "--end_frame",
    type=int,
    default=None,
    help="One past the last frame to play. Default: end of sequence.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac / torch imports after AppLauncher ---------------------------------
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from robotic_grounding.motion_schema import load_motion_data_parquet  # noqa: E402
from robotic_grounding.tasks.scene_utils.replay_data import (  # noqa: E402
    DualHandTrajectory,
    SingleRobotTrajectory,
    load_replay_trajectory,
)
from robotic_grounding.tasks.scene_utils.scene_viewer_env_cfg import (  # noqa: E402
    SceneViewerEnvCfg,
)
from scipy.spatial.transform import Rotation as R  # noqa: E402

# Foot-contact proxy: in kinematic replay there's no physics contact force,
# so "in contact" is a simple height check on the ankle-roll link. When the
# G1 foot is flat, left/right_ankle_roll_link sits ≈ 0.037 m above the
# ground (the LL_FOOT/LR_FOOT frames are defined at ``0.04 0 -0.037`` below
# their parent ankle-roll). 0.06 m gives a small tolerance for retarget noise
# without falsely triggering during swing.
FOOT_CONTACT_Z_THRESHOLD = 0.06


# =============================================================================
# Env configuration — kinematic replay variant
# =============================================================================


def _disable_gravity_in_articulation_cfg(cfg: Any) -> Any:
    """Return articulation cfg with gravity disabled in rigid properties."""
    spawn = getattr(cfg, "spawn", None)
    rigid_props = getattr(spawn, "rigid_props", None)
    if spawn is None or rigid_props is None:
        return cfg
    return cfg.replace(
        spawn=spawn.replace(
            rigid_props=rigid_props.replace(disable_gravity=True),
        )
    )


@configclass
class ReplayEnvCfg(SceneViewerEnvCfg):
    """Env cfg for kinematic trajectory replay.

    Note: This config disables collisions and gravity on all replayed bodies
    because the robot and object are teleported each step from the parquet
    trajectory (no physics forces act on them). Any contact-based rewards or
    termination signals would not be meaningful in this env; use a physics
    env (e.g. ManagerBasedRLEnv) for those.
    """

    def __post_init__(self) -> None:
        """Post-init: build viewer scene, then apply replay-only overrides."""
        super().__post_init__()

        # Replay: never resolve robot-object or robot-fixed collisions.
        self.events.setup_collision_groups.params[
            "disable_robot_to_object_collisions"
        ] = True
        self.events.setup_collision_groups.params[
            "disable_robot_to_fixed_object_collisions"
        ] = True

        # Replay object(s): collision OFF + kinematic + gravity OFF.
        object_names = self.events.setup_collision_groups.params.get("object_names", [])
        for name in object_names:
            obj_cfg = getattr(self.scene, name, None)
            if obj_cfg is None:
                continue
            spawn = getattr(obj_cfg, "spawn", None)
            rigid_props = getattr(spawn, "rigid_props", None)
            collision_props = getattr(spawn, "collision_props", None)
            if spawn is None or rigid_props is None:
                continue
            new_rigid_props = rigid_props.replace(
                disable_gravity=True,
                kinematic_enabled=True,
            )
            if collision_props is not None:
                spawn = spawn.replace(
                    rigid_props=new_rigid_props,
                    collision_props=collision_props.replace(collision_enabled=False),
                )
            else:
                spawn = spawn.replace(rigid_props=new_rigid_props)
            setattr(self.scene, name, obj_cfg.replace(spawn=spawn))

        # Replay robot(s): disable gravity for static kinematic playback.
        if hasattr(self.scene, "robot"):
            self.scene.robot = _disable_gravity_in_articulation_cfg(self.scene.robot)
        if hasattr(self.scene, "right_robot"):
            self.scene.right_robot = _disable_gravity_in_articulation_cfg(
                self.scene.right_robot
            )
        if hasattr(self.scene, "left_robot"):
            self.scene.left_robot = _disable_gravity_in_articulation_cfg(
                self.scene.left_robot
            )


# =============================================================================
# Trajectory loading
# =============================================================================


def _get_spawn_root_z(env_cfg: ReplayEnvCfg) -> float:
    """Read spawn root Z from the robot articulation cfg's initial state.

    Falls back to 0.0 for dual-hand layouts (floating wrist, no world-frame root).
    """
    robot_cfg = getattr(env_cfg.scene, "robot", None)
    if robot_cfg is not None:
        init_state = getattr(robot_cfg, "init_state", None)
        if init_state is not None:
            pos = getattr(init_state, "pos", None)
            if pos is not None and len(pos) >= 3:
                return float(pos[2])
    return 0.0


class Trajectory:
    """Load replay trajectory tensors for single-robot or dual-hand layouts."""

    def __init__(
        self,
        motion_file: str,
        device: torch.device,
        spawn_root_z: float = 0.0,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> None:
        """Load trajectory arrays from Parquet into GPU tensors.

        Args:
            motion_file: Path to Parquet partition directory.
            device: Target torch device.
            spawn_root_z: Initial root Z from the spawned robot cfg used to
                calibrate the height offset so the robot stands on the ground.
            start_frame: First frame to keep (motion_v1 only).
            end_frame: One past the last frame to keep. None = full.
        """
        replay = load_replay_trajectory(
            motion_file, start_frame=start_frame, end_frame=end_frame
        )
        self.fps = replay.fps
        self.num_frames = replay.num_frames
        self.robot_layout = replay.robot_layout

        if replay.object_traj is not None:
            self.object_root_pos = torch.tensor(
                replay.object_traj.root_position,
                dtype=torch.float32,
                device=device,
            )
            self.object_root_wxyz = torch.tensor(
                replay.object_traj.root_wxyz,
                dtype=torch.float32,
                device=device,
            )
        else:
            self.object_root_pos = torch.empty(0, 3, dtype=torch.float32, device=device)
            self.object_root_wxyz = torch.empty(
                0, 4, dtype=torch.float32, device=device
            )

        if isinstance(replay, SingleRobotTrajectory):
            self.robot_joint_names = list(replay.robot_joint_names)
            self.robot_root_pos = torch.tensor(
                replay.robot_root_position, dtype=torch.float32, device=device
            )
            self.robot_root_wxyz = torch.tensor(
                replay.robot_root_wxyz, dtype=torch.float32, device=device
            )
            self.robot_joint_pos = torch.tensor(
                replay.robot_joint_positions, dtype=torch.float32, device=device
            )

            first_root_z = float(self.robot_root_pos[0, 2].item())
            # Auto-detect legacy vs grounded retarget output:
            # - Legacy data had root Z near 0 and needs a global lift.
            # - New retargeted data is already world/ground-aligned and must
            #   keep robot/object in the same frame (no replay-time offset).
            legacy_root_z_threshold = 0.3
            if first_root_z < legacy_root_z_threshold:
                self.height_offset = spawn_root_z - first_root_z
                self.robot_root_pos[:, 2] += self.height_offset
                if self.object_root_pos.shape[0] > 0:
                    self.object_root_pos[:, 2] += self.height_offset
            else:
                self.height_offset = 0.0
            return

        if isinstance(replay, DualHandTrajectory):
            self.right_joint_names = list(replay.right_joint_names)
            self.left_joint_names = list(replay.left_joint_names)
            self.right_wrist_pos = torch.tensor(
                replay.right_wrist_position, dtype=torch.float32, device=device
            )
            self.left_wrist_pos = torch.tensor(
                replay.left_wrist_position, dtype=torch.float32, device=device
            )
            if replay.wrist_orientation_format == "wxyz":
                self.right_wrist_wxyz = torch.tensor(
                    replay.right_wrist_orientation, dtype=torch.float32, device=device
                )
                self.left_wrist_wxyz = torch.tensor(
                    replay.left_wrist_orientation, dtype=torch.float32, device=device
                )
            else:
                right_wxyz = [
                    R.from_euler("XYZ", xyz, degrees=False).as_quat(scalar_first=True)
                    for xyz in replay.right_wrist_orientation
                ]
                left_wxyz = [
                    R.from_euler("XYZ", xyz, degrees=False).as_quat(scalar_first=True)
                    for xyz in replay.left_wrist_orientation
                ]
                self.right_wrist_wxyz = torch.tensor(
                    right_wxyz, dtype=torch.float32, device=device
                )
                self.left_wrist_wxyz = torch.tensor(
                    left_wxyz, dtype=torch.float32, device=device
                )
            self.right_finger_joints = torch.tensor(
                replay.right_finger_joints, dtype=torch.float32, device=device
            )
            self.left_finger_joints = torch.tensor(
                replay.left_finger_joints, dtype=torch.float32, device=device
            )
            self.height_offset = 0.0
            return

        raise ValueError(f"Unsupported replay trajectory type: {type(replay).__name__}")


def _build_joint_reorder(
    parquet_names: list[str], sim_names: list[str]
) -> torch.Tensor | None:
    """Map from Parquet joint order → Isaac joint order. None if already identical."""
    if parquet_names == sim_names:
        return None
    sim_name_to_idx = {n: i for i, n in enumerate(sim_names)}
    indices: list[int] = []
    for pq_name in parquet_names:
        if pq_name not in sim_name_to_idx:
            raise ValueError(
                f"Parquet joint '{pq_name}' not found in spawned robot joints: "
                f"{sim_names}"
            )
        indices.append(sim_name_to_idx[pq_name])
    return torch.tensor(indices, dtype=torch.long)


def _write_single_robot_frame(
    robot: Any,
    traj: Trajectory,
    t: int,
    env_origins: torch.Tensor,
    num_envs: int,
    device: torch.device,
    sim_joint_names: list[str],
    reorder: torch.Tensor | None,
) -> None:
    root_pos = traj.robot_root_pos[t].unsqueeze(0).expand(num_envs, -1)
    root_wxyz = traj.robot_root_wxyz[t].unsqueeze(0).expand(num_envs, -1)
    root_pos_w = root_pos + env_origins
    root_pose = torch.cat([root_pos_w, root_wxyz], dim=-1)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device))

    joint_pos_t = traj.robot_joint_pos[t].unsqueeze(0).expand(num_envs, -1)
    if reorder is not None:
        sim_jpos = torch.zeros(num_envs, len(sim_joint_names), device=device)
        sim_jpos[:, reorder] = joint_pos_t
    else:
        sim_jpos = joint_pos_t
    robot.write_joint_state_to_sim(sim_jpos, torch.zeros_like(sim_jpos))


def _write_dual_hand_frame(
    right_robot: Any,
    left_robot: Any,
    traj: Trajectory,
    t: int,
    env_origins: torch.Tensor,
    num_envs: int,
    device: torch.device,
    right_sim_joint_names: list[str],
    left_sim_joint_names: list[str],
    right_reorder: torch.Tensor | None,
    left_reorder: torch.Tensor | None,
) -> None:
    right_pos = traj.right_wrist_pos[t].unsqueeze(0).expand(num_envs, -1) + env_origins
    right_wxyz = traj.right_wrist_wxyz[t].unsqueeze(0).expand(num_envs, -1)
    right_pose = torch.cat([right_pos, right_wxyz], dim=-1)
    right_robot.write_root_pose_to_sim(right_pose)
    right_robot.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device))

    left_pos = traj.left_wrist_pos[t].unsqueeze(0).expand(num_envs, -1) + env_origins
    left_wxyz = traj.left_wrist_wxyz[t].unsqueeze(0).expand(num_envs, -1)
    left_pose = torch.cat([left_pos, left_wxyz], dim=-1)
    left_robot.write_root_pose_to_sim(left_pose)
    left_robot.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device))

    right_joints = traj.right_finger_joints[t].unsqueeze(0).expand(num_envs, -1)
    if right_reorder is not None:
        right_sim_joints = torch.zeros(
            num_envs, len(right_sim_joint_names), device=device
        )
        right_sim_joints[:, right_reorder] = right_joints
    else:
        right_sim_joints = right_joints
    right_robot.write_joint_state_to_sim(
        right_sim_joints,
        torch.zeros_like(right_sim_joints),
    )

    left_joints = traj.left_finger_joints[t].unsqueeze(0).expand(num_envs, -1)
    if left_reorder is not None:
        left_sim_joints = torch.zeros(
            num_envs, len(left_sim_joint_names), device=device
        )
        left_sim_joints[:, left_reorder] = left_joints
    else:
        left_sim_joints = left_joints
    left_robot.write_joint_state_to_sim(
        left_sim_joints,
        torch.zeros_like(left_sim_joints),
    )


def _write_object_frame(
    obj: Any | None,
    traj: Trajectory,
    t: int,
    env_origins: torch.Tensor,
    num_envs: int,
    device: torch.device,
) -> None:
    if obj is None or t >= traj.object_root_pos.shape[0]:
        return
    obj_pos = traj.object_root_pos[t].unsqueeze(0).expand(num_envs, -1)
    obj_wxyz = traj.object_root_wxyz[t].unsqueeze(0).expand(num_envs, -1)
    obj_pos_w = obj_pos + env_origins
    obj_pose = torch.cat([obj_pos_w, obj_wxyz], dim=-1)
    obj.write_root_pose_to_sim(obj_pose)
    obj.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device))


# =============================================================================
# Contact-mask visualization
# =============================================================================


# Replay-time anchor for contact markers. `replay_data.load_replay_trajectory`
# deliberately drops per-side wrist + contact_active tensors, so we re-read
# them directly from the motion_v1 parquet via `load_motion_data_parquet`. This
# keeps the primary replay loader lean and single-purpose while allowing the
# script to overlay optional diagnostics.
class ContactOverlay:
    """Per-side wrist positions + binary contact mask, ready for per-frame draw.

    Attributes are set for every side that has both a wrist trajectory and a
    contact mask on disk. If either is missing for a side, `has_left`/`has_right`
    stays False and the marker for that side will never be shown.
    """

    def __init__(
        self,
        motion_file: str,
        device: torch.device,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> None:
        """Load contact overlay data, or no-op if the parquet lacks the fields."""
        self.has_left: bool = False
        self.has_right: bool = False
        self.left_wrist_pos: torch.Tensor | None = None
        self.right_wrist_pos: torch.Tensor | None = None
        self.left_active: torch.Tensor | None = None
        self.right_active: torch.Tensor | None = None

        try:
            md = load_motion_data_parquet(motion_file, device=str(device))
            md = md.trim(start_frame, end_frame)
        except Exception as exc:
            print(f"[WARN] Contact overlay disabled (motion_v1 load failed): {exc}")
            return

        for side in ("left", "right"):
            wrist = getattr(md, f"{side}_wrist_position", None)
            active = getattr(md, f"{side}_hand_contact_active", None)
            if wrist is None or active is None:
                continue
            # Align masks to the shorter of the two in case of drift.
            n = min(wrist.shape[0], active.shape[0])
            setattr(self, f"{side}_wrist_pos", wrist[:n].to(device).float())
            setattr(self, f"{side}_active", active[:n].to(device).float())
            setattr(self, f"has_{side}", True)

        if not (self.has_left or self.has_right):
            print(
                "[INFO] No per-side wrist + contact_active in motion file; "
                "skipping contact overlay."
            )


def _sphere_marker_cfg(
    prim_path: str,
    rgb: tuple[float, float, float],
    radius: float = 0.03,
) -> VisualizationMarkersCfg:
    """Colored sphere marker at a fixed radius (default ≈ 3 cm)."""
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
            ),
        },
    )


def _draw_contact_marker(
    marker: VisualizationMarkers | None,
    is_active: bool,
    wrist_pos_t: torch.Tensor,
    env_origins: torch.Tensor,
) -> None:
    """Place `marker` at the wrist, toggling visibility from `is_active`.

    We keep the sphere anchored at the wrist regardless of state so that the
    first `is_active=True` frame doesn't momentarily flash at (0, 0, 0) before
    USD picks up the new transform. Visibility alone drives on/off.
    """
    if marker is None:
        return
    translations = (
        wrist_pos_t.unsqueeze(0).expand(env_origins.shape[0], -1) + env_origins
    )
    marker.visualize(translations=translations)
    marker.set_visibility(bool(is_active))


def _find_body_idx(body_names: list[str], candidates: tuple[str, ...]) -> int | None:
    """Return the first body index matching one of ``candidates``, else None.

    Lets callers probe for a palm link that may be named differently across
    URDF variants (``*_hand_palm_link`` on the dex-hand G1, ``*_wrist_yaw_link``
    on the no-hand variants).
    """
    for name in candidates:
        if name in body_names:
            return body_names.index(name)
    return None


def _frame_marker_cfg(prim_path: str, scale: float) -> VisualizationMarkersCfg:
    """RGB xyz-axis frame marker (same asset used by the tracking command).

    Scale is uniform — FRAME_MARKER_CFG's ``markers["frame"].scale`` is a
    ``(sx, sy, sz)`` tuple. We replicate to keep axes equal-length.
    """
    cfg = FRAME_MARKER_CFG.replace(prim_path=prim_path)
    cfg.markers["frame"].scale = (scale, scale, scale)
    return cfg


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run kinematic trajectory replay."""
    cfg = ReplayEnvCfg(motion_file=args_cli.motion_file)
    cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedEnv(cfg=cfg)
    env.reset()
    device = env.device

    spawn_root_z = _get_spawn_root_z(cfg)
    traj = Trajectory(
        args_cli.motion_file,
        device,
        spawn_root_z=spawn_root_z,
        start_frame=args_cli.start_frame,
        end_frame=args_cli.end_frame,
    )

    # Resolve object handle by scene-config name (not hardcoded "object")
    object_names = cfg.events.setup_collision_groups.params.get("object_names", [])
    obj = env.scene[object_names[0]] if object_names else None

    # Resolve robot handle(s) and reorder maps based on replay layout.
    robot = None
    right_robot = None
    left_robot = None
    sim_joint_names = []
    right_sim_joint_names = []
    left_sim_joint_names = []
    reorder = None
    right_reorder = None
    left_reorder = None
    if traj.robot_layout == "single_robot":
        robot = env.scene["robot"]
        sim_joint_names = list(robot.joint_names)
        reorder = _build_joint_reorder(traj.robot_joint_names, sim_joint_names)
    else:
        right_robot = env.scene["right_robot"]
        left_robot = env.scene["left_robot"]
        right_sim_joint_names = list(right_robot.joint_names)
        left_sim_joint_names = list(left_robot.joint_names)
        right_reorder = _build_joint_reorder(
            traj.right_joint_names, right_sim_joint_names
        )
        left_reorder = _build_joint_reorder(traj.left_joint_names, left_sim_joint_names)

    num_envs = env.num_envs
    env_origins = env.scene.env_origins  # (num_envs, 3)

    # Contact overlay (motion_v1 parquets only). Anchors at each wrist; shows
    # while the per-frame contact-active mask is on.
    contact_overlay = ContactOverlay(
        args_cli.motion_file,
        device,
        start_frame=args_cli.start_frame,
        end_frame=args_cli.end_frame,
    )
    if traj.height_offset != 0.0:
        # Apply the same Z lift the replay loader used so markers sit on the
        # actual kinematic wrists (legacy whole-body data only).
        if contact_overlay.has_left and contact_overlay.left_wrist_pos is not None:
            contact_overlay.left_wrist_pos = contact_overlay.left_wrist_pos.clone()
            contact_overlay.left_wrist_pos[:, 2] += traj.height_offset
        if contact_overlay.has_right and contact_overlay.right_wrist_pos is not None:
            contact_overlay.right_wrist_pos = contact_overlay.right_wrist_pos.clone()
            contact_overlay.right_wrist_pos[:, 2] += traj.height_offset
    left_contact_marker: VisualizationMarkers | None = None
    right_contact_marker: VisualizationMarkers | None = None
    if contact_overlay.has_left:
        left_contact_marker = VisualizationMarkers(
            _sphere_marker_cfg(
                "/Visuals/Replay/left_hand_contact", rgb=(0.95, 0.25, 0.25)
            )
        )
        left_contact_marker.set_visibility(False)
    if contact_overlay.has_right:
        right_contact_marker = VisualizationMarkers(
            _sphere_marker_cfg(
                "/Visuals/Replay/right_hand_contact", rgb=(0.25, 0.85, 0.35)
            )
        )
        right_contact_marker.set_visibility(False)

    # Anchor frame markers (xyz axes, same asset the tracking command uses).
    # Palm markers read from ``robot.data.{body_pos_w, body_quat_w}`` so they
    # need an index resolved against the spawned articulation; if no palm-like
    # body is found for a side, that side's marker is skipped. For dual-hand
    # layouts we fall back to the wrist-root pose (which *is* the palm for the
    # floating hand robots).
    root_marker: VisualizationMarkers | None = None
    left_palm_marker: VisualizationMarkers | None = None
    right_palm_marker: VisualizationMarkers | None = None
    object_marker: VisualizationMarkers | None = None
    left_palm_idx: int | None = None
    right_palm_idx: int | None = None

    if traj.robot_layout == "single_robot":
        assert robot is not None
        root_marker = VisualizationMarkers(
            _frame_marker_cfg("/Visuals/Replay/robot_root", scale=0.20)
        )
        body_names = list(robot.body_names)
        left_palm_idx = _find_body_idx(
            body_names,
            ("left_hand_palm_link", "left_wrist_yaw_link", "left_wrist_roll_link"),
        )
        right_palm_idx = _find_body_idx(
            body_names,
            ("right_hand_palm_link", "right_wrist_yaw_link", "right_wrist_roll_link"),
        )
        if left_palm_idx is None:
            print("[WARN] No palm-like body found for left arm; marker disabled.")
        if right_palm_idx is None:
            print("[WARN] No palm-like body found for right arm; marker disabled.")

    if (traj.robot_layout == "single_robot" and left_palm_idx is not None) or (
        traj.robot_layout == "dual_hand"
    ):
        left_palm_marker = VisualizationMarkers(
            _frame_marker_cfg("/Visuals/Replay/left_palm", scale=0.10)
        )
    if (traj.robot_layout == "single_robot" and right_palm_idx is not None) or (
        traj.robot_layout == "dual_hand"
    ):
        right_palm_marker = VisualizationMarkers(
            _frame_marker_cfg("/Visuals/Replay/right_palm", scale=0.10)
        )

    if obj is not None and traj.object_root_pos.shape[0] > 0:
        object_marker = VisualizationMarkers(
            _frame_marker_cfg("/Visuals/Replay/object_root", scale=0.15)
        )

    # Foot-contact markers: small green spheres anchored to the ankle-roll
    # links; visibility toggles with the Z-threshold proxy defined above.
    # Single-robot only — dual-hand layouts have no legs.
    left_foot_marker: VisualizationMarkers | None = None
    right_foot_marker: VisualizationMarkers | None = None
    left_ankle_idx: int | None = None
    right_ankle_idx: int | None = None
    if traj.robot_layout == "single_robot":
        assert robot is not None
        body_names = list(robot.body_names)
        left_ankle_idx = _find_body_idx(body_names, ("left_ankle_roll_link",))
        right_ankle_idx = _find_body_idx(body_names, ("right_ankle_roll_link",))
        if left_ankle_idx is not None:
            left_foot_marker = VisualizationMarkers(
                _sphere_marker_cfg(
                    "/Visuals/Replay/left_foot_contact",
                    rgb=(0.2, 1.0, 0.3),
                    radius=0.02,
                )
            )
            left_foot_marker.set_visibility(False)
        if right_ankle_idx is not None:
            right_foot_marker = VisualizationMarkers(
                _sphere_marker_cfg(
                    "/Visuals/Replay/right_foot_contact",
                    rgb=(0.2, 1.0, 0.3),
                    radius=0.02,
                )
            )
            right_foot_marker.set_visibility(False)

    actions = torch.zeros(num_envs, 0, device=device)
    frame_f = 0.0
    frame_step = cfg.sim.dt * traj.fps * args_cli.speed
    loop = not args_cli.no_loop

    print(
        f"[INFO] Replaying {traj.num_frames} frames at {traj.fps:.0f} fps "
        f"(speed={args_cli.speed}x, loop={loop})"
    )
    if contact_overlay.has_left or contact_overlay.has_right:
        sides = [s for s in ("left", "right") if getattr(contact_overlay, f"has_{s}")]
        print(f"[INFO] Contact overlay enabled for: {', '.join(sides)}")
    print("[INFO] Close the window to exit.")

    while simulation_app.is_running():
        frame_i = int(frame_f)
        if frame_i >= traj.num_frames:
            if loop:
                frame_f -= traj.num_frames
                frame_i = int(frame_f)
            else:
                break
        t = frame_i

        if traj.robot_layout == "single_robot":
            _write_single_robot_frame(
                robot=robot,
                traj=traj,
                t=t,
                env_origins=env_origins,
                num_envs=num_envs,
                device=device,
                sim_joint_names=sim_joint_names,
                reorder=reorder,
            )
        else:
            _write_dual_hand_frame(
                right_robot=right_robot,
                left_robot=left_robot,
                traj=traj,
                t=t,
                env_origins=env_origins,
                num_envs=num_envs,
                device=device,
                right_sim_joint_names=right_sim_joint_names,
                left_sim_joint_names=left_sim_joint_names,
                right_reorder=right_reorder,
                left_reorder=left_reorder,
            )

        _write_object_frame(
            obj=obj,
            traj=traj,
            t=t,
            env_origins=env_origins,
            num_envs=num_envs,
            device=device,
        )

        if contact_overlay.has_left:
            assert contact_overlay.left_wrist_pos is not None
            assert contact_overlay.left_active is not None
            tl = min(t, contact_overlay.left_wrist_pos.shape[0] - 1)
            _draw_contact_marker(
                left_contact_marker,
                bool(contact_overlay.left_active[tl].item() > 0.5),
                contact_overlay.left_wrist_pos[tl],
                env_origins,
            )
        if contact_overlay.has_right:
            assert contact_overlay.right_wrist_pos is not None
            assert contact_overlay.right_active is not None
            tr = min(t, contact_overlay.right_wrist_pos.shape[0] - 1)
            _draw_contact_marker(
                right_contact_marker,
                bool(contact_overlay.right_active[tr].item() > 0.5),
                contact_overlay.right_wrist_pos[tr],
                env_origins,
            )

        env.step(actions)

        # Anchor frame markers: placed *after* env.step so body_pos_w /
        # body_quat_w reflect the joint state we just wrote (palms need FK,
        # which is refreshed during env.step). Root / object markers use
        # trajectory values directly; either timing works for those but we
        # keep them grouped for clarity.
        if root_marker is not None and traj.robot_layout == "single_robot":
            pos = traj.robot_root_pos[t].unsqueeze(0).expand(num_envs, -1) + env_origins
            quat = traj.robot_root_wxyz[t].unsqueeze(0).expand(num_envs, -1)
            root_marker.visualize(translations=pos, orientations=quat)

        if traj.robot_layout == "single_robot":
            assert robot is not None
            if left_palm_marker is not None and left_palm_idx is not None:
                left_palm_marker.visualize(
                    translations=robot.data.body_pos_w[:, left_palm_idx, :],
                    orientations=robot.data.body_quat_w[:, left_palm_idx, :],
                )
            if right_palm_marker is not None and right_palm_idx is not None:
                right_palm_marker.visualize(
                    translations=robot.data.body_pos_w[:, right_palm_idx, :],
                    orientations=robot.data.body_quat_w[:, right_palm_idx, :],
                )
        else:
            if left_palm_marker is not None:
                left_palm_marker.visualize(
                    translations=(
                        traj.left_wrist_pos[t].unsqueeze(0).expand(num_envs, -1)
                        + env_origins
                    ),
                    orientations=traj.left_wrist_wxyz[t]
                    .unsqueeze(0)
                    .expand(num_envs, -1),
                )
            if right_palm_marker is not None:
                right_palm_marker.visualize(
                    translations=(
                        traj.right_wrist_pos[t].unsqueeze(0).expand(num_envs, -1)
                        + env_origins
                    ),
                    orientations=traj.right_wrist_wxyz[t]
                    .unsqueeze(0)
                    .expand(num_envs, -1),
                )

        if object_marker is not None and t < traj.object_root_pos.shape[0]:
            pos = (
                traj.object_root_pos[t].unsqueeze(0).expand(num_envs, -1) + env_origins
            )
            quat = traj.object_root_wxyz[t].unsqueeze(0).expand(num_envs, -1)
            object_marker.visualize(translations=pos, orientations=quat)

        # Foot-contact markers: anchor at each ankle-roll link; visibility
        # toggles on when the ankle-roll Z (above env origin) is below the
        # contact threshold. All envs share the same kinematic trajectory, so
        # env-0's Z is sufficient to drive the single USD visibility flag.
        if left_foot_marker is not None and left_ankle_idx is not None:
            assert robot is not None
            ankle_pos = robot.data.body_pos_w[:, left_ankle_idx, :]
            left_foot_marker.visualize(translations=ankle_pos)
            z_rel = (ankle_pos[0, 2] - env_origins[0, 2]).item()
            left_foot_marker.set_visibility(z_rel < FOOT_CONTACT_Z_THRESHOLD)
        if right_foot_marker is not None and right_ankle_idx is not None:
            assert robot is not None
            ankle_pos = robot.data.body_pos_w[:, right_ankle_idx, :]
            right_foot_marker.visualize(translations=ankle_pos)
            z_rel = (ankle_pos[0, 2] - env_origins[0, 2]).item()
            right_foot_marker.set_visibility(z_rel < FOOT_CONTACT_Z_THRESHOLD)

        frame_f += frame_step
        if t > 0 and t % 100 == 0:
            secs = t / traj.fps if traj.fps else 0
            print(f"[INFO] Frame {t}/{traj.num_frames} ({secs:.1f}s)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
