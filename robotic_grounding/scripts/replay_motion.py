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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac / torch imports after AppLauncher ---------------------------------
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from robotic_grounding.tasks.scene_utils.replay_data import (  # noqa: E402
    DualHandTrajectory,
    SingleRobotTrajectory,
    load_replay_trajectory,
)
from robotic_grounding.tasks.scene_utils.scene_viewer_env_cfg import (  # noqa: E402
    SceneViewerEnvCfg,
)
from scipy.spatial.transform import Rotation as R  # noqa: E402

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
    """Env cfg for kinematic trajectory replay."""

    def __post_init__(self) -> None:
        """Post-init: build viewer scene, then apply replay-only overrides."""
        super().__post_init__()

        # Replay: never resolve robot-object or robot-fixed collisions.
        self.events.setup_collision_groups.params["disable_robot_to_object_collisions"] = (
            True
        )
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
    ) -> None:
        """Load trajectory arrays from Parquet into GPU tensors.

        Args:
            motion_file: Path to Parquet partition directory.
            device: Target torch device.
            spawn_root_z: Initial root Z from the spawned robot cfg used to
                calibrate the height offset so the robot stands on the ground.
        """
        replay = load_replay_trajectory(motion_file)
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
            self.object_root_wxyz = torch.empty(0, 4, dtype=torch.float32, device=device)

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
        right_sim_joints = torch.zeros(num_envs, len(right_sim_joint_names), device=device)
        right_sim_joints[:, right_reorder] = right_joints
    else:
        right_sim_joints = right_joints
    right_robot.write_joint_state_to_sim(
        right_sim_joints,
        torch.zeros_like(right_sim_joints),
    )

    left_joints = traj.left_finger_joints[t].unsqueeze(0).expand(num_envs, -1)
    if left_reorder is not None:
        left_sim_joints = torch.zeros(num_envs, len(left_sim_joint_names), device=device)
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
    traj = Trajectory(args_cli.motion_file, device, spawn_root_z=spawn_root_z)

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
        left_reorder = _build_joint_reorder(
            traj.left_joint_names, left_sim_joint_names
        )

    num_envs = env.num_envs
    env_origins = env.scene.env_origins  # (num_envs, 3)

    actions = torch.zeros(num_envs, 0, device=device)
    frame_f = 0.0
    frame_step = cfg.sim.dt * traj.fps * args_cli.speed
    loop = not args_cli.no_loop

    print(f"[INFO] Replaying {traj.num_frames} frames at {traj.fps:.0f} fps "
          f"(speed={args_cli.speed}x, loop={loop})")
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

        env.step(actions)

        frame_f += frame_step
        if t > 0 and t % 100 == 0:
            secs = t / traj.fps if traj.fps else 0
            print(f"[INFO] Frame {t}/{traj.num_frames} ({secs:.1f}s)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
