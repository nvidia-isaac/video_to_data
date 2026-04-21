import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from robotic_grounding.tasks.v2p_whole_body.mdp.commands import TrackingCommand


def reset_robot_to_trajectory_start(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg | None = None,
    trajectory_time_index: tuple[int, int] = (0, 0),
) -> None:
    """Reset robot and object to a frame in the motion trajectory.

    Frame selection:
    - always_reset_to_first_frame: force frame 0 (eval mode)
    - Otherwise: random within trajectory_time_index range

    Post-processing (all configurable on TrackingCommandCfg):
    - Optional root Z clamp (reset_root_height_min)
    - Optional yaw-only root quaternion (reset_yaw_only)
    - Optional shoulder spread + finger zeroing during freeze (reset_shoulder_spread)
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    command: TrackingCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]

    # --- Frame selection ---
    if command.cfg.always_reset_to_first_frame:
        reset_ts = torch.zeros(len(env_ids), dtype=torch.int32, device=env.device)
    else:
        low = max(0, trajectory_time_index[0])
        high = min(command.num_timesteps - 1, trajectory_time_index[1])
        reset_ts = torch.randint(
            low,
            max(low + 1, high),
            (len(env_ids),),
            dtype=torch.int32,
            device=env.device,
        )

    command.timestep[env_ids] = reset_ts
    command.reset_timestep[env_ids] = reset_ts

    # --- Read trajectory frame ---
    initial_root_pos = command.root_pos_w[reset_ts].clone()
    initial_root_quat = command.root_quat_w[reset_ts].clone()
    initial_joint_pos = command.joint_pos[reset_ts].clone()

    # --- Root Z clamp ---
    if command.cfg.reset_root_height_min is not None:
        initial_root_pos[:, 2] = initial_root_pos[:, 2].clamp(
            min=command.cfg.reset_root_height_min
        )

    # --- Yaw-only root quaternion ---
    if command.cfg.reset_yaw_only:
        w = initial_root_quat[:, 0]
        x = initial_root_quat[:, 1]
        y = initial_root_quat[:, 2]
        z = initial_root_quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        half_yaw = yaw * 0.5
        initial_root_quat = torch.stack(
            [
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
            ],
            dim=-1,
        )

    # --- Shoulder spread + finger zeroing ---
    freeze_steps = command.cfg.reset_freeze_steps
    shoulder_spread = command.cfg.reset_shoulder_spread
    if shoulder_spread > 0.0 and freeze_steps > 0:
        spread_offset = torch.zeros_like(initial_joint_pos)
        joint_names = robot.joint_names
        for i, name in enumerate(joint_names):
            if name == "left_shoulder_yaw_joint":
                spread_offset[:, i] = shoulder_spread
            elif name == "right_shoulder_yaw_joint":
                spread_offset[:, i] = -shoulder_spread
            elif any(
                finger in name
                for finger in ("thumb", "index", "middle", "ring", "pinky")
            ):
                spread_offset[:, i] = -initial_joint_pos[:, i]
        initial_joint_pos = initial_joint_pos + spread_offset
        command._spread_joint_offset[env_ids] = spread_offset
    else:
        command._spread_joint_offset[env_ids] = 0.0

    # --- Write to sim ---
    root_pos_w = initial_root_pos + env.scene.env_origins[env_ids]
    robot.write_root_pose_to_sim(
        torch.cat([root_pos_w, initial_root_quat], dim=-1), env_ids=env_ids
    )
    robot.write_root_velocity_to_sim(
        torch.zeros_like(robot.data.root_vel_w[env_ids]), env_ids=env_ids
    )
    robot.write_joint_state_to_sim(
        initial_joint_pos,
        torch.zeros_like(robot.data.joint_vel[env_ids]),
        env_ids=env_ids,
    )

    # --- Reset object ---
    scene_object = env.scene[command.cfg.object_name]
    if isinstance(scene_object, RigidObject):
        obj_pos = command._object_pos_w[reset_ts] + env.scene.env_origins[env_ids]
        obj_quat = command._object_quat_w[reset_ts]
        scene_object.write_root_pose_to_sim(
            torch.cat([obj_pos, obj_quat], dim=-1), env_ids=env_ids
        )
        scene_object.write_root_velocity_to_sim(
            torch.zeros_like(scene_object.data.root_vel_w[env_ids]), env_ids=env_ids
        )
    # TODO: generalize for ArticulatedObject (Arctic multi-body)
