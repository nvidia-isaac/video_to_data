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
    """
    Reset the robot to the start of the motion trajectory.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to reset.
        command_name: Name of the tracking command term.
        asset_cfg: Configuration for the robot asset. Defaults to SceneEntityCfg("robot").
        trajectory_time_index: (start, end) frame indices for trajectory segment.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    command: TrackingCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]

    # Reset timestep for these environments
    reset_ts = torch.randint(
        max(0, trajectory_time_index[0]),
        min(command.num_timesteps - 1, trajectory_time_index[1]),
        (len(env_ids),),
        dtype=torch.int32,
        device=env.device,
    )
    command.timestep[env_ids] = reset_ts
    command.reset_timestep[env_ids] = (
        reset_ts  # Track starting point for progress normalization
    )

    # Get the corresponding frame of motion data
    initial_root_pos = command.root_pos_w[command.timestep[env_ids]]
    initial_root_quat = command.root_quat_w[command.timestep[env_ids]]
    initial_joint_pos = command.joint_pos[command.timestep[env_ids]]

    # Apply env origins offset to root position
    root_pos_w = initial_root_pos + env.scene.env_origins[env_ids]

    # Set root pose and velocity
    robot.write_root_pose_to_sim(
        torch.cat([root_pos_w, initial_root_quat], dim=-1), env_ids=env_ids
    )
    robot.write_root_velocity_to_sim(
        torch.zeros_like(robot.data.root_vel_w[env_ids]), env_ids=env_ids
    )

    # Set joint positions and velocities
    robot.write_joint_state_to_sim(
        initial_joint_pos,
        torch.zeros_like(robot.data.joint_vel[env_ids]),
        env_ids=env_ids,
    )

    # Reset object to the corresponding frame in trajectory
    object: RigidObject = env.scene["object"]
    initial_object_pos = command._object_pos_w[command.timestep[env_ids]]
    initial_object_quat = command._object_quat_w[command.timestep[env_ids]]
    object_pos_w = initial_object_pos + env.scene.env_origins[env_ids]

    object.write_root_pose_to_sim(
        torch.cat([object_pos_w, initial_object_quat], dim=-1), env_ids=env_ids
    )
    object.write_root_velocity_to_sim(
        torch.zeros_like(object.data.root_vel_w[env_ids]), env_ids=env_ids
    )
