import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def hand_object_transform(
    env: ManagerBasedRLEnv, frame_transform_cfg: SceneEntityCfg, threshold: float = 0.10
) -> torch.Tensor:
    """
    Get the transform between the object and the hand.

    Args:
        env: The environment instance
        frame_transform_cfg: The frame transformer configuration
        threshold: The threshold for the distance between the object and the hand
    Returns:
        Transform (num_envs, num_bodies, 7)
    """
    frame_transform = env.scene[frame_transform_cfg.name]
    pos_source = frame_transform.data.target_pos_source
    quat_source = frame_transform.data.target_quat_source
    transform = torch.cat([pos_source, quat_source], dim=-1)

    # check if distance between object and hand is less than threshold
    distance = torch.norm(frame_transform.data.target_pos_source, dim=-1)
    distance_mask = (distance < threshold).unsqueeze(-1)
    filtered_transform = torch.where(
        distance_mask, transform, torch.zeros_like(transform)
    )

    return filtered_transform.reshape(env.num_envs, -1)


def contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Get contact force from a contact sensor.

    Args:
        env: The environment instance
        sensor_cfg: The contact sensor configuration
    Returns:
        Contact force (num_envs, num_contact_points)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    mean_contact_forces = contact_sensor.data.force_matrix_w_history.mean(dim=1)

    return mean_contact_forces.reshape(env.num_envs, -1)
