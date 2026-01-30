# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Debug environment configuration for Sharpa V2P with GUI control.

This environment provides interactive GUI controls for both the dual Sharpa hands
and the Arctic object, useful for debugging contact sensors and MDP components.
"""

import math

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robotic_grounding.tasks.debug import mdp as debug_mdp
from robotic_grounding.tasks.debug.mdp import (
    JointPositionGUIActionCfg,
    ObjectPoseGUIActionCfg,
    RewardVisualizerCfg,
)
from robotic_grounding.tasks.v2p.config.sharpa_wave.sharpa_v2p_env_cfg import (
    SharpaV2PEnvCfg,
)
from robotic_grounding.tasks.v2p.v2p_hand_env_cfg import ActionsCfg, RewardsCfg


@configclass
class DebugActionsCfg(ActionsCfg):
    """Actions configuration with GUI control for all robot joints and object pose."""

    # GUI control for ALL robot joints (wrist + finger)
    # Wrist joints (x/y/z/roll/pitch/yaw) control hand root pose
    # Finger joints control individual finger positions
    joint_pos = JointPositionGUIActionCfg(
        asset_name="robot",
        joint_names=[".*"],  # All joints
        scale=1.0,  # Will be overridden in __post_init__ with proper scale
        use_default_offset=True,
        preserve_order=True,
        max_stiffness=200.0,
        max_damping=25.0,
    )

    # Object pose GUI control (unified for both articulated and rigid objects)
    # - For articulated objects with floating base: provide position_joint_names and rotation_joint_names
    # - For rigid objects or articulations without floating base: omit joint names (uses direct root pose)
    object_pose = ObjectPoseGUIActionCfg(
        asset_name="object",
        position_limits={
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (0.5, 3.0),
        },
        rotation_limits={
            "roll": (-math.pi, math.pi),
            "pitch": (-math.pi, math.pi),
            "yaw": (-math.pi, math.pi),
        },
        # For articulated object with floating base joints:
        position_joint_names={
            "x": "base_x",
            "y": "base_y",
            "z": "base_z",
        },
        rotation_joint_names={
            "roll": "base_roll",
            "pitch": "base_pitch",
            "yaw": "base_yaw",
        },
        gui_window_title="Object Pose Controller",
    )

    # Reward visualizer GUI to monitor reward terms in real-time
    reward_visualizer = RewardVisualizerCfg(
        asset_name="robot",  # Required by ActionTermCfg but not used
        show_total_reward=True,
        show_weights=True,
        show_episode_sum=True,
        env_index=0,
        enable_history_plot=True,
        history_length=200,
        gui_window_title="Reward Monitor",
        gui_window_width=500,
        gui_window_height=450,
        update_interval=1,
    )


@configclass
class DebugRewardsCfg(RewardsCfg):
    """Minimal rewards for debugging contact sensors.

    Shows exact contact position (x, y, z) and contact force (x, y, z, magnitude)
    for the right pinky contact sensor.
    """

    # Keep alive reward
    is_alive = RewTerm(func=isaac_mdp.is_alive, weight=1.0)

    # --- Contact Position (world frame) ---
    contact_pos_x = RewTerm(
        func=debug_mdp.contact_pos,
        weight=1.0,
        params={"sensor_name": "right_pinky_contact_sensor_bottom", "axis": "x"},
    )
    contact_pos_y = RewTerm(
        func=debug_mdp.contact_pos,
        weight=1.0,
        params={"sensor_name": "right_pinky_contact_sensor_bottom", "axis": "y"},
    )
    contact_pos_z = RewTerm(
        func=debug_mdp.contact_pos,
        weight=1.0,
        params={"sensor_name": "right_pinky_contact_sensor_bottom", "axis": "z"},
    )

    # --- Contact Force (world frame, 0 when no contact) ---
    contact_force_x = RewTerm(
        func=debug_mdp.contact_force,
        weight=1.0,
        params={"sensor_name": "right_pinky_contact_sensor_bottom", "axis": "x"},
    )
    contact_force_y = RewTerm(
        func=debug_mdp.contact_force,
        weight=1.0,
        params={"sensor_name": "right_pinky_contact_sensor_bottom", "axis": "y"},
    )
    contact_force_z = RewTerm(
        func=debug_mdp.contact_force,
        weight=1.0,
        params={"sensor_name": "right_pinky_contact_sensor_bottom", "axis": "z"},
    )

    # Disable heavy reward computations for debugging
    action_rate_l2 = None
    joint_limit = None
    contact_force_penalty = None


@configclass
class DebugTerminationsCfg:
    """Minimal terminations for debugging."""

    # Only time out after very long episode for debugging
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class SharpaDebugEnvCfg(SharpaV2PEnvCfg):
    """Debug environment for Sharpa V2P with GUI control.

    This environment provides:
    - Joint GUI control for ALL robot joints with P/D gain adjustment:
      - Wrist joints (x/y/z/roll/pitch/yaw) for 6DoF hand root pose control
      - Finger joints for individual finger position control
    - Object pose GUI control for the Arctic object (6DoF via floating base)
    - Contact sensor visualization for debugging

    Useful for:
    - Verifying contact sensor setup
    - Debugging MDP observations and rewards
    - Testing robot-object interaction manually
    - Positioning hands and object to check contact forces
    """

    # Override actions with GUI-controlled versions
    actions: DebugActionsCfg = DebugActionsCfg()

    # Use minimal rewards and terminations for debugging
    rewards: DebugRewardsCfg = DebugRewardsCfg()
    terminations: DebugTerminationsCfg = DebugTerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        # Parent's __post_init__ sets DUAL_SHARPA_WAVE_ACTION_SCALE which is correct
        # for all joints (wrist + finger), so we don't override it here

        # Override initial positions for better debugging:
        # - Position hands to wrap around the object for contact testing
        # - Hands face inward toward object (palms facing object)
        self.scene.robot.init_state.joint_pos = {
            # Right hand: positioned to the right of object, rotated to face left
            "right_wrist_x": -0.15,  # Close to object
            "right_wrist_y": 0.0,
            "right_wrist_z": 1.25,
            "right_wrist_roll": -math.pi / 2,  # Palm facing left (toward object)
            "right_wrist_pitch": 0.0,
            "right_wrist_yaw": 0.0,
            # Left hand: positioned to the left of object, rotated to face right
            "left_wrist_x": 0.15,  # Close to object
            "left_wrist_y": 0.0,
            "left_wrist_z": 1.25,
            "left_wrist_roll": math.pi / 2,  # Palm facing right (toward object)
            "left_wrist_pitch": 0.0,
            "left_wrist_yaw": 0.0,
        }

        # Position object between the hands
        self.scene.object.init_state.joint_pos = {
            "base_x": -0.030,
            "base_y": 0.0,
            "base_z": 1.2,
            "base_roll": 0.0,
            "base_pitch": 0.0,
            "base_yaw": 0.7,
            "rotation": 0.0,
        }

        # Reduce number of environments for debugging (less GPU memory)
        self.scene.num_envs = 1

        # Enable contact sensor debug visualization
        for sensor_name in self.finger_sensor_names:
            sensor = getattr(self.scene, sensor_name, None)
            if sensor is not None:
                sensor.debug_vis = True

        # Extend episode length for extended debugging sessions
        self.episode_length_s = 3600.0  # 1 hour

        # Disable events that interfere with manual control
        if hasattr(self.events, "physics_material"):
            self.events.physics_material = None
        if hasattr(self.events, "reset_robot_and_object"):
            self.events.reset_robot_and_object = None

        # Viewer settings for close-up view of hands and object
        # Camera positioned to look at the manipulation area (z=1.25)
        self.viewer.eye = (0.8, 0.8, 1.6)  # Close diagonal view
        self.viewer.lookat = (0.0, 0.0, 1.25)  # Look at object/hands center
        self.viewer.origin_type = "world"
