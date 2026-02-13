import os

from isaaclab.envs.mdp import observations
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from robotic_grounding.assets import MOTION_ASSET_DIR
from robotic_grounding.assets.g1 import (
    G1_ACTION_SCALE,
    G1_CYLINDER_CFG,
    G1_CYLINDER_DEX_CFG,
    G1_CYLINDER_MODEL_12_CFG,
    G1_CYLINDER_MODEL_12_DEX_CFG,
    G1_CYLINDER_MODEL_12_DEX_DELAYED_CFG,
    G1_CYLINDER_MODEL_12_DEX_WAIST_CFG,
    G1_CYLINDER_MODEL_12_HANDS_DEX_DELAYED_CFG,
    G1_HAND_JOINT_NAMES,
    G1_MODEL_12_ACTION_SCALE,
    G1_MODEL_12_DEX_WAIST_ACTION_SCALE,
)
from robotic_grounding.assets.policies.grasp import (
    G1GraspPolicy,
    GraspPolicyCfg,
)
from robotic_grounding.tasks.v2p_whole_body.config.sonic.sonic_ee_env_cfg import (
    SonicEEEnvCfg,
)
from robotic_grounding.tasks.v2p_whole_body.config.sonic.sonic_env_cfg import (
    SonicEnvCfg,
)
from robotic_grounding.tasks.v2p_whole_body.mdp import observations as obs
from robotic_grounding.tasks.v2p_whole_body.mdp.actions import SONICActionCfg

# Path to G1 motion dataset
G1_MOTION_DATASET_DIR = os.path.join(MOTION_ASSET_DIR, "datasets", "g1")
G1_MOTION_DATASET_JOINT_ORDER_FILE = os.path.join(
    G1_MOTION_DATASET_DIR, "joint_order.txt"
)

# G1 joints that SONIC controls
G1_SONIC_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class G1ObservationsCfg:
    """G1-specific observation configuration for SONIC."""

    @configclass
    class G1SONICEncoderCfg(ObsGroup):
        """Observations for SONIC encoder group (tokenizer) - G1 specific.

        Note: Filters observations to 29 SONIC-controlled joints (excludes 14 hand joints).
        """

        encoder_index = ObsTerm(
            func=obs.encoder_mode,
            params={"command_name": "motion"},
        )

        command_joint_pos_multi_future = ObsTerm(
            func=obs.command_joint_pos,
            params={
                "command_name": "motion",
                "sonic_joints_only": True,
                "action_name": "joint_pos",
            },
        )

        command_joint_vel_multi_future = ObsTerm(
            func=obs.command_joint_vel,
            params={
                "command_name": "motion",
                "sonic_joints_only": True,
                "action_name": "joint_pos",
            },
        )

        padding_1 = ObsTerm(
            func=obs.encoder_padding,
            params={"dim": 17},
        )

        motion_anchor_ori_b = ObsTerm(
            func=obs.motion_anchor_ori_b,
            params={"command_name": "motion"},
        )

        padding_2 = ObsTerm(
            func=obs.encoder_padding,
            params={"dim": 1772 - 17 - 644},
        )

        concatenate_terms = True

    @configclass
    class G1SONICDecoderCfg(ObsGroup):
        """Observations for SONIC decoder group (policy) - G1 specific.

        Note: Filters observations to 29 SONIC-controlled joints (excludes 14 hand joints).
        """

        base_ang_vel = ObsTerm(
            func=observations.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        joint_pos = ObsTerm(
            func=obs.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sonic_joints_only": True,
                "action_name": "joint_pos",
            },
        )

        joint_vel = ObsTerm(
            func=obs.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sonic_joints_only": True,
                "action_name": "joint_pos",
            },
        )

        actions = ObsTerm(
            func=obs.last_action,
            params={"action_name": "joint_pos", "sonic_joints_only": True},
        )

        gravity_dir = ObsTerm(
            func=observations.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        concatenate_terms = True
        history_length = 4

    @configclass
    class G1HandPolicyCfg(ObsGroup):
        """Observations for G1 hand policy."""

        left_hand_object_transform = ObsTerm(
            func=obs.hand_object_transform,
            params={
                "frame_transform_cfg": SceneEntityCfg("left_hand_object_transform"),
                "threshold": 10.0,
            },
        )
        right_hand_object_transform = ObsTerm(
            func=obs.hand_object_transform,
            params={
                "frame_transform_cfg": SceneEntityCfg("right_hand_object_transform"),
                "threshold": 10.0,
            },
        )

    sonic_tokenizer: G1SONICEncoderCfg = G1SONICEncoderCfg()
    sonic_policy: G1SONICDecoderCfg = G1SONICDecoderCfg()
    hand_policy: G1HandPolicyCfg = G1HandPolicyCfg()


@configclass
class G1SonicEnvCfg(SonicEnvCfg):
    """Configuration for the G1 Sonic environment.

    Note: When using G1 with hands, SONIC controls 29 joints (legs, torso, arms).
    The 14 hand joints are controlled directly from commands, bypassing SONIC.
    """

    # scene_config_path must be provided via --scene_config

    def __post_init__(self) -> None:
        """Post-initialization: set robot, actions, and observations for G1 Sonic."""
        robot_mapping = {
            "g1": {"robot_cfg": G1_CYLINDER_CFG, "action_scale": G1_ACTION_SCALE},
            "g1_dex": {
                "robot_cfg": G1_CYLINDER_DEX_CFG,
                "action_scale": G1_ACTION_SCALE,
            },
            "g1_model_12": {
                "robot_cfg": G1_CYLINDER_MODEL_12_CFG,
                "action_scale": G1_MODEL_12_ACTION_SCALE,
            },
            "g1_model_12_dex": {
                "robot_cfg": G1_CYLINDER_MODEL_12_DEX_CFG,
                "action_scale": G1_MODEL_12_ACTION_SCALE,
            },
            "g1_model_12_dex_delayed": {
                "robot_cfg": G1_CYLINDER_MODEL_12_DEX_DELAYED_CFG,
                "action_scale": G1_MODEL_12_ACTION_SCALE,
            },
            "g1_model_12_hands_dex_delayed": {
                "robot_cfg": G1_CYLINDER_MODEL_12_HANDS_DEX_DELAYED_CFG,
                "action_scale": G1_MODEL_12_ACTION_SCALE,
            },
            "g1_model_12_dex_waist": {
                "robot_cfg": G1_CYLINDER_MODEL_12_DEX_WAIST_CFG,
                "action_scale": G1_MODEL_12_DEX_WAIST_ACTION_SCALE,
            },
        }

        self.scene.robot = robot_mapping["g1_model_12_hands_dex_delayed"][
            "robot_cfg"
        ].replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = robot_mapping["g1_model_12_hands_dex_delayed"][
            "action_scale"
        ]

        # G1-specific frame transformers for hand-object transforms
        self.scene.left_hand_object_transform = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/object",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_hand_palm_link"
                )
            ],
        )
        self.scene.right_hand_object_transform = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/object",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand_palm_link"
                )
            ],
        )

        # if sonic action, set sonic joint names
        if isinstance(self.actions.joint_pos, SONICActionCfg):
            self.actions.joint_pos.sonic_joint_names = G1_SONIC_JOINT_NAMES

        # Set G1-specific observation groups
        g1_obs = G1ObservationsCfg()
        self.observations.sonic_tokenizer = g1_obs.sonic_tokenizer
        self.observations.sonic_policy = g1_obs.sonic_policy
        self.observations.hand_policy = g1_obs.hand_policy

        super().__post_init__()


@configclass
class G1SonicEEEnvCfg(SonicEEEnvCfg):
    """Configuration for the G1 Sonic EE tracking environment.

    Note: When using G1 with hands, SONIC controls 29 joints (legs, torso, arms).
    The 14 hand joints are controlled directly from commands, bypassing SONIC.
    """

    # scene_config_path must be provided via --scene_config

    def __post_init__(self) -> None:
        """Post-initialization: set robot, actions, and observations for G1 Sonic EE."""
        robot_mapping = {
            "g1": {"robot_cfg": G1_CYLINDER_CFG, "action_scale": G1_ACTION_SCALE},
            "g1_dex": {
                "robot_cfg": G1_CYLINDER_DEX_CFG,
                "action_scale": G1_ACTION_SCALE,
            },
            "g1_model_12": {
                "robot_cfg": G1_CYLINDER_MODEL_12_CFG,
                "action_scale": G1_MODEL_12_ACTION_SCALE,
            },
            "g1_model_12_dex": {
                "robot_cfg": G1_CYLINDER_MODEL_12_DEX_CFG,
                "action_scale": G1_MODEL_12_ACTION_SCALE,
            },
            "g1_model_12_dex_delayed": {
                "robot_cfg": G1_CYLINDER_MODEL_12_DEX_DELAYED_CFG,
                "action_scale": G1_MODEL_12_ACTION_SCALE,
            },
            "g1_model_12_hands_dex_delayed": {
                "robot_cfg": G1_CYLINDER_MODEL_12_HANDS_DEX_DELAYED_CFG,
                "action_scale": G1_MODEL_12_ACTION_SCALE,
            },
            "g1_model_12_dex_waist": {
                "robot_cfg": G1_CYLINDER_MODEL_12_DEX_WAIST_CFG,
                "action_scale": G1_MODEL_12_DEX_WAIST_ACTION_SCALE,
            },
        }

        self.scene.robot = robot_mapping["g1_model_12_hands_dex_delayed"][
            "robot_cfg"
        ].replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = robot_mapping["g1_model_12_hands_dex_delayed"][
            "action_scale"
        ]

        # G1-specific frame transformers for hand-object transforms
        self.scene.left_hand_object_transform = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/object",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_hand_palm_link"
                )
            ],
        )
        self.scene.right_hand_object_transform = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/object",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand_palm_link"
                )
            ],
        )

        # if sonic action, set sonic joint names
        if isinstance(self.actions.joint_pos, SONICActionCfg):
            self.actions.joint_pos.sonic_joint_names = G1_SONIC_JOINT_NAMES
            self.actions.joint_pos.hand_policy_class = G1GraspPolicy
            self.actions.joint_pos.hand_policy_cfg = GraspPolicyCfg(
                asset_name="robot",
                joint_names=G1_HAND_JOINT_NAMES,
            )

        self.commands.motion.ee_link_names = [
            "left_hand_palm_link",
            "right_hand_palm_link",
        ]

        # Set G1-specific observation groups
        g1_obs = G1ObservationsCfg()
        self.observations.sonic_tokenizer = g1_obs.sonic_tokenizer
        self.observations.sonic_policy = g1_obs.sonic_policy
        self.observations.hand_policy = g1_obs.hand_policy

        # # kernel similarity reward for natural motion
        # self.rewards.natural_motion_similarity = RewTerm(
        #     func=regularization_rewards.kernel_similarity_reward,
        #     weight=0.5,
        #     params={
        #         "dataset_path": G1_MOTION_DATASET_DIR,
        #         "joint_order_file": G1_MOTION_DATASET_JOINT_ORDER_FILE,
        #         "bandwidth_list": [0.01, 0.1, 1.0],
        #         "num_expert_samples": 5000,
        #         "top_k": 500,
        #         "temperature": 10.0,
        #     },
        # )

        super().__post_init__()
