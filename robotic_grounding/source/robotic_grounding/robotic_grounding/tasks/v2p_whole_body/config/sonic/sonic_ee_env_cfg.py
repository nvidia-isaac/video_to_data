"""SONIC EE-focused environment configuration.

This module extends the base V2P EE-focused configuration with SONIC-specific
action and observation configurations for end-effector tracking.
"""

from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass

from robotic_grounding.assets import POLICY_ASSET_DIR
from robotic_grounding.tasks.v2p_whole_body.base_ee_env_cfg import (
    EEObservationsCfg as BaseEEObservationsCfg,
)
from robotic_grounding.tasks.v2p_whole_body.base_ee_env_cfg import (
    EERewardsCfg,
    EETerminationsCfg,
    V2PEEEnvCfg,
)
from robotic_grounding.tasks.v2p_whole_body.mdp.actions import (
    SONICActionCfg,
    SONICActionType,
)

POLICY_DIR = f"{POLICY_ASSET_DIR}/sonic"


@configclass
class SonicEEActionsCfg:
    """SONIC EE-focused action configuration with hand policy."""

    joint_pos = SONICActionCfg(
        action_type=SONICActionType.LATENT_HAND_POLICY,
        policy_dir=POLICY_DIR,
        asset_name="robot",
        joint_names=[".*"],
        sonic_joint_names=MISSING,
        command_name="motion",
        use_default_offset=True,
        hand_policy_class=MISSING,
        hand_policy_cfg=MISSING,
    )


@configclass
class SonicEEObservationsCfg(BaseEEObservationsCfg):
    """SONIC EE observation configuration.

    Extends base EE observations with SONIC tokenizer and policy observation groups.
    These groups are populated by robot-specific configs.
    """

    # SONIC tokenizer observations
    sonic_tokenizer: ObsGroup = MISSING

    # SONIC policy observations
    sonic_policy: ObsGroup = MISSING

    # Hand policy observations (optional)
    hand_policy: ObsGroup | None = None


@configclass
class SonicEEEnvCfg(V2PEEEnvCfg):
    """SONIC EE-focused environment configuration.

    Uses end-effector tracking instead of joint tracking. Robot-specific
    configs should:
    1. Set the robot articulation config
    2. Set sonic_joint_names and hand_policy in actions
    3. Set ee_link_names in commands
    4. Populate sonic_tokenizer and sonic_policy observation groups
    """

    actions: SonicEEActionsCfg = SonicEEActionsCfg()
    observations: SonicEEObservationsCfg = SonicEEObservationsCfg()
    rewards: EERewardsCfg = EERewardsCfg()
    terminations: EETerminationsCfg = EETerminationsCfg()
