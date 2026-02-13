"""SONIC controller environment configuration.

This module extends the base V2P whole-body configuration with SONIC-specific
action and observation configurations.
"""

from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass

from robotic_grounding.assets import POLICY_ASSET_DIR
from robotic_grounding.tasks.v2p_whole_body.base_env_cfg import (
    BaseObservationsCfg,
    V2PEnvCfg,
)
from robotic_grounding.tasks.v2p_whole_body.mdp.actions import (
    SONICActionCfg,
    SONICActionType,
)

POLICY_DIR = f"{POLICY_ASSET_DIR}/sonic"


@configclass
class SonicActionsCfg:
    """SONIC-specific action configuration."""

    joint_pos = SONICActionCfg(
        action_type=SONICActionType.LATENT,
        policy_dir=POLICY_DIR,
        asset_name="robot",
        joint_names=[".*"],
        sonic_joint_names=MISSING,
        command_name="motion",
        use_default_offset=True,
    )


@configclass
class SonicObservationsCfg(BaseObservationsCfg):
    """SONIC observation configuration."""

    # SONIC tokenizer observations
    sonic_tokenizer: ObsGroup = MISSING

    # SONIC policy observations
    sonic_policy: ObsGroup = MISSING

    # Hand policy observations (optional)
    hand_policy: ObsGroup | None = None


@configclass
class SonicEnvCfg(V2PEnvCfg):
    """SONIC controller environment configuration.

    Extends the base V2P whole-body configuration with SONIC-specific actions and observations.
    """

    actions: SonicActionsCfg = SonicActionsCfg()
    observations: SonicObservationsCfg = SonicObservationsCfg()
